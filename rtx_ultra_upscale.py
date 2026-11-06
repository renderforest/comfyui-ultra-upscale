"""
Ultra Upscale — one-node video/image upscaler built on NVIDIA RTX Video Super Resolution.

Per frame, entirely on the GPU:
    [denoise @ source size] → [RTX upscale] → [supersample downscale] → [deblur @ output size]
    → [luma unsharp mask with noise gate + halo guard] → [film grain]
Only the finished frame leaves the GPU, so a full chain costs about as much as a single RTX node.
"""

import math
import time
from contextlib import ExitStack, nullcontext

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils

try:
    import nvvfx  # shipped by the `nvidia-vfx` package
    _NVVFX_ERROR = None
except Exception as e:  # pragma: no cover - depends on the host
    nvvfx = None
    _NVVFX_ERROR = e


# --------------------------------------------------------------------------- SDK levels
_FALLBACK_LEVELS = [
    "BICUBIC", "LOW", "MEDIUM", "HIGH", "ULTRA",
    "DENOISE_LOW", "DENOISE_MEDIUM", "DENOISE_HIGH", "DENOISE_ULTRA",
    "DEBLUR_LOW", "DEBLUR_MEDIUM", "DEBLUR_HIGH", "DEBLUR_ULTRA",
    "HIGHBITRATE_LOW", "HIGHBITRATE_MEDIUM", "HIGHBITRATE_HIGH", "HIGHBITRATE_ULTRA",
    "STREAMING_MEDIUM", "STREAMING_ULTRA",
]


def _sdk_levels():
    if nvvfx is not None:
        try:
            return list(nvvfx.QualityLevel.__members__)
        except Exception:
            pass
    return list(_FALLBACK_LEVELS)


_LEVELS = _sdk_levels()
_UPSCALE_ORDER = ["HIGHBITRATE_ULTRA", "HIGHBITRATE_HIGH", "HIGHBITRATE_MEDIUM", "HIGHBITRATE_LOW",
                  "ULTRA", "HIGH", "MEDIUM", "LOW", "STREAMING_ULTRA", "STREAMING_MEDIUM", "BICUBIC"]
UPSCALE_LEVELS = [l for l in _UPSCALE_ORDER if l in _LEVELS] + \
                 [l for l in _LEVELS if l not in _UPSCALE_ORDER and not l.startswith(("DENOISE_", "DEBLUR_"))]
DENOISE_LEVELS = ["off"] + [l for l in _LEVELS if l.startswith("DENOISE_")]
DEBLUR_LEVELS = ["off"] + [l for l in _LEVELS if l.startswith("DEBLUR_")]


# --------------------------------------------------------------------------- presets
# Built on NVIDIA's own guidance for the VSR models:
#  · HIGHBITRATE_*  — clean / high-bitrate sources, no unnecessary artifact suppression (keeps texture)
#  · ULTRA          — typical compressed footage (web, phone, social)
#  · DENOISE_*      — same-size noise & compression clean-up; higher levels soften texture → run BEFORE upscaling
#  · DEBLUR_*       — same-size sharpening for soft sources → run AFTER upscaling, at the output size
PRESETS = {
    "Clean · Natural": dict(   # AI-generated / camera-original; faithful, barely touched
        denoise="off", denoise_strength=0.0,
        upscale="HIGHBITRATE_ULTRA", upscale_strength=1.0,
        deblur="off", deblur_strength=0.0,
        sharpen=0.15, radius=1.0, gate=0.010, halo_guard=True, grain=0.0, supersample=False),
    "Clean · Crisp": dict(     # default: clean source, visibly sharper
        denoise="off", denoise_strength=0.0,
        upscale="HIGHBITRATE_ULTRA", upscale_strength=1.0,
        deblur="DEBLUR_LOW", deblur_strength=0.5,
        sharpen=0.30, radius=1.0, gate=0.010, halo_guard=True, grain=0.0, supersample=False),
    "Web · Compressed": dict(  # YouTube, phone, social re-uploads: blocks and smear
        denoise="DENOISE_LOW", denoise_strength=0.4,
        upscale="ULTRA", upscale_strength=1.0,
        deblur="DEBLUR_LOW", deblur_strength=0.35,
        sharpen=0.25, radius=1.0, gate=0.015, halo_guard=True, grain=0.0, supersample=False),
    "Noisy · Low light": dict( # grain/sensor noise: clean first, sharpen only real edges, add back fine grain
        denoise="DENOISE_MEDIUM", denoise_strength=0.8,
        upscale="ULTRA", upscale_strength=1.0,
        deblur="off", deblur_strength=0.0,
        sharpen=0.15, radius=1.2, gate=0.030, halo_guard=True, grain=0.012, supersample=False),
    "Max detail": dict(        # clean source, maximum crispness: supersampled RTX + stronger deblur
        denoise="off", denoise_strength=0.0,
        upscale="HIGHBITRATE_ULTRA", upscale_strength=1.0,
        deblur="DEBLUR_MEDIUM", deblur_strength=0.6,
        sharpen=0.35, radius=0.8, gate=0.008, halo_guard=True, grain=0.008, supersample=True),
}
AUTO = "Detect Mode"
CUSTOM = "Custom"
PRESET_NAMES = [AUTO] + list(PRESETS) + [CUSTOM]
TARGETS = ["720p", "1080p", "1440p (2K)"]
SUPERSAMPLE = 1.25
MAX_SIDE = 8192
CHUNK_PIXELS = 1024 * 1024 * 16
PIN_LIMIT_BYTES = 2 * 1024 ** 3
EDGE_PROTECT_LOW, EDGE_PROTECT_HIGH = 0.15, 0.45   # local luma range: full sharpening below, none above


# --------------------------------------------------------------------------- helpers
def _r8(v):
    return max(8, int(round(v / 8.0)) * 8)


def _output_size(w, h, target):
    """Short-side presets keep the aspect ratio: 1080p → 1920×1080, 1080×1920, 1080×1080."""
    if target.endswith("p") or "(2K)" in target or "(4K)" in target:
        side = int(target.split("p")[0])
        if w == h:
            return side, side, side / w
        if w > h:
            return _r8(w * side / h), side, side / h
        return side, _r8(h * side / w), side / w
    if target.startswith("x"):
        s = float(target[1:])
        return _r8(w * s), _r8(h * s), s
    return w, h, 1.0


def _device():
    device = mm.get_torch_device()
    if device.type != "cuda":
        raise RuntimeError("Ultra Upscale needs an NVIDIA RTX GPU (CUDA device).")
    return device, (device.index if device.index is not None else torch.cuda.current_device())


def _pack_rgb10a2(hwc):
    rgb = hwc.clamp(0.0, 1.0).mul(1023.0).round().to(torch.int32)
    packed = rgb[..., 0] | (rgb[..., 1] << 10) | (rgb[..., 2] << 20) | -1073741824
    return packed.view(torch.uint32).contiguous()


def _unpack_rgb10a2(packed):
    packed = packed.view(torch.int32)
    r = packed.bitwise_and(0x3FF)
    g = packed.bitwise_right_shift(10).bitwise_and(0x3FF)
    b = packed.bitwise_right_shift(20).bitwise_and(0x3FF)
    return torch.stack((r, g, b), dim=-1).to(torch.float32).div_(1023.0)


def _make_effect(stack, level, strength, out_w, out_h, device_index, ten_bit):
    eff = stack.enter_context(nvvfx.VideoSuperRes(
        quality=nvvfx.QualityLevel[level],
        strength=float(strength),
        device=device_index,
        image_encoding=nvvfx.ImageEncoding.RGB10A2 if ten_bit else nvvfx.ImageEncoding.RGB8,
    ))
    eff.output_width = int(out_w)
    eff.output_height = int(out_h)
    eff.load()
    return eff


def _run_effect(eff, chw, ten_bit):
    """chw: float32 [3,H,W] on CUDA → float32 [3,H',W'] on CUDA."""
    if ten_bit:
        out = torch.from_dlpack(eff.run(_pack_rgb10a2(chw.movedim(0, -1))).image).clone()
        return _unpack_rgb10a2(out).movedim(-1, 0).contiguous()
    out = torch.from_dlpack(eff.run(chw.contiguous()).image).clone()
    return out.to(torch.float32).contiguous()


def _luma(chw):
    return (0.2126 * chw[0] + 0.7152 * chw[1] + 0.0722 * chw[2])[None, None]


_KERNELS = {}


def _gauss(sigma, device):
    key = (round(float(sigma), 4), str(device))
    if key not in _KERNELS:
        r = max(1, int(math.ceil(3.0 * sigma)))
        x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
        k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
        _KERNELS[key] = ((k / k.sum()), r)
    return _KERNELS[key]


def _blur(y, sigma):
    k, r = _gauss(sigma, y.device)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))


def _sharpen(chw, amount, sigma, gate, halo_guard, protect_edges=True):
    """Luma-only unsharp mask.
    · noise gate   — detail below `gate` is not sharpened, so noise and flicker are not amplified
    · edge protect — contrast-adaptive (CAS-style): full strength on low-contrast texture (skin, fabric, hair),
                     fading to zero on edges that are already hard (text, logos, UI, thin lines)
    · halo guard   — clamps overshoot to the local 3×3 range, so edges get no bright/dark outline
    """
    if amount <= 0:
        return chw
    y = _luma(chw)
    detail = y - _blur(y, sigma)
    if gate > 0:
        t = ((detail.abs() - gate) / gate).clamp(0.0, 1.0)
        detail = detail * (t * t * (3.0 - 2.0 * t))
    need_range = halo_guard or protect_edges
    if need_range:
        pad = F.pad(y, (2, 2, 2, 2), mode="replicate")
        mx = F.max_pool2d(pad, 5, 1)
        mn = -F.max_pool2d(-pad, 5, 1)
    if protect_edges:
        t = (((mx - mn) - EDGE_PROTECT_LOW) / (EDGE_PROTECT_HIGH - EDGE_PROTECT_LOW)).clamp(0.0, 1.0)
        detail = detail * (1.0 - t * t * (3.0 - 2.0 * t))
    ys = y + amount * detail
    if halo_guard:
        ys = torch.maximum(torch.minimum(ys, mx + 0.01), mn - 0.01)
    return (chw + (ys - y)[0]).clamp_(0.0, 1.0)


def _grain(chw, amount, generator):
    """Fine luma grain, strongest in mid-tones, none in clipped blacks/whites."""
    if amount <= 0:
        return chw
    _, h, w = chw.shape
    n = torch.randn((1, 1, h, w), generator=generator, device=chw.device)
    n = _blur(n, 0.6)
    n = n / n.std().clamp_min(1e-6)
    y = _luma(chw)
    return (chw + (amount * n * (4.0 * y * (1.0 - y)))[0]).clamp_(0.0, 1.0)


def _analyse(images, device):
    """Robust noise sigma (median Immerkær) and 8×8 blockiness, measured on up to 5 frames."""
    b = images.shape[0]
    idx = sorted({int(round(i * (b - 1) / 4)) for i in range(5)}) if b > 1 else [0]
    sigmas, blocks = [], []
    lap = torch.tensor([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]], device=device).view(1, 1, 3, 3)
    for i in idx:
        f = images[i, ..., :3].to(device=device, dtype=torch.float32).movedim(-1, 0)
        y = _luma(f) * 255.0
        resp = F.conv2d(y, lap).abs()
        sigmas.append(float(resp.median()) / (0.6745 * 6.0))
        dh = (y[..., :, 1:] - y[..., :, :-1]).abs()
        dv = (y[..., 1:, :] - y[..., :-1, :]).abs()
        ratios = []
        for d, dim in ((dh, -1), (dv, -2)):
            n = d.shape[dim]
            if n < 16:
                continue
            pos = torch.arange(n, device=device)
            mask = (pos % 8) == 7
            on = d.index_select(dim, pos[mask]).mean()
            off = d.index_select(dim, pos[~mask]).mean().clamp_min(1e-6)
            ratios.append(float(on / off))
        if ratios:
            blocks.append(sum(ratios) / len(ratios))
    sigma = sorted(sigmas)[len(sigmas) // 2]
    block = sorted(blocks)[len(blocks) // 2] if blocks else 1.0
    if sigma > 3.0:
        choice = "Noisy · Low light"
    elif block > 1.12:
        choice = "Web · Compressed"
    else:
        choice = "Clean · Crisp"
    return choice, sigma, block


# --------------------------------------------------------------------------- node
class RTXUltraUpscale:
    DESCRIPTION = ("One-node RTX upscaler: denoise → RTX Video Super Resolution → deblur → edge-aware sharpen → grain, "
                   "all on the GPU per frame. 5 presets + Detect Mode + Custom. Images or video frames in, upscaled frames out.")
    CATEGORY = "image/upscaling"
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "preset": (PRESET_NAMES, {"default": "Clean · Crisp", "tooltip":
                    "Detect Mode — measures noise and compression blocks, then picks a preset.\n"
                    "Clean · Natural — AI / camera-original, faithful.\n"
                    "Clean · Crisp — clean source, visibly sharper (default).\n"
                    "Web · Compressed — YouTube, phone, social re-uploads.\n"
                    "Noisy · Low light — sensor noise, night, dark rooms.\n"
                    "Max detail — clean source, maximum crispness (supersampled, a bit slower).\n"
                    "Custom — uses the custom_* settings below."}),
                "target": (TARGETS, {"default": "1080p", "tooltip":
                    "720p / 1080p / 1440p (2K) set the SHORT side, aspect kept: 1080p → 1920×1080, 1080×1920, 1080×1080."}),
            },
            "optional": {
                "custom_denoise": (DENOISE_LEVELS, {"default": "off", "tooltip": "Runs at source size, before upscaling."}),
                "custom_denoise_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "custom_upscale_model": (UPSCALE_LEVELS, {"default": "HIGHBITRATE_ULTRA", "tooltip":
                    "HIGHBITRATE_* — clean sources · ULTRA/HIGH… — compressed · STREAMING_* — webcam/broadcast · BICUBIC — non-AI reference."}),
                "custom_upscale_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "custom_deblur": (DEBLUR_LEVELS, {"default": "DEBLUR_LOW", "tooltip": "Runs at output size, after upscaling."}),
                "custom_deblur_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "custom_sharpen": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Luma unsharp mask amount."}),
                "custom_sharpen_radius": ("FLOAT", {"default": 1.0, "min": 0.4, "max": 3.0, "step": 0.1, "tooltip": "Gaussian sigma in output pixels."}),
                "custom_noise_gate": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.1, "step": 0.002, "tooltip":
                    "Detail smaller than this is not sharpened — keeps noise and flicker from being amplified."}),
                "custom_halo_guard": ("BOOLEAN", {"default": True, "tooltip": "Clamps overshoot so edges get no bright/dark outline."}),
                "custom_grain": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.08, "step": 0.002, "tooltip": "Fine mid-tone film grain."}),
                "custom_supersample": ("BOOLEAN", {"default": False, "tooltip": "RTX upscales 1.25× past the target, then downsamples: cleaner edges, finer detail."}),
                "ten_bit": ("BOOLEAN", {"default": False, "tooltip": "10-bit processing inside the SDK: smoother gradients, slightly slower."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31 - 1, "tooltip": "Grain pattern seed."}),
                "sharpness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05, "tooltip":
                    "Works on top of ANY preset, Detect Mode included.\n"
                    "+0.3 — a bit crisper · +0.6 — clearly crisper · +1.0 — maximum · negative — softer.\n"
                    "Scales the sharpen amount and the RTX deblur strength."}),
                "protect_edges": ("BOOLEAN", {"default": True, "tooltip":
                    "Contrast-adaptive sharpening: texture gets crisper, hard edges stay as they are.\n"
                    "Keeps text, logos, UI and thin lines from thickening, stair-stepping or getting outlines."}),
            },
        }

    def upscale(self, images, preset, target, custom_denoise="off", custom_denoise_strength=0.5,
                custom_upscale_model="HIGHBITRATE_ULTRA", custom_upscale_strength=1.0,
                custom_deblur="DEBLUR_LOW", custom_deblur_strength=0.5,
                custom_sharpen=0.3, custom_sharpen_radius=1.0, custom_noise_gate=0.01,
                custom_halo_guard=True, custom_grain=0.0, custom_supersample=False,
                ten_bit=False, seed=0, sharpness=0.0, protect_edges=True):
        if nvvfx is None:
            raise RuntimeError(f"nvvfx is not installed (pip install nvidia-vfx): {_NVVFX_ERROR}")
        t0 = time.perf_counter()
        device, device_index = _device()

        alpha = images[..., 3:4] if images.shape[-1] > 3 else None
        rgb = images[..., :3]
        b, h, w, _ = rgb.shape

        auto_note = ""
        if preset == AUTO:
            chosen, sigma, block = _analyse(rgb, device)
            auto_note = f"detect: noise σ={sigma:.2f}, blockiness {block:.2f} → "
            p = dict(PRESETS[chosen]); preset_label = chosen
        elif preset == CUSTOM:
            p = dict(denoise=custom_denoise, denoise_strength=custom_denoise_strength,
                     upscale=custom_upscale_model, upscale_strength=custom_upscale_strength,
                     deblur=custom_deblur, deblur_strength=custom_deblur_strength,
                     sharpen=custom_sharpen, radius=custom_sharpen_radius, gate=custom_noise_gate,
                     halo_guard=custom_halo_guard, grain=custom_grain, supersample=custom_supersample)
            preset_label = CUSTOM
        else:
            p = dict(PRESETS[preset]); preset_label = preset

        # global sharpness on top of the preset: sharpen amount ×(1+s), deblur strength ×(1+s/2), capped at 1
        sharpness = max(-1.0, min(1.0, float(sharpness)))
        if sharpness:
            p["sharpen"] = max(0.0, p["sharpen"] * (1.0 + sharpness))
            if p["deblur"] != "off":
                p["deblur_strength"] = max(0.0, min(1.0, p["deblur_strength"] * (1.0 + 0.5 * sharpness)))

        out_w, out_h, scale = _output_size(w, h, target)
        out_w, out_h = min(out_w, MAX_SIDE), min(out_h, MAX_SIDE)

        # RTX stage size: up to 4× source, optional 1.25× supersample, never above 8192
        upscaling = out_w > w or out_h > h
        downscaling = out_w < w and out_h < h
        rtx_w, rtx_h = out_w, out_h
        if upscaling:
            k = SUPERSAMPLE if p["supersample"] else 1.0
            k = min(k, (4.0 * w) / out_w, (4.0 * h) / out_h, MAX_SIDE / out_w, MAX_SIDE / out_h)
            k = max(k, min(1.0, (4.0 * w) / out_w, (4.0 * h) / out_h))  # >4× targets: RTX does 4×, resize does the rest
            rtx_w, rtx_h = _r8(out_w * k), _r8(out_h * k)
        resize_after = (rtx_w, rtx_h) != (out_w, out_h)

        # Page-locked RAM speeds up GPU→CPU copies but cannot be swapped out, so only small outputs get it:
        # pinning a multi-GB video would starve the OS of memory and can freeze the machine.
        out_bytes = b * out_h * out_w * 3 * 4
        output = None
        if out_bytes <= PIN_LIMIT_BYTES:
            try:
                output = torch.empty((b, out_h, out_w, 3), dtype=torch.float32, pin_memory=True)
            except Exception:
                output = None
        pinned = output is not None
        if output is None:
            output = torch.empty((b, out_h, out_w, 3), dtype=torch.float32)

        gen = torch.Generator(device=device)
        pbar = comfy.utils.ProgressBar(b)
        chunk = max(1, CHUNK_PIXELS // max(1, w * h))

        with (torch.cuda.device(device) if device.type == "cuda" else nullcontext()), ExitStack() as stack:
            den = _make_effect(stack, p["denoise"], p["denoise_strength"], w, h, device_index, ten_bit) \
                if p["denoise"] != "off" and p["denoise_strength"] > 0 else None
            up = _make_effect(stack, p["upscale"], p["upscale_strength"], rtx_w, rtx_h, device_index, ten_bit) \
                if upscaling else None
            deb = _make_effect(stack, p["deblur"], p["deblur_strength"], out_w, out_h, device_index, ten_bit) \
                if p["deblur"] != "off" and p["deblur_strength"] > 0 else None

            for i in range(0, b, chunk):
                batch = rgb[i:i + chunk].to(device=device, dtype=torch.float32, non_blocking=True)
                for j in range(batch.shape[0]):
                    mm.throw_exception_if_processing_interrupted()
                    f = batch[j].movedim(-1, 0).contiguous()
                    if den is not None:
                        f = _run_effect(den, f, ten_bit)
                    if up is not None:
                        f = _run_effect(up, f, ten_bit)
                    if resize_after or downscaling:
                        # antialiased bicubic in both directions: "area" at non-integer ratios averages uneven pixel
                        # windows in a repeating pattern, which deblur/sharpen then turn into a visible block grid
                        f = F.interpolate(f[None], size=(out_h, out_w), mode="bicubic",
                                          align_corners=False, antialias=True)[0].clamp_(0.0, 1.0)
                    if deb is not None:
                        f = _run_effect(deb, f.clamp(0.0, 1.0), ten_bit)
                    f = _sharpen(f, p["sharpen"], p["radius"], p["gate"], p["halo_guard"], protect_edges)
                    if p["grain"] > 0:
                        gen.manual_seed(int(seed) + i + j)
                        f = _grain(f, p["grain"], gen)
                    output[i + j].copy_(f.clamp_(0.0, 1.0).movedim(0, -1), non_blocking=pinned)
                    pbar.update(1)
                del batch
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        if alpha is not None:
            a = F.interpolate(alpha.movedim(-1, 1).float(), size=(out_h, out_w), mode="bilinear", align_corners=False)
            output = torch.cat((output, a.movedim(1, -1).clamp(0, 1)), dim=-1)

        dt = time.perf_counter() - t0
        stages = [
            f"denoise {p['denoise']} {p['denoise_strength']:.2f}" if den is not None else "denoise off",
            (f"{p['upscale']} → {rtx_w}×{rtx_h}" + (" (supersampled)" if p["supersample"] and resize_after else "")) if up is not None else "no RTX upscale",
            f"deblur {p['deblur']} {p['deblur_strength']:.2f}" if deb is not None else "deblur off",
            f"sharpen {p['sharpen']:.2f} r{p['radius']:.1f} gate {p['gate']:.3f}" + (" halo-guard" if p["halo_guard"] else "") + (" edge-protect" if protect_edges else ""),
            f"grain {p['grain']:.3f}" if p["grain"] > 0 else "no grain",
        ]
        note = ""
        if downscaling:
            note = (f"\nnote: source {w}×{h} is already larger than {target} — it was downscaled, not upscaled. "
                    "Pick a bigger target (1440p (2K)) to upscale.")
        info = (f"{auto_note}{preset_label}" + (f" · sharpness {sharpness:+.2f}" if sharpness else "") + f" | {w}×{h} → {out_w}×{out_h} | {b} frame(s) | "
                f"{dt:.1f} s ({b / dt:.1f} fps)\n" + " · ".join(stages) + note)
        print(f"[Ultra Upscale] {info}")
        return (output, info)


NODE_CLASS_MAPPINGS = {"RTXUltraUpscale": RTXUltraUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"RTXUltraUpscale": "Ultra Upscale ⚡"}
