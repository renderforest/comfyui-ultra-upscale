"""
SeedVR2 Ultra Upscale — one node: Detect Mode → official SeedVR2 pipeline → temporal stabiliser → RTX finish.

· Detect Mode measures each video (skin share, fine texture, motion, how the clip starts, source softness)
  and picks every setting for that clip; the choice is printed in `info`.
· SeedVR2 runs exactly the official native ComfyUI way (preprocess → tiled VAE → temporal chunks →
  conditioning → 1-step sampler → merge → tiled decode → colour post-processing), so memory stays managed by ComfyUI.
· The stabiliser removes SeedVR2's 4-frame shimmer: the detail SeedVR2 added is averaged over a 5-frame window
  only where the picture is still (where shimmer is visible); moving areas keep their per-frame detail (no ghosting).
  The source returns on soft / out-of-focus areas and on busy moving texture; source colour is locked at large scale.
· Finish: edge-safe sharpen at the output size — halo guard + edge protect, so edges get crisper without outlines.
"""
import math
import time
from contextlib import ExitStack

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils

from .rtx_ultra_upscale import _blur, _luma, _sharpen, _r8, _make_effect, _run_effect, nvvfx
from .detail_mix import _detail_map

TARGETS = ["720p", "1080p", "1440p (2K)"]
MEDIUM, STRONG = "Detect · Medium", "Detect · Strong"
MODES = [MEDIUM, STRONG]
COLORS = ["lab", "wavelet", "adain", "none"]

# Detect Mode measures each clip and sets every parameter for it; the strength level then shifts the whole set:
#   Gentle — maximum smoothness · Medium — balance · Strong — maximum upscale effect.
# The temporal stabiliser stays fully on at every level, so stronger levels add detail without bringing the 4-frame shimmer back.
# Base values tuned on real SeedVR2 output of Seedance clips (face in fast motion, slow foliage) against the source's own 4-frame unevenness.


def _ss(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


def _final_size(w, h, target):
    side = int(target.split("p")[0])
    if w == h:
        return side, side
    if w > h:
        return _r8(w * side / h), side
    return side, _r8(h * side / w)


# --------------------------------------------------------------------------- analysis
@torch.no_grad()
def analyse_video(frames, device):
    """Metrics at a fixed 360-px short side so they mean the same for any resolution."""
    n, h, w, _ = frames.shape
    s = 360.0 / min(h, w)
    size = (max(16, int(round(h * s))), max(16, int(round(w * s))))
    def small(i):
        f = frames[i, ..., :3].to(device, torch.float32).movedim(-1, 0)[None]
        return F.interpolate(f, size=size, mode="bicubic", align_corners=False, antialias=True)[0].clamp(0, 1)
    step = max(1, (n - 1) // 120)
    tc, tx, mo = [], [], []
    prev = small(0)
    for i in range(0, n - 1, step):
        a = prev if i == 0 or step == 1 else small(i)
        b = small(i + 1)
        ya, yb = _luma(a), _luma(b)
        ha, hb = ya - _blur(ya, 1.0), yb - _blur(yb, 1.0)
        tc.append(float((hb - ha).abs().mean())); tx.append(float(ha.abs().mean())); mo.append(float((yb - ya).abs().mean()))
        prev = b
    skin = []
    for i in range(0, n, max(1, n // 12)):
        f = small(i) * 255.0
        R, G, B = f[0], f[1], f[2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
        Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
        skin.append(float(((Cb > 77) & (Cb < 127) & (Cr > 133) & (Cr < 173) & (Y > 40)).float().mean()))
    k = max(1, min(8, len(mo)))
    mean_mo = sum(mo) / max(1, len(mo))
    return dict(
        sharp=255.0 * sum(tx) / max(1, len(tx)),               # fine texture energy (source softness)
        motion=255.0 * mean_mo,                                 # overall motion
        skin=sorted(skin)[len(skin) // 2] if skin else 0.0,     # share of skin pixels (faces, people)
        start=(sum(mo[:k]) / k) / max(mean_mo, 1e-6),           # how lively the first frames are vs the rest
        frames=n,
    )


def detect_params(m):
    busy = _ss(1.6, 2.6, m["sharp"]) * (1.0 - _ss(8.0, 20.0, m["motion"]))   # textured AND slow → shimmer is most visible
    face = _ss(0.03, 0.20, m["skin"])
    soft = 1.0 - _ss(1.2, 2.4, m["sharp"])
    p = dict(
        mix=0.07 + 0.15 * busy + 0.08 * face,
        stab=1.0, m0=0.03, m1=0.16,
        guard=0.10 + 0.25 * busy,
        protect=0.25 + 0.20 * face,
        denoise=max(0.85, min(1.0, 1.0 - 0.08 * (1.0 - soft) - 0.05 * face)),
        color="lab" if (face > 0.3 or busy > 0.3 or soft > 0.5) else "wavelet",
        warm=12 if m["start"] > 1.3 else (4 if m["start"] < 0.6 else 8),
        rtx_deblur=0.0,
        rtx_sharpen=max(0.0, 0.18 + 0.10 * soft - 0.06 * busy - 0.04 * face),
    )
    tags = []
    if face > 0.5: tags.append("people")
    if busy > 0.5: tags.append("busy texture")
    if soft > 0.5: tags.append("soft source")
    if m["motion"] > 12: tags.append("fast motion")
    if m["start"] > 1.3: tags.append("lively start")
    return p, (", ".join(tags) or "general")


def apply_level(p, level):
    """One upscale style at two strengths (like a pro upscaler's strength slider):
    SeedVR2's own detail kept and lifted, mid-scale clarity, crisp edges — the stabiliser fully on, and the whole finish
    clamped to the local range so there are no outlines. Medium is the same look, gentler; Strong is the full effect."""
    p = dict(p)
    p["denoise"] = 1.0
    p["stab"], p["m0"], p["m1"] = 1.0, 0.040, 0.220
    p["lock"] = 0.0                                       # colour is kept by SeedVR2's lab post-processing
    if level == STRONG:
        p["mix"], p["protect"], p["guard"] = 0.02, 0.0, p["guard"] * 0.3
        p["rtx_sharpen"] = p["rtx_sharpen"] + 0.35
        p.update(edge_lo=0.50, edge_hi=0.90, gain=1.15, clarity=0.30, sharp_sigma=0.9, gate=0.008)
    else:  # MEDIUM
        p["mix"], p["protect"], p["guard"] = 0.05, p["protect"] * 0.25, p["guard"] * 0.5
        p["rtx_sharpen"] = p["rtx_sharpen"] + 0.22
        p.update(edge_lo=0.40, edge_hi=0.85, gain=1.07, clarity=0.15, sharp_sigma=1.0, gate=0.010)
    return p


def apply_sharpness(p, sharpness):
    if not sharpness:
        return p
    p = dict(p)
    p["mix"] = max(0.0, min(0.6, p["mix"] * (1.0 - 0.5 * sharpness)))
    p["rtx_sharpen"] = max(0.0, p["rtx_sharpen"] * (1.0 + sharpness))
    return p


def pick_tile(device):
    try:
        free = mm.get_free_memory(device) / 1024 ** 3
    except Exception:
        free = 0.0
    return (1024 if free >= 60 else 768 if free >= 36 else 512), free


# --------------------------------------------------------------------------- official SeedVR2 path
def _args(out):
    return out.args if hasattr(out, "args") else out


def run_seedvr2(resized, model, vae, p, seed, tile):
    """Exactly the official native graph: preprocess → tiled VAE → chunks → conditioning → 1 step → merge → decode → colour."""
    import nodes
    from comfy_extras.nodes_seedvr import (SeedVR2Preprocess, SeedVR2Conditioning, SeedVR2TemporalChunk,
                                           SeedVR2TemporalMerge, SeedVR2PostProcessing)
    padded = _args(SeedVR2Preprocess.execute(resized))[0]
    latent = nodes.VAEEncodeTiled().encode(vae, padded, tile, 128, 64, 8)[0]
    chunks, overlap = _args(SeedVR2TemporalChunk.execute(latent, 2, {"chunking_mode": "auto"}))
    sampled = []
    for chunk in chunks:
        mm.throw_exception_if_processing_interrupted()
        pos, neg = _args(SeedVR2Conditioning.execute(model, chunk))
        sampled.append(nodes.common_ksampler(model, int(seed), 1, 1.0, "euler", "simple", pos, neg, chunk, denoise=float(p["denoise"]))[0])
    merged = _args(SeedVR2TemporalMerge.execute(sampled, [overlap]))[0]
    decoded = nodes.VAEDecodeTiled().decode(vae, merged, tile, 128, 64, 8)[0]
    out = _args(SeedVR2PostProcessing.execute(decoded, resized, p["color"]))[0]
    if out.dim() == 5:
        out = out.reshape(-1, *out.shape[-3:])
    return out, len(chunks)


# --------------------------------------------------------------------------- finishing tools
def _local_range(y, k):
    pad = F.pad(y, (k // 2,) * 4, mode="replicate")
    return F.max_pool2d(pad, k, 1), -F.max_pool2d(-pad, k, 1)


def _crisp(chw, amount, sigma, gate, edge_lo, edge_hi, k=5):
    """Luma unsharp mask; noise gate; contrast-adaptive fade only on the hardest edges; overshoot clamped to the local range."""
    if amount <= 0:
        return chw
    y = _luma(chw)
    detail = y - _blur(y, sigma)
    if gate > 0:
        t = ((detail.abs() - gate) / gate).clamp(0.0, 1.0)
        detail = detail * (t * t * (3.0 - 2.0 * t))
    mx, mn = _local_range(y, k)
    if edge_hi > edge_lo:
        t = (((mx - mn) - edge_lo) / (edge_hi - edge_lo)).clamp(0.0, 1.0)
        detail = detail * (1.0 - t * t * (3.0 - 2.0 * t))
    ys = torch.maximum(torch.minimum(y + amount * detail, mx + 0.01), mn - 0.01)
    return (chw + (ys - y)[0]).clamp_(0.0, 1.0)


# --------------------------------------------------------------------------- stabiliser + mix + finish
_TAPS = (1.0, 2.0, 3.0, 2.0, 1.0)


@torch.no_grad()
def stabilise_and_finish(up, src, p, device, rtx=True, progress=None):
    """up, src: [N,H,W,3] same size. Returns finished frames on CPU."""
    n, h, w, _ = up.shape
    out = torch.empty((n, h, w, 3), dtype=torch.float32)
    scale = max(1.0, min(h, w) / 1080.0)
    cache = {}

    def S(i):
        i = min(max(i, 0), n - 1)
        if ("s", i) not in cache:
            cache[("s", i)] = src[i, ..., :3].to(device, torch.float32).movedim(-1, 0)
        return cache[("s", i)]

    def R(i):
        i = min(max(i, 0), n - 1)
        if ("r", i) not in cache:
            cache[("r", i)] = up[i, ..., :3].to(device, torch.float32).movedim(-1, 0) - S(i)
        return cache[("r", i)]

    use_rtx = bool(rtx)
    with ExitStack() as stack:
        deb = None
        for t in range(n):
            mm.throw_exception_if_processing_interrupted()
            for key in [k for k in cache if k[1] < t - 2]:
                del cache[key]
            s = S(t)
            # 1) temporal stabiliser on the added detail — only where the picture is still
            r_avg = sum(wt * R(t + k) for wt, k in zip(_TAPS, range(-2, 3))) / sum(_TAPS)
            motion = _blur((_luma(S(t + 1)) - _luma(S(t - 1))).abs(), 2.0 * scale)
            m0, m1 = p.get("m0", 0.012), p.get("m1", 0.05)
            still = 1.0 - ((motion - m0) / (m1 - m0)).clamp(0, 1).pow(2).mul(-2).add(3).mul(((motion - m0) / (m1 - m0)).clamp(0, 1).pow(2))
            a = (p["stab"] * still)[0]
            r = a * r_avg + (1.0 - a) * R(t)
            # 2) where the source comes back: soft / out-of-focus areas, busy moving texture, and a base share everywhere
            d = _detail_map(s, scale)[0]
            busy = (1.0 - still[0]) * d
            share = (p["mix"] + (1.0 - p["mix"]) * p["protect"] * (1.0 - d) + p["guard"] * busy).clamp(0.0, 0.95)
            gain = 1.0 + (p.get("gain", 1.0) - 1.0) * d            # lift SeedVR2's own detail where the source has detail
            o = (s + r * (1.0 - share) * gain).clamp(0.0, 1.0)
            # 3) optional colour/tone lock at large scale (lab inside SeedVR2 already keeps colour)
            if p.get("lock", 6.0) > 0:
                sig = p.get("lock", 6.0) * scale
                o = (o + (_blur(s[:, None], sig) - _blur(o[:, None], sig))[:, 0]).clamp(0.0, 1.0)
            # 4) RTX finish
            if deb is not None:
                o = _run_effect(deb, o.contiguous(), False).clamp(0.0, 1.0)
            if use_rtx and (p.get("clarity", 0.0) > 0 or p["rtx_sharpen"] > 0):
                y0 = _luma(o)
                fmx, fmn = _local_range(y0, 5)                    # one shared limit for the whole finish: no outlines
                if p.get("clarity", 0.0) > 0:
                    o = _crisp(o, p["clarity"], 2.5 * scale, 0.004, 0.55, 0.95, k=7)      # mid-scale clarity
                if p["rtx_sharpen"] > 0:
                    o = _crisp(o, p["rtx_sharpen"], p.get("sharp_sigma", 1.0), p.get("gate", 0.012), p.get("edge_lo", 0.15), p.get("edge_hi", 0.45))
                y1 = _luma(o)
                y1c = torch.maximum(torch.minimum(y1, fmx + 0.01), fmn - 0.01)
                o = (o + (y1c - y1)[0]).clamp_(0.0, 1.0)
            out[t] = o.movedim(0, -1).cpu()
            if progress:
                progress()
    return out, use_rtx


# --------------------------------------------------------------------------- node
class SeedVR2UltraUpscale:
    DESCRIPTION = ("One-node SeedVR2 upscaler. Detect Mode reads each video and picks every setting, at two strengths (Medium / Strong); SeedVR2 runs the official "
                   "native way; a temporal stabiliser removes SeedVR2's 4-frame shimmer; an edge-safe finish adds crispness without outlines.")
    CATEGORY = "image/upscaling"
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Video frames (Get Video Components → images) or a single picture."}),
                "model": ("MODEL", {"tooltip": "Load Diffusion Model → seedvr2_7b_int8_convrot / 7b_fp16 / 3b_int8."}),
                "vae": ("VAE", {"tooltip": "Load VAE → seedvr2_ema_vae_fp16."}),
                "mode": (MODES, {"default": MEDIUM, "tooltip":
                    "Detect Mode measures every clip (people, fine texture, motion, how it starts, source softness) and sets everything for it.\n"
                    "Medium — the same upscale look, gentler (default).\nStrong — full effect: SeedVR2 detail lifted, mid-scale clarity, crisp edges — no outlines.\n"
                    "The 4-frame shimmer stabiliser is on at every level."}),
                "target": (TARGETS, {"default": "1080p", "tooltip": "Short side; aspect kept: 1080p → 1920×1080, 1080×1920, 1080×1080."}),
            },
            "optional": {
                "rtx_finish": ("BOOLEAN", {"default": True, "tooltip": "Edge-safe finishing sharpen: crisper detail with no outlines (halo guard + edge protect)."}),
                "sharpness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05, "tooltip": "Fine-tune on top of the level: + crisper, − softer."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2 ** 31 - 1}),
            },
        }

    def upscale(self, images, model, vae, mode, target, rtx_finish=True, sharpness=0.0, seed=42):
        t0 = time.perf_counter()
        device = mm.get_torch_device()
        frames = images[..., :3]
        n, h, w, _ = frames.shape

        metrics = analyse_video(frames, device)
        level = mode if mode in MODES else MEDIUM
        p, scene = detect_params(metrics)
        p = apply_level(p, level)
        head = f"{level} [{scene}]"
        p = apply_sharpness(p, sharpness)
        warm = int(p["warm"]) if n > 1 else 0
        tile, free_gb = pick_tile(device)

        out_w, out_h = _final_size(w, h, target)
        src = comfy.utils.common_upscale(frames.movedim(-1, 1), out_w, out_h, "lanczos", "disabled").movedim(1, -1).clamp(0, 1)
        if warm:
            resized = torch.cat([src[:1].repeat(warm, 1, 1, 1), src], dim=0)
        else:
            resized = src

        t1 = time.perf_counter()
        up, n_chunks = run_seedvr2(resized, model, vae, p, seed, tile)
        up = up[warm:warm + n].float().cpu()
        if up.shape[1:3] != src.shape[1:3]:
            up = F.interpolate(up.movedim(-1, 1), size=(out_h, out_w), mode="bicubic", align_corners=False, antialias=True).movedim(1, -1).clamp(0, 1)
        t2 = time.perf_counter()

        pbar = comfy.utils.ProgressBar(n)
        result, rtx_used = stabilise_and_finish(up, src, p, device, rtx=rtx_finish, progress=lambda: pbar.update(1))
        t3 = time.perf_counter()

        rtx_note = (("finish sharpen %.2f" % p["rtx_sharpen"]) + (" · clarity %.2f" % p["clarity"] if p.get("clarity") else "")
                    + (" · detail ×%.2f" % p["gain"] if p.get("gain", 1.0) != 1.0 else "") + (" · colour lock" if p.get("lock") else "")) if rtx_used else "finish off"
        info = (f"{head} | {w}×{h} → {out_w}×{out_h} | {n} frames | SeedVR2 {t2 - t1:.1f} s, finish {t3 - t2:.1f} s, total {t3 - t0:.1f} s\n"
                f"measured: skin {metrics['skin']:.2f} · texture {metrics['sharp']:.2f} · motion {metrics['motion']:.1f} · start {metrics['start']:.2f}\n"
                f"settings: denoise {p['denoise']:.2f} · colour {p['color']} · warm-up {warm} · mix {p['mix']:.2f} · stabilise {p['stab']:.2f} · "
                f"motion guard {p['guard']:.2f} · protect soft {p['protect']:.2f} · {rtx_note}"
                + (f" · sharpness {sharpness:+.2f}" if sharpness else "")
                + f"\nmemory: {free_gb:.0f} GB free → VAE tiles {tile} · {n_chunks} temporal chunk(s)")
        print(f"[SeedVR2 Ultra Upscale] {info}")
        return (result, info)
