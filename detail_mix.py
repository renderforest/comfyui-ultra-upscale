"""
Detail-Aware Mix — blends a restored/upscaled frame with the plain-resized source, region by region.

Where the source already has real detail (eyes, hair, edges, in-focus texture) the restored frame wins.
Where the source is smooth or out of focus (bokeh backgrounds, soft skin, sky) the source comes back,
so a generative upscaler cannot paint grain onto blur or over-texture skin. Runs on the GPU.
"""
import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.utils

from .rtx_ultra_upscale import _blur, _luma


def _detail_map(chw, scale):
    """0 = smooth / out of focus, 1 = clearly detailed. Relative to the frame, robust to exposure."""
    y = _luma(chw)
    hf = (y - _blur(y, 1.2 * scale)).abs()
    energy = _blur(hf, 4.0 * scale)
    ref = torch.quantile(energy.flatten()[:: max(1, energy.numel() // 200000)], 0.90).clamp_min(1e-4)
    t = ((energy / ref - 0.15) / (0.70 - 0.15)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class DetailAwareMix:
    DESCRIPTION = ("Mixes an upscaled result (SeedVR2, FlashVSR, RTX…) with the plain-resized source by region: keeps the new detail "
                   "where the source had detail, brings the source back on smooth skin and out-of-focus backgrounds.")
    CATEGORY = "image/upscaling"
    FUNCTION = "mix"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "detail_mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscaled": ("IMAGE", {"tooltip": "Output of the upscaler."}),
                "source": ("IMAGE", {"tooltip": "The same frames resized to the same size (e.g. the Resize node before SeedVR2)."}),
                "base_mix": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
                    "Source amount everywhere, detailed areas included. 0 = keep all new detail there."}),
                "protect_soft_areas": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
                    "How much of the source returns on smooth / out-of-focus areas: bokeh, soft skin, sky.\n"
                    "0.5 — a little · 0.75 — natural (default) · 1.0 — fully original there."}),
                "color_lock": ("BOOLEAN", {"default": True, "tooltip":
                    "Brings back the source colour and tone at large scale (no warmer skin, no contrast drift); fine detail untouched."}),
            },
        }

    def mix(self, upscaled, source, base_mix, protect_soft_areas, color_lock=True):
        device = mm.get_torch_device()
        b, h, w, c = upscaled.shape
        if source.shape[1:3] != upscaled.shape[1:3]:
            source = F.interpolate(source.movedim(-1, 1).float(), size=(h, w), mode="bicubic",
                                   align_corners=False, antialias=True).movedim(1, -1).clamp(0, 1)
        if source.shape[0] != b:
            raise ValueError(f"Detail-Aware Mix: {b} upscaled frames but {source.shape[0]} source frames.")
        out = torch.empty_like(upscaled, device="cpu")
        mask = torch.empty((b, h, w), dtype=torch.float32)
        scale = max(1.0, min(h, w) / 1080.0)
        pbar = comfy.utils.ProgressBar(b)
        for i in range(b):
            mm.throw_exception_if_processing_interrupted()
            up = upscaled[i, ..., :3].to(device, torch.float32).movedim(-1, 0)
            src = source[i, ..., :3].to(device, torch.float32).movedim(-1, 0)
            d = _detail_map(src, scale)[0, 0]
            src_w = base_mix + (1.0 - base_mix) * protect_soft_areas * (1.0 - d)
            res = up * (1.0 - src_w) + src * src_w
            if color_lock:
                sig = 6.0 * scale
                res = res + (_blur(src[:, None], sig) - _blur(res[:, None], sig))[:, 0]
            out[i, ..., :3] = res.clamp(0, 1).movedim(0, -1).cpu()
            if c > 3:
                out[i, ..., 3:] = upscaled[i, ..., 3:].cpu()
            mask[i] = d.cpu()
            pbar.update(1)
        return (out, mask)
