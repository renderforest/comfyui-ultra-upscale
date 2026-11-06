# Ultra Upscale ⚡ · SeedVR2 Ultra Upscale 🎬

## SeedVR2 Ultra Upscale 🎬 — one node for SeedVR2 video upscaling

`images` + `model` (Load Diffusion Model → `seedvr2_7b_int8_convrot`, `7b_fp16` or `3b_int8`) + `vae` (Load VAE → `seedvr2_ema_vae_fp16`) → upscaled `images` + `info`.

1. **Detect Mode** (Medium / Strong) measures every clip — skin share (people), fine texture, motion, how the clip starts, source softness — and picks all settings for that clip. The measurements and the choice are printed in `info`.
2. **SeedVR2 runs the official native way** inside the node: preprocess → tiled VAE → temporal chunks (auto, overlap 2) → conditioning → 1-step sampler → merge → tiled decode → colour post-processing. Memory is managed by ComfyUI; VAE tiles are picked from free VRAM (512 on 32 GB cards like the RTX 5090, 768 / 1024 on bigger ones).
3. **Warm-up**: copies of the first frame go in front (4–12, chosen per clip) and are cut off after — SeedVR2's unstable start lands on them.
4. **Temporal stabiliser**: SeedVR2 redraws detail in 4-frame groups, so detail jumps every 4 frames. The detail it added is averaged over a 5-frame window wherever the picture is still enough for the jump to be seen; fast-moving areas keep per-frame detail (no ghosting). The source comes back on soft / out-of-focus areas and on busy moving texture, and the source colour is locked at large scale.
5. **Finish**: edge-safe sharpen at the output size — halo guard + edge protect, crisper detail with no outlines. (No RTX deblur here: after SeedVR2 it adds outlines rather than detail.)

Measured on real SeedVR2 output of Seedance clips (4-frame unevenness of fine detail; the source's own level in brackets): face in fast motion 11.8 % → 6.7 % (3.2 %), slow foliage 21.3 % → 8.8 % (6.9 %).

**One mode — Detect — at two strengths.** Both measure the clip and adapt to it:

| level | what it does | measured on real clips (detail vs source · 4-frame unevenness) |
|---|---|---|
| **Detect · Medium** *(default)* | the upscale look, gentler: SeedVR2 detail lifted ×1.07, clarity 0.15, crisp edges | face ×1.44 · 6.7 % — foliage ×1.28 · 8.9 % |
| **Detect · Strong** | the full effect: SeedVR2 detail lifted ×1.15, clarity 0.30, crisper edges | face ×1.64 · 6.9 % — foliage ×1.40 · 9.2 % |

One style, two strengths. Both keep the stabiliser fully on and clamp the whole finish to the local range, so there are no outlines; colour is kept by SeedVR2's `lab` post-processing.

For reference, raw SeedVR2 on the same clips: face ×1.27 · 11.8 %, foliage ×1.16 · 21.3 % (the source itself: 3.2 % and 6.9 %). The stabiliser is fully on at both levels.

`target` 720p / 1080p / 1440p (2K) = short side, aspect kept. `sharpness` (−1…+1) works on top of any mode. Audio and fps are not touched by the node — take them from Get Video Components.

---


One ComfyUI node that turns any image or video-frame batch into a clean, sharp upscale with **NVIDIA RTX Video Super Resolution** — and does the whole chain on the GPU, per frame:

```
denoise (source size) → RTX upscale → supersample downscale → deblur (output size)
→ edge-aware sharpen (luma, noise-gated, halo-guarded) → fine film grain
```

Only the finished frame leaves the GPU, so the full chain costs about as much as a single RTX node — a chain of separate nodes copies every frame back and forth at every step.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo> comfyui-ultra-upscale       # or unzip it here
python -m pip install -r comfyui-ultra-upscale/requirements.txt   # nvidia-vfx
```

Needs an NVIDIA RTX GPU and CUDA-enabled PyTorch (cu130+). Restart ComfyUI; the node is under **image/upscaling → Ultra Upscale ⚡**.

## Inputs → outputs

`images` (IMAGE batch: one picture or all video frames) → `images` (upscaled) + `info` (what was applied, sizes, speed).

For video: **Load Video → Get Video Components → Ultra Upscale → Video Combine**, with audio and fps taken from Get Video Components. See `example_workflows/`.

## Target

`720p`, `1080p`, `1440p (2K)` set the **short side** and keep the aspect ratio:

| source | 720p | 1080p | 1440p (2K) |
|---|---|---|---|
| 854×480 | 1280×720 | 1920×1080 | 2560×1440 |
| 1280×720 | — (same size: clean-up and sharpen only) | 1920×1080 | 2560×1440 |
| 720×1280 | — | 1080×1920 | 1440×2560 |
| 720×720 | — | 1080×1080 | 1440×1440 |

If the source is already larger than the target, it is downscaled (antialiased) and `info` says so.

## Presets

| preset | for | denoise | RTX model | deblur | sharpen | extras |
|---|---|---|---|---|---|---|
| **Detect Mode** | anything | measures the source and picks one of the presets below | | | | |
| **Clean · Natural** | AI-generated, camera-original — faithful | off | HIGHBITRATE_ULTRA | off | 0.15 | |
| **Clean · Crisp** *(default)* | clean source, visibly sharper | off | HIGHBITRATE_ULTRA | LOW 0.5 | 0.30 | |
| **Web · Compressed** | YouTube, phone, social re-uploads | LOW 0.4 | ULTRA | LOW 0.35 | 0.25 | higher noise gate |
| **Noisy · Low light** | sensor noise, night, dark rooms | MEDIUM 0.8 | ULTRA | off | 0.15 | strong noise gate, fine grain |
| **Max detail** | clean source, maximum crispness | off | HIGHBITRATE_ULTRA | MEDIUM 0.6 | 0.35 | 1.25× supersampling, fine grain |
| **Custom** | your settings | `custom_*` inputs | | | | |

Why these choices — NVIDIA's own guidance for the VSR models:
- **HIGHBITRATE_*** is for clean, high-bitrate sources and avoids unnecessary artifact suppression, so texture survives. **ULTRA** is for typical compressed footage.
- **DENOISE_*** cleans noise and compression at the same size; higher levels soften texture — so it runs **before** upscaling and gently.
- **DEBLUR_*** is same-size sharpening for soft sources — so it runs **after** upscaling, at the final size.

### Detect Mode
Measures up to 5 frames: robust noise σ (median Immerkær estimator) and 8×8 blockiness. σ > 3 → *Noisy · Low light*; blockiness > 1.12 → *Web · Compressed*; otherwise *Clean · Crisp*. The measured values are printed in `info`, so you can see why it chose what it chose.

## What makes the sharpening different

- **Luma only** — the same correction goes to R, G and B, so edges never get coloured fringes.
- **Noise gate** — detail smaller than `noise_gate` is not sharpened, so noise, compression mush and frame-to-frame flicker are not amplified.
- **Edge protect** *(on by default)* — contrast-adaptive, CAS-style: full sharpening on low-contrast texture (skin, fabric, hair), fading to zero on edges that are already hard, so text, logos, UI and thin lines never thicken or stair-step.
- **Halo guard** — the result is clamped to the local 3×3 min/max (+0.02), so edges get crisper without bright or dark outlines.
- **Grain** — fine, slightly soft, strongest in mid-tones and absent in clipped blacks and whites; brings back a natural look after denoising. Seeded, so re-runs are identical.

## Sharpness — on top of any preset

`sharpness` (−1 … +1) works with **every** preset, including **Detect Mode**: the preset decides *what* to do, `sharpness` decides *how crisp*. +0.3 a bit crisper · +0.6 clearly crisper · +1.0 maximum · negative values soften. It scales the sharpen amount and the RTX deblur strength (capped at 1).

## Custom settings

`custom_denoise` + strength · `custom_upscale_model` + strength · `custom_deblur` + strength · `custom_sharpen` (0–2) · `custom_sharpen_radius` (σ in output px) · `custom_noise_gate` · `custom_halo_guard` · `custom_grain` · `custom_supersample`. Used only when **preset = Custom**.

`ten_bit` (all presets) runs the SDK in 10-bit: smoother gradients, slightly slower. `seed` sets the grain pattern.

## Tips
- Fastest: *Clean · Natural* (one RTX pass). *Max detail* is the slowest (bigger RTX pass + deblur), still seconds per clip on a modern RTX card.
- Faces flicker in other upscalers because details are invented per frame. Here nothing is invented by a diffusion model, and the noise gate keeps small flicker from being sharpened.
- 4K video needs RAM for the output frames (~100 MB per frame in ComfyUI's float format).

## Detail-Aware Mix 🎯 (second node)

For **generative** upscalers — SeedVR2, FlashVSR — that paint texture everywhere. Feed it the upscaled frames and the same frames plainly resized to the same size:

- **keeps the new detail where the source had detail** — eyes, hair, edges, in-focus texture
- **brings the source back on smooth or out-of-focus areas** — bokeh backgrounds stay soft, skin is not over-textured
- **color_lock** — restores the source colour and tone at large scale (no warmer skin, no contrast drift), fine detail untouched

`base_mix` 0.15 — source everywhere · `protect_soft_areas` 0.75 — how much source returns on soft areas. Also outputs the `detail_mask` it used.

## License
Node code: MIT. `nvidia-vfx` is distributed by NVIDIA under its own license.
