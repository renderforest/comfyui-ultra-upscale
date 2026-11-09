# Renderforest production workflow: `tool.upscale`

The `ai_workflows` row with code `tool.upscale` (type `upscale`) that ai-workers-service runs for an upscale tool job.

- `tool.upscale.api.json` → the row's `config` (ComfyUI API format): **RelayVSR x4** (Hayk, 2026-11-06).
- `tool.upscale.mapping.json` → the row's `mapping`. ai-workers-service fills:
  - `videoUrl`: URL of the base video (node 1, `VHS_LoadVideoPath`, which reads http/https).
  - `targetWidth` / `targetHeight`: the exact output size, computed by ai-video-api (`getUpscaleTargetSize`): the
    base's shape snapped to the nearest standard ratio, short side = the requested tier (e.g. 1280×720, 720×1280,
    960×720). Node 22 (core `ImageScale`, crop `center`) covers that size and crops the overflow at the center, so
    an off-ratio vendor frame (Seedance 480p 16:9 is 864×496) comes out exactly 16:9 with no side bars. Without
    them the node's default 1280×720 is used.
- `tool.upscale.seedvr2.*`: the previous SeedVR2 workflow (uses this repo's `SeedVR2UltraUpscale` node), kept for
  rollback; it maps `target` (`'720p'`/`'1080p'`) instead.

Changes from Hayk's `relayvsr_upscale_presets.json` (2026-11-06 17:00 version: load format `None` for the frame fix, seed 42, sharpen 0.3):
- Node 1 `VHS_LoadVideo` (a file in ComfyUI's input folder) → `VHS_LoadVideoPath` (URL input, same outputs).
- Node 22 `ImageScaleToMaxDimension` (long side 1280/1920) → core `ImageScale` (area, crop center) to
  `targetWidth`×`targetHeight`: the long-side size gave the wrong output for square and 4:3 videos, and scaling the
  short side kept the vendor's off-ratio frame (864×496 → 1254×720, side bars in the 16:9 composite).

Deploy order for a change of the mapped keys: ai-workers-service, then ai-video-api, then this row. A job queued
before ai-video-api sends the size gets the default 1280×720 and falls back to native generation if that is not its
size.

Speed (Hayk, 5 s clip): 480→720 ~8 s, 720→1080 ~20 s.

Requirements on the worker: ComfyUI with the core RelayVSR nodes (`RelayVSRConditioning`, `VSRDecode`,
`comfy_extras/nodes_wan.py`) and `ImageScale`, ComfyUI-VideoHelperSuite, and the models
(https://huggingface.co/Comfy-Org/RelayVSR):
- `models/diffusion_models/relayvsr_x4_int8_convrot.safetensors` (5.2 GB)
- `models/vae/relayvsr_flash_vae_x4_fp32.safetensors` (128 MB)

No custom node pack is needed for this workflow.
