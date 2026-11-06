# Renderforest production workflow: `tool.upscale`

The `ai_workflows` row with code `tool.upscale` (type `upscale`) that ai-workers-service runs for an upscale tool job.

- `tool.upscale.api.json` → the row's `config` (ComfyUI API format): **RelayVSR x4** (Hayk, 2026-11-06).
- `tool.upscale.mapping.json` → the row's `mapping`. ai-workers-service fills:
  - `videoUrl`: URL of the base video (node 1, `VHS_LoadVideoPath`, which reads http/https).
  - `targetShortSide`: `720` or `1080`, the short side of the output (node 22), so one workflow serves both
    resolutions and every aspect ratio keeps its shape.
- `tool.upscale.seedvr2.*`: the previous SeedVR2 workflow (uses this repo's `SeedVR2UltraUpscale` node), kept for
  rollback; it maps `target` (`'720p'`/`'1080p'`) instead.

Changes from Hayk's `relayvsr_upscale_presets.json`:
- Node 1 `VHS_LoadVideo` (a file in ComfyUI's input folder) → `VHS_LoadVideoPath` (URL input, same outputs).
- Node 22 `ImageScaleToMaxDimension` (long side 1280/1920) → core `ResizeImageMaskNode` "scale shorter dimension":
  the long-side size gives the wrong output for square and 4:3 videos (a 720×720 base would come out 1280×1280).

Speed (Hayk, 5 s clip): 480→720 ~8 s, 720→1080 ~20 s.

Requirements on the worker: ComfyUI with the core RelayVSR nodes (`RelayVSRConditioning`, `VSRDecode`,
`comfy_extras/nodes_wan.py`) and `ResizeImageMaskNode`, ComfyUI-VideoHelperSuite, and the models
(https://huggingface.co/Comfy-Org/RelayVSR):
- `models/diffusion_models/relayvsr_x4_int8_convrot.safetensors` (5.2 GB)
- `models/vae/relayvsr_flash_vae_x4_fp32.safetensors` (128 MB)

No custom node pack is needed for this workflow.
