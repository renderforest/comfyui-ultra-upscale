# Renderforest production workflow: `tool.upscale`

The `ai_workflows` row with code `tool.upscale` (type `upscale`) that ai-workers-service runs for an upscale tool job.

- `tool.upscale.api.json` → the row's `config` (ComfyUI API format).
- `tool.upscale.mapping.json` → the row's `mapping`. ai-workers-service fills:
  - `videoUrl`: URL of the base video (node 1, `VHS_LoadVideoPath`, which reads http/https).
  - `target`: `720p` or `1080p` (node 10), so one workflow serves both resolutions.

Changes from Hayk's `Ultra_Upscale 720/1080.json`:
- Nodes 1+2 (`LoadVideo` + `GetVideoComponents`, which read a file from ComfyUI's input folder) are replaced by
  `VHS_LoadVideoPath` (URL input; frames, audio, video_info) + `VHS_VideoInfoLoaded` (fps). With `force_rate: 0`
  the source fps and every frame are kept.
- Node 7 (source preview) is removed: ai-workers-service takes the first video output of the prompt.
- Node 6 (`ShowText|pysssss`, info display) is removed: it needs ComfyUI-Custom-Scripts, which the workers do not have.

Requirements on the worker: ComfyUI-VideoHelperSuite, this node pack, ComfyUI with SeedVR2 `int8_convrot` support,
and the models `diffusion_models/seedvr2_7b_int8_convrot.safetensors` and `vae/seedvr2_ema_vae_fp16.safetensors`
(https://huggingface.co/Comfy-Org/SeedVR2).
