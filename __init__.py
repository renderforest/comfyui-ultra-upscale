from .rtx_ultra_upscale import NODE_CLASS_MAPPINGS as _A, NODE_DISPLAY_NAME_MAPPINGS as _AD
from .detail_mix import DetailAwareMix
from .seedvr2_ultra import SeedVR2UltraUpscale

NODE_CLASS_MAPPINGS = {**_A, "DetailAwareMix": DetailAwareMix, "SeedVR2UltraUpscale": SeedVR2UltraUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {**_AD, "DetailAwareMix": "Detail-Aware Mix 🎯", "SeedVR2UltraUpscale": "SeedVR2 Ultra Upscale 🎬"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
