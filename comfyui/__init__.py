"""ComfyUI custom-node pack for ByteDance Lance + its quantized variants.

This pack exposes:
  * LanceModelLoader   — loads Lance (bf16 / AWQ-INT4 / NVFP4) with the same
                         meta-init + streaming bf16 loader we use offline
  * LanceT2I           — text-to-image
  * LanceT2V           — text-to-video
  * LanceImageEdit     — instruction-guided image edit
  * LanceX2TImage      — image understanding / VQA

The node pack assumes you have ByteDance/Lance cloned and accessible. Set
`LANCE_SRC_PATH` env var to its path, or place the repo at
`ComfyUI/custom_nodes/ComfyUI-Lance/Lance/`.

Model weights (Lance_3B, Lance_3B_Video, Qwen2.5-VL-ViT, Wan2.2_VAE.pth)
should be in `ComfyUI/models/lance/`. Quantized variants land in subdirs
`ComfyUI/models/lance/<repo-name>/`.
"""

from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
