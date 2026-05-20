"""Runtime helpers for loading and exercising the Lance model with reduced
VRAM during calibration / quantization.

The stock Lance loader (`init_from_model_path_if_needed` in inference_lance.py)
does three things badly for our 16 GB GPU:
  1. reads the F32 safetensors (~24 GB) entirely into CPU RAM,
  2. copies it into the F32-initialised model (also F32),
  3. `model.to(device=cuda, dtype=bfloat16)` peak-allocates F32 + BF16 copies
     on the GPU.

We rewrite the loader to:
  - construct the Lance modules with empty weights via `torch.device("meta")`
  - stream tensors from the safetensors file ONE AT A TIME, cast to bf16, then
    direct-copy into the model parameter on the target device.

This keeps GPU peak at ~13 GB (LLM + ViT in bf16) for x2t tasks and ~17 GB
for gen tasks if we keep VAE on CPU (it's only needed for the final decode).
"""

from __future__ import annotations

import gc
import os
import os.path as osp
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open


def stream_load_into_model(
    model: torch.nn.Module,
    safetensors_path: str | Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    skip_keys: Iterable[str] = (),
    verbose: bool = True,
) -> dict:
    """Load weights tensor-by-tensor from a safetensors file into `model`,
    casting to `dtype` and copying directly to `device`. Returns a load report.

    Tensors NOT present in the safetensors are left at whatever initial value
    the model construction set (e.g. random init for `latent_pos_embed`,
    which Lance intentionally regenerates as a sinusoid).
    """
    skip = set(skip_keys)
    own_state = dict(model.state_dict(keep_vars=True))
    missing = set(own_state.keys())
    unexpected = []
    loaded = 0
    t0 = time.time()

    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        keys = f.keys()
        for k in keys:
            if k in skip:
                missing.discard(k)
                continue
            if k not in own_state:
                unexpected.append(k)
                continue
            target = own_state[k]
            src = f.get_tensor(k)
            # cast on CPU first (smaller memory footprint than F32->bf16 on GPU)
            if src.is_floating_point() and src.dtype != dtype:
                src = src.to(dtype)
            target.data = src.to(device, non_blocking=True)
            missing.discard(k)
            loaded += 1
            del src
            if loaded % 200 == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gc.collect()

    if verbose:
        elapsed = time.time() - t0
        print(f"  loaded {loaded} tensors in {elapsed:.1f}s")
        print(f"  missing ({len(missing)}): {sorted(missing)[:6]}{'...' if len(missing) > 6 else ''}")
        print(f"  unexpected ({len(unexpected)}): {unexpected[:6]}{'...' if len(unexpected) > 6 else ''}")
    return {"loaded": loaded, "missing": sorted(missing), "unexpected": unexpected}


def build_lance_skeleton(
    lance_src: str | Path,
    model_dir: str | Path,
    vit_dir: str | Path,
    vae_path: str | Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Construct an uninitialised Lance model on `meta` device, then move to
    `device` (which allocates real but uninitialised storage), so that
    `stream_load_into_model` can fill it tensor by tensor.
    """
    sys.path.insert(0, str(lance_src))
    from modeling.lance import LanceConfig, Lance
    from modeling.lance.qwen2_navit import Qwen2ForCausalLM
    from modeling.qwen2.modeling_qwen2 import Qwen2Config
    from modeling.qwen2 import Qwen2Tokenizer
    from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    from common.utils.misc import AutoEncoderParams

    # 1) tokenizer + token id table (needed before model build to set vocab)
    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_dir))

    # 2) configs
    import json
    llm_cfg_dict = json.loads(Path(model_dir).joinpath("llm_config.json").read_text())
    # Lance's Qwen2Config has extra fields; pass through as kwargs
    llm_cfg = Qwen2Config(**{k: v for k, v in llm_cfg_dict.items() if not isinstance(v, dict)})
    # add fields Lance expects
    llm_cfg.layer_module = "Qwen2MoTDecoderLayer"
    llm_cfg.qk_norm = True
    llm_cfg.qk_norm_und = True
    llm_cfg.qk_norm_gen = True
    llm_cfg.freeze_und = False
    llm_cfg.tie_word_embeddings = False
    llm_cfg.apply_qwen_2_5_vl_pos_emb = True
    for k in ("vision_config",):
        if k in llm_cfg_dict:
            setattr(llm_cfg, k, llm_cfg_dict[k])
    vit_cfg = Qwen2_5_VLVisionConfig.from_pretrained(str(vit_dir))

    vae_cfg = AutoEncoderParams(
        z_channels=48,
        downsample_temporal=4,
        downsample_spatial=8,
    )
    lance_cfg = LanceConfig(
        visual_gen=True, visual_und=True,
        llm_config=llm_cfg, vit_config=vit_cfg, vae_config=vae_cfg,
        latent_patch_size=(1, 1, 1),
        max_latent_size=64, max_num_frames=121,
        vit_max_num_patch_per_side=70,
        interpolate_pos=False, timestep_shift=3.5,
    )

    # 3) build modules on meta then assign empty real tensors via .to_empty()
    with torch.device("meta"):
        llm = Qwen2ForCausalLM(llm_cfg)
        vit = Qwen2_5_VisionTransformerPretrainedModel(vit_cfg)

    # tiny training args mock
    class _TArgs:
        use_task_embedding = False
        use_modality_embedding = False
    lance_cfg._dummy_targs = _TArgs()

    with torch.device("meta"):
        # Lance __init__ touches a `training_args` attr; pass via kwargs path
        model = Lance(
            language_model=llm, vit_model=vit, vit_type="qwen_2_5_vl_original",
            config=lance_cfg, training_args=_TArgs(),
        )
    # Materialise on real device (uninitialised storage, no copies)
    model = model.to_empty(device=device).to(dtype=dtype)
    return model, tokenizer, lance_cfg


def load_lance_for_calibration(
    lance_src: str | Path,
    model_dir: str | Path,
    vit_dir: str | Path,
    vae_path: str | Path,
    device: str = "cuda",
):
    """Build + load Lance in bf16 on `device`, with VAE kept on CPU.
    Returns the model (eval mode), tokenizer, and lance_cfg.
    """
    print("[load] building skeleton")
    model, tokenizer, cfg = build_lance_skeleton(
        lance_src, model_dir, vit_dir, vae_path, device=device, dtype=torch.bfloat16,
    )
    print("[load] streaming LLM weights from", model_dir)
    stream_load_into_model(
        model, Path(model_dir) / "model.safetensors",
        device=device, dtype=torch.bfloat16,
        skip_keys={"latent_pos_embed.pos_embed"},   # Lance regenerates this
    )
    # Some video checkpoints bundle vit_model in their main safetensors; only
    # load the standalone ViT for image-only checkpoints
    vit_in_main = any(k.startswith("vit_model.") for k in dict(model.state_dict()))
    if not vit_in_main:
        print("[load] streaming ViT weights from", vit_dir)
        stream_load_into_model(
            model.vit_model, Path(vit_dir) / "vit.safetensors",
            device=device, dtype=torch.bfloat16,
        )

    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"[load] cuda mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
              f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak")
    return model, tokenizer, cfg


def estimate_mem_savings() -> None:
    """Quick sanity print of expected mem reductions per quant scheme."""
    bf16_gb = 6.17 * 2
    int4_grp = 6.17 * 0.5 + 6.17 * 0.5 / 128 * 2  # 4-bit data + bf16 scales
    nvfp4 = 6.17 * 0.5 + 6.17 * 1 / 16            # 4-bit + fp8 scales/16
    print(f"bf16 LLM size:  {bf16_gb:.2f} GB")
    print(f"AWQ INT4 g128:  {int4_grp:.2f} GB")
    print(f"NVFP4 g16:      {nvfp4:.2f} GB")


if __name__ == "__main__":
    estimate_mem_savings()
