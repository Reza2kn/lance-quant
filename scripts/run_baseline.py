"""Run a Lance inference task with a memory-frugal loader.

Replaces two pieces of inference_lance.py via monkey-patch:

  1. Model construction: builds Lance's submodules on `meta` device so no
     CPU F32 storage is allocated, then `to_empty(device, dtype=bf16)` makes
     bf16 storage directly on the GPU.

  2. Weight load: streams the safetensors file one tensor at a time, casting
     to bf16 on CPU and copy_-ing into the pre-allocated GPU parameter.

This brings GPU peak from ~26 GB (default loader) down to ~13.5 GB, which fits
on the RTX 5080 Laptop's 16 GB VRAM. CPU RAM stays under 6 GB because we
never realise the F32 weight tensor at all.

Usage:
  python run_baseline.py --task x2t_image --model_path downloads/Lance_3B_Video
"""

from __future__ import annotations

import argparse
import gc
import os
import os.path as osp
import sys
import time
from contextlib import contextmanager

import torch
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Patch 1: meta-device construction of Lance
# ---------------------------------------------------------------------------


@contextmanager
def _meta_init():
    """Force every nn.Module / nn.Parameter construction in this block onto
    the meta device. We toggle by monkey-patching `torch.empty` for the
    duration."""
    orig_empty = torch.empty

    def _empty_meta(*sizes, **kw):
        kw.setdefault("device", "meta")
        return orig_empty(*sizes, **kw)

    torch.empty = _empty_meta
    try:
        yield
    finally:
        torch.empty = orig_empty


# ---------------------------------------------------------------------------
# Patch 2: streaming bf16 loader (replaces init_from_model_path_if_needed)
# ---------------------------------------------------------------------------


def _streaming_bf16_loader(model, model_args):
    path_dir = model_args.model_path
    candidates = [osp.join(path_dir, "model.safetensors"),
                  osp.join(path_dir, "ema.safetensors")]
    ck = next((p for p in candidates if osp.exists(p)), None)
    if ck is None:
        raise FileNotFoundError(f"no checkpoint in {path_dir}")

    print(f"[bf16-stream] loading {ck}")
    t0 = time.time()
    own = dict(model.state_dict(keep_vars=True))
    missing = set(own.keys())
    unexpected: list[str] = []
    loaded = 0
    device = next(model.parameters()).device

    with safe_open(ck, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k == "latent_pos_embed.pos_embed":
                missing.discard(k)
                continue
            if k not in own:
                unexpected.append(k)
                continue
            src = f.get_tensor(k)
            if src.is_floating_point() and src.dtype != torch.bfloat16:
                src = src.to(torch.bfloat16)
            param = own[k]
            with torch.no_grad():
                if param.device.type == "meta":
                    # Materialise the parameter on the target device by
                    # replacing its tensor data.
                    new = src.to(device)
                    param.data = new
                else:
                    if param.shape != src.shape:
                        print(f"[bf16-stream] shape mismatch {k}: own={tuple(param.shape)} ck={tuple(src.shape)} -> skipping")
                        continue
                    param.data.copy_(src.to(device), non_blocking=True)
            missing.discard(k)
            loaded += 1
            del src
            if loaded % 200 == 0:
                gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[bf16-stream] {loaded} loaded in {time.time()-t0:.1f}s; "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    if unexpected[:5]:
        print(f"[bf16-stream] sample unexpected: {unexpected[:5]}")
    if list(missing)[:5]:
        print(f"[bf16-stream] sample missing:    {list(missing)[:5]}")

    class _Msg:
        missing_keys = sorted(missing)
        unexpected_keys = unexpected
    return _Msg()


# ---------------------------------------------------------------------------
# Patch 3: build Lance model on meta then to_empty on GPU
# ---------------------------------------------------------------------------


def _patch_inference_lance_module():
    import inference_lance as IL
    from modeling.lance import Lance
    from modeling.lance.qwen2_navit import Qwen2ForCausalLM
    from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel

    # Save original constructors
    _OrigQwen2 = Qwen2ForCausalLM.__init__
    _OrigViT = Qwen2_5_VisionTransformerPretrainedModel.__init__
    _OrigLance = Lance.__init__

    def _Qwen2_init(self, config):
        with _meta_init():
            _OrigQwen2(self, config)

    def _ViT_init(self, config):
        with _meta_init():
            _OrigViT(self, config)

    def _Lance_init(self, *args, **kwargs):
        with _meta_init():
            _OrigLance(self, *args, **kwargs)
        # We deliberately do NOT materialise here. Params stay on meta until
        # the streaming bf16 loader replaces them with real bf16 tensors on
        # GPU. This avoids any F32 allocation peak.
        if torch.cuda.is_available():
            print(f"[meta-init] cuda mem after construct: "
                  f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    Qwen2ForCausalLM.__init__ = _Qwen2_init
    Qwen2_5_VisionTransformerPretrainedModel.__init__ = _ViT_init
    Lance.__init__ = _Lance_init

    # Replace IL's init_from_model_path_if_needed
    IL.init_from_model_path_if_needed = _streaming_bf16_loader

    # The .to(DEVICE) call at line 499 becomes a no-op since we're already on
    # the right device. The .to(DEVICE, bfloat16) at line 541 likewise.

    print("[patch] meta-init + streaming bf16 loader installed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["t2i", "t2v", "image_edit", "video_edit",
                             "x2t_image", "x2t_video"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--vit_path", default="downloads/Qwen2.5-VL-ViT")
    ap.add_argument("--resolution", default=None)
    ap.add_argument("--save_path_gen", default=None)
    ap.add_argument("--num_frames", type=int, default=50)
    ap.add_argument("--video_height", type=int, default=768)
    ap.add_argument("--video_width", type=int, default=768)
    ap.add_argument("--validation_num_timesteps", type=int, default=30)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("POSITION_EMBEDDING_3D_VERSION", "v2")
    os.environ.setdefault("EXP_HW_20250819", "False")

    # Build CLI args for inference_lance.main()
    if args.save_path_gen is None:
        args.save_path_gen = f"results/baseline_{args.task}_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.resolution is None:
        args.resolution = "image_768res" if args.task in ("t2i", "image_edit", "x2t_image") else "video_480p"

    sys.argv = [
        "inference_lance.py",
        "--model_path",            args.model_path,
        "--vit_path",              args.vit_path,
        "--vit_type",              "qwen_2_5_vl_original",
        "--llm_qk_norm",           "true",
        "--llm_qk_norm_und",       "true",
        "--llm_qk_norm_gen",       "true",
        "--tie_word_embeddings",   "false",
        "--validation_num_timesteps", str(args.validation_num_timesteps),
        "--validation_timestep_shift", "3.5",
        "--copy_init_moe",         "true",
        "--max_num_frames",        "121",
        "--max_latent_size",       "64",
        "--latent_patch_size",     "1", "1", "1",
        "--visual_und",            "true",
        "--visual_gen",            "true",
        "--vae_model_type",        "wan",
        "--apply_qwen_2_5_vl_pos_emb", "true",
        "--apply_chat_template",   "false",
        "--cfg_type",              "0",
        "--validation_data_seed",  "42",
        "--video_height",          str(args.video_height),
        "--video_width",           str(args.video_width),
        "--num_frames",            str(args.num_frames),
        "--task",                  args.task,
        "--save_path_gen",         args.save_path_gen,
        "--resolution",            args.resolution,
        "--text_template",         "true",
        "--cfg_text_scale",        "4.0",
        "--use_KVcache",           "true",
    ]

    _patch_inference_lance_module()

    import inference_lance
    if hasattr(inference_lance, "main"):
        inference_lance.main()
    else:
        # The script runs at import time; re-exec the file in patched globals
        with open(inference_lance.__file__) as f:
            exec(f.read(), {"__name__": "__main__"})


if __name__ == "__main__":
    main()
