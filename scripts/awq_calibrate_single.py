"""Single-task activation calibration for AWQ on Lance.

Loads Lance, installs forward hooks on every quant-target Linear in
language_model.*, runs the validation loop for ONE Lance task end-to-end,
and saves the accumulated activation magnitudes to <out>.pt.

We run this multiple times (once per task) and then merge with
`awq_merge_stats.py` to get full coverage of both und and gen-expert paths
in the Mixture-of-Tasks decoder.

Usage:
  python awq_calibrate_single.py \
      --task x2t_image \
      --model_path /dev/shm/lance-weights/Lance_3B_Video \
      --vit_path /dev/shm/lance-weights/Qwen2.5-VL-ViT \
      --example_json config/examples/x2t_image_example.json \
      --out ../calib/x2t_image_stats.pt \
      --num_timesteps 30
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch


QUANT_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "self_attn.q_proj_moe_gen", "self_attn.k_proj_moe_gen",
    "self_attn.v_proj_moe_gen", "self_attn.o_proj_moe_gen",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj",
)


# ---------------------------------------------------------------------------
# Meta-init + streaming bf16 loader (same as run_baseline.py)
# ---------------------------------------------------------------------------


@contextmanager
def _meta_init():
    orig_empty = torch.empty
    def _empty_meta(*sizes, **kw):
        kw.setdefault("device", "meta")
        return orig_empty(*sizes, **kw)
    torch.empty = _empty_meta
    try:
        yield
    finally:
        torch.empty = orig_empty


def _streaming_bf16_loader(model, model_args):
    from safetensors import safe_open
    path_dir = model_args.model_path
    ck = next((p for p in [osp.join(path_dir, "model.safetensors"),
                            osp.join(path_dir, "ema.safetensors")]
                if osp.exists(p)), None)
    if ck is None:
        raise FileNotFoundError(f"no checkpoint in {path_dir}")

    print(f"[bf16-stream] loading {ck}")
    t0 = time.time()
    own = dict(model.state_dict(keep_vars=True))
    missing = set(own.keys())
    unexpected: list[str] = []
    loaded = 0
    device = next(iter(model.parameters())).device

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
            p = own[k]
            with torch.no_grad():
                if p.device.type == "meta":
                    p.data = src.to(device)
                else:
                    p.data.copy_(src.to(device), non_blocking=True)
            missing.discard(k)
            loaded += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[bf16-stream] {loaded} loaded in {time.time()-t0:.1f}s")
    class _M:
        missing_keys = sorted(missing); unexpected_keys = unexpected
    return _M()


def _patch_inference_lance():
    import inference_lance as IL
    from modeling.lance import Lance
    from modeling.lance.qwen2_navit import Qwen2ForCausalLM
    from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel

    _OQ = Qwen2ForCausalLM.__init__
    _OV = Qwen2_5_VisionTransformerPretrainedModel.__init__
    _OL = Lance.__init__

    def _Q(self, c):
        with _meta_init(): _OQ(self, c)
    def _V(self, c):
        with _meta_init(): _OV(self, c)
    def _L(self, *a, **k):
        with _meta_init(): _OL(self, *a, **k)

    Qwen2ForCausalLM.__init__ = _Q
    Qwen2_5_VisionTransformerPretrainedModel.__init__ = _V
    Lance.__init__ = _L
    IL.init_from_model_path_if_needed = _streaming_bf16_loader


# ---------------------------------------------------------------------------
# Activation stats
# ---------------------------------------------------------------------------


class ActStats:
    __slots__ = ("sum_abs", "n_tokens", "n_calls")

    def __init__(self):
        self.sum_abs: torch.Tensor | None = None
        self.n_tokens: int = 0
        self.n_calls: int = 0

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().to(torch.float32).reshape(-1, x.shape[-1])
        if x.numel() == 0:
            return
        s = x.abs().sum(dim=0).cpu()
        if self.sum_abs is None:
            self.sum_abs = s
        else:
            self.sum_abs += s
        self.n_tokens += x.shape[0]
        self.n_calls += 1


def install_hooks(model):
    targets = []
    for n, m in model.named_modules():
        if (isinstance(m, torch.nn.Linear)
                and "language_model" in n
                and any(n.endswith(s) for s in QUANT_SUFFIXES)):
            targets.append(n)
    print(f"[hooks] installing on {len(targets)} Linear modules")
    stats = {n: ActStats() for n in targets}
    handles = []
    mods = dict(model.named_modules())
    for n in targets:
        def make_hook(name):
            def h(module, inputs, output):
                if isinstance(inputs, tuple) and len(inputs) > 0:
                    stats[name].update(inputs[0])
            return h
        handles.append(mods[n].register_forward_hook(make_hook(n)))
    return stats, handles


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["t2i", "t2v", "image_edit", "video_edit",
                             "x2t_image", "x2t_video"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--vit_path", required=True)
    ap.add_argument("--example_json", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num_timesteps", type=int, default=30,
                    help="for gen tasks; lower = faster calibration")
    ap.add_argument("--video_height", type=int, default=768)
    ap.add_argument("--video_width", type=int, default=768)
    ap.add_argument("--num_frames", type=int, default=50)
    ap.add_argument("--resolution", default=None)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("POSITION_EMBEDDING_3D_VERSION", "v2")
    os.environ.setdefault("EXP_HW_20250819", "False")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    resolution = args.resolution or ("image_768res" if args.task in ("t2i", "image_edit", "x2t_image") else "video_480p")

    sys.argv = [
        "inference_lance.py",
        "--model_path",            args.model_path,
        "--vit_path",              args.vit_path,
        "--vit_type",              "qwen_2_5_vl_original",
        "--llm_qk_norm",           "true",
        "--llm_qk_norm_und",       "true",
        "--llm_qk_norm_gen",       "true",
        "--tie_word_embeddings",   "false",
        "--validation_num_timesteps", str(args.num_timesteps),
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
        "--save_path_gen",         f"results/calib_{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",
        "--resolution",            resolution,
        "--text_template",         "true",
        "--cfg_text_scale",        "4.0",
        "--use_KVcache",           "true",
        "--val_dataset_config_file", args.example_json,
    ]

    _patch_inference_lance()
    import inference_lance
    from modeling.vae.wan.model import WanVideoVAE

    state = {"stats": None, "handles": None}
    orig_validate = inference_lance.validate_on_fixed_batch

    def _validate_with_hooks(*a, **kw):
        if state["stats"] is None:
            model = kw.get("fsdp_model") or a[0]
            state["stats"], state["handles"] = install_hooks(model)
        return orig_validate(*a, **kw)

    # Skip VAE decode entirely during calibration — it's memory-hungry and we
    # don't care about the actual pixel output, only the LLM activation magnitudes
    # already collected by hooks during the diffusion forward passes.
    def _stub_vae_decode(self, latents):
        out = []
        for lat in latents:
            # Lance expects [C, T, H, W] with H = 8 * lat.shape[-2]
            c, t, h, w = lat.shape if lat.dim() == 4 else (lat.shape[0], 1, lat.shape[-2], lat.shape[-1])
            out.append(torch.zeros((3, t, h * 8, w * 8), dtype=torch.bfloat16, device="cpu"))
        return out
    WanVideoVAE.vae_decode = _stub_vae_decode
    print("[patch] WanVideoVAE.vae_decode stubbed (returns zeros — calibration only)")

    inference_lance.validate_on_fixed_batch = _validate_with_hooks
    inference_lance.main()

    if state["stats"] is None:
        raise RuntimeError("validate was never called")
    for h in state["handles"]:
        h.remove()

    # Persist
    out_dict = {
        "format": 1,
        "task": args.task,
        "n_linears": len(state["stats"]),
        "stats": {
            n: {"sum_abs": s.sum_abs, "n_tokens": s.n_tokens, "n_calls": s.n_calls}
            for n, s in state["stats"].items() if s.sum_abs is not None
        },
    }
    torch.save(out_dict, args.out)
    total_tokens = sum(s.n_tokens for s in state["stats"].values())
    print(f"[done] {out_dict['n_linears']} linears, "
          f"{len([s for s in state['stats'].values() if s.sum_abs is not None])} with data, "
          f"{total_tokens} total tokens")
    print(f"[done] saved {args.out}")


if __name__ == "__main__":
    main()
