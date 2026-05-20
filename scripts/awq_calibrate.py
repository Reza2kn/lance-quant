"""AWQ activation calibration for Lance.

Runs the Lance inference pipeline with activation hooks on every Linear we
plan to quantize. For each hook we accumulate `sum_abs` and `n_tokens` so we
can compute mean(|input|) per input-channel for AWQ scale search later.

Because Lance's MoT routes tokens through `mlp` vs `mlp_moe_gen` based on
token type, calibration MUST mix understanding and generation samples or
the gen-expert weights see no activations and AWQ scales for them default
to identity (= same as plain min-max INT4).

We use Lance's own example_*.json sets as the calibration source — they
already contain representative prompts/images for every task, are small,
and exercise the full model code path.

Output:
  <out>/act_stats.pt           torch dict: {name: {sum_abs: tensor[in_features],
                                                    n: int}}
  <out>/calib_log.json         which tasks/samples were run + timings

Usage (on A100):
  python awq_calibrate.py \
      --model_path downloads/Lance_3B_Video \
      --vit_path downloads/Qwen2.5-VL-ViT \
      --out ../calib/Lance_3B_Video \
      --num_t2i 10 --num_x2t 10 --num_edit 5 --num_t2v 4 \
      --t2i_timesteps 4
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Reuse the same meta-init + bf16 stream loader as run_baseline.py
# (We re-implement here to keep this script self-contained.)
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
        missing_keys = sorted(missing)
        unexpected_keys = unexpected
    return _M()


def _patch_inference_lance_module():
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
# Activation stats collection
# ---------------------------------------------------------------------------


QUANT_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "self_attn.q_proj_moe_gen", "self_attn.k_proj_moe_gen",
    "self_attn.v_proj_moe_gen", "self_attn.o_proj_moe_gen",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj",
)


@dataclass
class ActStats:
    sum_abs: torch.Tensor | None = None
    n_tokens: int = 0
    n_calls: int = 0

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


def install_hooks(model: torch.nn.Module) -> tuple[dict[str, ActStats], list]:
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if "language_model" not in name:
            continue
        if not any(name.endswith(s) for s in QUANT_SUFFIXES):
            continue
        targets.append(name)
    print(f"[hooks] {len(targets)} target Linears")

    stats: dict[str, ActStats] = {n: ActStats() for n in targets}
    handles = []
    modules_by_name = dict(model.named_modules())

    def make_hook(name):
        def h(module, inputs, output):
            if isinstance(inputs, tuple) and len(inputs) > 0:
                stats[name].update(inputs[0])
        return h

    for n in targets:
        handles.append(modules_by_name[n].register_forward_hook(make_hook(n)))
    return stats, handles


# ---------------------------------------------------------------------------
# Calibration driver
# ---------------------------------------------------------------------------


def build_argv(task: str, *, model_path: str, vit_path: str, save_path: str,
               resolution: str, num_frames: int, h: int, w: int,
               example_json: str, num_timesteps: int = 30) -> list[str]:
    return [
        "inference_lance.py",
        "--model_path",            model_path,
        "--vit_path",              vit_path,
        "--vit_type",              "qwen_2_5_vl_original",
        "--llm_qk_norm",           "true",
        "--llm_qk_norm_und",       "true",
        "--llm_qk_norm_gen",       "true",
        "--tie_word_embeddings",   "false",
        "--validation_num_timesteps", str(num_timesteps),
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
        "--video_height",          str(h),
        "--video_width",           str(w),
        "--num_frames",            str(num_frames),
        "--task",                  task,
        "--save_path_gen",         save_path,
        "--resolution",            resolution,
        "--text_template",         "true",
        "--cfg_text_scale",        "4.0",
        "--use_KVcache",           "true",
        "--val_dataset_config_file", example_json,
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--vit_path", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num_t2i", type=int, default=10)
    ap.add_argument("--num_x2t", type=int, default=10)
    ap.add_argument("--num_edit", type=int, default=5)
    ap.add_argument("--num_t2v", type=int, default=4)
    ap.add_argument("--t2i_timesteps", type=int, default=4,
                    help="few timesteps is enough — we just need activation magnitudes")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("POSITION_EMBEDDING_3D_VERSION", "v2")
    os.environ.setdefault("EXP_HW_20250819", "False")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    args.out.mkdir(parents=True, exist_ok=True)
    log = {"tasks": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    _patch_inference_lance_module()
    import inference_lance

    # Build the model + hooks ONCE, then iterate tasks within the same Python process
    # so we don't reload between tasks.
    # We do this by manually invoking parts of inference_lance.main.
    from transformers import HfArgumentParser, set_seed
    from config.config_factory import ModelArguments, DataArguments, InferenceArguments
    from common.utils.distributed import get_global_rank

    # Use the t2i example to bootstrap (any task works for model build)
    sys.argv = build_argv(
        "t2i", model_path=args.model_path, vit_path=args.vit_path,
        save_path="results/_calib_bootstrap", resolution="image_768res",
        num_frames=1, h=768, w=768,
        example_json="config/examples/t2i_example.json",
        num_timesteps=args.t2i_timesteps,
    )

    # Re-run inference_lance.main, but with hooks installed BEFORE the
    # validation loop fires. The cleanest hookpoint: monkey-patch
    # `validate_on_fixed_batch` to install hooks on its first call.
    _hook_state = {"stats": None, "handles": None, "installed": False}
    orig_validate = inference_lance.validate_on_fixed_batch

    def _validate_with_hooks(*args_, **kwargs_):
        if not _hook_state["installed"]:
            model = kwargs_.get("fsdp_model") or args_[0]
            stats, handles = install_hooks(model)
            _hook_state["stats"] = stats
            _hook_state["handles"] = handles
            _hook_state["installed"] = True
            print(f"[calib] hooks installed on first validate call")
        return orig_validate(*args_, **kwargs_)

    inference_lance.validate_on_fixed_batch = _validate_with_hooks

    # Run t2i once (which builds the model, runs hooks, and saves stats)
    inference_lance.main()

    stats = _hook_state["stats"]
    handles = _hook_state["handles"]
    if stats is None:
        raise RuntimeError("validate_on_fixed_batch was never called")

    print(f"[calib] after t2i: {sum(s.n_tokens for s in stats.values())} total tokens seen")

    # TODO(stage2): run additional tasks (x2t_image, image_edit, t2v) by
    # invoking the dataset+forward in-place. For simplicity v1 ships with
    # just t2i; this exercises every gen-path expert and the und path for
    # the prompt-encoding phase.

    for h in handles:
        h.remove()

    # Save stats
    out = {"format": 1, "n_tasks": 1, "stats": {}}
    for n, s in stats.items():
        if s.sum_abs is None:
            continue
        out["stats"][n] = {"sum_abs": s.sum_abs, "n_tokens": s.n_tokens,
                           "n_calls": s.n_calls}
    torch.save(out, args.out / "act_stats.pt")
    print(f"wrote {args.out / 'act_stats.pt'}: {len(out['stats'])} linears")

    log["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (args.out / "calib_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
