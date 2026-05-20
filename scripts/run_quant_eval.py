"""Run a Lance inference task with the quantized language_model swapped in.

Loads everything that wasn't quantized (ViT, VAE projections, norms, embeds,
time_embedder, latent_pos_embed) from the original Lance_3B_Video weights at
bf16, then swaps in WQLinearINT4 for every quantized linear and streams the
INT4 packed buffers from `awq_state_dict.safetensors`.

Usage:
  python run_quant_eval.py \
      --task x2t_image \
      --model_path downloads/Lance_3B_Video \
      --awq_dir ../models/Lance_3B_Video-INT4-MinMax
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Reuse the meta-init + streaming bf16 loader from run_baseline.py
# (copy-pasted to keep this a self-contained script)
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
            param = own[k]
            with torch.no_grad():
                if param.device.type == "meta":
                    param.data = src.to(device)
                else:
                    if param.shape != src.shape:
                        continue
                    param.data.copy_(src.to(device), non_blocking=True)
            missing.discard(k)
            loaded += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[bf16-stream] {loaded} loaded in {time.time()-t0:.1f}s; "
          f"missing={len(missing)}, unexpected={len(unexpected)}")

    class _Msg:
        missing_keys = sorted(missing)
        unexpected_keys = unexpected
    return _Msg()


# ---------------------------------------------------------------------------
# AWQ swap-in (similar to patches/quantized_linear.py but reads buffers
# in-stream rather than loading the whole file)
# ---------------------------------------------------------------------------


class WQLinearINT5(torch.nn.Module):
    """5-bit asymmetric grouped, uint8-per-code storage.
    Matches output of scripts/awq_apply_5bit.py."""
    MODE = "ondemand"

    def __init__(self, in_features, out_features, group_size, bias, device, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.compute_dtype = dtype
        n_groups = in_features // group_size
        self.register_buffer("qweight", torch.zeros(
            (out_features, in_features), dtype=torch.uint8, device=device))
        self.register_buffer("scales",  torch.zeros(
            (out_features, n_groups), dtype=dtype, device=device))
        self.register_buffer("zeros",   torch.zeros(
            (out_features, n_groups), dtype=torch.uint8, device=device))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def _dequantize(self):
        codes = self.qweight.to(self.compute_dtype)
        u = codes.reshape(self.out_features, -1, self.group_size)
        z = self.zeros.unsqueeze(-1).to(self.compute_dtype)
        s = self.scales.unsqueeze(-1)
        return ((u - z) * s).reshape(self.out_features, self.in_features)

    def forward(self, x):
        return torch.nn.functional.linear(x, self._dequantize(), self.bias)


class WQLinearINT4(torch.nn.Module):
    """Drop-in for nn.Linear with 4-bit grouped asymmetric weight storage.

    Three execution modes (set via class attribute `MODE`):
      - "ondemand": dequantize on every forward; minimal VRAM, slow
      - "cached":   dequantize once on first forward, cache bf16; fast, peaks VRAM
      - "fused":    use torch._weight_int4pack_mm if available; fast, low VRAM
    """
    MODE = "ondemand"

    def __init__(self, in_features, out_features, group_size, bias, device, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.compute_dtype = dtype
        n_groups = in_features // group_size
        self.register_buffer("qweight", torch.zeros(
            (out_features, in_features // 2), dtype=torch.uint8, device=device))
        self.register_buffer("scales", torch.zeros(
            (out_features, n_groups), dtype=dtype, device=device))
        self.register_buffer("zeros", torch.zeros(
            (out_features, n_groups), dtype=torch.uint8, device=device))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)
        self._cached_weight: torch.Tensor | None = None

    def _dequantize(self) -> torch.Tensor:
        packed = self.qweight
        lo = (packed & 0xF).to(torch.int16)
        hi = ((packed >> 4) & 0xF).to(torch.int16)
        unpacked = torch.stack([lo, hi], dim=-1).reshape(self.out_features, self.in_features)
        u = unpacked.reshape(self.out_features, -1, self.group_size).to(self.compute_dtype)
        z = self.zeros.unsqueeze(-1).to(self.compute_dtype)
        s = self.scales.unsqueeze(-1)
        return ((u - z) * s).reshape(self.out_features, self.in_features)

    def forward(self, x):
        if WQLinearINT4.MODE == "cached":
            if self._cached_weight is None:
                with torch.no_grad():
                    self._cached_weight = self._dequantize().contiguous()
            return torch.nn.functional.linear(x, self._cached_weight, self.bias)
        # ondemand: dequant every call
        return torch.nn.functional.linear(x, self._dequantize(), self.bias)


def swap_to_awq(model, awq_dir: Path, compute_dtype=torch.bfloat16):
    meta_doc = json.loads((awq_dir / "awq_meta.json").read_text())
    meta = meta_doc["per_weight"]
    # n_bit: 4 (default) uses WQLinearINT4 with nibble packing;
    # n_bit: 5 uses WQLinearINT5 with byte-per-code storage.
    n_bit = meta_doc.get("n_bit", 4)
    LinClass = WQLinearINT5 if n_bit == 5 else WQLinearINT4
    print(f"[awq-swap] n_bit={n_bit}, swapping {len(meta)} linears to {LinClass.__name__}")
    modules_by_name = dict(model.named_modules())
    swapped = 0
    for wkey in meta:
        info = meta[wkey]
        lin_name = wkey[:-len(".weight")]
        parent_path, lin_attr = lin_name.rsplit(".", 1)
        parent = modules_by_name.get(parent_path)
        if parent is None:
            continue
        old = getattr(parent, lin_attr, None)
        if not isinstance(old, torch.nn.Linear):
            continue
        device = old.weight.device if old.weight.device.type != "meta" else (
            next((p.device for p in model.parameters() if p.device.type != "meta"), torch.device("cuda")))
        new = LinClass(
            in_features=info["shape"][1],
            out_features=info["shape"][0],
            group_size=info["group_size"],
            bias=old.bias is not None,
            device=device,
            dtype=compute_dtype,
        )
        setattr(parent, lin_attr, new)
        modules_by_name[lin_name] = new
        swapped += 1
    print(f"[awq-swap] swapped {swapped} linears")
    return modules_by_name


def stream_awq_buffers(modules_by_name, awq_dir: Path):
    sd_path = awq_dir / "awq_state_dict.safetensors"
    print(f"[awq-stream] loading {sd_path}")
    t0 = time.time()
    loaded_q = 0
    loaded_pass = 0
    own = {}                                # name -> param

    # We need to find which modules expect which buffers.
    with safe_open(str(sd_path), framework="pt", device="cpu") as f:
        for k in f.keys():
            # quant-buffer key like ".../q_proj.qweight"
            if k.endswith((".qweight", ".scales", ".zeros")):
                base, suffix = k.rsplit(".", 1)
                mod = modules_by_name.get(base)
                if mod is None or not hasattr(mod, suffix):
                    continue
                src = f.get_tensor(k)
                target = getattr(mod, suffix)
                with torch.no_grad():
                    target.data.copy_(src.to(target.device, non_blocking=True))
                loaded_q += 1
            else:
                # pass-through bf16 weight: copy into the model's existing param
                # (these are ViT, norms, embeds, time_embedder, etc.)
                # We'll resolve via name later through a second pass on model.state_dict
                pass

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[awq-stream] {loaded_q} quant buffers loaded in {time.time()-t0:.1f}s")


def stream_pass_through_weights(model, awq_dir: Path):
    """Load the non-quantized (bf16 pass-through) weights from awq_state_dict.
    For Lance the pass-through covers ViT, norms, embeds, time_embedder, llm2vae,
    vae2llm, latent_pos_embed (but that one's regen'd from sinusoid)."""
    sd_path = awq_dir / "awq_state_dict.safetensors"
    own = dict(model.state_dict(keep_vars=True))
    loaded = 0
    skipped_quant = 0
    device = next(iter(model.parameters())).device

    with safe_open(str(sd_path), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith((".qweight", ".scales", ".zeros")):
                skipped_quant += 1
                continue
            if k == "latent_pos_embed.pos_embed":
                continue
            if k not in own:
                continue
            src = f.get_tensor(k)
            if src.is_floating_point() and src.dtype != torch.bfloat16:
                src = src.to(torch.bfloat16)
            target = own[k]
            with torch.no_grad():
                if target.device.type == "meta":
                    target.data = src.to(device)
                else:
                    target.data.copy_(src.to(device), non_blocking=True)
            loaded += 1
    print(f"[bf16-stream] pass-through bf16 weights loaded: {loaded}")


# ---------------------------------------------------------------------------
# Patching inference_lance.main()
# ---------------------------------------------------------------------------


def _patch_for_meta_then_awq(awq_dir: Path):
    import inference_lance as IL
    from modeling.lance import Lance
    from modeling.lance.qwen2_navit import Qwen2ForCausalLM
    from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel

    _OQwen = Qwen2ForCausalLM.__init__
    _OViT = Qwen2_5_VisionTransformerPretrainedModel.__init__
    _OLance = Lance.__init__

    def _Q(self, c):
        with _meta_init():
            _OQwen(self, c)
    def _V(self, c):
        with _meta_init():
            _OViT(self, c)
    def _L(self, *a, **k):
        with _meta_init():
            _OLance(self, *a, **k)

    Qwen2ForCausalLM.__init__ = _Q
    Qwen2_5_VisionTransformerPretrainedModel.__init__ = _V
    Lance.__init__ = _L

    # Replace the loader with one that:
    #   1. swaps Linear -> WQLinearINT4 (allocating quant buffers on device)
    #   2. streams the pass-through bf16 weights into the model
    #   3. streams the quant buffers into the swapped modules
    def _awq_loader(model, model_args):
        print("[awq-loader] swapping linears -> WQLinearINT4")
        modules_by_name = swap_to_awq(model, awq_dir)
        print("[awq-loader] streaming pass-through bf16 weights")
        stream_pass_through_weights(model, awq_dir)
        print("[awq-loader] streaming quant buffers")
        stream_awq_buffers(modules_by_name, awq_dir)
        if torch.cuda.is_available():
            print(f"[awq-loader] cuda mem: "
                  f"{torch.cuda.memory_allocated()/1e9:.2f} GB")
        class _M:
            missing_keys: list[str] = []
            unexpected_keys: list[str] = []
        return _M()

    IL.init_from_model_path_if_needed = _awq_loader
    print("[patch] meta-init + AWQ-aware loader installed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model_path", required=True,
                    help="dir with original Lance llm_config.json + tokenizer")
    ap.add_argument("--awq_dir", type=Path, required=True)
    ap.add_argument("--vit_path", default="downloads/Qwen2.5-VL-ViT")
    ap.add_argument("--resolution", default=None)
    ap.add_argument("--save_path_gen", default=None)
    ap.add_argument("--num_frames", type=int, default=50)
    ap.add_argument("--video_height", type=int, default=768)
    ap.add_argument("--video_width", type=int, default=768)
    ap.add_argument("--validation_num_timesteps", type=int, default=30)
    ap.add_argument("--mode", choices=["ondemand", "cached"], default="cached",
                    help="ondemand = dequant every fwd (low VRAM, ~10x slow); "
                         "cached = dequant once per linear (peaks VRAM, ~baseline speed)")
    args = ap.parse_args()
    WQLinearINT4.MODE = args.mode
    print(f"[mode] WQLinearINT4.MODE = {args.mode}")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("POSITION_EMBEDDING_3D_VERSION", "v2")
    os.environ.setdefault("EXP_HW_20250819", "False")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    if args.save_path_gen is None:
        tag = args.awq_dir.name
        args.save_path_gen = f"results/{tag}_{args.task}_{time.strftime('%Y%m%d_%H%M%S')}"
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

    _patch_for_meta_then_awq(args.awq_dir.resolve())

    import inference_lance
    inference_lance.main()


if __name__ == "__main__":
    main()
