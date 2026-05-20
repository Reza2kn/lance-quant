"""Plain per-group INT4 quantization of the Lance language_model.

NO AWQ scale search — just min-max per group_size channels. Use this as a
quality floor before layering on AWQ calibration. The output format matches
`patches/quantized_linear.py` so we can verify end-to-end the swap-in works.

Usage:
  python quantize_int4_minmax.py \
      --src downloads/Lance_3B_Video/model.safetensors \
      --out ../models/Lance_3B_Video-INT4-MinMax \
      --group_size 128
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


QUANT_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "self_attn.q_proj_moe_gen", "self_attn.k_proj_moe_gen",
    "self_attn.v_proj_moe_gen", "self_attn.o_proj_moe_gen",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj",
)


def is_quantize_target(key: str) -> bool:
    """Targets the LLM Linear weights but NOT vit, norms, embeddings, time_embedder,
    llm2vae/vae2llm projections, latent_pos_embed, or biases."""
    if not key.endswith(".weight"):
        return False
    if key.startswith("vit_model."):
        return False
    if any(p in key for p in ("time_embedder", "llm2vae", "vae2llm",
                              "latent_pos_embed", "embed_tokens", "rotary_emb",
                              "layernorm", "norm")):
        return False
    base = key[:-len(".weight")]
    if base.endswith(QUANT_SUFFIXES):
        return True
    # NOTE: lm_head intentionally kept in bf16. Inference_lance asserts on
    # `lm_head.weight.data_ptr()` and quantizing it would break that check.
    # It's also numerically the most sensitive linear (vocab projection).
    return False


def pack_int4(q_uint8: torch.Tensor) -> torch.Tensor:
    """Pack pairs of nibbles along the last axis into bytes.
    Input shape: [..., in_features], values 0..15. Output: [..., in_features // 2].
    Even index goes to low nibble, odd to high nibble (matches WQLinearINT4 unpack)."""
    assert q_uint8.dtype == torch.uint8
    assert q_uint8.shape[-1] % 2 == 0
    lo = q_uint8[..., 0::2] & 0xF
    hi = (q_uint8[..., 1::2] & 0xF) << 4
    return (lo | hi).to(torch.uint8)


@torch.no_grad()
def quantize_weight(w_bf16: torch.Tensor, n_bit: int, group_size: int):
    """Returns (qweight_packed, scales_bf16, zeros_uint8). Asymmetric per-group."""
    w = w_bf16.to(torch.float32)
    out_features, in_features = w.shape

    # If group_size doesn't divide, fall back to a finer one
    if in_features % group_size != 0:
        for g in (128, 64, 32, 16, in_features):
            if in_features % g == 0:
                group_size = g
                break

    n_groups = in_features // group_size
    w_grp = w.reshape(out_features, n_groups, group_size)
    max_v = w_grp.amax(dim=-1)
    min_v = w_grp.amin(dim=-1)
    qmax = (1 << n_bit) - 1
    scales = ((max_v - min_v) / qmax).clamp(min=1e-8)
    zeros = (-min_v / scales).round().clamp(0, qmax).to(torch.uint8)
    q = torch.clamp(
        (w_grp / scales.unsqueeze(-1) + zeros.unsqueeze(-1)).round(),
        0, qmax,
    ).to(torch.uint8)
    q = q.reshape(out_features, in_features)
    packed = pack_int4(q)
    return packed, scales.to(torch.bfloat16), zeros, group_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="Lance model.safetensors (the unquantized one)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output dir for awq_state_dict.safetensors + awq_meta.json")
    ap.add_argument("--n_bit", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    out_sd: dict[str, torch.Tensor] = {}
    meta: dict[str, dict] = {}
    n_quant, n_pass = 0, 0
    t0 = time.time()
    bytes_before = 0
    bytes_after = 0

    print(f"reading {args.src}")
    with safe_open(str(args.src), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"  {len(keys)} tensors")
        for i, k in enumerate(keys):
            t = f.get_tensor(k)
            # always cast floating point pass-through to bf16
            if not is_quantize_target(k):
                if t.is_floating_point():
                    out_sd[k] = t.to(torch.bfloat16)
                else:
                    out_sd[k] = t
                n_pass += 1
                bytes_before += t.numel() * t.element_size()
                bytes_after += out_sd[k].numel() * out_sd[k].element_size()
                continue
            # quantize
            t_bf16 = t.to(torch.bfloat16)
            t_gpu = t_bf16.to(args.device)
            packed, scales, zeros, gs = quantize_weight(t_gpu, args.n_bit, args.group_size)
            base = k[:-len(".weight")]
            out_sd[f"{base}.qweight"] = packed.cpu()
            out_sd[f"{base}.scales"]  = scales.cpu()
            out_sd[f"{base}.zeros"]   = zeros.cpu()
            meta[k] = {
                "shape": list(t.shape),
                "n_bit": args.n_bit,
                "group_size": gs,
                "scheme": "asym_int4_grouped_minmax",
            }
            n_quant += 1
            bytes_before += t.numel() * t.element_size()
            bytes_after += (packed.numel() + scales.numel() * 2 + zeros.numel())
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(keys)}] quant={n_quant} pass={n_pass} "
                      f"so_far_GB={bytes_after/1e9:.2f}")
            del t, t_bf16, t_gpu, packed, scales, zeros
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"writing {args.out / 'awq_state_dict.safetensors'}")
    save_file(out_sd, str(args.out / "awq_state_dict.safetensors"))
    (args.out / "awq_meta.json").write_text(json.dumps({
        "n_bit": args.n_bit,
        "group_size": args.group_size,
        "scheme": "asym_int4_grouped_minmax",
        "calibration_samples": 0,
        "per_weight": meta,
    }, indent=2))

    print()
    print(f"Quantized {n_quant} weights, passed through {n_pass}.")
    print(f"Source size:       {bytes_before / 1e9:7.2f} GB")
    print(f"Quant+pass size:   {bytes_after / 1e9:7.2f} GB "
          f"(saving {100*(1-bytes_after/bytes_before):.1f}%)")
    print(f"Time:              {time.time() - t0:7.1f} s")
    print(f"Output:            {args.out}")


if __name__ == "__main__":
    main()
