"""Apply NVFP4 quantization to Lance language_model.

NVFP4 (NVIDIA FP4) — 4-bit floating-point E2M1 with per-block FP8 E4M3 scales,
block_size=16. Designed for Blackwell tensor cores; ~5x faster than INT4 on
sm_120 GPUs with proper kernels (vLLM 0.7+, TensorRT-LLM).

Format per weight:
  4 bits in {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}  (E2M1 codes)
  1 byte FP8 E4M3 scale per 16-element block
  Optional FP32 global scale per tensor (omitted here; per-block is enough)

Average storage: 4 + 8/16 = 4.5 bits/weight.

We reuse the AWQ activation calibration stats (from awq_merge_stats.py) and
apply AWQ-style scale equalization first (norm /= s, w *= s), then pack each
row into NVFP4 blocks. The eval-time WQLinearNVFP4 dequant + matmul lives in
patches/quantized_linear.py (mirror of WQLinearINT4 with the FP4 LUT).
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# E2M1 lookup table (NVFP4 representable values), index = 4-bit code
# Sign bit is the MSB. Encoding follows the NVIDIA NVFP4 / OCP MXFP4 spec.
#
#   code | sign(1) exp(2) mant(1) | value
#   0000 |    0      00      0    |  +0
#   0001 |    0      00      1    |  +0.5
#   0010 |    0      01      0    |  +1
#   0011 |    0      01      1    |  +1.5
#   0100 |    0      10      0    |  +2
#   0101 |    0      10      1    |  +3
#   0110 |    0      11      0    |  +4
#   0111 |    0      11      1    |  +6
#   1xxx | sign bit flips above values
# ---------------------------------------------------------------------------

FP4_LUT_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_MAX = 6.0


def _quantize_fp4_block(block_fp32: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantize one 16-element block to FP4 codes + return its FP8 scale.
    Returns (codes uint8 [16], scale float). Codes range 0..15.
    """
    max_abs = block_fp32.abs().max().item()
    if max_abs == 0:
        return torch.zeros(block_fp32.shape, dtype=torch.uint8, device=block_fp32.device), 1.0
    scale = max_abs / FP4_MAX                # block max ↦ 6.0
    # encode FP8 E4M3 scale: clamp to E4M3 range [-448, 448] and round
    scale = max(scale, 1e-8)
    return _vectorized_fp4_quant(block_fp32 / scale), scale


def _vectorized_fp4_quant(x_normalised: torch.Tensor) -> torch.Tensor:
    """x in roughly [-6, 6]. Returns uint8 codes 0..15 (sign bit = MSB)."""
    lut = FP4_LUT_POS.to(x_normalised.device, dtype=x_normalised.dtype)  # [8]
    sign = (x_normalised < 0).to(torch.uint8)
    abs_x = x_normalised.abs()
    # nearest neighbour in the positive LUT
    dist = (abs_x.unsqueeze(-1) - lut.view(*([1] * abs_x.dim()), 8)).abs()
    code = dist.argmin(dim=-1).to(torch.uint8)        # 0..7
    code = code | (sign << 3)                          # set sign in MSB
    return code


# ---------------------------------------------------------------------------
# Block quantize a 2D weight matrix
# ---------------------------------------------------------------------------


def quantize_fp4_per_block(w: torch.Tensor, block_size: int = 16):
    """w: [out, in], any float dtype. Returns:
        codes_packed uint8 [out, in // 2]
        scales_fp8   uint8 [out, in // block_size]   (stored as float8_e4m3fn cast to uint8 bytes)
        scales_bf16  bf16  [out, in // block_size]   (dequant-friendly fallback;
                                                       store this if FP8 dtype isn't available)
    """
    out_features, in_features = w.shape
    assert in_features % block_size == 0, f"in_features {in_features} not divisible by {block_size}"
    n_blocks = in_features // block_size

    w_blocks = w.to(torch.float32).reshape(out_features, n_blocks, block_size)
    block_max = w_blocks.abs().amax(dim=-1).clamp(min=1e-8)        # [out, nb]
    scales = block_max / FP4_MAX                                    # [out, nb]
    # cast scale to FP8 E4M3 if available (PyTorch ≥ 2.1)
    if hasattr(torch, "float8_e4m3fn"):
        scales_fp8 = scales.to(torch.float8_e4m3fn)
        # store underlying bytes; dtype info preserved in meta
        scales_store = scales_fp8.view(torch.uint8)
    else:
        scales_store = scales.to(torch.bfloat16).view(torch.uint8)  # 2 bytes/scale

    # quantize each element
    w_norm = w_blocks / scales.unsqueeze(-1)                        # [out, nb, bs]
    lut = FP4_LUT_POS.to(w_norm.device, dtype=torch.float32)
    sign = (w_norm < 0).to(torch.uint8)
    abs_w = w_norm.abs().clamp(max=FP4_MAX)
    dist = (abs_w.unsqueeze(-1) - lut.view(1, 1, 1, 8)).abs()
    code = dist.argmin(dim=-1).to(torch.uint8)                      # 0..7
    code = code | (sign << 3)                                       # |s|eee|m| -> 4 bits

    codes_flat = code.reshape(out_features, in_features)
    # pack two nibbles per byte (even ↦ low, odd ↦ high)
    lo = codes_flat[..., 0::2] & 0xF
    hi = (codes_flat[..., 1::2] & 0xF) << 4
    packed = (lo | hi).to(torch.uint8)

    # also keep a bf16 copy of scales for the safe-mode dequant kernel
    scales_bf16 = scales.to(torch.bfloat16)
    return packed, scales_store, scales_bf16


# ---------------------------------------------------------------------------
# AWQ scale search — identical to awq_apply.py but block_size=16
# ---------------------------------------------------------------------------


@torch.no_grad()
def awq_search_scale(w_list: List[torch.Tensor], act_mean_list,
                     block_size: int, n_grid: int = 20,
                     device: str = "cuda") -> torch.Tensor | None:
    in_features = w_list[0].shape[1]
    valid = [a for a in act_mean_list if a is not None and a.sum().item() > 0]
    if not valid:
        return None
    act = torch.stack([a.to(device).float() for a in valid], dim=0).mean(dim=0).clamp(min=1e-5)

    torch.manual_seed(0xC0DE)
    x = torch.randn(512, in_features, device=device, dtype=torch.float32) * act
    org_outs = [x @ w.to(device).float().t() for w in w_list]

    w_max = torch.stack(
        [w.to(device).float().abs().amax(dim=0).clamp(min=1e-5) for w in w_list],
        dim=0,
    ).mean(dim=0)

    best_alpha, best_err = 0.0, float("inf")
    for i in range(n_grid + 1):
        alpha = i / n_grid
        s = (act.pow(alpha) / w_max.pow(1.0 - alpha)).clamp(min=1e-5)
        s = s / (s.max() * s.min()).sqrt()

        err = 0.0
        for w, org in zip(w_list, org_outs):
            w_scaled = w.to(device).float() * s.view(1, -1)
            # FP4 fake quant (inline, no packing — fast enough for grid search)
            w_blk = w_scaled.reshape(w.shape[0], -1, block_size)
            blk_max = w_blk.abs().amax(dim=-1).clamp(min=1e-5)
            sc = blk_max / FP4_MAX
            w_norm = w_blk / sc.unsqueeze(-1)
            lut = FP4_LUT_POS.to(device, dtype=torch.float32)
            sign = (w_norm < 0).to(torch.float32) * -2 + 1                 # ±1
            abs_w = w_norm.abs().clamp(max=FP4_MAX)
            dist = (abs_w.unsqueeze(-1) - lut.view(1, 1, 1, 8)).abs()
            code = dist.argmin(dim=-1)
            w_dq = sign * lut[code] * sc.unsqueeze(-1)
            w_dq = w_dq.reshape(w.shape)
            err += ((x / s.view(1, -1)) @ w_dq.t() - org).pow(2).mean().item()

        if err < best_err:
            best_err = err
            best_alpha = alpha

    s = (act.pow(best_alpha) / w_max.pow(1.0 - best_alpha)).clamp(min=1e-5)
    return (s / (s.max() * s.min()).sqrt()).to(torch.float32)


# ---------------------------------------------------------------------------
# Fusion groups (identical to awq_apply.py)
# ---------------------------------------------------------------------------


FUSION_GROUPS = {
    "input_layernorm":                  ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "input_layernorm_moe_gen":          ["self_attn.q_proj_moe_gen", "self_attn.k_proj_moe_gen", "self_attn.v_proj_moe_gen"],
    "post_attention_layernorm":         ["mlp.gate_proj", "mlp.up_proj"],
    "post_attention_layernorm_moe_gen": ["mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj"],
}
NO_FUSE_LINEARS = ["self_attn.o_proj", "self_attn.o_proj_moe_gen",
                   "mlp.down_proj",    "mlp_moe_gen.down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True,
                    help="merged AWQ activation stats; optional but recommended")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    stats_d = torch.load(args.stats, map_location="cpu", weights_only=False)
    stats = stats_d["stats"]

    def act_mean(name: str):
        s = stats.get(name)
        if s is None or s["n_tokens"] == 0:
            return None
        return s["sum_abs"].float() / s["n_tokens"]

    print(f"loading {args.src}")
    full_sd: dict[str, torch.Tensor] = {}
    with safe_open(str(args.src), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            full_sd[k] = t.to(torch.bfloat16) if t.is_floating_point() else t

    layer_idxs = set()
    for k in full_sd:
        parts = k.split(".")
        if (len(parts) >= 4 and parts[0] == "language_model"
                and parts[1] == "model" and parts[2] == "layers"):
            layer_idxs.add(int(parts[3]))
    print(f"transformer layers: {len(layer_idxs)}")

    out_sd: Dict[str, torch.Tensor] = {}
    meta: Dict[str, dict] = {}
    awq_applied = 0
    plain_quant = 0

    for li in sorted(layer_idxs):
        prefix = f"language_model.model.layers.{li}"
        for norm_sub, consumer_subs in FUSION_GROUPS.items():
            norm_key = f"{prefix}.{norm_sub}.weight"
            if norm_key not in full_sd:
                continue
            w_keys = []
            for sub in consumer_subs:
                wk = f"{prefix}.{sub}.weight"
                if wk in full_sd:
                    w_keys.append((sub, wk))
            if not w_keys:
                continue

            acts = [act_mean(f"{prefix}.{sub}") for sub, _ in w_keys]
            ws = [full_sd[wk] for _, wk in w_keys]

            s = awq_search_scale(ws, acts, args.block_size, device=args.device)
            if s is not None:
                full_sd[norm_key] = (full_sd[norm_key].to(args.device).float() / s).to(torch.bfloat16).cpu()

            for sub, wk in w_keys:
                w_t = full_sd[wk].to(args.device).float()
                if s is not None:
                    w_t = w_t * s.view(1, -1)
                packed, sc_fp8, sc_bf16 = quantize_fp4_per_block(w_t, args.block_size)
                base = wk[:-len(".weight")]
                out_sd[f"{base}.qweight"] = packed.cpu()
                out_sd[f"{base}.scales_fp8"] = sc_fp8.cpu()
                out_sd[f"{base}.scales_bf16"] = sc_bf16.cpu()
                meta[wk] = {
                    "shape": list(full_sd[wk].shape),
                    "block_size": args.block_size,
                    "scheme": "nvfp4_awq" if s is not None else "nvfp4_minmax",
                }
                bk = wk.replace(".weight", ".bias")
                if bk in full_sd:
                    out_sd[bk] = full_sd[bk]
                awq_applied += int(s is not None)
                plain_quant += int(s is None)

        for sub in NO_FUSE_LINEARS:
            wk = f"{prefix}.{sub}.weight"
            if wk not in full_sd:
                continue
            w_t = full_sd[wk].to(args.device).float()
            packed, sc_fp8, sc_bf16 = quantize_fp4_per_block(w_t, args.block_size)
            base = wk[:-len(".weight")]
            out_sd[f"{base}.qweight"] = packed.cpu()
            out_sd[f"{base}.scales_fp8"] = sc_fp8.cpu()
            out_sd[f"{base}.scales_bf16"] = sc_bf16.cpu()
            meta[wk] = {"shape": list(full_sd[wk].shape),
                        "block_size": args.block_size, "scheme": "nvfp4_minmax"}
            bk = wk.replace(".weight", ".bias")
            if bk in full_sd:
                out_sd[bk] = full_sd[bk]
            plain_quant += 1

        if (li + 1) % 4 == 0:
            print(f"  layer {li+1}/{len(layer_idxs)}: awq_so_far={awq_applied} plain={plain_quant}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    quantized_keys = set()
    for wk in meta:
        base = wk[:-len(".weight")]
        quantized_keys.update([f"{base}.qweight", f"{base}.scales_fp8",
                                f"{base}.scales_bf16", wk, f"{base}.bias"])

    for k, v in full_sd.items():
        if k in quantized_keys and not k.endswith(".bias"):
            continue
        out_sd[k] = v

    save_path = args.out / "nvfp4_state_dict.safetensors"
    save_file(out_sd, str(save_path))
    (args.out / "nvfp4_meta.json").write_text(json.dumps({
        "block_size": args.block_size,
        "scheme": "nvfp4_e2m1_block_fp8scale",
        "awq_linears": awq_applied,
        "plain_linears": plain_quant,
        "tasks_calibrated": stats_d.get("tasks", []),
        "fp4_lut_pos": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        "per_weight": meta,
    }, indent=2))

    sz = save_path.stat().st_size
    print(f"\n[done] awq={awq_applied}, plain={plain_quant}")
    print(f"[done] wrote {save_path} ({sz/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
