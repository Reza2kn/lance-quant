"""Apply AWQ scales + INT4 grouped quantization to Lance language_model weights.

Input:
  --src        path to Lance model.safetensors (the unquantized model)
  --stats      path to merged act_stats.pt (from awq_merge_stats.py)
  --out        output directory; writes awq_state_dict.safetensors + awq_meta.json

For each transformer layer, we:

  1) For (input_layernorm, [q_proj, k_proj, v_proj]) AND
        (post_attention_layernorm, [mlp.gate_proj, mlp.up_proj]) AND
        same for `_moe_gen` variants:
     - average activation magnitudes across the consumer linears
     - grid-search alpha in {0..1} for s = act^alpha / w_max^(1-alpha)
     - normalize s so its geometric mean is 1
     - divide the preceding norm's weight by s
     - multiply each consumer linear's weight columns by s
     - quantize the scaled weight to per-group INT4 asymmetric

  2) For o_proj, mlp.down_proj (no clean fuse target): plain per-group INT4.

  3) lm_head: kept in bf16 (inference_lance asserts on its .weight pointer).

Output layout matches `patches/quantized_linear.py` so the same swap-in
loader works.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

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


FUSION_GROUPS = {
    "input_layernorm":                  ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "input_layernorm_moe_gen":          ["self_attn.q_proj_moe_gen", "self_attn.k_proj_moe_gen", "self_attn.v_proj_moe_gen"],
    "post_attention_layernorm":         ["mlp.gate_proj", "mlp.up_proj"],
    "post_attention_layernorm_moe_gen": ["mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj"],
}

# Linears that get plain per-group INT4 (no fusion target):
NO_FUSE_LINEARS = [
    "self_attn.o_proj", "self_attn.o_proj_moe_gen",
    "mlp.down_proj",    "mlp_moe_gen.down_proj",
]


# ---------------------------------------------------------------------------


def pack_int4(q_uint8: torch.Tensor) -> torch.Tensor:
    assert q_uint8.dtype == torch.uint8 and q_uint8.shape[-1] % 2 == 0
    lo = q_uint8[..., 0::2] & 0xF
    hi = (q_uint8[..., 1::2] & 0xF) << 4
    return (lo | hi).to(torch.uint8)


@torch.no_grad()
def quantize_per_group(w_fp: torch.Tensor, n_bit: int, group_size: int):
    """Per-group asymmetric INT4. Returns (packed, scales_bf16, zeros_uint8, gs_used)."""
    w = w_fp.to(torch.float32)
    out_features, in_features = w.shape
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
        (w_grp / scales.unsqueeze(-1) + zeros.unsqueeze(-1)).round(), 0, qmax,
    ).to(torch.uint8)
    q = q.reshape(out_features, in_features)
    packed = pack_int4(q)
    return packed, scales.to(torch.bfloat16), zeros, group_size


@torch.no_grad()
def awq_search_scale(w_list: List[torch.Tensor], act_mean_list: List[torch.Tensor | None],
                     n_bit: int, group_size: int, n_grid: int = 20,
                     device: str = "cuda") -> torch.Tensor | None:
    """Search alpha that minimizes per-output quant error summed over the
    consumer linears that share this fusion group. Returns s [in_features]
    or None if no valid activation data is available."""

    # Use first weight's in_features
    in_features = w_list[0].shape[1]

    # average activation magnitudes (per input channel) across consumers
    valid_acts = [a for a in act_mean_list if a is not None and a.sum().item() > 0]
    if not valid_acts:
        return None
    act = torch.stack([a.to(device).float() for a in valid_acts], dim=0).mean(dim=0)
    act = act.clamp(min=1e-5)

    # synthetic input matching activation magnitudes
    torch.manual_seed(0xC0DE)
    x = torch.randn(512, in_features, device=device, dtype=torch.float32) * act

    # baseline outputs (full precision)
    org_outs = []
    for w in w_list:
        org_outs.append(x @ w.to(device).float().t())

    best_alpha, best_err = 0.0, float("inf")
    # per-output channel max abs of each weight (needed for s formula)
    w_max_list = [w.to(device).float().abs().amax(dim=0).clamp(min=1e-5) for w in w_list]
    w_max = torch.stack(w_max_list, dim=0).mean(dim=0)  # avg across consumers

    for i in range(n_grid + 1):
        alpha = i / n_grid
        s = (act.pow(alpha) / w_max.pow(1.0 - alpha)).clamp(min=1e-5)
        s = s / (s.max() * s.min()).sqrt()           # normalize geom-mean ~1

        err = 0.0
        for w, org in zip(w_list, org_outs):
            w_scaled = w.to(device).float() * s.view(1, -1)
            # per-group fake quant
            n_groups = in_features // group_size
            w_grp = w_scaled.reshape(w.shape[0], n_groups, group_size)
            max_v, min_v = w_grp.amax(-1), w_grp.amin(-1)
            qmax = (1 << n_bit) - 1
            sc = ((max_v - min_v) / qmax).clamp(min=1e-5)
            z = (-min_v / sc).round()
            q = torch.clamp((w_grp / sc.unsqueeze(-1) + z.unsqueeze(-1)).round(), 0, qmax)
            w_dq = (q - z.unsqueeze(-1)) * sc.unsqueeze(-1)
            w_dq = w_dq.reshape(w.shape)
            out = (x / s.view(1, -1)) @ w_dq.t()
            err += (out - org).pow(2).mean().item()

        if err < best_err:
            best_err = err
            best_alpha = alpha

    s = (act.pow(best_alpha) / w_max.pow(1.0 - best_alpha)).clamp(min=1e-5)
    s = s / (s.max() * s.min()).sqrt()
    return s.to(torch.float32)


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="Lance model.safetensors (unquantized)")
    ap.add_argument("--stats", type=Path, required=True,
                    help="merged activation stats .pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n_bit", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # 1) Load activation stats
    print(f"loading {args.stats}")
    stats_d = torch.load(args.stats, map_location="cpu", weights_only=False)
    stats = stats_d["stats"]

    def act_mean(name: str) -> torch.Tensor | None:
        s = stats.get(name)
        if s is None or s["n_tokens"] == 0:
            return None
        return s["sum_abs"].float() / s["n_tokens"]

    # 2) Load full state dict (we need to modify norms then re-save)
    print(f"loading {args.src}")
    full_sd: dict[str, torch.Tensor] = {}
    with safe_open(str(args.src), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            full_sd[k] = t.to(torch.bfloat16) if t.is_floating_point() else t

    # 3) Determine layer indices present
    layer_idxs = set()
    for k in full_sd:
        parts = k.split(".")
        if (len(parts) >= 4 and parts[0] == "language_model"
                and parts[1] == "model" and parts[2] == "layers"):
            layer_idxs.add(int(parts[3]))
    print(f"transformer layers: {len(layer_idxs)}")

    # 4) For each layer + fusion group, compute scale and apply
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

            s = awq_search_scale(ws, acts, args.n_bit, args.group_size, device=args.device)
            if s is None:
                # No activation data for any of these consumers -> plain per-group INT4
                for sub, wk in w_keys:
                    packed, sc, zr, gs = quantize_per_group(
                        full_sd[wk].to(args.device), args.n_bit, args.group_size)
                    base = wk[:-len(".weight")]
                    out_sd[f"{base}.qweight"] = packed.cpu()
                    out_sd[f"{base}.scales"] = sc.cpu()
                    out_sd[f"{base}.zeros"] = zr.cpu()
                    meta[wk] = {"shape": list(full_sd[wk].shape), "n_bit": args.n_bit,
                                "group_size": gs, "scheme": "asym_int4_grouped_no_awq"}
                    plain_quant += 1
                continue

            # Apply: norm /= s, weight *= s on input axis, then quant
            new_norm = full_sd[norm_key].to(args.device).float() / s
            full_sd[norm_key] = new_norm.to(torch.bfloat16).cpu()

            for sub, wk in w_keys:
                w_scaled = full_sd[wk].to(args.device).float() * s.view(1, -1)
                packed, sc, zr, gs = quantize_per_group(w_scaled, args.n_bit, args.group_size)
                base = wk[:-len(".weight")]
                out_sd[f"{base}.qweight"] = packed.cpu()
                out_sd[f"{base}.scales"] = sc.cpu()
                out_sd[f"{base}.zeros"] = zr.cpu()
                meta[wk] = {"shape": list(full_sd[wk].shape), "n_bit": args.n_bit,
                            "group_size": gs, "scheme": "asym_int4_grouped_awq"}
                # carry bias if present
                bk = wk.replace(".weight", ".bias")
                if bk in full_sd:
                    out_sd[bk] = full_sd[bk]
                awq_applied += 1

        # Quantize no-fuse linears (o_proj, down_proj)
        for sub in NO_FUSE_LINEARS:
            wk = f"{prefix}.{sub}.weight"
            if wk not in full_sd:
                continue
            packed, sc, zr, gs = quantize_per_group(
                full_sd[wk].to(args.device), args.n_bit, args.group_size)
            base = wk[:-len(".weight")]
            out_sd[f"{base}.qweight"] = packed.cpu()
            out_sd[f"{base}.scales"] = sc.cpu()
            out_sd[f"{base}.zeros"] = zr.cpu()
            meta[wk] = {"shape": list(full_sd[wk].shape), "n_bit": args.n_bit,
                        "group_size": gs, "scheme": "asym_int4_grouped_minmax"}
            bk = wk.replace(".weight", ".bias")
            if bk in full_sd:
                out_sd[bk] = full_sd[bk]
            plain_quant += 1

        if (li + 1) % 4 == 0:
            print(f"  layer {li+1}/{len(layer_idxs)}: awq_so_far={awq_applied} plain={plain_quant}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 5) Copy through all non-quantized tensors (norms with fused scale already
    # updated in-place above; everything else pass-through bf16)
    quantized_keys = set()
    for wk in meta:
        base = wk[:-len(".weight")]
        quantized_keys.update([f"{base}.qweight", f"{base}.scales", f"{base}.zeros",
                                wk, f"{base}.bias"])

    for k, v in full_sd.items():
        if k in quantized_keys and not k.endswith(".bias"):
            continue
        out_sd[k] = v

    # 6) Save
    save_path = args.out / "awq_state_dict.safetensors"
    save_file(out_sd, str(save_path))
    (args.out / "awq_meta.json").write_text(json.dumps({
        "n_bit": args.n_bit,
        "group_size": args.group_size,
        "scheme": "asym_int4_grouped_awq",
        "awq_linears": awq_applied,
        "plain_linears": plain_quant,
        "tasks_calibrated": stats_d.get("tasks", []),
        "per_weight": meta,
    }, indent=2))

    sz = save_path.stat().st_size
    print(f"\n[done] awq_applied={awq_applied}, plain={plain_quant}")
    print(f"[done] wrote {save_path} ({sz/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
