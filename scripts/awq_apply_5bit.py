"""5-bit AWQ INT5 quantization variant.

Differs from awq_apply.py only in:
  - n_bit = 5 (32 levels instead of 16)
  - Storage: uint8 (1 byte per code, wastes 3 bits/code — simpler than
    packed 5-bit. Storage ~2x larger than packed INT4 but still 3x smaller
    than bf16. We pick simplicity over the last 30% of compression.)
  - meta scheme tag: asym_int5_grouped_awq

Runtime: see patches/quantized_linear.py WQLinearINT5 (the byte-per-code
storage means dequant is `(byte - zero) * scale`, no nibble unpack needed).
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


FUSION_GROUPS = {
    "input_layernorm":                  ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "input_layernorm_moe_gen":          ["self_attn.q_proj_moe_gen", "self_attn.k_proj_moe_gen", "self_attn.v_proj_moe_gen"],
    "post_attention_layernorm":         ["mlp.gate_proj", "mlp.up_proj"],
    "post_attention_layernorm_moe_gen": ["mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj"],
}
NO_FUSE_LINEARS = ["self_attn.o_proj", "self_attn.o_proj_moe_gen",
                   "mlp.down_proj",    "mlp_moe_gen.down_proj"]


@torch.no_grad()
def quantize_per_group_5bit(w_fp: torch.Tensor, group_size: int):
    """Per-group asymmetric INT5 quantization. Returns
    (codes uint8 [out, in],  scales bf16 [out, n_groups],  zeros uint8 [out, n_groups]).
    Codes stored as full bytes (5 bits used, top 3 zero)."""
    w = w_fp.to(torch.float32)
    out_features, in_features = w.shape
    if in_features % group_size != 0:
        for g in (64, 32, 16, in_features):
            if in_features % g == 0:
                group_size = g
                break
    n_groups = in_features // group_size
    w_grp = w.reshape(out_features, n_groups, group_size)
    max_v = w_grp.amax(dim=-1)
    min_v = w_grp.amin(dim=-1)
    qmax = 31                                                  # 5-bit max
    scales = ((max_v - min_v) / qmax).clamp(min=1e-8)
    zeros = (-min_v / scales).round().clamp(0, qmax).to(torch.uint8)
    q = torch.clamp((w_grp / scales.unsqueeze(-1) + zeros.unsqueeze(-1)).round(),
                     0, qmax).to(torch.uint8)
    codes = q.reshape(out_features, in_features)
    return codes, scales.to(torch.bfloat16), zeros, group_size


@torch.no_grad()
def awq_search_scale_5bit(w_list: List[torch.Tensor], act_mean_list,
                          group_size: int, n_grid: int = 20,
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
    qmax = 31
    for i in range(n_grid + 1):
        alpha = i / n_grid
        s = (act.pow(alpha) / w_max.pow(1.0 - alpha)).clamp(min=1e-5)
        s = s / (s.max() * s.min()).sqrt()
        err = 0.0
        for w, org in zip(w_list, org_outs):
            w_scaled = w.to(device).float() * s.view(1, -1)
            w_grp = w_scaled.reshape(w.shape[0], -1, group_size)
            max_v, min_v = w_grp.amax(-1), w_grp.amin(-1)
            sc = ((max_v - min_v) / qmax).clamp(min=1e-5)
            z = (-min_v / sc).round()
            q = torch.clamp((w_grp / sc.unsqueeze(-1) + z.unsqueeze(-1)).round(), 0, qmax)
            w_dq = (q - z.unsqueeze(-1)) * sc.unsqueeze(-1)
            w_dq = w_dq.reshape(w.shape)
            err += ((x / s.view(1, -1)) @ w_dq.t() - org).pow(2).mean().item()
        if err < best_err:
            best_err, best_alpha = err, alpha

    s = (act.pow(best_alpha) / w_max.pow(1.0 - best_alpha)).clamp(min=1e-5)
    return (s / (s.max() * s.min()).sqrt()).to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--group_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stats_d = torch.load(args.stats, map_location="cpu", weights_only=False)
    stats = stats_d["stats"]

    def act_mean(name):
        s = stats.get(name)
        return None if (s is None or s["n_tokens"] == 0) else s["sum_abs"].float() / s["n_tokens"]

    print(f"loading {args.src}")
    full_sd: dict[str, torch.Tensor] = {}
    with safe_open(str(args.src), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            full_sd[k] = t.to(torch.bfloat16) if t.is_floating_point() else t

    layer_idxs = sorted({int(k.split(".")[3])
                          for k in full_sd
                          if k.startswith("language_model.model.layers.")})
    print(f"layers: {len(layer_idxs)}")

    out_sd: Dict[str, torch.Tensor] = {}
    meta: Dict[str, dict] = {}
    awq_applied = plain_quant = 0

    for li in layer_idxs:
        prefix = f"language_model.model.layers.{li}"
        for norm_sub, consumer_subs in FUSION_GROUPS.items():
            norm_key = f"{prefix}.{norm_sub}.weight"
            if norm_key not in full_sd:
                continue
            w_keys = [(s, f"{prefix}.{s}.weight") for s in consumer_subs
                       if f"{prefix}.{s}.weight" in full_sd]
            if not w_keys:
                continue
            acts = [act_mean(f"{prefix}.{s}") for s, _ in w_keys]
            ws = [full_sd[wk] for _, wk in w_keys]
            s = awq_search_scale_5bit(ws, acts, args.group_size, device=args.device)
            if s is not None:
                full_sd[norm_key] = (full_sd[norm_key].to(args.device).float() / s).to(torch.bfloat16).cpu()
            for sub, wk in w_keys:
                w_t = full_sd[wk].to(args.device).float()
                if s is not None:
                    w_t = w_t * s.view(1, -1)
                codes, sc, zr, gs = quantize_per_group_5bit(w_t, args.group_size)
                base = wk[:-len(".weight")]
                out_sd[f"{base}.qweight"] = codes.cpu()        # uint8 [out, in]
                out_sd[f"{base}.scales"] = sc.cpu()
                out_sd[f"{base}.zeros"] = zr.cpu()
                meta[wk] = {"shape": list(full_sd[wk].shape), "n_bit": 5,
                            "group_size": gs,
                            "scheme": "asym_int5_grouped_awq" if s is not None else "asym_int5_grouped_minmax",
                            "storage": "uint8_per_code"}
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
            codes, sc, zr, gs = quantize_per_group_5bit(w_t, args.group_size)
            base = wk[:-len(".weight")]
            out_sd[f"{base}.qweight"] = codes.cpu()
            out_sd[f"{base}.scales"] = sc.cpu()
            out_sd[f"{base}.zeros"] = zr.cpu()
            meta[wk] = {"shape": list(full_sd[wk].shape), "n_bit": 5,
                        "group_size": gs, "scheme": "asym_int5_grouped_minmax",
                        "storage": "uint8_per_code"}
            bk = wk.replace(".weight", ".bias")
            if bk in full_sd:
                out_sd[bk] = full_sd[bk]
            plain_quant += 1

        if (li + 1) % 8 == 0:
            print(f"  layer {li+1}/{len(layer_idxs)}: awq={awq_applied} plain={plain_quant}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    quantized_keys = set()
    for wk in meta:
        base = wk[:-len(".weight")]
        quantized_keys.update([f"{base}.qweight", f"{base}.scales",
                                f"{base}.zeros", wk, f"{base}.bias"])

    for k, v in full_sd.items():
        if k in quantized_keys and not k.endswith(".bias"):
            continue
        out_sd[k] = v

    save_path = args.out / "awq_state_dict.safetensors"
    save_file(out_sd, str(save_path))
    (args.out / "awq_meta.json").write_text(json.dumps({
        "n_bit": 5, "group_size": args.group_size,
        "scheme": "asym_int5_grouped_awq_uint8_storage",
        "awq_linears": awq_applied, "plain_linears": plain_quant,
        "tasks_calibrated": stats_d.get("tasks", []),
        "per_weight": meta,
    }, indent=2))
    print(f"\n[done] awq={awq_applied}, plain={plain_quant}")
    print(f"[done] wrote {save_path} ({save_path.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
