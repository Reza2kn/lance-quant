"""AWQ-style INT4 quantization for the Lance language_model.

Why hand-rolled instead of AutoAWQ / llm-compressor?
  * Lance is not a standard HF transformers model. Its decoder layer holds
    parallel `_moe_gen` Linear modules (Mixture-of-Tasks). AutoAWQ's model
    registry doesn't know how to walk these, and llm-compressor's `oneshot()`
    assumes `AutoModelForCausalLM.from_pretrained` works (it doesn't here).
  * AutoAWQ was officially deprecated in 2025 and force-upgrades torch in a way
    that breaks our cu128 / Blackwell environment.
  * The AWQ algorithm itself is small. Doing it inline lets us:
      - calibrate the und path AND the gen path in one model,
      - keep the projection/embedder/VAE-side weights at bf16 untouched,
      - emit a `quantized.safetensors` that loads back into Lance with a
        single-file monkey-patch (`patches/quantized_linear.py`).

Algorithm (per Lin et al., AWQ: Activation-aware Weight Quantization for LLM
Compression and Acceleration, MLSys 2024):

  For each Linear with weight W [out, in] preceded by a norm/embedding whose
  output X feeds it, we
    1. compute s_act = mean(|X|, dim=batch) -> per-input-channel magnitude
       (collected via forward hooks across the calibration set)
    2. find alpha* in {0.0, 0.1, ..., 1.0} that minimizes the MSE
       ||Q(W * s) @ (x / s) - W @ x|| over a small grid, with
            s = s_act^alpha / s_weight^(1-alpha),
       s_weight = max(|W|, dim=0).
    3. fuse s into the preceding layer's output projection (i.e. divide the
       prev layer's last linear weight rows by s), and multiply this layer's
       input cols by s -> mathematically equivalent, but now W's effective
       distribution per input channel is smoother and group-INT4 quantizes
       with much less error.
    4. quantize W with symmetric per-group INT4, group_size=128.

We skip steps 1-3 for layers we can't pair to a "preceding" op (e.g. the lm_head
and the very first input embedding). For those we just do per-group INT4 with
no scale search.

Output layout (in `models/Lance_3B_Video-AWQ-INT4/`):
  - awq_state_dict.safetensors   # quantized weights, packed INT4 + scales
  - awq_meta.json                # which keys are quantized, group_size, etc.
  - rest of weights in bf16 (vit_model, time_embedder, latent_pos_embed,
    llm2vae, vae2llm, norms)

Loader (see patches/quantized_linear.py) swaps the targeted nn.Linear modules
for `WQLinear_INT4` at runtime. Total LLM size goes from ~12 GB (bf16) to
~3.5 GB (4-bit + per-group scales).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------


def pseudo_quantize_tensor(
    w: torch.Tensor,
    n_bit: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-group quantize + immediately dequantize (returns float tensor).
    Used during scale search to evaluate quantization error.
    Returns (w_dq, scales, zeros).
    """
    org_shape = w.shape
    assert w.dim() == 2, "expected 2D linear weight"
    # reshape to (rows, n_groups, group_size)
    if group_size > 0:
        assert w.shape[1] % group_size == 0, f"in_features {w.shape[1]} not divisible by group_size {group_size}"
        w = w.reshape(-1, group_size)

    if symmetric:
        max_abs = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        qmax = (1 << (n_bit - 1)) - 1            # e.g. 7 for 4-bit symmetric
        scales = max_abs / qmax
        zeros = torch.zeros_like(scales)
        q = torch.clamp((w / scales).round(), -qmax - 1, qmax)
        w_dq = q * scales
    else:
        max_v = w.amax(dim=1, keepdim=True)
        min_v = w.amin(dim=1, keepdim=True)
        qmax = (1 << n_bit) - 1                  # 15 for 4-bit
        scales = ((max_v - min_v) / qmax).clamp(min=1e-5)
        zeros = (-min_v / scales).round()
        q = torch.clamp((w / scales + zeros).round(), 0, qmax)
        w_dq = (q - zeros) * scales

    w_dq = w_dq.reshape(org_shape)
    scales = scales.reshape(org_shape[0], -1)
    zeros = zeros.reshape(org_shape[0], -1)
    return w_dq, scales, zeros


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack two INT4 values per byte. q is uint8 [..., 2k] with values 0..15."""
    assert q.dtype == torch.uint8
    assert q.shape[-1] % 2 == 0
    lo = q[..., 0::2] & 0xF
    hi = (q[..., 1::2] & 0xF) << 4
    return (lo | hi).to(torch.uint8)


def unpack_int4(packed: torch.Tensor, last_dim: int) -> torch.Tensor:
    assert packed.dtype == torch.uint8
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    out = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], last_dim)
    return out


# ---------------------------------------------------------------------------
# Linear module families to quantize
#
# Per modeling/lance/qwen2_navit.py:
#   - Standard und path:  q_proj, k_proj, v_proj, o_proj,
#                         mlp.gate_proj, mlp.up_proj, mlp.down_proj
#   - Gen-expert path:    *_moe_gen counterparts of all of the above
#   - lm_head             -> quantize last (no preceding op fusion)
# We skip:
#   - input/output embeddings (tied), norms, time_embedder, llm2vae, vae2llm,
#     latent_pos_embed, vit_model.*  (small or numerically sensitive)
# ---------------------------------------------------------------------------

QUANT_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "self_attn.q_proj_moe_gen", "self_attn.k_proj_moe_gen",
    "self_attn.v_proj_moe_gen", "self_attn.o_proj_moe_gen",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj",
)


def is_quantize_target(name: str) -> bool:
    if name.endswith(".lm_head"):
        return True
    return any(name.endswith(s) for s in QUANT_SUFFIXES) and "language_model" in name


# ---------------------------------------------------------------------------
# Activation magnitude collection
# ---------------------------------------------------------------------------


@dataclass
class ActStats:
    """Running EMA of mean |x| per input channel, plus token count seen."""
    sum_abs: torch.Tensor = None     # [in_features]
    n: int = 0

    def update(self, x: torch.Tensor) -> None:
        # x: [..., in_features]
        x = x.detach().to(torch.float32).reshape(-1, x.shape[-1])
        if x.numel() == 0:
            return
        s = x.abs().sum(dim=0)
        if self.sum_abs is None:
            self.sum_abs = s
        else:
            self.sum_abs = self.sum_abs + s
        self.n += x.shape[0]

    def mean(self) -> torch.Tensor:
        return self.sum_abs / max(self.n, 1)


def install_act_hooks(model: nn.Module, targets: List[str]) -> Tuple[Dict[str, ActStats], list]:
    stats: Dict[str, ActStats] = {n: ActStats() for n in targets}
    handles = []
    name_lookup = {id(m): n for n, m in model.named_modules() if n in targets}

    def make_hook(name: str):
        def hook(module, inputs, output):
            if isinstance(inputs, tuple) and len(inputs) > 0:
                stats[name].update(inputs[0])
        return hook

    for name, module in model.named_modules():
        if name in targets:
            handles.append(module.register_forward_hook(make_hook(name)))
    return stats, handles


# ---------------------------------------------------------------------------
# AWQ scale search
# ---------------------------------------------------------------------------


@torch.no_grad()
def awq_search_scale(
    w: torch.Tensor,           # [out, in]
    act_mean_abs: torch.Tensor,  # [in]
    n_bit: int,
    group_size: int,
    n_grid: int = 20,
) -> torch.Tensor:
    """Find s [in] that minimizes ||Q(W*s) @ (x/s) - W @ x|| under the AWQ
    activation-aware criterion. We approximate the loss with a per-output
    Frobenius error of pseudo-quantized W*s vs W on random x ~ |act_mean_abs|."""

    device = w.device
    if act_mean_abs is None or act_mean_abs.numel() == 0 or act_mean_abs.sum() == 0:
        # no activation info; fall back to plain per-group INT4
        return torch.ones(w.shape[1], device=device, dtype=torch.float32)

    act = act_mean_abs.to(device=device, dtype=torch.float32).clamp(min=1e-5)
    # synthetic input matching activation magnitudes
    x = torch.randn(512, w.shape[1], device=device, dtype=torch.float32) * act
    org_out = x @ w.t().to(torch.float32)

    best_alpha = 0.0
    best_err = float("inf")
    w_max = w.abs().amax(dim=0).clamp(min=1e-5).to(torch.float32)

    for i in range(n_grid + 1):
        alpha = i / n_grid
        s = (act.pow(alpha) / w_max.pow(1.0 - alpha)).clamp(min=1e-5)
        s = s / (s.max() * s.min()).sqrt()        # normalize to avoid drift

        w_scaled = w.to(torch.float32) * s.view(1, -1)
        w_dq, _, _ = pseudo_quantize_tensor(w_scaled, n_bit, group_size, symmetric=False)
        out = (x / s.view(1, -1)) @ w_dq.t()
        err = (out - org_out).pow(2).mean().item()
        if err < best_err:
            best_err = err
            best_alpha = alpha

    s = (act.pow(best_alpha) / w_max.pow(1.0 - best_alpha)).clamp(min=1e-5)
    s = s / (s.max() * s.min()).sqrt()
    return s.to(w.dtype)


# ---------------------------------------------------------------------------
# Per-layer pairing
#
# In Qwen2-style transformer:
#   input_layernorm -> q_proj, k_proj, v_proj
#   o_proj          (preceded by attention, no clean prev linear -> skip fuse)
#   post_attention_layernorm -> gate_proj, up_proj
#   down_proj       (preceded by silu(gate)*up, no scalar fuse -> skip fuse)
#
# We fuse s by:
#   - dividing the corresponding RMSNorm.weight by s   (so its output already
#     has the per-channel scaling baked in)
#   - the consumer linear's weight is multiplied by s along input axis
# The two ops cancel mathematically; weights' effective rows for the
# critical channels are amplified, then quantized.
# ---------------------------------------------------------------------------


# layer_name (with .self_attn / .mlp etc) -> (norm_module_attr, [linear suffixes])
FUSION_GROUPS = {
    "input_layernorm":           ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "input_layernorm_moe_gen":   ["self_attn.q_proj_moe_gen", "self_attn.k_proj_moe_gen", "self_attn.v_proj_moe_gen"],
    "post_attention_layernorm":  ["mlp.gate_proj", "mlp.up_proj"],
    "post_attention_layernorm_moe_gen": ["mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj"],
}


@torch.no_grad()
def quantize_language_model(
    state_dict: Dict[str, torch.Tensor],
    act_stats: Dict[str, ActStats],
    n_bit: int = 4,
    group_size: int = 128,
    device: str = "cuda",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, dict]]:
    """Produces (packed_state_dict, meta).

    packed_state_dict contains:
      <orig_key>.qweight   uint8  [out, in/2]   packed INT4
      <orig_key>.scales    bf16   [out, in/group_size]
      <orig_key>.zeros     uint8  [out, in/group_size]   (asymmetric)
      <orig_key>.bias      bf16   [out]                  (if present)
    and all NON-quantized tensors are passed through (norms, embeddings, etc.).
    """
    meta: Dict[str, dict] = {}
    out_sd: Dict[str, torch.Tensor] = {}

    # First pass: figure out which layers belong to which decoder layer
    # by parsing keys like language_model.model.layers.<i>.<...>
    layer_keys: Dict[int, Dict[str, str]] = {}
    for k in state_dict:
        parts = k.split(".")
        if len(parts) >= 4 and parts[0] == "language_model" and parts[1] == "model" and parts[2] == "layers":
            idx = int(parts[3])
            sub = ".".join(parts[4:])
            layer_keys.setdefault(idx, {})[sub] = k

    # Pass-through buffer for things we don't quantize.
    for k, v in state_dict.items():
        if is_quantize_target(k):
            continue
        out_sd[k] = v.to(torch.bfloat16) if v.is_floating_point() else v

    # Walk each transformer layer
    for li in sorted(layer_keys.keys()):
        sub_to_full = layer_keys[li]
        prefix = f"language_model.model.layers.{li}"

        # 1) fuse-then-quantize for grouped linears with a known preceding norm
        for norm_sub, lin_subs in FUSION_GROUPS.items():
            norm_key = sub_to_full.get(f"{norm_sub}.weight")
            if norm_key is None:
                continue
            # Build s by averaging across these consumers' act stats
            collected = []
            for lsub in lin_subs:
                wkey = sub_to_full.get(f"{lsub}.weight")
                if wkey is None:
                    continue
                name = f"{prefix}.{lsub}"
                stat = act_stats.get(name)
                if stat is None or stat.sum_abs is None:
                    continue
                w = state_dict[wkey].to(device=device, dtype=torch.float32)
                s = awq_search_scale(w, stat.mean().to(device), n_bit, group_size)
                collected.append(s)
            if not collected:
                # No stats for any of them — still quantize without fusion
                for lsub in lin_subs:
                    _quant_linear_no_fuse(sub_to_full, lsub, state_dict, out_sd, meta,
                                          n_bit, group_size, device)
                continue
            s_layer = torch.stack(collected, dim=0).mean(dim=0)
            # Fuse: divide norm weight by s, multiply each consumer's weight by s
            new_norm = state_dict[norm_key].to(device=device, dtype=torch.float32) / s_layer
            out_sd[norm_key] = new_norm.to(torch.bfloat16).cpu()

            for lsub in lin_subs:
                wkey = sub_to_full.get(f"{lsub}.weight")
                if wkey is None:
                    continue
                bkey = sub_to_full.get(f"{lsub}.bias")
                w = state_dict[wkey].to(device=device, dtype=torch.float32) * s_layer.view(1, -1)
                _emit_quant_weight(wkey, w, n_bit, group_size, out_sd, meta)
                if bkey is not None:
                    out_sd[bkey] = state_dict[bkey].to(torch.bfloat16)

        # 2) o_proj and down_proj — no clean fuse target. Quantize per-group only.
        for lsub in ["self_attn.o_proj", "self_attn.o_proj_moe_gen",
                     "mlp.down_proj", "mlp_moe_gen.down_proj"]:
            _quant_linear_no_fuse(sub_to_full, lsub, state_dict, out_sd, meta,
                                  n_bit, group_size, device)

    # lm_head
    lm_key = "language_model.lm_head.weight"
    if lm_key in state_dict:
        w = state_dict[lm_key].to(device=device, dtype=torch.float32)
        _emit_quant_weight(lm_key, w, n_bit, group_size, out_sd, meta)

    return out_sd, meta


@torch.no_grad()
def _quant_linear_no_fuse(sub_to_full, lsub, state_dict, out_sd, meta, n_bit, group_size, device):
    wkey = sub_to_full.get(f"{lsub}.weight")
    if wkey is None or wkey in {k.rsplit(".", 1)[0] + "." + k.rsplit(".", 1)[1] for k in out_sd if k.endswith(".qweight")}:
        return
    bkey = sub_to_full.get(f"{lsub}.bias")
    w = state_dict[wkey].to(device=device, dtype=torch.float32)
    _emit_quant_weight(wkey, w, n_bit, group_size, out_sd, meta)
    if bkey is not None:
        out_sd[bkey] = state_dict[bkey].to(torch.bfloat16)


def _emit_quant_weight(wkey, w_fp, n_bit, group_size, out_sd, meta):
    """Quantize w_fp [out, in] per-group INT4 asymmetric, pack, and add to out_sd
    under keys derived from wkey."""
    out_features, in_features = w_fp.shape
    if in_features % group_size != 0:
        # pick the largest divisor of in_features that's <= group_size
        for g in [128, 64, 32, 16, in_features]:
            if in_features % g == 0:
                group_size = g
                break
    n_groups = in_features // group_size

    w_grp = w_fp.reshape(out_features, n_groups, group_size)
    max_v = w_grp.amax(dim=-1)
    min_v = w_grp.amin(dim=-1)
    qmax = (1 << n_bit) - 1
    scales = ((max_v - min_v) / qmax).clamp(min=1e-8)
    zeros = (-min_v / scales).round().clamp(0, qmax).to(torch.uint8)
    q = torch.clamp((w_grp / scales.unsqueeze(-1) + zeros.unsqueeze(-1)).round(), 0, qmax).to(torch.uint8)
    q = q.reshape(out_features, in_features)

    packed = pack_int4(q)
    base = wkey[:-len(".weight")]
    out_sd[f"{base}.qweight"] = packed.cpu()
    out_sd[f"{base}.scales"] = scales.to(torch.bfloat16).cpu()
    out_sd[f"{base}.zeros"] = zeros.cpu()
    meta[wkey] = {
        "shape": [out_features, in_features],
        "n_bit": n_bit,
        "group_size": group_size,
        "scheme": "asym_int4_grouped",
    }


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance_src", type=Path, required=True,
                    help="path to Lance GitHub source (added to sys.path)")
    ap.add_argument("--model_dir", type=Path, required=True,
                    help="dir with llm_config.json + model.safetensors (Lance_3B_Video)")
    ap.add_argument("--vit_dir", type=Path, required=True)
    ap.add_argument("--vae_path", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True,
                    help="dir produced by build_calib.py (manifest.json + assets/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--n_bit", type=int, default=4)
    ap.add_argument("--n_samples", type=int, default=64,
                    help="how many calibration samples to run forward on")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(args.lance_src))
    # imports happen here so PYTHONPATH is set first
    from quant_runtime import (  # type: ignore  (local module, see below)
        load_lance_for_calibration,
        run_calibration_forward,
    )

    args.out.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading bf16 model for calibration")
    model = load_lance_for_calibration(
        model_dir=args.model_dir, vit_dir=args.vit_dir, vae_path=args.vae_path,
        device=args.device,
    )

    print("[2/4] installing activation hooks on quant targets")
    targets = [n for n, _ in model.named_modules() if is_quantize_target(n)]
    print(f"      {len(targets)} target Linear modules")
    stats, handles = install_act_hooks(model, targets)

    print(f"[3/4] running {args.n_samples} calibration forwards")
    manifest = json.loads((args.calib / "manifest.json").read_text())
    t0 = time.time()
    run_calibration_forward(model, manifest["samples"][:args.n_samples],
                            calib_dir=args.calib)
    for h in handles:
        h.remove()
    print(f"      done in {time.time() - t0:.1f}s")

    print("[4/4] computing scales + quantizing")
    state_dict = {k: v.detach() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    torch.cuda.empty_cache()

    out_sd, meta = quantize_language_model(
        state_dict, stats,
        n_bit=args.n_bit, group_size=args.group_size, device=args.device,
    )

    save_file(out_sd, str(args.out / "awq_state_dict.safetensors"))
    (args.out / "awq_meta.json").write_text(json.dumps({
        "n_bit": args.n_bit,
        "group_size": args.group_size,
        "scheme": "asym_int4_grouped_awq",
        "calibration_samples": args.n_samples,
        "per_weight": meta,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
