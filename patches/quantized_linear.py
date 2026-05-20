"""Runtime swap-in for AWQ INT4 quantized linears.

Produces a `WQLinearINT4` nn.Module that matches the forward signature of
nn.Linear, decompresses on demand to bf16 for the matmul. Designed to be
swapped in by name into a Lance model whose Linear modules we want quantized.

Storage layout (per linear):
  qweight  uint8   [out_features, in_features // 2]   # 2x INT4 per byte
  scales   bf16    [out_features, in_features // group_size]
  zeros    uint8   [out_features, in_features // group_size]    # asymmetric
  bias     bf16    [out_features]   (optional)

Forward: dequantize qweight to bf16 once per call (we don't cache —  Lance
runs each linear at most a handful of times per generation step), then a
standard linear matmul.

For maximum throughput swap this for the WQLinear_GEMM/WQLinear_GEMV kernels
from AutoAWQ/exllamav2 (they batch the dequant into the gemm). We avoid that
dep here since we're not in a transformers-compatible model anyway.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from safetensors import safe_open


class WQLinearINT4(nn.Module):
    """4-bit grouped, asymmetric, packed weight-only quantized linear."""

    def __init__(self, in_features: int, out_features: int,
                 group_size: int = 128, bias: bool = False,
                 device=None, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.compute_dtype = dtype

        n_groups = in_features // group_size
        # Stored as buffers (not Parameters; we don't train them).
        self.register_buffer(
            "qweight",
            torch.zeros((out_features, in_features // 2),
                        dtype=torch.uint8, device=device),
        )
        self.register_buffer(
            "scales",
            torch.zeros((out_features, n_groups),
                        dtype=dtype, device=device),
        )
        self.register_buffer(
            "zeros",
            torch.zeros((out_features, n_groups),
                        dtype=torch.uint8, device=device),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype,
                                                 device=device))
        else:
            self.register_parameter("bias", None)

    def _dequantize(self) -> torch.Tensor:
        # Unpack: low nibble = even, high nibble = odd index
        packed = self.qweight                 # [out, in/2]
        lo = (packed & 0xF).to(torch.int16)
        hi = (packed >> 4 & 0xF).to(torch.int16)
        # interleave -> [out, in]
        unpacked = torch.stack([lo, hi], dim=-1).reshape(
            self.out_features, self.in_features)
        # Reshape to groups for per-group dequant
        u = unpacked.reshape(self.out_features, -1, self.group_size).to(self.compute_dtype)
        z = self.zeros.unsqueeze(-1).to(self.compute_dtype)
        s = self.scales.unsqueeze(-1)
        w = (u - z) * s
        return w.reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantize()
        return torch.nn.functional.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"group={self.group_size}, bias={self.bias is not None}")


# ---------------------------------------------------------------------------
# Lance model patcher: swap Linear -> WQLinearINT4 in-place
# ---------------------------------------------------------------------------


def patch_lance_with_awq(model: nn.Module, awq_dir: str | Path,
                        compute_dtype: torch.dtype = torch.bfloat16) -> None:
    """In-place swap of every Linear listed in `awq_meta.json` for the
    WQLinearINT4 equivalent, then stream-load the quantized buffers.
    """
    awq_dir = Path(awq_dir)
    meta = json.loads((awq_dir / "awq_meta.json").read_text())["per_weight"]
    sd_path = awq_dir / "awq_state_dict.safetensors"

    # Build lookup: param-name -> module + attr
    modules_by_name = dict(model.named_modules())
    swapped = 0
    skipped = 0

    for wkey, info in meta.items():
        # wkey like "language_model.model.layers.0.self_attn.q_proj.weight"
        parent_name, attr = wkey[:-len(".weight")].rsplit(".", 1)
        # parent_name should be the linear module, attr should be 'weight'
        # but we want to replace the PARENT of the linear: pull lin_name
        lin_name = wkey[:-len(".weight")]
        parent_path, lin_attr = lin_name.rsplit(".", 1)
        parent = modules_by_name.get(parent_path)
        if parent is None or not hasattr(parent, lin_attr):
            print(f"[patch] skip {lin_name}: parent or attr missing")
            skipped += 1
            continue
        old: nn.Linear = getattr(parent, lin_attr)
        if not isinstance(old, nn.Linear):
            skipped += 1
            continue
        new = WQLinearINT4(
            in_features=info["shape"][1],
            out_features=info["shape"][0],
            group_size=info["group_size"],
            bias=old.bias is not None,
            device=next(old.parameters()).device,
            dtype=compute_dtype,
        )
        setattr(parent, lin_attr, new)
        modules_by_name[lin_name] = new
        swapped += 1

    print(f"[patch] swapped {swapped} Linear -> WQLinearINT4 ({skipped} skipped)")

    # Stream quantized buffers
    with safe_open(str(sd_path), framework="pt", device="cpu") as f:
        for k in f.keys():
            # k like .../q_proj.qweight, .../q_proj.scales, etc.
            base, suffix = k.rsplit(".", 1)
            mod = modules_by_name.get(base)
            if mod is None:
                continue
            tensor = f.get_tensor(k)
            if hasattr(mod, suffix):
                target = getattr(mod, suffix)
                target_dev = target.device if isinstance(target, torch.Tensor) else next(mod.parameters()).device
                with torch.no_grad():
                    target.data.copy_(tensor.to(target_dev, non_blocking=True))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"[patch] cuda mem after AWQ swap: "
              f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")
