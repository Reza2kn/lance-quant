"""Convert the Lance und-path Qwen2 LLM to CoreML with 4-bit palettization.

The output is a `.mlpackage` you can load with `coremltools` and run on
M-series Macs / iPhone via the Apple Neural Engine + GPU.

Why CoreML in addition to MLX?
  - MLX is strictly better for LLM inference on Apple Silicon Macs (more
    flexible, faster on GPU for irregular shapes), but it does **not** run on
    iOS. CoreML is the only path for iPhone / iPad deployment.
  - Even on Mac, CoreML can target the ANE (Apple Neural Engine), which is
    power-efficient for batched/sequential workloads.

The conversion targets fp16 weights with 4-bit per-grouped-channel
palettization (kmeans, group_size=32). On disk that's ~1 GB for our 3B-param
LLM, down from 6 GB fp16.

Usage:
    python convert_to_coreml.py \\
        --hf-path /path/to/Lance_3B-und-qwen \\
        --out /path/to/Lance_3B-und-CoreML-4bit.mlpackage
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def _build_wrapped(hf_path: Path, max_seq: int) -> tuple[torch.nn.Module, torch.Tensor]:
    """Load model and wrap forward to return only logits (CoreML can't deal
    with HF's full ModelOutput dataclass)."""
    print(f"[load] {hf_path}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(hf_path), torch_dtype=torch.float16, use_cache=False,
    ).eval()
    print(f"[load] done in {time.time()-t0:.1f}s; "
          f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            # input_ids: [B, L] int32
            return self.m(input_ids).logits

    wrapped = _Wrap(model).eval()
    example = torch.zeros((1, max_seq), dtype=torch.int32)
    # Run once to populate caches / verify forward
    with torch.no_grad():
        wrapped(example)
    return wrapped, example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-path", type=Path, required=True,
                    help="extracted und-Qwen HF model dir")
    ap.add_argument("--out", type=Path, required=True,
                    help="output .mlpackage path")
    ap.add_argument("--max_seq", type=int, default=512,
                    help="max input sequence length to trace for")
    ap.add_argument("--no_palettize", action="store_true",
                    help="skip the 4-bit palettization step (debug)")
    ap.add_argument("--nbits", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=32)
    args = ap.parse_args()

    import coremltools as ct
    import coremltools.optimize.coreml as cto
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        register_torch_op,
    )

    @register_torch_op(override=True)
    def bitwise_or(context, node):
        # boolean tensors → logical_or; integer tensors → element-wise OR via
        # ((a OR b) == nonzero) cast back. Qwen2's mask code only uses this on
        # bool / uint8 tensors so logical_or is the correct translation.
        inputs = _get_inputs(context, node, expected=2)
        a, b = inputs[0], inputs[1]
        out = mb.logical_or(x=a, y=b, name=node.name)
        context.add(out)

    print("[ops] registered bitwise_or as logical_or")

    wrapped, example = _build_wrapped(args.hf_path, args.max_seq)

    print("[trace] torch.jit.trace …")
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example)
    print(f"[trace] done in {time.time()-t0:.1f}s")

    print("[convert] coremltools.convert (this is slow, ~5–15 min) …")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids",
                                shape=(1, ct.RangeDim(1, args.max_seq)),
                                dtype=np.int32)],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    print(f"[convert] done in {time.time()-t0:.1f}s")

    if not args.no_palettize:
        print(f"[palettize] {args.nbits}-bit per-grouped-channel "
              f"(group_size={args.group_size}) …")
        t0 = time.time()
        cfg = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(
                nbits=args.nbits, mode="kmeans",
                granularity="per_grouped_channel",
                group_size=args.group_size,
            ),
        )
        mlmodel = cto.palettize_weights(mlmodel, cfg)
        print(f"[palettize] done in {time.time()-t0:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.out))
    sz = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"[done] saved {args.out} ({sz/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
