"""Merge multiple per-task activation stats files into one.

Inputs: any number of .pt files produced by awq_calibrate_single.py.
Output: single .pt with combined sum_abs / n_tokens per linear.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    merged = {}
    tasks_seen = []
    for p in args.inputs:
        d = torch.load(p, map_location="cpu", weights_only=False)
        tasks_seen.append(d.get("task", str(p.name)))
        for name, s in d["stats"].items():
            if name not in merged:
                merged[name] = {
                    "sum_abs": s["sum_abs"].clone(),
                    "n_tokens": s["n_tokens"],
                    "n_calls": s["n_calls"],
                }
            else:
                merged[name]["sum_abs"] += s["sum_abs"]
                merged[name]["n_tokens"] += s["n_tokens"]
                merged[name]["n_calls"] += s["n_calls"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": 1,
        "tasks": tasks_seen,
        "n_linears": len(merged),
        "stats": merged,
    }, args.out)
    n_und = sum(1 for k in merged if "_moe_gen" not in k)
    n_gen = sum(1 for k in merged if "_moe_gen" in k)
    n_und_data = sum(1 for k, v in merged.items() if "_moe_gen" not in k and v["n_tokens"] > 0)
    n_gen_data = sum(1 for k, v in merged.items() if "_moe_gen" in k and v["n_tokens"] > 0)
    print(f"merged {len(merged)} linears from {len(args.inputs)} tasks: {tasks_seen}")
    print(f"  und path: {n_und} linears, {n_und_data} with data")
    print(f"  gen path: {n_gen} linears, {n_gen_data} with data")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
