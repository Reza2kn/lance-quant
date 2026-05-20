"""Side-by-side quality eval: compare any number of Lance inference runs
against a bf16 baseline.

For each task's `prompt.json` we compute, per sample:
  * exact-string match (after stripping trailing chat tokens)
  * char-level Levenshtein similarity (1 − distance / max_len)
  * SequenceMatcher ratio (Python's difflib, robust to small edits)
  * token-level Jaccard similarity on whitespace-split words

Run this on stallion or anywhere with access to the result JSON files. No
GPU needed.

Usage:
    python eval_compare.py \\
        --baseline results/baseline_x2t_image_*/prompt.json \\
        --variants results/Lance_3B_Video-AWQ-INT4_x2t_image_*/prompt.json:AWQ \\
                   results/Lance_3B_Video-INT4-MinMax_x2t_image_*/prompt.json:MinMax \\
        --out eval_report.md
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path


def _strip(s: str) -> str:
    return re.sub(r"<\|im_end\|>.*$", "", s).strip()


def _levenshtein(a: str, b: str) -> int:
    """O(len(a)·len(b)) edit distance. Fine for short LLM outputs."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                          prev[j - 1] + (0 if ca == cb else 1))
        prev = cur
    return prev[-1]


def _score(baseline: str, candidate: str) -> dict[str, float]:
    a, b = _strip(baseline), _strip(candidate)
    if not a and not b:
        return {"exact": 1.0, "levenshtein": 1.0, "ratio": 1.0, "jaccard": 1.0}
    L = max(len(a), len(b), 1)
    dist = _levenshtein(a, b)
    lev = 1 - dist / L
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    j = (len(set_a & set_b) / len(set_a | set_b)) if (set_a or set_b) else 1.0
    return {"exact": 1.0 if a == b else 0.0,
            "levenshtein": lev, "ratio": ratio, "jaccard": j}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True,
                    help="bf16 baseline prompt.json")
    ap.add_argument("--variants", nargs="+", required=True,
                    help="<prompt.json>:<label> per variant")
    ap.add_argument("--out", type=Path, default=Path("eval_report.md"))
    args = ap.parse_args()

    baseline = json.loads(Path(args.baseline).read_text())
    variants = {}
    for spec in args.variants:
        path, _, label = spec.rpartition(":")
        variants[label] = json.loads(Path(path).read_text())

    keys = sorted(baseline.keys())
    print(f"baseline={args.baseline}  samples={len(keys)}  "
          f"variants={list(variants.keys())}")

    # Per-sample table + per-variant aggregates
    rows = []
    agg = {label: {"exact": 0.0, "levenshtein": 0.0,
                   "ratio": 0.0, "jaccard": 0.0, "n": 0}
           for label in variants}

    for k in keys:
        base = baseline.get(k, "")
        row = {"sample": k, "baseline": _strip(base)}
        for label, vd in variants.items():
            cand = vd.get(k, "")
            s = _score(base, cand)
            row[label] = {"text": _strip(cand), **s}
            for m in ("exact", "levenshtein", "ratio", "jaccard"):
                agg[label][m] += s[m]
            agg[label]["n"] += 1
        rows.append(row)

    # Render markdown report
    lines = ["# Lance quantization eval", "",
             f"_baseline_: `{args.baseline}`  ({len(keys)} samples)", ""]
    lines.append("## Aggregate scores")
    lines.append("| variant | exact-match | char Levenshtein sim | "
                  "difflib ratio | word Jaccard |")
    lines.append("|---|---|---|---|---|")
    for label, a in agg.items():
        n = a["n"] or 1
        lines.append(f"| **{label}** | {a['exact']/n:.3f} | "
                      f"{a['levenshtein']/n:.3f} | "
                      f"{a['ratio']/n:.3f} | {a['jaccard']/n:.3f} |")
    lines.append("")

    lines.append("## Per-sample side-by-side")
    for r in rows:
        lines.append(f"### `{r['sample']}`")
        lines.append("")
        lines.append(f"**baseline**: {r['baseline']}")
        lines.append("")
        for label in variants:
            v = r[label]
            lines.append(f"**{label}** (lev={v['levenshtein']:.2f}, "
                          f"ratio={v['ratio']:.2f}, jaccard={v['jaccard']:.2f}):")
            lines.append("")
            lines.append("> " + v["text"][:600].replace("\n", "\n> "))
            lines.append("")
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")

    # Also print summary to stdout
    print("\n=== AGGREGATE ===")
    for label, a in agg.items():
        n = a["n"] or 1
        print(f"  {label:20s} exact={a['exact']/n:.3f}  "
              f"lev={a['levenshtein']/n:.3f}  "
              f"ratio={a['ratio']/n:.3f}  jaccard={a['jaccard']/n:.3f}")


if __name__ == "__main__":
    main()
