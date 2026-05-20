# Lance_3B AWQ g64 text perplexity

Quick held-out text-only benchmark for the image checkpoint. This uses the
LLM-only path in `scripts/eval_perplexity.py`, so it avoids constructing the
ViT/VAE and fits on the 16 GB 5080 laptop GPU.

## Result

| variant | perplexity | tokens | delta vs bf16 |
|---|---:|---:|---:|
| bf16 Lance_3B | 854.662 | 480 | - |
| AWQ INT4 g64 | 927.366 | 480 | +8.51% |

## Interpretation

The g64 AWQ checkpoint is substantially better than the original g128 on the
bundled x2t_image eval, and it fixes the case-4 long-form hallucination. This
perplexity pass still shows a measurable text-only degradation: +8.51% PPL on a
small 480-token corpus. Treat this as a quick regression gauge, not a full
language benchmark.

The remaining quality work should focus on more diverse calibration data,
5-bit/mixed-bit AWQ, or distillation-style refinement for long-form text.

Raw outputs:

- `docs/perplexity_bf16_lance3b.json`
- `docs/perplexity_awq_g64_lance3b.json`
