---
license: apache-2.0
base_model:
- bytedance-research/Lance
base_model_relation: quantized
pipeline_tag: text-generation
library_name: mlx
tags:
- multimodal
- quantized
- mlx
- 4-bit
- apple-silicon
language:
- en
- zh
---

# Lance LLM (understanding path) — MLX 4-bit

MLX-format 4-bit quantization of the **understanding-path language model** extracted from [bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance). Runs on Apple Silicon (M1/M2/M3/M4) via [`mlx-lm`](https://github.com/ml-explore/mlx-lm).

## What's quantized

Lance ships a custom modified Qwen2.5-VL with **Mixture-of-Tasks** routing: understanding tokens flow through one set of layer weights, generation tokens flow through `_moe_gen` siblings. This checkpoint contains **only the understanding path** weights, re-packaged as a standard Qwen2 LLM so `mlx-lm` accepts it.

That means:
- ✓ Text generation, instruction-following, VQA on text-only (use vision via the original Lance pipeline)
- ✗ Image/video generation, which lives in the `_moe_gen` path (separate quantization, not in this repo)

The Lance team's actual full inference loop is the canonical way to use this model — these MLX weights are most useful for **text-decoder-only experimentation** and for benchmarking the LLM half independently.

## Variants in this repo family

| Repo | Format | Group size | Bits/weight | DWQ refined |
|---|---|---|---|---|
| `…-MLX-4bit` | affine INT4 | 64 | 4.50 | no |
| `…-MLX-4bit-DWQ` | affine INT4 (distilled) | 64 | 4.50 | **yes** |
| `…-MLX-NVFP4` | NVFP4 (E2M1) | 16 | 4.50 | no |

DWQ ([Distillation-aware Weight Quantization](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md)) optimises the per-group scales/biases via KL-divergence distillation from the bf16 teacher. Typically recovers ~0.6 bits-per-weight of quality vs plain post-training quantization at the same bit budget.

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("Reza2kn/Lance-3B-Video-und-MLX-4bit-DWQ")
print(generate(
    model, tokenizer,
    prompt="What is the capital of France?",
    max_tokens=64, verbose=True,
))
```

Or via CLI:
```bash
mlx_lm.generate --model Reza2kn/Lance-3B-Video-und-MLX-4bit-DWQ \
    --prompt "Describe Persian cuisine in one paragraph."
```

## Extraction recipe

```python
# scripts/extract_und_to_qwen.py from https://github.com/Reza2kn/lance-quant
python extract_und_to_qwen.py \
    --src downloads/Lance_3B_Video/model.safetensors \
    --llm_config downloads/Lance_3B_Video/llm_config.json \
    --tokenizer_src downloads/Lance_3B_Video \
    --out Lance_3B_Video-und-qwen \
    --variant und

# Then drop the qk_norm weights (mlx-lm's Qwen2 doesn't have them) and
# convert to MLX 4-bit
mlx_lm.convert --hf-path Lance_3B_Video-und-qwen \
    --mlx-path Lance_3B_Video-und-MLX-4bit \
    -q --q-bits 4 --q-group-size 64

# Optional: DWQ refinement
mlx_lm.dwq --model Lance_3B_Video-und-qwen \
    --quantized-model Lance_3B_Video-und-MLX-4bit \
    --mlx-path Lance_3B_Video-und-MLX-4bit-DWQ \
    --bits 4 --group-size 64 --num-samples 256
```

## Limitations

- **Only the understanding path is quantized here.** Image/video generation uses `_moe_gen` weights which aren't in this checkpoint.
- The qk_norm weights from the original Lance modified-Qwen2.5-VL had to be dropped (mlx-lm's `qwen2` model class doesn't define them). Small but measurable quality cost vs the original FP32.
- Vision encoding (ViT, Wan VAE) must come from the original Lance pipeline.

For full multimodal use, see the AWQ INT4 / NVFP4 sibling repos which preserve the entire Lance architecture:

- [`Reza2kn/Lance-3B-AWQ-INT4`](https://huggingface.co/Reza2kn/Lance-3B-AWQ-INT4)
- [`Reza2kn/Lance-3B-Video-AWQ-INT4`](https://huggingface.co/Reza2kn/Lance-3B-Video-AWQ-INT4)
- [`Reza2kn/Lance-3B-NVFP4`](https://huggingface.co/Reza2kn/Lance-3B-NVFP4)
- [`Reza2kn/Lance-3B-Video-NVFP4`](https://huggingface.co/Reza2kn/Lance-3B-Video-NVFP4)

Reproduction toolkit: **https://github.com/Reza2kn/lance-quant**

## License

Apache 2.0, inherited from the base model.
