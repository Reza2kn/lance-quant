---
license: apache-2.0
base_model:
- bytedance-research/Lance
base_model_relation: quantized
pipeline_tag: any-to-any
library_name: Lance
tags:
- multimodal
- image-generation
- image-editing
- image-understanding
- any-to-any
- quantized
- nvfp4
- fp4
- 4-bit
- blackwell
language:
- en
- zh
---

# Lance-3B NVFP4 (image checkpoint)

4-bit floating-point quantized variant of [bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance), the **`Lance_3B` image-focused checkpoint**, using NVIDIA's **NVFP4** format (E2M1 weights + FP8 E4M3 per-block scales).

Targets Blackwell tensor cores (RTX 50-series, B100/B200) where it gets hardware-accelerated dequantization with 5–10× the throughput of INT4 once paired with TensorRT-LLM / vLLM ≥ 0.8.

**File-size: 24.7 GB → ~6 GB (4×)**

Companion to the AWQ INT4 image variant: [`Reza2kn/Lance-3B-AWQ-INT4`](https://huggingface.co/Reza2kn/Lance-3B-AWQ-INT4).
Video-flavoured sibling: [`Reza2kn/Lance-3B-Video-NVFP4`](https://huggingface.co/Reza2kn/Lance-3B-Video-NVFP4).

## Format

- 4-bit E2M1 codes per weight (LUT {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6})
- FP8 E4M3 scale per 16-element block (1 byte per 16 weights)
- Average **4.5 bits per weight**
- Both `scales_fp8` (uint8 bytes carrying float8_e4m3fn) and `scales_bf16` (redundant copy) are stored — drop one for slimmer storage if your runtime supports the other.

See the [video sibling NVFP4 README](https://huggingface.co/Reza2kn/Lance-3B-Video-NVFP4) for the full FP4 LUT and storage layout — identical here.

## Calibration

Same AWQ activation statistics as the AWQ-INT4 image variant — 252 und-path + 252 gen-path Linears, all with activation data, calibrated on Lance's bundled `x2t_image` + `t2i` example sets (108.5 M tokens total).

## File layout

```
Lance_3B-NVFP4/
├── nvfp4_state_dict.safetensors   # ~6 GB: packed FP4 + FP8 + bf16 scales + pass-through
├── nvfp4_meta.json                # per-weight scheme + block_size + shape + FP4 LUT
└── README.md
```

## How to use

Production: vLLM ≥ 0.8 / TensorRT-LLM on Blackwell (Lance not yet wired in but format is compatible).
Verification: reference `WQLinearNVFP4` swap-in module at **https://github.com/Reza2kn/lance-quant**.

## License

Apache 2.0, inherited from the base model.
