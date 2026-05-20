---
license: apache-2.0
base_model:
- bytedance-research/Lance
base_model_relation: quantized
pipeline_tag: text-generation
library_name: transformers
tags:
- multimodal
- quantized
- coreml
- palettized
- 4-bit
- apple-silicon
- ios
language:
- en
- zh
---

# Lance LLM (understanding path) — 4-bit kmeans palettized (CoreML-ready)

4-bit per-grouped-channel k-means palettization of the **understanding-path LLM** extracted from [bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance), via [`coremltools.optimize.torch.palettization.PostTrainingPalettizer`](https://apple.github.io/coremltools/docs-guides/source/optimizing-models.html).

Each Linear weight is clustered with k-means to **16 codes per group** (`group_size=32`, granularity = `per_grouped_channel`). The codes + LUT are then dequantized back to **fp16 for storage**, so this safetensors loads as a normal HuggingFace model with the *numerical quality* of a 4-bit palettized checkpoint — useful for:

- **Quality probing**: see how 4-bit kmeans palettization affects outputs without writing a custom CoreML pipeline
- **CoreML deployment**: the same numerical scheme is what `coremltools.optimize.coreml.OpPalettizerConfig(nbits=4, mode="kmeans", granularity="per_grouped_channel", group_size=32)` produces inside a `.mlpackage`. A custom converter that traces this model into CoreML will get the same weights losslessly compressed back to 4-bit on disk.
- **Apple Neural Engine targeting**: the kmeans LUT scheme is ANE-friendly; weight decode is hardware-accelerated.

## Why fp16 storage instead of true 4-bit on disk

Compressing to actual 4-bit indices + per-group LUT requires a custom on-disk format that no standard runtime (transformers, MLX) reads directly. The CoreML `.mlpackage` IS that custom format, but producing it requires tracing the model through coremltools — which currently hits unimplemented torch ops in modern Qwen2's mask construction (`bitwise_or_`, `_int` of multi-dim tensors).

So this checkpoint ships the **dequantized fp16 weights** for drop-in usability, with the same quality as a true 4-bit deployment. Total size: **~6 GB** (vs. 6.8 GB bf16 source — roughly the same because both are 2 bytes/weight on disk; the difference is in the *effective precision* of the values).

If you want **true 4-bit on-disk storage** for the same Lance LLM, use the MLX siblings:

- [`Reza2kn/Lance-3B-und-MLX-4bit`](https://huggingface.co/Reza2kn/Lance-3B-und-MLX-4bit) (~1.6 GB, ANE not used; Metal GPU)
- [`Reza2kn/Lance-3B-und-MLX-4bit-DWQ`](https://huggingface.co/Reza2kn/Lance-3B-und-MLX-4bit-DWQ) (~1.6 GB + distilled scales)

## Companion: full Lance multimodal pipeline

This checkpoint is the **understanding path only** — image/video generation lives in the `_moe_gen` expert path which isn't extracted here. For full multimodal inference, use:

- [`Reza2kn/Lance-3B-AWQ-INT4`](https://huggingface.co/Reza2kn/Lance-3B-AWQ-INT4) — image, AWQ INT4, 4.2 GB
- [`Reza2kn/Lance-3B-Video-AWQ-INT4`](https://huggingface.co/Reza2kn/Lance-3B-Video-AWQ-INT4) — video, AWQ INT4, 6.0 GB
- [`Reza2kn/Lance-3B-NVFP4`](https://huggingface.co/Reza2kn/Lance-3B-NVFP4) — image, NVFP4 (Blackwell), 5.1 GB
- [`Reza2kn/Lance-3B-Video-NVFP4`](https://huggingface.co/Reza2kn/Lance-3B-Video-NVFP4) — video, NVFP4, 6.9 GB

## Reproduction

```bash
# scripts/palettize_weights_coreml.py from https://github.com/Reza2kn/lance-quant
python palettize_weights_coreml.py \
    --hf-path Lance_3B-und-qwen \
    --out Lance_3B-und-CoreML-palettized-4bit \
    --nbits 4 --group_size 32
```

## License

Apache 2.0, inherited from the base model.
