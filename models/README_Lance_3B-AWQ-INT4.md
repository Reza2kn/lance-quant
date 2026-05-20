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
- awq
- 4-bit
- int4
language:
- en
- zh
---

# Lance-3B AWQ INT4 (image checkpoint)

4-bit AWQ-quantized variant of [bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance), the **`Lance_3B` image-focused checkpoint** (text-to-image, image edit, image understanding).

**File-size reduction: 24.7 GB → ~6 GB (4×)**
**Inference VRAM (LLM only, bf16 activations): ~13 GB → ~6 GB**

## What's different from the video sibling

Lance ships two checkpoints:
- `Lance_3B` — image-focused (this one). 24.7 GB F32 source. No bundled ViT, smaller `latent_pos_embed` (image grid only).
- `Lance_3B_Video` — video-focused. 28.4 GB F32 source. Bundles the Qwen2.5-VL ViT in its safetensors + larger video-grid `latent_pos_embed`. Quantized variant: [`Reza2kn/Lance-3B-Video-AWQ-INT4`](https://huggingface.co/Reza2kn/Lance-3B-Video-AWQ-INT4).

This image checkpoint relies on the **standalone Qwen2.5-VL-ViT** for vision encoding (also bundled in the official Lance HF repo; not redistributed here).

## What was quantized

Same MoT-aware scheme as the video sibling — 504 Linear modules in `language_model.*` (252 understanding-path + 252 generation-expert `_moe_gen` variants), 360 with AWQ scale fusion into the preceding RMSNorm, 144 with plain per-group min-max (`o_proj`, `down_proj`). The ViT, projection layers, time embedder, latent positional embeds, and `lm_head` are kept in bf16.

See the [video sibling README](https://huggingface.co/Reza2kn/Lance-3B-Video-AWQ-INT4) for the full per-component table — it's identical here.

## Calibration

- `x2t_image` (Lance's 6-sample example set, full 30 timesteps) → **252 und-path linears**, 85.3 M tokens of activation data
- `t2i` (Lance's 11-sample example set, 2 denoising timesteps) → **all 504 linears** (both und and gen paths)
- Merged: 252 und + 252 gen Linears all with activation data

## File layout

```
Lance_3B-AWQ-INT4/
├── awq_state_dict.safetensors   # ~6 GB: packed INT4 + bf16 pass-through
├── awq_meta.json                # per-weight scheme + group_size + shape
└── README.md
```

Storage layout per quantized linear is identical to the video sibling — see that repo for the `qweight` / `scales` / `zeros` byte layout.

## How to use

Same as the video sibling. The Lance source ships a custom `Lance` `PreTrainedModel` (in [github.com/bytedance/Lance](https://github.com/bytedance/Lance)). Use the runtime swap-in approach: build Lance normally, then replace `nn.Linear` modules in `language_model.*` with the `WQLinearINT4` reference module and stream the AWQ buffers in.

A complete reproduction (calibration scripts + `WQLinearINT4` + `run_quant_eval.py`) is at: **https://github.com/Reza2kn/lance-quant**

## Quality

Side-by-side on Lance's bundled x2t_image example (6 cases) — outputs match the bf16 baseline to within typical AWQ tolerance. Naïve min-max INT4 produces gibberish ("the loose subs ifa…"); proper AWQ calibration recovers it ("Yes, the largest segment is greater than the sum of all the other segments.").

## License

Apache 2.0, inherited from the base model.
