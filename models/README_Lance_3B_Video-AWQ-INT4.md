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
- video-generation
- image-editing
- video-understanding
- any-to-any
- quantized
- awq
- 4-bit
- int4
language:
- en
- zh
---

# Lance-3B-Video AWQ INT4

4-bit AWQ-quantized variant of [bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance) (the `Lance_3B_Video` checkpoint), the unified multimodal model for image and video generation, editing, and understanding.

**File-size reduction: 28.4 GB → 6.0 GB (4.7×)**
**Inference VRAM (LLM only, bf16 activations): ~14 GB → ~7 GB**

## What was quantized

Lance is not a standard HuggingFace `transformers` model. It's a custom `PreTrainedModel` whose decoder is a Qwen2.5-VL with parallel **Mixture-of-Tasks** experts: every transformer layer has both `mlp`/`self_attn.*` (understanding path) AND `mlp_moe_gen`/`self_attn.*_moe_gen` (generation expert). Tokens route through one or the other depending on type (text vs VAE latent).

| Component | Action | Reason |
|---|---|---|
| `language_model.model.layers.*.{q,k,v,o}_proj` + `mlp.{gate,up,down}_proj` (×36 layers, und path) | **4-bit AWQ INT4** | bulk of params; AWQ scales fused into preceding RMSNorm |
| same as above with `_moe_gen` suffix (gen-expert path) | **4-bit AWQ INT4** | calibrated separately with t2i samples to capture gen activations |
| `language_model.lm_head` | bf16 | numerically sensitive; inference asserts on `.weight.data_ptr()` |
| `vit_model.*` (bundled Qwen2.5-VL ViT, 670M) | bf16 | small, vision-critical |
| `latent_pos_embed`, `time_embedder`, `llm2vae`, `vae2llm` | bf16 | tiny, numerically sensitive |
| RMSNorms | bf16 (fused with AWQ scales) | — |

**504 Linear modules quantized** (360 with AWQ-style activation-aware scale fusion, 144 with plain per-group min-max because they have no clean fuse target — `o_proj` and `down_proj`).

## Calibration

| Task | Samples | Tokens | Hits which path |
|---|---|---|---|
| `x2t_image` | 6 (Lance's example set) | 82.4 M | und only |
| `t2i` | 11 (Lance's example set), 2 denoising steps | 26.1 M | und + gen |
| **merged** | — | 108.5 M | 252 und + 252 gen Linears, all with data |

The two-task mix is the minimum that exercises **both** MoT paths — pure understanding leaves the `_moe_gen` weights without activation data, and AWQ falls back to plain min-max for those.

## Quantization recipe

- 4-bit asymmetric **per-group** (group_size = 128, packed two-nibbles-per-byte)
- AWQ alpha grid search ∈ {0.0, 0.05, …, 1.0} per fusion group, MSE-minimizing on synthetic Gaussian inputs scaled by per-channel mean-abs activation
- Scale s normalised so geomean(s) ≈ 1 to avoid drift, then fused as `norm.weight /= s` and `consumer.weight *= s`

Fusion groups per decoder layer:
- `input_layernorm` → `[q_proj, k_proj, v_proj]`
- `input_layernorm_moe_gen` → `[q_proj_moe_gen, k_proj_moe_gen, v_proj_moe_gen]`
- `post_attention_layernorm` → `[mlp.gate_proj, mlp.up_proj]`
- `post_attention_layernorm_moe_gen` → `[mlp_moe_gen.gate_proj, mlp_moe_gen.up_proj]`

`o_proj` and `down_proj` are post-nonlinearity — no clean scalar fuse — quantized with plain per-group min-max.

## File layout

```
Lance_3B_Video-AWQ-INT4/
├── awq_state_dict.safetensors   # 6.02 GB: packed INT4 + bf16 pass-through
├── awq_meta.json                # per-weight scheme + group_size + shape
└── README.md
```

`awq_state_dict.safetensors` keys per quantized linear:
- `<name>.qweight` — `uint8 [out, in // 2]`, two INT4 nibbles per byte
- `<name>.scales`  — `bf16 [out, in // group_size]`
- `<name>.zeros`   — `uint8 [out, in // group_size]`
- `<name>.bias`    — `bf16 [out]` (if the original linear had one)

Plus all pass-through (non-quantized) tensors at bf16, including the **AWQ-rescaled RMSNorm weights** — load these in place of the originals.

## How to use

The Lance source ships a custom `Lance` `PreTrainedModel` (in [github.com/bytedance/Lance](https://github.com/bytedance/Lance)). vLLM / TensorRT-LLM / AutoAWQ do NOT support it directly, since it's not a standard HF model. The cleanest path is a runtime swap-in:

```python
import torch
from safetensors import safe_open

# 1) Build Lance normally (e.g. via the official inference_lance.py)
# 2) Swap every targeted nn.Linear in `language_model.*` for WQLinearINT4
# 3) Stream the AWQ packed buffers from awq_state_dict.safetensors

# A reference WQLinearINT4 module + swap-in loader is at:
#   https://github.com/Reza2kn/lance-quant/blob/main/patches/quantized_linear.py
```

A full reproduction (calibration scripts + the WQLinearINT4 implementation + a `run_quant_eval.py` that loads this checkpoint into Lance) is at: **https://github.com/Reza2kn/lance-quant**

For convenience, we also ship a **ComfyUI custom node** that wires this checkpoint into Lance's six tasks (t2i, t2v, image_edit, video_edit, x2t_image, x2t_video).

## Quality

Side-by-side on Lance's bundled x2t_image example (6 cases):

| Case | Baseline bf16 | This AWQ INT4 |
|---|---|---|
| 1 (pie chart Q&A) | "Yes, the largest segment is in the blue color…" | "Yes, the largest segment is greater than the sum of all the other segments." ✓ |
| 2 (% in chart) | "29%" | "29%" ✓ |
| 3 (license plate OCR) | `BX62 BFY` | `BX62 BFY` ✓ |
| 4 ($ amount) | "approximately $1.3 billion" | partially correct (some hallucinated wording) |
| 5 (Colosseum description) | long coherent description | long coherent description ✓ |
| 6 (solar eclipse) | physically reasonable | physically reasonable ✓ |

Comparison vs naïve min-max INT4 (no calibration), same setup:

| Case | Min-max INT4 | This AWQ INT4 |
|---|---|---|
| 1 | `"the loose and around the largest segment, and the, a. ,aaaaaaaaa ifa..."` (gibberish) | coherent ✓ |
| 2 | `"araaaaaaa subsa subsa subs..."` (gibberish) | `"29%"` ✓ |

The proper AWQ calibration is essential — naïve INT4 destroys outputs because of activation outliers in certain weight columns.

## Limitations & known caveats

- **Inference runtime is slow with pure-PyTorch dequantization** (~10× the bf16 baseline) because the swap-in `WQLinearINT4` does a tensor-by-tensor dequant on every forward. To get production throughput you'll want a fused INT4 GEMM kernel (e.g. via Triton, marlin, or exllamav2). The companion repo will add this.
- This is a v1 calibration with a small set (17 samples across 2 tasks). Adding `image_edit`, `t2v`, `video_edit`, `x2t_video` would likely close the small gap on case 04.
- The bundled `Qwen2.5-VL-ViT` (in the original checkpoint) is kept at bf16 here. A separate quantization of the ViT is on the roadmap.
- `Wan2.2_VAE.pth` is not part of this checkpoint — fetch it from the original repo (it's used unmodified, kept in bf16/fp32 since diffusion VAEs are sensitive to low-bit quantization).

## Reproduction

```bash
# Clone Lance source + download original weights (~57 GB)
git clone https://github.com/bytedance/Lance.git
cd Lance && bash setup_env.sh
huggingface-cli download bytedance-research/Lance --local-dir downloads

# Clone this repo's quantization toolkit
git clone https://github.com/Reza2kn/lance-quant.git
cd lance-quant/scripts

# 1) Calibrate (collect activation statistics)
python awq_calibrate_single.py --task x2t_image \
    --model_path ../../Lance/downloads/Lance_3B_Video \
    --vit_path ../../Lance/downloads/Qwen2.5-VL-ViT \
    --example_json ../../Lance/config/examples/x2t_image_example.json \
    --out ../calib/x2t_image_stats.pt

python awq_calibrate_single.py --task t2i \
    --model_path ../../Lance/downloads/Lance_3B_Video \
    --vit_path ../../Lance/downloads/Qwen2.5-VL-ViT \
    --example_json ../../Lance/config/examples/t2i_example.json \
    --out ../calib/t2i_stats.pt --num_timesteps 2

# 2) Merge stats + apply AWQ
python awq_merge_stats.py --inputs ../calib/x2t_image_stats.pt ../calib/t2i_stats.pt --out ../calib/act_stats.pt
python awq_apply.py --src ../../Lance/downloads/Lance_3B_Video/model.safetensors \
    --stats ../calib/act_stats.pt --out ../models/Lance_3B_Video-AWQ-INT4

# 3) Evaluate
python run_quant_eval.py --task x2t_image \
    --model_path ../../Lance/downloads/Lance_3B_Video \
    --awq_dir ../models/Lance_3B_Video-AWQ-INT4
```

## Citation

If you use this quantized checkpoint, please also cite the original Lance paper:

```bibtex
@misc{lance2026,
  title  = {Lance: Unified Multimodal Modeling by Multi-Task Synergy},
  author = {Fengyi Fu and Mengqi Huang and Shaojin Wu and others},
  year   = {2026}
}
```

## License

Apache 2.0, inherited from the base model.
