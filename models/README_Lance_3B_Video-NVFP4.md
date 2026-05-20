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
- nvfp4
- fp4
- 4-bit
- blackwell
language:
- en
- zh
---

# Lance-3B-Video NVFP4 (E2M1 + FP8 block scales)

4-bit floating-point quantized variant of [bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance) (`Lance_3B_Video` checkpoint), using NVIDIA's **NVFP4** format. Targets Blackwell tensor cores (RTX 50-series, B100/B200) where it gets hardware-accelerated dequantization.

**File-size: 28.4 GB → 6.93 GB (4.1×)**

Companion to the AWQ INT4 variant: [`Reza2kn/Lance-3B-Video-AWQ-INT4`](https://huggingface.co/Reza2kn/Lance-3B-Video-AWQ-INT4).

## Format

Per quantized linear weight:
- **4 bits per element**, codes from E2M1 LUT {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
- **FP8 E4M3 scale per 16-element block** (1 byte per 16 weights = 0.5 bits/weight)
- Average **4.5 bits per weight** (vs 4.1875 bits/weight for the AWQ INT4 sibling)

Stored as both `.scales_fp8` (uint8 bytes carrying float8_e4m3fn) and `.scales_bf16` (bf16 redundant copy for runtimes that lack FP8 dtype support). You can drop one for slimmer storage.

## Why NVFP4 over INT4 AWQ

| | NVFP4 | AWQ INT4 |
|---|---|---|
| Hardware path | **Blackwell tensor cores** (5–10× INT4 throughput on sm_120) | Triton / marlin / exllamav2 INT4 GEMM |
| Distribution | Non-linear (FP) — denser near 0, wider range | Linear (INT) — uniform spacing |
| Block size | 16 (more granular) | 128 |
| Storage | 4.5 bits/weight | 4.19 bits/weight |
| Calibration | activation-aware AWQ scaling on top | activation-aware AWQ scaling |

On Blackwell, **NVFP4 wins on speed** by a wide margin once you're on a fused kernel (TensorRT-LLM, vLLM ≥ 0.8). On older GPUs and our pure-PyTorch reference loader, INT4 is slightly faster because the dequant math is simpler.

## What was quantized

Same scope as the AWQ INT4 sibling — see that README for the full breakdown. Same 504 Linear modules in `language_model.*` (360 with AWQ-style scale fusion, 144 with plain min-max for `o_proj` / `down_proj`). ViT, VAE, projections, time embedder, latent positional embeds, and lm_head are kept at bf16.

## Calibration

Identical activation statistics to the AWQ INT4 variant — calibrated on Lance's bundled examples for `x2t_image` (und path, 82.4M tokens) + `t2i` (gen path, 26.1M tokens). 504 linears, all with activation data.

## File layout

```
Lance_3B_Video-NVFP4/
├── nvfp4_state_dict.safetensors   # 6.93 GB: packed FP4 + FP8 + bf16 scales + pass-through
├── nvfp4_meta.json                # per-weight scheme + block_size + shape + FP4 LUT
└── README.md
```

Quantized linear keys:
- `<name>.qweight` — `uint8 [out, in // 2]`, two FP4 codes per byte
- `<name>.scales_fp8` — `uint8 [out, in // 16]` (cast `view(torch.float8_e4m3fn)` to read as FP8)
- `<name>.scales_bf16` — `bf16 [out, in // 16]` (same values, for runtimes lacking FP8)
- `<name>.bias` — `bf16 [out]` if present

FP4 code encoding (MSB = sign):
```
bits  3 2 1 0  | value
0 0 0 0       |  +0
0 0 0 1       |  +0.5
0 0 1 0       |  +1
0 0 1 1       |  +1.5
0 1 0 0       |  +2
0 1 0 1       |  +3
0 1 1 0       |  +4
0 1 1 1       |  +6
1 x x x       |  -<value above>
```

## How to use

Production path:
- **vLLM ≥ 0.8 on Blackwell GPUs** — supports compressed-tensors NVFP4 directly. Lance is not yet wired into vLLM (custom arch), but the format follows the same convention.
- **TensorRT-LLM** — has full NVFP4 support; export Lance to TRT engine and pull in these weights.

Verification path (pure-PyTorch, no special kernels):
A reference `WQLinearNVFP4` (mirror of the `WQLinearINT4` in the AWQ sibling) lives at **https://github.com/Reza2kn/lance-quant/blob/main/patches/quantized_linear.py**. It does on-the-fly dequant + linear, slow but correctness-checking.

## Reproduction

```bash
# Same calibration as the AWQ variant (act_stats.pt) is reused.
python nvfp4_apply.py \
    --src downloads/Lance_3B_Video/model.safetensors \
    --stats calib/act_stats.pt \
    --out models/Lance_3B_Video-NVFP4 \
    --block_size 16
```

## Limitations

- Pure-PyTorch dequant is slow — meant for compatibility, not speed
- FP4 quantization preserves outliers better than INT4 (E2M1 max = ±6), but per-block scales must absorb the rest. If a block has a large outlier, the remaining 15 values lose precision
- A separate calibration with more diverse tasks (image_edit, t2v) would likely close any quality gap

## Citation

Same as the AWQ sibling — please cite the original Lance paper.

## License

Apache 2.0, inherited from the base model.
