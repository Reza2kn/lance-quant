# lance-quant

4-bit quantization toolkit for **ByteDance Lance** ([model card](https://huggingface.co/bytedance-research/Lance), [paper](https://arxiv.org/abs/2605.18678)) — a unified 3B-active multimodal model for image / video generation, editing, and understanding.

Lance has a **custom architecture** (modified Qwen2.5-VL with parallel `_moe_gen` experts on every transformer layer — a "Mixture-of-Tasks" routing where understanding tokens flow through one expert and generation tokens through another). Off-the-shelf quantizers (AutoAWQ, llm-compressor's `oneshot()`) don't know how to walk it, and inference runtimes that accept their outputs (vLLM, TensorRT-LLM) don't know how to run it. So this repo hand-rolls the calibration + packing + runtime swap-in.

## What's in the box

| | |
|---|---|
| `scripts/awq_calibrate_single.py` | runs Lance inference on one task with activation hooks on all 504 target Linears (`q/k/v/o_proj`, `mlp.{gate,up,down}_proj`, and the `_moe_gen` siblings of each), saves per-channel mean-abs activation magnitudes |
| `scripts/awq_merge_stats.py` | combines per-task stats into a single calibration set |
| `scripts/awq_apply.py` | grid-searches the AWQ scale-equalization α per fusion group (norm + consumer linears), fuses the scale into the preceding RMSNorm, packs weights to per-group INT4 |
| `scripts/nvfp4_apply.py` | same calibration but packs to NVFP4 (E2M1 codes + FP8 E4M3 per-16-element-block scales) for Blackwell tensor cores |
| `scripts/quantize_int4_minmax.py` | naïve per-group INT4 (no AWQ scaling). Use to see the quality floor — outputs are gibberish without calibration; demonstrates that AWQ scale-search is essential, not optional, for Lance |
| `scripts/run_baseline.py` | runs Lance inference in bf16 with a memory-frugal loader (meta-init + streaming bf16 cast). Lets a 12.3 GB bf16 model fit on a 16 GB GPU. |
| `scripts/run_quant_eval.py` | swaps Linears for `WQLinearINT4`/`WQLinearNVFP4` and runs the same inference for direct A/B comparison vs `run_baseline.py` |
| `patches/quantized_linear.py` | reference `WQLinearINT4` + `WQLinearNVFP4` modules (pure PyTorch, on-demand dequant). Slow but correctness-checking. Hook a fused INT4 GEMM kernel for production. |
| `comfyui/` | ComfyUI custom node pack (v1 scaffold; loader auto-detects Lance source). |

## Quantized model checkpoints

Uploaded to Hugging Face under the original Lance base model:

| Variant | Source | Quantized | Reduction |
|---|---|---|---|
| [`Reza2kn/Lance-3B-Video-AWQ-INT4`](https://huggingface.co/Reza2kn/Lance-3B-Video-AWQ-INT4) | 28.4 GB (F32) | 6.02 GB | 4.7× |
| [`Reza2kn/Lance-3B-Video-NVFP4`](https://huggingface.co/Reza2kn/Lance-3B-Video-NVFP4) | 28.4 GB (F32) | 6.93 GB | 4.1× |
| [`Reza2kn/Lance-3B-AWQ-INT4`](https://huggingface.co/Reza2kn/Lance-3B-AWQ-INT4) | 24.7 GB (F32) | 4.18 GB | 5.9× |
| [`Reza2kn/Lance-3B-NVFP4`](https://huggingface.co/Reza2kn/Lance-3B-NVFP4) | 24.7 GB (F32) | 5.09 GB | 4.8× |

## Pipeline

```
       ┌──────────────────────────────────────────────────────┐
       │  bytedance-research/Lance/{Lance_3B,Lance_3B_Video}  │   (F32 safetensors)
       └────────┬─────────────────────────────────────────────┘
                │
                ▼
       ┌──────────────────────────────────────────────────────┐
       │  awq_calibrate_single.py --task x2t_image            │   (und path: 252 linears)
       │  awq_calibrate_single.py --task t2i  --num_timesteps 2 │   (gen path: all 504)
       └────────┬─────────────────────────────────────────────┘
                │
                ▼
       ┌────────────────────────────────────┐
       │  awq_merge_stats.py                │
       └────────┬───────────────────────────┘
                │ (act_stats.pt)
                ├─────────────────────────┐
                ▼                         ▼
       ┌─────────────────┐       ┌─────────────────┐
       │ awq_apply.py    │       │ nvfp4_apply.py  │
       │  → INT4         │       │  → NVFP4 / FP8  │
       └─────────────────┘       └─────────────────┘
                │                         │
                ▼                         ▼
   Reza2kn/Lance-3B…-AWQ-INT4   Reza2kn/Lance-3B…-NVFP4
```

## Key design decisions

**Why not autoawq / llm-compressor?** Lance is not in their architecture registries (their `AutoModelForCausalLM.from_pretrained` flow fails on Lance's custom `PreTrainedModel`). AutoAWQ is also officially deprecated; its `pip install` upgrades torch in ways that broke our cu128 / Blackwell env.

**Why calibrate with TWO tasks?** Lance's Mixture-of-Tasks router sends understanding tokens through `mlp` / `q_proj` / etc. and generation tokens through `mlp_moe_gen` / `q_proj_moe_gen` / etc. A pure x2t calibration leaves the `_moe_gen` weights with no activation data, so AWQ falls back to plain min-max for them — which is exactly the case that produces gibberish (see `quantize_int4_minmax.py`'s outputs for evidence). Adding t2i routes activations through the gen path too.

**Why skip lm_head?** `inference_lance.py` line 539 asserts on `model.language_model.get_output_embeddings().weight.data.data_ptr()`. Swapping `lm_head` for `WQLinearINT4` (which has no `.weight` attribute) breaks that assert. It's also numerically sensitive (vocab projection) and only saves ~600 MB. Not worth the carve-out.

**Why a custom runtime swap-in instead of converting to vLLM format?** vLLM / TensorRT-LLM don't have Lance in their model registry. Until they do, the swap-in approach is the only way to load these checkpoints into the real Lance forward path.

## Memory-frugal loader

Lance's stock loader peaks GPU at ~26 GB (loads F32 24 GB safetensors, then `.to(cuda, bfloat16)` holds both copies temporarily). `scripts/run_baseline.py` patches the loader with:
- meta-device construction of every nn.Module (no real allocation until weight load)
- streaming bf16 cast from safetensors directly to the meta param's data, one tensor at a time

Net: GPU peak drops from ~26 GB to ~13.5 GB. Fits on a 16 GB laptop GPU.

## Reproduction

```bash
# 1. Clone Lance + download weights
git clone https://github.com/bytedance/Lance.git
cd Lance && bash setup_env.sh
huggingface-cli download bytedance-research/Lance --local-dir downloads

# 2. Clone this repo
cd .. && git clone https://github.com/Reza2kn/lance-quant.git
cd lance-quant
cp scripts/*.py ../Lance/        # the patched inference scripts need to import Lance's source

cd ../Lance
# Patch inference_lance.py for the memory-frugal loader
python patch_inference_lance.py inference_lance.py

# 3. Calibrate (collects activation statistics through Lance's actual forward)
python awq_calibrate_single.py --task x2t_image \
    --model_path downloads/Lance_3B_Video \
    --vit_path  downloads/Qwen2.5-VL-ViT \
    --example_json config/examples/x2t_image_example.json \
    --out ../calib/x2t_image_stats.pt

python awq_calibrate_single.py --task t2i \
    --model_path downloads/Lance_3B_Video \
    --vit_path  downloads/Qwen2.5-VL-ViT \
    --example_json config/examples/t2i_example.json \
    --out ../calib/t2i_stats.pt --num_timesteps 2

python awq_merge_stats.py --inputs ../calib/x2t_image_stats.pt ../calib/t2i_stats.pt \
    --out ../calib/act_stats.pt

# 4. Apply AWQ INT4
python awq_apply.py --src downloads/Lance_3B_Video/model.safetensors \
    --stats ../calib/act_stats.pt \
    --out ../models/Lance_3B_Video-AWQ-INT4

# 5. Apply NVFP4 (reuses the same act_stats)
python nvfp4_apply.py --src downloads/Lance_3B_Video/model.safetensors \
    --stats ../calib/act_stats.pt \
    --out ../models/Lance_3B_Video-NVFP4 --block_size 16

# 6. Evaluate either against bf16 baseline
python run_baseline.py --task x2t_image \
    --model_path downloads/Lance_3B_Video

python run_quant_eval.py --task x2t_image \
    --model_path downloads/Lance_3B_Video \
    --awq_dir ../models/Lance_3B_Video-AWQ-INT4 \
    --mode ondemand
```

## Limitations

- **Runtime speed** with the pure-PyTorch `WQLinear*` modules is ~10× slower than bf16 because dequant runs every forward call. Production needs a fused INT4/FP4 GEMM kernel (Triton / marlin / exllamav2 for INT4; TensorRT-LLM / vLLM ≥ 0.8 for NVFP4).
- **Calibration set is small** (17 samples across 2 tasks). Adding `image_edit`, `t2v`, `video_edit`, `x2t_video` would close residual quality gaps.
- **MLX and CoreML variants** for the same checkpoints are work-in-progress.

## License

Apache 2.0, matching the base Lance model.
