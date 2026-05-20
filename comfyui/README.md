# ComfyUI-Lance

ComfyUI custom-node pack for [ByteDance Lance](https://huggingface.co/bytedance-research/Lance) and its quantized variants.

Hosted at:

- GitHub code: https://github.com/Reza2kn/lance-quant
- ComfyUI node folder inside the repo: `comfyui/`
- Hugging Face models: https://huggingface.co/Reza2kn

**v2 — resident worker with subprocess fallback.** The model loader can start a
long-lived Lance worker per checkpoint/precision combo, so repeated prompts do
not pay the full model-load cost. The older subprocess backend is still
available and can be used as a fallback when a local Lance checkout is missing
the resident-worker hooks.

## Which model should I use?

ComfyUI uses Lance's PyTorch runtime, so it can use the full multimodal bf16,
AWQ, and NVFP4 checkpoints. MLX-DWQ and CoreML are Apple deployment formats and
are not loaded by this ComfyUI node.

| Goal | Download this quant | Works in ComfyUI? | Notes |
|---|---|---:|---|
| Image generation, image edit, image VQA on a 16 GB NVIDIA GPU | `Reza2kn/Lance-3B-AWQ-INT4` | Yes | Recommended default. Smallest full multimodal checkpoint. |
| Same tasks on Blackwell / RTX 50-series | `Reza2kn/Lance-3B-NVFP4` | Yes | Correctness path works; fused FP4 kernels would be faster later. |
| Video generation + image/video-family Lance checkpoint | `Reza2kn/Lance-3B-Video-AWQ-INT4` | Yes | Use `flavor=Lance_3B_Video`. |
| Video checkpoint on Blackwell / RTX 50-series | `Reza2kn/Lance-3B-Video-NVFP4` | Yes | Use `flavor=Lance_3B_Video`. |
| Apple Silicon text/image understanding experiments | `*-MLX-4bit-DWQ` | No | Use with MLX tooling, not ComfyUI. |
| iOS / CoreML app deployment | `*-CoreML-palettized-4bit` | No | Use with CoreML tooling, not ComfyUI. |

## Nodes shipped

| Node | What it does |
|---|---|
| **Lance: Model Loader** | Picks checkpoint flavor (`Lance_3B` or `Lance_3B_Video`), precision (`bf16` / `awq_int4` / `nvfp4`), and backend (`resident_worker` / `subprocess`); returns a `LANCE_MODEL` handle. |
| **Lance: Text → Image** | t2i, 768² default |
| **Lance: Text → Video** | t2v (needs `Lance_3B_Video`) |
| **Lance: Image Edit** | instruction-guided edit |
| **Lance: Image Understanding (VQA)** | image + question → answer string |

`video_edit` and `x2t_video` are doable with the same wrapper but require additional input plumbing (video frame stacks); coming in a follow-up.

## Setup

Beginner quick start:

1. Install or open ComfyUI.
2. Put this repo in `ComfyUI/custom_nodes/ComfyUI-Lance`.
3. Put Lance and one quantized checkpoint in `ComfyUI/models/lance/`.
4. Restart ComfyUI.
5. Add **Lance: Model Loader**, choose `flavor`, `precision`, and
   `resident_worker`.
6. Connect the loader to **Lance: Text -> Image**, **Lance: Image Edit**, or
   **Lance: Image Understanding (VQA)**.

1. **Clone Lance source** somewhere ComfyUI can see:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/Reza2kn/lance-quant ComfyUI-Lance
   cd ComfyUI-Lance
   git clone https://github.com/bytedance/Lance.git
   ```
   Or set `LANCE_SRC_PATH` env var to point at an existing clone.

2. **Install Lance dependencies into ComfyUI's Python**:
   ```bash
   pip install transformers==4.49.0 diffusers==0.29.1 flash-attn \
              accelerate safetensors einops decord opencv-python \
              imageio imageio-ffmpeg qwen-vl-utils kornia \
              omegaconf pydantic timm sentencepiece tiktoken
   ```
   See `lance-quant/scripts/bootstrap_a100.sh` for the full list.

3. **Apply our memory-frugal patch** to Lance:
   ```bash
   cd ComfyUI-Lance/Lance
   python ../patches/patch_inference_lance.py inference_lance.py
   ```

4. **Place model weights**:
   ```
   ComfyUI/models/lance/
   ├── Lance_3B/                    # or Lance_3B_Video/
   │   ├── model.safetensors
   │   ├── llm_config.json
   │   ├── tokenizer.json
   │   └── ...
   ├── Qwen2.5-VL-ViT/
   │   ├── vit.safetensors
   │   └── config.json
   ├── Wan2.2_VAE.pth
   ├── Lance_3B-AWQ-INT4/           # optional, 4× smaller
   │   ├── awq_state_dict.safetensors
   │   ├── awq_meta.json
   │   └── README.md
   └── Lance_3B-NVFP4/              # optional, Blackwell-friendly
       ├── nvfp4_state_dict.safetensors
       ├── nvfp4_meta.json
       └── README.md
   ```

   Download with:
   ```bash
   huggingface-cli download bytedance-research/Lance --local-dir ComfyUI/models/lance
   huggingface-cli download Reza2kn/Lance-3B-AWQ-INT4 --local-dir ComfyUI/models/lance/Lance_3B-AWQ-INT4
   huggingface-cli download Reza2kn/Lance-3B-NVFP4    --local-dir ComfyUI/models/lance/Lance_3B-NVFP4
   huggingface-cli download Reza2kn/Lance-3B-Video-AWQ-INT4 --local-dir ComfyUI/models/lance/Lance_3B_Video-AWQ-INT4
   huggingface-cli download Reza2kn/Lance-3B-Video-NVFP4    --local-dir ComfyUI/models/lance/Lance_3B_Video-NVFP4
   ```

   You only need the quant repo you plan to select in the loader. For example,
   if you choose `flavor=Lance_3B` and `precision=awq_int4`, download
   `Reza2kn/Lance-3B-AWQ-INT4`.

5. **Restart ComfyUI**. Nodes appear under the `Lance` category.

## Backend modes

| Backend | Behavior |
|---|---|
| `resident_worker` | Starts `lance_worker.py` once per selected checkpoint and sends later requests over line-delimited JSON. Best for interactive use. |
| `subprocess` | Runs `scripts/run_baseline.py` or `scripts/run_quant_eval.py` for every node execution. Slower, but useful for debugging and for Lance checkouts where resident execution fails. |

The loader has `fallback_to_subprocess` enabled by default. Turn it off if you
want resident-worker errors to fail loudly during development.

Set `LANCE_QUANT_PATH` if the ComfyUI custom node has been copied out of the
repo and cannot find `scripts/run_baseline.py` / `scripts/run_quant_eval.py`.

## VRAM requirements

| Precision | Approx GPU VRAM (LLM only, bf16 activations) |
|---|---|
| bf16 | 14 GB (fits on 16 GB GPU with our memory-frugal loader) |
| awq_int4 | 7 GB |
| nvfp4 | 7 GB |

(Add ~2 GB for VAE during generation tasks.)

## Limitations

- No support yet for `x2t_video` or `video_edit` (need video-frame plumbing).
- Quantized inference path uses pure-PyTorch on-demand dequant for AWQ and
  NVFP4 correctness. A fused INT4/FP4 GEMM kernel (Triton / marlin /
  exllamav2 for INT4; TensorRT-LLM for NVFP4) would be much faster.

## License

Apache 2.0, matching Lance.
