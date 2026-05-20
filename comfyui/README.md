# ComfyUI-Lance

ComfyUI custom-node pack for [ByteDance Lance](https://huggingface.co/bytedance-research/Lance) and its quantized variants.

**v1 — subprocess wrapper, slow but correct.** Each task node shells out to a fresh `inference_lance.py` invocation (~30–60 s per generation due to model reload). v2 will hold the model resident across calls.

## Nodes shipped

| Node | What it does |
|---|---|
| **Lance: Model Loader** | Picks checkpoint flavor (`Lance_3B` or `Lance_3B_Video`) and precision (`bf16` / `awq_int4` / `nvfp4`); returns a `LANCE_MODEL` handle. |
| **Lance: Text → Image** | t2i, 768² default |
| **Lance: Text → Video** | t2v (needs `Lance_3B_Video`) |
| **Lance: Image Edit** | instruction-guided edit |
| **Lance: Image Understanding (VQA)** | image + question → answer string |

`video_edit` and `x2t_video` are doable with the same wrapper but require additional input plumbing (video frame stacks); coming in a follow-up.

## Setup

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
   ```

5. **Restart ComfyUI**. Nodes appear under the `Lance` category.

## VRAM requirements

| Precision | Approx GPU VRAM (LLM only, bf16 activations) |
|---|---|
| bf16 | 14 GB (fits on 16 GB GPU with our memory-frugal loader) |
| awq_int4 | 7 GB |
| nvfp4 | 7 GB |

(Add ~2 GB for VAE during generation tasks.)

## Limitations (v1)

- Per-call model reload makes each generation ~30–60 s, even on a fast GPU. v2 will refactor `inference_lance.main` to expose a `build` step separately from the per-sample `validate` step so the loader can hold the model resident.
- No support yet for `x2t_video` or `video_edit` (need video-frame plumbing).
- Quantized inference path uses pure-PyTorch on-demand dequant; a fused INT4 GEMM kernel (Triton / marlin / exllamav2 for INT4; TensorRT-LLM for NVFP4) would be 5–10× faster.

## License

Apache 2.0, matching Lance.
