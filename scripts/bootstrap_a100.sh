#!/bin/bash
# A100 bootstrap: install deps + apply patches in one shot, logs to disk.
# Idempotent — safe to re-run.
set -e
exec >> /workspace/bootstrap.log 2>&1
echo "===== $(date) BOOTSTRAP START ====="

PY=/workspace/.venv-lance/bin/python
unset VIRTUAL_ENV

# 1) Deps install
echo "[1/4] installing Lance deps"
uv pip install --python "$PY" \
  "transformers==4.49.0" \
  "tokenizers==0.21.4" \
  "diffusers==0.29.1" \
  "huggingface_hub" hf_transfer \
  accelerate safetensors einops einops-exts \
  omegaconf pydantic timm sentencepiece tiktoken \
  qwen-vl-utils decord opencv-python imageio imageio-ffmpeg \
  pillow scipy scikit-image albumentations kornia torchmetrics \
  librosa soundfile tabulate tqdm psutil gpustat ftfy webdataset \
  protobuf datasets peft || { echo "deps install FAILED"; exit 1; }
echo "[1/4] deps OK"

# 2) flash-attn (prebuilt wheel for torch 2.6+cu124+py3.11)
echo "[2/4] installing flash-attn"
"$PY" -c "import torch; print('torch', torch.__version__, 'abi', torch.compiled_with_cxx11_abi())"
TORCH_MAJ=$("$PY" -c "import torch; print(torch.__version__.split('+')[0].split('.')[:2])" | tr -d "[],' " | sed 's/\([0-9]\)\([0-9]\)/\1.\2/')
ABI=$("$PY" -c "import torch; print('TRUE' if torch.compiled_with_cxx11_abi() else 'FALSE')")
WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch${TORCH_MAJ}cxx11abi${ABI}-cp311-cp311-linux_x86_64.whl"
echo "  selecting $WHL"
uv pip install --python "$PY" "$WHL" || echo "  flash-attn install FAILED (non-fatal; SDPA fallback works)"

# 3) Verify imports
echo "[3/4] verifying imports"
"$PY" -c "
import torch, transformers, diffusers, accelerate, safetensors, datasets
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devs', torch.cuda.device_count())
print('transformers', transformers.__version__)
print('diffusers', diffusers.__version__)
try:
  import flash_attn
  print('flash_attn', flash_attn.__version__)
except ImportError:
  print('flash_attn NOT installed')
"

# 4) Symlink weights into Lance dir layout
echo "[4/4] linking weights to downloads/"
cd /workspace/lance/src
mkdir -p downloads
for x in Lance_3B_Video Qwen2.5-VL-ViT Wan2.2_VAE.pth; do
  ln -sfn "/dev/shm/lance-weights/$x" "downloads/$x"
done
ls -la downloads/

# 5) Apply the inference_lance.py patch (skip if already patched)
if grep -q "Lance model move+bf16 to GPU" inference_lance.py; then
  echo "  inference_lance.py already patched"
else
  python3 -c "
src = open('inference_lance.py').read()
old = '    model = model.to(DEVICE)\n    log_stage(\"Lance model move to GPU\", stage_start)'
new = '    model = model.to(device=DEVICE, dtype=torch.bfloat16)\n    log_stage(\"Lance model move+bf16 to GPU\", stage_start)'
assert old in src, 'patch target not found'
open('inference_lance.py','w').write(src.replace(old, new, 1))
print('patched inference_lance.py')
"
fi

echo "===== $(date) BOOTSTRAP DONE ====="
touch /workspace/bootstrap.done
