"""Drop-in monkey-patch that replaces Lance's `init_from_model_path_if_needed`
with a streaming bf16 loader, so we don't peak at 24 GB F32 on the GPU.

Usage: import this module BEFORE running inference_lance.py.
  python -c "import sys; sys.path.insert(0,'/path/to/patches'); import load_bf16_inplace; exec(open('inference_lance.py').read())"

Or simpler, run via the wrapper `scripts/run_baseline.py`.
"""

from __future__ import annotations

import gc
import os.path as osp
import time

import torch
from safetensors import safe_open


def _patched_init_from_model_path_if_needed(model, model_args):
    """Streams safetensors -> bf16 -> model.<param>.data in place.
    Skips `latent_pos_embed.pos_embed` (Lance regenerates it on init).
    """
    path_dir = model_args.model_path
    candidates = [osp.join(path_dir, "model.safetensors"),
                  osp.join(path_dir, "ema.safetensors")]
    ck = next((p for p in candidates if osp.exists(p)), None)
    if ck is None:
        raise FileNotFoundError(
            f"No model.safetensors / ema.safetensors in {path_dir}")

    print(f"[bf16-loader] streaming {ck}")
    t0 = time.time()
    own = dict(model.state_dict(keep_vars=True))
    missing = set(own.keys())
    unexpected = []
    loaded = 0

    with safe_open(ck, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            if k == "latent_pos_embed.pos_embed":
                missing.discard(k)
                continue
            if k not in own:
                unexpected.append(k)
                continue
            src = f.get_tensor(k)
            if src.is_floating_point() and src.dtype != torch.bfloat16:
                src = src.to(torch.bfloat16)
            param = own[k]
            # In-place copy, keeps device/storage of the existing param.
            with torch.no_grad():
                if param.device.type == "meta":
                    # parameter wasn't materialised yet; force-allocate on CPU
                    param.data = src
                else:
                    param.data.copy_(src.to(param.device), non_blocking=True)
            missing.discard(k)
            loaded += 1
            if loaded % 250 == 0:
                gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[bf16-loader] {loaded} tensors in {time.time()-t0:.1f}s; "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    if unexpected:
        print(f"  unexpected: {unexpected[:5]}...")

    class _Msg:
        pass
    msg = _Msg()
    msg.missing_keys = sorted(missing)
    msg.unexpected_keys = unexpected
    return msg


def apply():
    import inference_lance
    inference_lance.init_from_model_path_if_needed = _patched_init_from_model_path_if_needed
    print("[patch] init_from_model_path_if_needed -> bf16 streaming loader")

    # Also patch the .to(device, dtype) chain — when called on an already-bf16
    # model, this is a no-op. But we wrap it so that VAE stays on CPU.
    _orig_to = torch.nn.Module.to

    def _patched_to(self, *args, **kwargs):
        # If the call is moving the *whole* Lance model to cuda, don't move
        # the VAE submodule with it.
        return _orig_to(self, *args, **kwargs)

    # We don't actually need to patch .to() — Lance doesn't store the VAE
    # inside the model. The vae_model variable in main() is separate.
    return True


if __name__ == "__main__":
    apply()
