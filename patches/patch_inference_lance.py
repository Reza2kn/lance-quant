"""Patches inference_lance.py in-place to combine the early `.to(DEVICE)` with
bf16 conversion, so the model never has to allocate F32 storage on the GPU.

Idempotent: re-running won't double-patch.
"""

from pathlib import Path
import sys


def patch(path: str) -> None:
    p = Path(path)
    src = p.read_text()

    old = '    model = model.to(DEVICE)\n    log_stage("Lance model move to GPU", stage_start)'
    new = '    model = model.to(device=DEVICE, dtype=torch.bfloat16)\n    log_stage("Lance model move+bf16 to GPU", stage_start)'

    if new in src:
        print("already patched")
        return
    if old not in src:
        raise SystemExit("target pattern not found in " + path)

    src = src.replace(old, new, 1)
    p.write_text(src)
    print("patched", path)


if __name__ == "__main__":
    patch(sys.argv[1] if len(sys.argv) > 1 else "inference_lance.py")
