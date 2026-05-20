"""ComfyUI nodes for ByteDance Lance + AWQ-INT4 / NVFP4 quantized variants.

v2 design: the default backend starts a resident Lance worker per
checkpoint/precision combo and sends requests over line-delimited JSON. The
older subprocess backend remains available for debugging and as a fallback.

Setup:
  - Clone github.com/bytedance/Lance to either:
      * `<this dir>/Lance/`  (default search path)
      * or set LANCE_SRC_PATH env var
  - Place model weights under `ComfyUI/models/lance/`:
        Lance_3B/                          (or Lance_3B_Video/)
        Qwen2.5-VL-ViT/
        Wan2.2_VAE.pth
        Lance_3B-AWQ-INT4/                 (optional, for 4-bit inference)
        Lance_3B-NVFP4/                    (optional)
  - The Lance conda env must be activated in the ComfyUI process; see top-level
    README of lance-quant for the env recipe.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Paths & helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _lance_src_path() -> Path:
    env = os.environ.get("LANCE_SRC_PATH")
    if env and Path(env).exists():
        return Path(env)
    here = Path(__file__).parent
    for cand in (here / "Lance", here.parent / "Lance",
                  Path.home() / "lance-quant" / "src"):
        if (cand / "inference_lance.py").exists():
            return cand
    raise RuntimeError(
        "Lance source not found. Either set LANCE_SRC_PATH env var to your "
        "git clone of github.com/bytedance/Lance, or place the repo at "
        f"{here / 'Lance'} or {here.parent / 'Lance'}."
    )


def _model_root() -> Path:
    try:
        import folder_paths
        root = Path(folder_paths.models_dir) / "lance"
    except Exception:
        root = Path(os.environ.get("LANCE_MODELS_DIR", "models/lance"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _script_path(script_name: str) -> Path:
    """Find lance-quant helper scripts whether installed as a repo or copied."""
    env = os.environ.get("LANCE_QUANT_PATH")
    candidates = []
    if env:
        candidates.append(Path(env) / "scripts" / script_name)
    candidates.extend([
        _repo_root() / "scripts" / script_name,
        Path(__file__).resolve().parent / script_name,
        _lance_src_path() / script_name,
    ])
    for cand in candidates:
        if cand.exists():
            return cand
    raise RuntimeError(
        f"Cannot find {script_name}. Set LANCE_QUANT_PATH to the lance-quant "
        "checkout, or keep the ComfyUI node inside the repo."
    )


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE: [B, H, W, C] float 0..1 -> PIL."""
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    arr = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> ComfyUI IMAGE: [1, H, W, C] float 0..1."""
    arr = np.asarray(img.convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ---------------------------------------------------------------------------
# Backends: v1 subprocess runner and v2 persistent worker
# ---------------------------------------------------------------------------


class _WorkerClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.proc: subprocess.Popen[str] | None = None
        self._stderr_q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()

    def _start(self):
        if self.proc and self.proc.poll() is None:
            return
        src = _lance_src_path()
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "lance_worker.py"),
            "--lance_src", str(src),
            "--script_root", str(_repo_root()),
            "--model_path", self.cfg["bf16_path"],
            "--vit_path", self.cfg["vit_path"],
            "--save_path_gen", tempfile.mkdtemp(prefix="lance_worker_boot_"),
        ]
        if self.cfg.get("awq_dir"):
            cmd.extend(["--awq_dir", self.cfg["awq_dir"]])

        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("TORCHDYNAMO_DISABLE", "1")
        env.setdefault("TORCH_COMPILE_DISABLE", "1")
        self.proc = subprocess.Popen(
            cmd, cwd=str(src), env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )

        def _drain_stderr():
            assert self.proc and self.proc.stderr
            for line in self.proc.stderr:
                self._stderr_q.put(line)
                sys.stderr.write(line)

        threading.Thread(target=_drain_stderr, daemon=True).start()
        assert self.proc.stdout
        deadline = time.time() + int(os.environ.get("LANCE_WORKER_START_TIMEOUT", "900"))
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if line == "" and self.proc.poll() is not None:
                raise RuntimeError(self._worker_error("worker exited before READY"))
            if line.strip() == "READY":
                return
            if line:
                sys.stderr.write(f"[lance-worker] {line}")
        raise TimeoutError(self._worker_error("timed out waiting for READY"))

    def _worker_error(self, prefix: str) -> str:
        tail = []
        while not self._stderr_q.empty():
            tail.append(self._stderr_q.get_nowait())
        return prefix + ("\n" + "".join(tail[-40:]) if tail else "")

    def request(self, payload: dict) -> dict:
        with self._lock:
            self._start()
            assert self.proc and self.proc.stdin and self.proc.stdout
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
            while True:
                line = self.proc.stdout.readline()
                if line == "" and self.proc.poll() is not None:
                    raise RuntimeError(self._worker_error("worker exited during request"))
                line = line.strip()
                if not line:
                    continue
                try:
                    res = json.loads(line)
                except json.JSONDecodeError:
                    sys.stderr.write(f"[lance-worker] {line}\n")
                    continue
                if not res.get("ok"):
                    raise RuntimeError(res.get("error", "Lance worker failed") + "\n" + res.get("trace", ""))
                return res


_WORKERS: dict[str, _WorkerClient] = {}


def _worker_for(cfg: dict) -> _WorkerClient:
    key = json.dumps({
        "src": str(_lance_src_path()),
        "model": cfg["bf16_path"],
        "vit": cfg["vit_path"],
        "precision": cfg["precision"],
        "awq": cfg.get("awq_dir"),
    }, sort_keys=True)
    if key not in _WORKERS:
        _WORKERS[key] = _WorkerClient(cfg)
    return _WORKERS[key]


def _run_lance_cli(*, task: str, model_path: str, vit_path: str,
                   awq_dir: str | None, example_json: str, save_dir: str,
                   num_steps: int, num_frames: int, height: int, width: int,
                   cfg_scale: float, seed: int) -> dict:
    """Call our run_baseline.py / run_quant_eval.py and return the parsed result."""
    py = sys.executable
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    env.setdefault("TORCH_COMPILE_DISABLE", "1")

    if awq_dir:
        script = "run_quant_eval.py"
        cmd = [py, str(_script_path(script)), "--task", task,
                "--model_path", model_path, "--vit_path", vit_path,
                "--awq_dir", awq_dir, "--example_json", example_json,
                "--save_path_gen", save_dir,
                "--validation_num_timesteps", str(num_steps),
                "--cfg_scale", str(cfg_scale), "--seed", str(seed),
                "--video_height", str(height), "--video_width", str(width),
                "--num_frames", str(num_frames), "--mode", "ondemand"]
    else:
        script = "run_baseline.py"
        cmd = [py, str(_script_path(script)), "--task", task,
                "--model_path", model_path, "--vit_path", vit_path,
                "--example_json", example_json,
                "--save_path_gen", save_dir,
                "--validation_num_timesteps", str(num_steps),
                "--cfg_scale", str(cfg_scale), "--seed", str(seed),
                "--video_height", str(height), "--video_width", str(width),
                "--num_frames", str(num_frames)]

    print(f"[lance-comfy] running {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(_lance_src_path()), env=env, capture_output=True, text=True)
    print(f"[lance-comfy] returned {res.returncode} in {time.time()-t0:.1f}s")
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-2000:])
        raise RuntimeError(f"Lance CLI failed: {res.stderr[-500:]}")

    prompt_json = Path(save_dir) / "prompt.json"
    if prompt_json.exists():
        return json.loads(prompt_json.read_text())
    return {}


def _run_lance(*, lance_model: dict, task: str, example_json: str, save_dir: str,
               num_steps: int, num_frames: int, height: int, width: int,
               cfg_scale: float, seed: int) -> dict:
    backend = lance_model.get("backend", "resident_worker")
    if backend == "resident_worker":
        try:
            return _worker_for(lance_model).request({
                "task": task,
                "manifest_path": example_json,
                "save_dir": save_dir,
                "num_steps": num_steps,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "cfg_scale": cfg_scale,
                "seed": seed,
            }).get("outputs", {})
        except Exception:
            if not lance_model.get("fallback_to_subprocess", True):
                raise
            print("[lance-comfy] resident worker failed; falling back to subprocess", file=sys.stderr)

    return _run_lance_cli(
        task=task,
        model_path=lance_model["bf16_path"],
        vit_path=lance_model["vit_path"],
        awq_dir=lance_model["awq_dir"],
        example_json=example_json,
        save_dir=save_dir,
        num_steps=num_steps,
        num_frames=num_frames,
        height=height,
        width=width,
        cfg_scale=cfg_scale,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


class LanceModelLoader:
    """Selects which Lance checkpoint + precision to use; returns a config dict.
    The actual model is reloaded per inference call (see top-level docstring)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flavor": (["Lance_3B", "Lance_3B_Video"], {"default": "Lance_3B"}),
                "precision": (["bf16", "awq_int4", "nvfp4"], {"default": "awq_int4"}),
                "backend": (["resident_worker", "subprocess"], {"default": "resident_worker"}),
                "fallback_to_subprocess": ("BOOLEAN", {"default": True}),
            },
        }
    RETURN_TYPES = ("LANCE_MODEL",)
    RETURN_NAMES = ("lance_model",)
    FUNCTION = "load"
    CATEGORY = "Lance"

    def load(self, flavor, precision, backend, fallback_to_subprocess):
        root = _model_root()
        vit = root / "Qwen2.5-VL-ViT"
        vae = root / "Wan2.2_VAE.pth"
        # the bf16 source dir is needed for tokenizer/config in all modes
        bf16 = root / flavor
        if precision == "bf16":
            awq_dir = None
        elif precision == "awq_int4":
            awq_dir = root / f"{flavor}-AWQ-INT4"
        else:
            awq_dir = root / f"{flavor}-NVFP4"

        for p, name in [(vit, "ViT"), (vae, "VAE"), (bf16, flavor)]:
            if not p.exists():
                raise RuntimeError(
                    f"missing {name} at {p}. Download "
                    f"bytedance-research/Lance into {root}.")
        if awq_dir and not awq_dir.exists():
            raise RuntimeError(
                f"missing quantized weights at {awq_dir}. Download "
                f"Reza2kn/{awq_dir.name} into {root}.")

        cfg = {
            "flavor": flavor, "precision": precision,
            "backend": backend,
            "fallback_to_subprocess": fallback_to_subprocess,
            "bf16_path": str(bf16), "vit_path": str(vit), "vae_path": str(vae),
            "awq_dir": str(awq_dir) if awq_dir else None,
        }
        print(f"[lance-comfy] loaded config: {cfg}")
        return (cfg,)


class _BaseLanceTask:
    CATEGORY = "Lance"
    TASK_NAME: str = ""

    def _save_example_json(self, save_dir: Path, payload: dict) -> str:
        save_dir.mkdir(parents=True, exist_ok=True)
        p = save_dir / "_input_manifest.json"
        p.write_text(json.dumps(payload))
        return str(p)


class LanceT2I(_BaseLanceTask):
    TASK_NAME = "t2i"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lance_model": ("LANCE_MODEL",),
            "prompt": ("STRING", {"multiline": True,
                                    "default": "A beautiful landscape painting."}),
            "num_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 15.0, "step": 0.1}),
            "seed": ("INT", {"default": 42}),
            "size": (["768x768", "1024x1024", "512x512"], {"default": "768x768"}),
        }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self, lance_model, prompt, num_steps, cfg_scale, seed, size):
        h, w = map(int, size.split("x"))
        save = Path(tempfile.mkdtemp(prefix="lance_t2i_"))
        # Lance's example JSON for t2i is {filename: prompt}
        manifest = {"000000.png": prompt}
        (save / "_input.json").write_text(json.dumps(manifest))

        _run_lance(
            lance_model=lance_model, task="t2i",
            example_json=str(save / "_input.json"),
            save_dir=str(save), num_steps=num_steps,
            num_frames=1, height=h, width=w,
            cfg_scale=cfg_scale, seed=seed,
        )

        # Lance saves PNG/MP4 as 000000.{png,mp4}
        for fname in ("000000.png", "000000.mp4"):
            p = save / fname
            if p.exists():
                if fname.endswith(".png"):
                    return (_pil_to_tensor(Image.open(p)),)
        raise RuntimeError(f"no output in {save}")


class LanceX2TImage(_BaseLanceTask):
    TASK_NAME = "x2t_image"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lance_model": ("LANCE_MODEL",),
            "image": ("IMAGE",),
            "question": ("STRING", {"multiline": True,
                                      "default": "Describe this image."}),
        }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    def run(self, lance_model, image, question):
        save = Path(tempfile.mkdtemp(prefix="lance_x2t_"))
        pil = _tensor_to_pil(image)
        ipath = save / "input.png"
        pil.save(ipath)

        # Lance's x2t_image expects an interleave_array with image + Q
        manifest = {"0001": {
            "interleave_array": [str(ipath),
                                  ["Look at the image carefully and answer.",
                                   question, ""]],
            "element_dtype_array": ["image", "text"],
            "istarget_in_interleave": [0, 1],
        }}
        (save / "_input.json").write_text(json.dumps(manifest))

        results = _run_lance(
            lance_model=lance_model, task="x2t_image",
            example_json=str(save / "_input.json"),
            save_dir=str(save), num_steps=1, num_frames=1,
            height=768, width=768, cfg_scale=4.0, seed=42,
        )
        # results: {filename: answer_text}
        if results:
            return (next(iter(results.values())).replace("<|im_end|>", "").strip(),)
        return ("(no output)",)


class LanceImageEdit(_BaseLanceTask):
    TASK_NAME = "image_edit"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lance_model": ("LANCE_MODEL",),
            "image": ("IMAGE",),
            "instruction": ("STRING", {"multiline": True,
                                         "default": "Make it look like a watercolor painting."}),
            "num_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 15.0, "step": 0.1}),
            "seed": ("INT", {"default": 42}),
        }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self, lance_model, image, instruction, num_steps, cfg_scale, seed):
        save = Path(tempfile.mkdtemp(prefix="lance_edit_"))
        pil = _tensor_to_pil(image)
        ipath = save / "input.png"
        pil.save(ipath)

        manifest = {"0001": {
            "interleave_array": [instruction, str(ipath), str(ipath)],
            "element_dtype_array": ["text", "image", "image"],
            "istarget_in_interleave": [0, 0, 1],
        }}
        (save / "_input.json").write_text(json.dumps(manifest))

        _run_lance(
            lance_model=lance_model, task="image_edit",
            example_json=str(save / "_input.json"),
            save_dir=str(save), num_steps=num_steps, num_frames=1,
            height=pil.height, width=pil.width,
            cfg_scale=cfg_scale, seed=seed,
        )
        out = save / "0001.png"
        if out.exists():
            return (_pil_to_tensor(Image.open(out)),)
        raise RuntimeError(f"no edited image found in {save}")


# t2v / video_edit / x2t_video follow the same pattern; keeping them simple

class LanceT2V(_BaseLanceTask):
    TASK_NAME = "t2v"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lance_model": ("LANCE_MODEL",),
            "prompt": ("STRING", {"multiline": True}),
            "num_frames": ("INT", {"default": 50, "min": 5, "max": 121}),
            "height": ("INT", {"default": 480, "min": 256, "max": 768, "step": 32}),
            "width": ("INT", {"default": 832, "min": 256, "max": 1280, "step": 32}),
            "num_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 15.0, "step": 0.1}),
            "seed": ("INT", {"default": 42}),
        }}
    RETURN_TYPES = ("STRING",)   # returns path to generated video file
    FUNCTION = "run"
    def run(self, lance_model, prompt, num_frames, height, width, num_steps, cfg_scale, seed):
        if lance_model["flavor"] != "Lance_3B_Video":
            raise RuntimeError("t2v requires the Lance_3B_Video checkpoint")
        save = Path(tempfile.mkdtemp(prefix="lance_t2v_"))
        manifest = {"000000.mp4": prompt}
        (save / "_input.json").write_text(json.dumps(manifest))
        _run_lance(lance_model=lance_model, task="t2v",
                       example_json=str(save / "_input.json"),
                       save_dir=str(save), num_steps=num_steps,
                       num_frames=num_frames, height=height, width=width,
                       cfg_scale=cfg_scale, seed=seed)
        return (str(save / "000000.mp4"),)


# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LanceModelLoader": LanceModelLoader,
    "LanceT2I":         LanceT2I,
    "LanceT2V":         LanceT2V,
    "LanceImageEdit":   LanceImageEdit,
    "LanceX2TImage":    LanceX2TImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LanceModelLoader": "Lance: Model Loader",
    "LanceT2I":         "Lance: Text → Image",
    "LanceT2V":         "Lance: Text → Video",
    "LanceImageEdit":   "Lance: Image Edit",
    "LanceX2TImage":    "Lance: Image Understanding (VQA)",
}
