"""ComfyUI nodes for ByteDance Lance + AWQ-INT4 / NVFP4 quantized variants."""

from __future__ import annotations

import gc
import json
import os
import os.path as osp
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Lance source location
# ---------------------------------------------------------------------------


def _lance_src_path() -> Path:
    """Resolve where Lance's source code lives. Search order:
      1) LANCE_SRC_PATH env var
      2) <this dir>/Lance/   (git submodule or extracted clone)
      3) ComfyUI/custom_nodes/Lance/   (sibling node)
    """
    env = os.environ.get("LANCE_SRC_PATH")
    if env:
        return Path(env)
    here = Path(__file__).parent
    for cand in (here / "Lance", here.parent / "Lance"):
        if (cand / "inference_lance.py").exists():
            return cand
    raise RuntimeError(
        "Lance source not found. Set LANCE_SRC_PATH env var to your "
        "git clone of github.com/bytedance/Lance, or place the repo at "
        f"{here / 'Lance'}."
    )


def _model_root() -> Path:
    """ComfyUI's models/lance/ directory."""
    # Try standard ComfyUI layout first
    try:
        import folder_paths  # ComfyUI's built-in
        root = Path(folder_paths.models_dir) / "lance"
    except Exception:
        # Fallback: env or relative
        root = Path(os.environ.get("LANCE_MODELS_DIR", "models/lance"))
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Meta-init + streaming bf16 loader (mirror of scripts/run_baseline.py)
# ---------------------------------------------------------------------------


@contextmanager
def _meta_init():
    orig_empty = torch.empty
    def _empty_meta(*sizes, **kw):
        kw.setdefault("device", "meta")
        return orig_empty(*sizes, **kw)
    torch.empty = _empty_meta
    try:
        yield
    finally:
        torch.empty = orig_empty


def _streaming_bf16_loader(model, ck_path: str):
    print(f"[lance] streaming bf16 weights from {ck_path}")
    t0 = time.time()
    own = dict(model.state_dict(keep_vars=True))
    device = next(iter(model.parameters())).device
    n_loaded = 0
    with safe_open(ck_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k == "latent_pos_embed.pos_embed":
                continue
            if k not in own:
                continue
            src = f.get_tensor(k)
            if src.is_floating_point() and src.dtype != torch.bfloat16:
                src = src.to(torch.bfloat16)
            p = own[k]
            with torch.no_grad():
                if p.device.type == "meta":
                    p.data = src.to(device)
                else:
                    p.data.copy_(src.to(device), non_blocking=True)
            n_loaded += 1
    print(f"[lance] {n_loaded} tensors loaded in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# WQLinearINT4 / WQLinearNVFP4 (mirror of patches/quantized_linear.py)
# ---------------------------------------------------------------------------


class WQLinearINT4(torch.nn.Module):
    MODE = "ondemand"

    def __init__(self, in_features, out_features, group_size, bias, device, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.compute_dtype = dtype
        n_groups = in_features // group_size
        self.register_buffer("qweight", torch.zeros((out_features, in_features // 2), dtype=torch.uint8, device=device))
        self.register_buffer("scales",  torch.zeros((out_features, n_groups), dtype=dtype, device=device))
        self.register_buffer("zeros",   torch.zeros((out_features, n_groups), dtype=torch.uint8, device=device))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def _dequantize(self):
        packed = self.qweight
        lo = (packed & 0xF).to(torch.int16)
        hi = ((packed >> 4) & 0xF).to(torch.int16)
        unpacked = torch.stack([lo, hi], dim=-1).reshape(self.out_features, self.in_features)
        u = unpacked.reshape(self.out_features, -1, self.group_size).to(self.compute_dtype)
        z = self.zeros.unsqueeze(-1).to(self.compute_dtype)
        s = self.scales.unsqueeze(-1)
        return ((u - z) * s).reshape(self.out_features, self.in_features)

    def forward(self, x):
        return torch.nn.functional.linear(x, self._dequantize(), self.bias)


FP4_LUT_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


class WQLinearNVFP4(torch.nn.Module):
    """4-bit FP (E2M1) + per-block FP8/bf16 scale, block_size=16."""

    def __init__(self, in_features, out_features, block_size, bias, device, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.compute_dtype = dtype
        n_blocks = in_features // block_size
        self.register_buffer("qweight",     torch.zeros((out_features, in_features // 2), dtype=torch.uint8, device=device))
        self.register_buffer("scales_bf16", torch.zeros((out_features, n_blocks), dtype=dtype, device=device))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def _dequantize(self):
        packed = self.qweight
        lo = packed & 0xF
        hi = (packed >> 4) & 0xF
        codes = torch.stack([lo, hi], dim=-1).reshape(self.out_features, self.in_features)
        sign = (codes >> 3).to(torch.float32) * -2 + 1
        mag = (codes & 0x7).to(torch.int64)
        lut = FP4_LUT_POS.to(codes.device, dtype=torch.float32)
        vals = lut[mag] * sign
        vals = vals.to(self.compute_dtype).reshape(self.out_features, -1, self.block_size)
        return (vals * self.scales_bf16.unsqueeze(-1)).reshape(self.out_features, self.in_features)

    def forward(self, x):
        return torch.nn.functional.linear(x, self._dequantize(), self.bias)


# ---------------------------------------------------------------------------
# Lance loader (used by LanceModelLoader)
# ---------------------------------------------------------------------------


def _patch_inference_lance(use_meta_init: bool = True):
    sys.path.insert(0, str(_lance_src_path()))
    import inference_lance as IL
    from modeling.lance import Lance
    from modeling.lance.qwen2_navit import Qwen2ForCausalLM
    from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel

    if not use_meta_init:
        return IL

    _OQ = Qwen2ForCausalLM.__init__
    _OV = Qwen2_5_VisionTransformerPretrainedModel.__init__
    _OL = Lance.__init__

    def _Q(self, c):
        with _meta_init(): _OQ(self, c)
    def _V(self, c):
        with _meta_init(): _OV(self, c)
    def _L(self, *a, **k):
        with _meta_init(): _OL(self, *a, **k)

    Qwen2ForCausalLM.__init__ = _Q
    Qwen2_5_VisionTransformerPretrainedModel.__init__ = _V
    Lance.__init__ = _L
    return IL


def _swap_to_awq_int4(model, awq_dir: Path):
    meta = json.loads((awq_dir / "awq_meta.json").read_text())["per_weight"]
    modules_by_name = dict(model.named_modules())
    sd_path = awq_dir / "awq_state_dict.safetensors"

    for wkey in meta:
        info = meta[wkey]
        lin_name = wkey[:-len(".weight")]
        parent_path, lin_attr = lin_name.rsplit(".", 1)
        parent = modules_by_name.get(parent_path)
        if parent is None:
            continue
        old = getattr(parent, lin_attr, None)
        if not isinstance(old, torch.nn.Linear):
            continue
        device = old.weight.device if old.weight.device.type != "meta" else torch.device("cuda")
        new = WQLinearINT4(
            in_features=info["shape"][1], out_features=info["shape"][0],
            group_size=info["group_size"], bias=old.bias is not None, device=device,
        )
        setattr(parent, lin_attr, new)
        modules_by_name[lin_name] = new

    # Load pass-through + quant buffers
    own = dict(model.state_dict(keep_vars=True))
    with safe_open(str(sd_path), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k.endswith((".qweight", ".scales", ".zeros")):
                base, suf = k.rsplit(".", 1)
                mod = modules_by_name.get(base)
                if mod is not None and hasattr(mod, suf):
                    target = getattr(mod, suf)
                    with torch.no_grad():
                        target.data.copy_(t.to(target.device, non_blocking=True))
            elif k in own and k != "latent_pos_embed.pos_embed":
                if t.is_floating_point():
                    t = t.to(torch.bfloat16)
                p = own[k]
                with torch.no_grad():
                    if p.device.type == "meta":
                        p.data = t.to("cuda" if torch.cuda.is_available() else "cpu")
                    else:
                        p.data.copy_(t.to(p.device), non_blocking=True)


def _swap_to_nvfp4(model, nvfp4_dir: Path):
    meta = json.loads((nvfp4_dir / "nvfp4_meta.json").read_text())["per_weight"]
    modules_by_name = dict(model.named_modules())
    sd_path = nvfp4_dir / "nvfp4_state_dict.safetensors"

    for wkey in meta:
        info = meta[wkey]
        lin_name = wkey[:-len(".weight")]
        parent_path, lin_attr = lin_name.rsplit(".", 1)
        parent = modules_by_name.get(parent_path)
        if parent is None:
            continue
        old = getattr(parent, lin_attr, None)
        if not isinstance(old, torch.nn.Linear):
            continue
        device = old.weight.device if old.weight.device.type != "meta" else torch.device("cuda")
        new = WQLinearNVFP4(
            in_features=info["shape"][1], out_features=info["shape"][0],
            block_size=info["block_size"], bias=old.bias is not None, device=device,
        )
        setattr(parent, lin_attr, new)
        modules_by_name[lin_name] = new

    own = dict(model.state_dict(keep_vars=True))
    with safe_open(str(sd_path), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k.endswith((".qweight", ".scales_bf16")):
                base, suf = k.rsplit(".", 1)
                mod = modules_by_name.get(base)
                if mod is not None and hasattr(mod, suf):
                    with torch.no_grad():
                        getattr(mod, suf).data.copy_(t.to(getattr(mod, suf).device, non_blocking=True))
            elif k.endswith(".scales_fp8"):
                continue   # skip; we use bf16 scales path
            elif k in own and k != "latent_pos_embed.pos_embed":
                if t.is_floating_point():
                    t = t.to(torch.bfloat16)
                p = own[k]
                with torch.no_grad():
                    if p.device.type == "meta":
                        p.data = t.to("cuda" if torch.cuda.is_available() else "cpu")
                    else:
                        p.data.copy_(t.to(p.device), non_blocking=True)


# ---------------------------------------------------------------------------
# ComfyUI nodes
# ---------------------------------------------------------------------------


class LanceModelLoader:
    """Loads Lance (image or video flavor) at the requested precision.
    Returns an opaque LANCE_MODEL handle used by the inference nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flavor": (["Lance_3B", "Lance_3B_Video"], {"default": "Lance_3B_Video"}),
                "precision": (["bf16", "awq_int4", "nvfp4"], {"default": "awq_int4"}),
            },
            "optional": {
                "vit_path":  ("STRING", {"default": "auto"}),
                "vae_path":  ("STRING", {"default": "auto"}),
                "weight_dir": ("STRING", {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("LANCE_MODEL",)
    RETURN_NAMES = ("lance_model",)
    FUNCTION = "load"
    CATEGORY = "Lance"

    def load(self, flavor, precision, vit_path="auto", vae_path="auto", weight_dir="auto"):
        root = _model_root()
        if vit_path == "auto":
            vit_path = str(root / "Qwen2.5-VL-ViT")
        if vae_path == "auto":
            vae_path = str(root / "Wan2.2_VAE.pth")
        if weight_dir == "auto":
            if precision == "bf16":
                weight_dir = str(root / flavor)
            elif precision == "awq_int4":
                # Quantized weights live alongside flavor as a sibling repo
                weight_dir = str(root / f"{flavor}-AWQ-INT4")
            elif precision == "nvfp4":
                weight_dir = str(root / f"{flavor}-NVFP4")

        IL = _patch_inference_lance(use_meta_init=True)
        # Build minimal argv for inference_lance.main
        sys.argv = [
            "inference_lance.py",
            "--model_path", weight_dir if precision == "bf16" else str(root / flavor),
            "--vit_path", vit_path,
            "--vit_type", "qwen_2_5_vl_original",
            "--llm_qk_norm", "true",
            "--llm_qk_norm_und", "true",
            "--llm_qk_norm_gen", "true",
            "--tie_word_embeddings", "false",
            "--validation_num_timesteps", "30",
            "--validation_timestep_shift", "3.5",
            "--copy_init_moe", "true",
            "--max_num_frames", "121",
            "--max_latent_size", "64",
            "--latent_patch_size", "1", "1", "1",
            "--visual_und", "true", "--visual_gen", "true",
            "--vae_model_type", "wan",
            "--apply_qwen_2_5_vl_pos_emb", "true",
            "--apply_chat_template", "false",
            "--cfg_type", "0",
            "--validation_data_seed", "42",
            "--video_height", "768", "--video_width", "768", "--num_frames", "50",
            "--task", "x2t_image",     # placeholder; reset per-call below
            "--save_path_gen", "results/comfyui_dummy",
            "--resolution", "image_768res",
            "--text_template", "true", "--cfg_text_scale", "4.0", "--use_KVcache", "true",
        ]

        # Override init_from_model_path_if_needed
        if precision == "bf16":
            IL.init_from_model_path_if_needed = lambda m, ma: _streaming_bf16_loader(
                m, osp.join(ma.model_path, "model.safetensors"))
        elif precision == "awq_int4":
            IL.init_from_model_path_if_needed = lambda m, ma: _swap_to_awq_int4(m, Path(weight_dir))
        elif precision == "nvfp4":
            IL.init_from_model_path_if_needed = lambda m, ma: _swap_to_nvfp4(m, Path(weight_dir))

        # We don't want to actually run main() yet — just build the model.
        # The simplest path is to delegate building to a helper that mirrors
        # the first ~530 lines of inference_lance.main without the val loop.
        # For brevity here, we run main() but with a one-sample dummy dataset
        # — too invasive to extract cleanly. The clean refactor is a TODO.
        raise NotImplementedError(
            "ComfyUI inline model loader requires refactoring inference_lance.main "
            "to split build-model from run-loop. See scripts/run_baseline.py for "
            "the offline equivalent. v1 of this node pack ships placeholder nodes; "
            "the loader will land in v2."
        )


class LanceT2I:
    """Text → Image via Lance."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL",),
                "prompt": ("STRING", {"multiline": True}),
                "num_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 15.0, "step": 0.1}),
                "seed": ("INT", {"default": 42}),
                "height": ("INT", {"default": 768, "min": 256, "max": 1024, "step": 64}),
                "width":  ("INT", {"default": 768, "min": 256, "max": 1024, "step": 64}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Lance"

    def generate(self, lance_model, prompt, num_steps, cfg_scale, seed, height, width):
        raise NotImplementedError("Pending v2 — see Lance source.")


class LanceX2TImage:
    """Image → Text (VQA / captioning) via Lance."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL",),
                "image": ("IMAGE",),
                "question": ("STRING", {"multiline": True,
                                          "default": "Describe this image."}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 1024}),
            },
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "answer"
    CATEGORY = "Lance"

    def answer(self, lance_model, image, question, max_new_tokens):
        raise NotImplementedError("Pending v2 — see Lance source.")


class LanceImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lance_model": ("LANCE_MODEL",),
            "image": ("IMAGE",),
            "instruction": ("STRING", {"multiline": True}),
            "num_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 15.0, "step": 0.1}),
            "seed": ("INT", {"default": 42}),
        }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit"
    CATEGORY = "Lance"

    def edit(self, lance_model, image, instruction, num_steps, cfg_scale, seed):
        raise NotImplementedError("Pending v2.")


class LanceT2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lance_model": ("LANCE_MODEL",),
            "prompt": ("STRING", {"multiline": True}),
            "num_frames": ("INT", {"default": 50, "min": 5, "max": 121}),
            "height": ("INT", {"default": 480, "min": 256, "max": 768, "step": 32}),
            "width":  ("INT", {"default": 832, "min": 256, "max": 1280, "step": 32}),
            "num_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 15.0, "step": 0.1}),
            "seed": ("INT", {"default": 42}),
        }}
    RETURN_TYPES = ("IMAGE",)            # ComfyUI returns video as a batch of frames
    FUNCTION = "generate"
    CATEGORY = "Lance"

    def generate(self, lance_model, prompt, num_frames, height, width, num_steps, cfg_scale, seed):
        raise NotImplementedError("Pending v2.")


class LanceVideoEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lance_model": ("LANCE_MODEL",),
            "video_frames": ("IMAGE",),
            "instruction": ("STRING", {"multiline": True}),
            "num_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 15.0, "step": 0.1}),
            "seed": ("INT", {"default": 42}),
        }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit"
    CATEGORY = "Lance"

    def edit(self, lance_model, video_frames, instruction, num_steps, cfg_scale, seed):
        raise NotImplementedError("Pending v2.")


class LanceX2TVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lance_model": ("LANCE_MODEL",),
            "video_frames": ("IMAGE",),
            "question": ("STRING", {"multiline": True,
                                      "default": "What is happening in this video?"}),
            "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 1024}),
        }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "answer"
    CATEGORY = "Lance"

    def answer(self, lance_model, video_frames, question, max_new_tokens):
        raise NotImplementedError("Pending v2.")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LanceModelLoader": LanceModelLoader,
    "LanceT2I":         LanceT2I,
    "LanceT2V":         LanceT2V,
    "LanceImageEdit":   LanceImageEdit,
    "LanceVideoEdit":   LanceVideoEdit,
    "LanceX2TImage":    LanceX2TImage,
    "LanceX2TVideo":    LanceX2TVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LanceModelLoader": "Lance Model Loader",
    "LanceT2I":         "Lance: Text → Image",
    "LanceT2V":         "Lance: Text → Video",
    "LanceImageEdit":   "Lance: Image Edit",
    "LanceVideoEdit":   "Lance: Video Edit",
    "LanceX2TImage":    "Lance: Image Understanding",
    "LanceX2TVideo":    "Lance: Video Understanding",
}
