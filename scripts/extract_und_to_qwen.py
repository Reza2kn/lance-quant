"""Extract Lance's und-path (understanding) weights into a standard Qwen2.5-VL
HF-compatible model directory, so we can run it through mlx_lm.convert /
coremltools / etc. that expect a vanilla transformers checkpoint.

What gets renamed:
  language_model.model.embed_tokens.weight     -> model.embed_tokens.weight
  language_model.model.layers.<i>.<x>          -> model.layers.<i>.<x>          (drop _moe_gen variants)
  language_model.model.norm.weight             -> model.norm.weight
  language_model.lm_head.weight                -> lm_head.weight

What gets dropped:
  language_model.*._moe_gen variants  (used by Lance's gen-expert path only)
  vit_model.*                         (separate Qwen2.5-VL-ViT model)
  latent_pos_embed, time_embedder, llm2vae, vae2llm (Lance-specific)

This produces an HF-loadable Qwen2_5_VLForConditionalGeneration *language_model*
half (sans vision tower). It's ~3.5B params in bf16 (~7 GB). Good enough as
input to mlx_lm.convert / mlx_lm.dwq and to coremltools.

A separate script (extract_gen_to_qwen.py) does the same for the `_moe_gen`
variant if you want to quantize the generation expert path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="Lance model.safetensors (e.g. downloads/Lance_3B_Video/model.safetensors)")
    ap.add_argument("--llm_config", type=Path, required=True,
                    help="Lance llm_config.json (sibling of src)")
    ap.add_argument("--tokenizer_src", type=Path, required=True,
                    help="dir with tokenizer.json + vocab.json + merges.txt")
    ap.add_argument("--out", type=Path, required=True,
                    help="output dir for the rebuilt Qwen2.5-VL model")
    ap.add_argument("--variant", choices=["und", "gen"], default="und",
                    help="which expert path to extract")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(args.llm_config.read_text())
    # Build a standard Qwen2.5-VL config without the Lance-only fields
    hf_cfg = {
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "attention_dropout": cfg.get("attention_dropout", 0.0),
        "bos_token_id": cfg.get("bos_token_id", 151643),
        "eos_token_id": cfg.get("eos_token_id", 151645),
        "vision_start_token_id": cfg.get("vision_start_token_id", 151652),
        "vision_end_token_id": cfg.get("vision_end_token_id", 151653),
        "vision_token_id": cfg.get("vision_token_id", 151654),
        "image_token_id": cfg.get("image_token_id", 151655),
        "video_token_id": cfg.get("video_token_id", 151656),
        "hidden_act": cfg.get("hidden_act", "silu"),
        "hidden_size": cfg["hidden_size"],
        "initializer_range": cfg.get("initializer_range", 0.02),
        "intermediate_size": cfg["intermediate_size"],
        "max_position_embeddings": cfg.get("max_position_embeddings", 128000),
        "max_window_layers": cfg.get("max_window_layers", 70),
        "model_type": "qwen2_5_vl",
        "num_attention_heads": cfg["num_attention_heads"],
        "num_hidden_layers": cfg["num_hidden_layers"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "rms_norm_eps": cfg.get("rms_norm_eps", 1e-6),
        "rope_theta": cfg.get("rope_theta", 1000000.0),
        "sliding_window": cfg.get("sliding_window", 32768),
        "tie_word_embeddings": cfg.get("tie_word_embeddings", False),
        "torch_dtype": "bfloat16",
        "transformers_version": "4.49.0",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": cfg["vocab_size"],
        "rope_scaling": cfg.get("rope_scaling", {"type": "mrope", "mrope_section": [16, 24, 24]}),
        "vision_config": cfg.get("vision_config"),
    }
    (args.out / "config.json").write_text(json.dumps(hf_cfg, indent=2))

    # Generation config
    gen_cfg = json.loads(args.llm_config.with_name("generation_config.json").read_text()) \
        if args.llm_config.with_name("generation_config.json").exists() else {}
    (args.out / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2))

    # Copy tokenizer files
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        src = args.tokenizer_src / fn
        if src.exists():
            (args.out / fn).write_bytes(src.read_bytes())

    # Stream + rename weights
    t0 = time.time()
    new_sd: dict[str, torch.Tensor] = {}
    n_in = 0
    n_kept = 0
    bytes_kept = 0

    KEEP_SUFFIXES_BASE = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.q_proj.bias",
        "self_attn.k_proj.weight", "self_attn.k_proj.bias",
        "self_attn.v_proj.weight", "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    )

    # If variant == 'gen', we look for the _moe_gen suffix on every above
    def rename_key(k: str) -> str | None:
        if not k.startswith("language_model."):
            return None
        rest = k[len("language_model."):]
        # Standard top-level
        if rest in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            if args.variant == "und":
                return rest
            else:  # gen variant — keep these too (shared) so the model is loadable
                if rest == "model.norm.weight":
                    # gen path uses model.norm_moe_gen; rest fall back to und norm
                    return None
                return rest
        if rest == "model.norm_moe_gen.weight" and args.variant == "gen":
            return "model.norm.weight"
        # Layer-level
        if rest.startswith("model.layers."):
            parts = rest.split(".", 3)        # model, layers, idx, suffix
            if len(parts) != 4:
                return None
            idx, suffix = parts[2], parts[3]
            if args.variant == "und":
                if suffix.endswith("_moe_gen.weight") or suffix.endswith("_moe_gen.bias"):
                    return None
                if suffix in [s for s in KEEP_SUFFIXES_BASE]:
                    return f"model.layers.{idx}.{suffix}"
                return None
            else:  # gen variant
                # we want suffixes that END with _moe_gen → strip suffix and use as base
                for base in KEEP_SUFFIXES_BASE:
                    base_no_ext = base[:-len(".weight")] if base.endswith(".weight") else base[:-len(".bias")]
                    ext = ".weight" if base.endswith(".weight") else ".bias"
                    gen_suffix = base_no_ext + "_moe_gen" + ext
                    if suffix == gen_suffix:
                        return f"model.layers.{idx}.{base}"
                return None
        return None

    with safe_open(str(args.src), framework="pt", device="cpu") as f:
        for k in f.keys():
            n_in += 1
            new_name = rename_key(k)
            if new_name is None:
                continue
            t = f.get_tensor(k)
            if t.is_floating_point() and t.dtype != torch.bfloat16:
                t = t.to(torch.bfloat16)
            new_sd[new_name] = t
            bytes_kept += t.numel() * t.element_size()
            n_kept += 1

    print(f"input keys: {n_in}, kept: {n_kept}, size: {bytes_kept/1e9:.2f} GB")
    save_file(new_sd, str(args.out / "model.safetensors"))
    print(f"wrote {args.out / 'model.safetensors'} in {time.time()-t0:.1f}s")
    print(f"\nyou can now run on this dir:")
    print(f"  mlx_lm.convert --hf-path {args.out} --mlx-path {args.out}-mlx-4bit -q --q-bits 4 --q-group-size 64")


if __name__ == "__main__":
    main()
