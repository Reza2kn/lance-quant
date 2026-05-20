"""Perplexity benchmark for Lance variants.

For text completion only — runs the language_model on chunks of held-out
text, sums log-likelihood per token, returns perplexity = exp(-mean_logp).

Uses an LLM-only memory-frugal loader (meta-init + streaming bf16). For a
4-bit AWQ run, the WQLinearINT4 swap happens via run_quant_eval.py. The
wrapper is intentionally named `language_model` so AWQ keys match the full
Lance checkpoint without constructing ViT/VAE modules.

Comparison: run on the bf16 baseline, then the AWQ variant; report PPL diff.

Usage:
    python eval_perplexity.py \\
        --model_path downloads/Lance_3B_Video \\
        --out_perplexity ../docs/perplexity.json

    python eval_perplexity.py \\
        --model_path downloads/Lance_3B_Video \\
        --awq_dir ../models/Lance_3B_Video-AWQ-INT4-g64 \\
        --out_perplexity ../docs/perplexity_awq.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open


@contextmanager
def _meta_init():
    orig = torch.empty
    def _e(*s, **kw):
        kw.setdefault("device", "meta")
        return orig(*s, **kw)
    torch.empty = _e
    try: yield
    finally: torch.empty = orig


CORPUS = [
    # Wikipedia-style factual paragraphs (covers diverse domains)
    "The Roman Empire was the post-Republican period of ancient Rome. As a polity, it included large territorial holdings around the Mediterranean Sea in Europe, North Africa, and Western Asia, ruled by emperors. From the accession of Caesar Augustus as the first Roman emperor to the military anarchy of the 3rd century, it was a Principate with Italy as the metropole of its provinces and the city of Rome as its sole capital.",
    "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science.",
    "The Industrial Revolution was the transition to new manufacturing processes in Great Britain, continental Europe, and the United States, that occurred during the period from around 1760 to about 1820–1840. This transition included going from hand production methods to machines; new chemical manufacturing and iron production processes; the increasing use of steam power and water power; the development of machine tools; and the rise of the mechanized factory system.",
    "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities. Some of this chemical energy is stored in carbohydrate molecules, such as sugars and starches, which are synthesized from carbon dioxide and water.",
    "The Internet is the global system of interconnected computer networks that uses the Internet protocol suite to communicate between networks and devices. It is a network of networks that consists of private, public, academic, business, and government networks of local to global scope, linked by a broad array of electronic, wireless, and optical networking technologies.",
    "Climate change includes both global warming driven by human-induced emissions of greenhouse gases and the resulting large-scale shifts in weather patterns. Though there have been previous periods of climatic change, since the mid-20th century humans have had an unprecedented impact on Earth's climate system and caused change on a global scale.",
    "The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2. The novel virus was first identified in an outbreak in the Chinese city of Wuhan in December 2019.",
    "Artificial intelligence is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals and humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.",
]


class LLMWrapper(torch.nn.Module):
    def __init__(self, language_model: torch.nn.Module):
        super().__init__()
        self.language_model = language_model


def _build_llm_config(model_path: Path):
    from modeling.qwen2.modeling_qwen2 import Qwen2Config

    cfg_dict = json.loads((model_path / "llm_config.json").read_text())
    cfg = Qwen2Config(**{k: v for k, v in cfg_dict.items() if not isinstance(v, dict)})
    cfg.layer_module = "Qwen2MoTDecoderLayer"
    cfg.qk_norm = True
    cfg.qk_norm_und = True
    cfg.qk_norm_gen = True
    cfg.freeze_und = False
    cfg.tie_word_embeddings = False
    cfg.apply_qwen_2_5_vl_pos_emb = True
    cfg.rope_scaling = cfg_dict.get("rope_scaling")
    if cfg.rope_scaling is not None and "type" in cfg.rope_scaling:
        cfg.rope_scaling.setdefault("rope_type", cfg.rope_scaling["type"])
    return cfg


def _stream_bf16_llm(llm: torch.nn.Module, model_path: Path, device: torch.device):
    own = dict(llm.state_dict(keep_vars=True))
    loaded = 0
    t0 = time.time()
    with safe_open(str(model_path / "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            if not k.startswith("language_model."):
                continue
            local = k[len("language_model."):]
            if local not in own:
                continue
            src = f.get_tensor(k)
            if src.is_floating_point() and src.dtype != torch.bfloat16:
                src = src.to(torch.bfloat16)
            target = own[local]
            with torch.no_grad():
                if target.device.type == "meta":
                    target.data = src.to(device)
                else:
                    target.data.copy_(src.to(device), non_blocking=True)
            loaded += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[load] streamed {loaded} LLM tensors in {time.time()-t0:.1f}s")


def _materialize_meta_tensors(module: torch.nn.Module, device: torch.device):
    """Allocate still-meta tensors in-place without touching real tensors."""
    n_params = 0
    n_buffers = 0
    for child in module.modules():
        for name, param in list(child.named_parameters(recurse=False)):
            if param is None or param.device.type != "meta":
                continue
            dtype = torch.bfloat16 if param.dtype.is_floating_point else param.dtype
            child._parameters[name] = torch.nn.Parameter(
                torch.empty(param.shape, dtype=dtype, device=device),
                requires_grad=param.requires_grad,
            )
            n_params += 1
        for name, buf in list(child.named_buffers(recurse=False)):
            if buf is None or buf.device.type != "meta":
                continue
            dtype = torch.bfloat16 if buf.dtype.is_floating_point else buf.dtype
            child._buffers[name] = torch.empty(buf.shape, dtype=dtype, device=device)
            n_buffers += 1
    if n_params or n_buffers:
        print(f"[alloc] materialized meta tensors: params={n_params}, buffers={n_buffers}")


def build_lance(model_path: Path, vit_path: Path, awq_dir: Path | None):
    """Build only Lance's language_model. Returns an LLMWrapper + tokenizer."""
    src_root = Path(__file__).resolve().parent
    from modeling.lance.qwen2_navit import Qwen2ForCausalLM
    from modeling.qwen2 import Qwen2Tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_llm_config(model_path)
    print("[build] meta-init Qwen2ForCausalLM only (no ViT/VAE)")
    with _meta_init():
        llm = Qwen2ForCausalLM(cfg)
    llm.to(dtype=torch.bfloat16)
    llm.to_empty(device=device)
    wrapper = LLMWrapper(llm)

    if awq_dir:
        sys.path.insert(0, str(src_root))
        from run_quant_eval import (swap_to_awq, stream_pass_through_weights,
                                      stream_awq_buffers, WQLinearINT4)
        WQLinearINT4.MODE = "ondemand"
        mods = swap_to_awq(wrapper, Path(awq_dir))
        _materialize_meta_tensors(wrapper, device)
        stream_pass_through_weights(wrapper, Path(awq_dir))
        stream_awq_buffers(mods, Path(awq_dir))
    else:
        _materialize_meta_tensors(llm, device)
        _stream_bf16_llm(llm, model_path, device)

    wrapper.eval()
    if torch.cuda.is_available():
        print(f"[load] GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return wrapper, Qwen2Tokenizer.from_pretrained(model_path)


@torch.no_grad()
def compute_perplexity(model, tokenizer, corpus, max_seq: int = 256):
    """Compute corpus perplexity via teacher-forced loss on language_model.

    For our memory-frugal flow, model is the Lance wrapper; we want the
    inner Qwen2ForCausalLM. The lm_head is the output projection; we feed
    embeddings through the model directly and compute cross-entropy.
    """
    inner = model.language_model
    device = next(p.device for p in inner.parameters() if p.device.type != "meta")
    total_logp = 0.0
    total_tokens = 0

    for i, text in enumerate(corpus):
        ids = tokenizer(text, return_tensors="pt", max_length=max_seq,
                         truncation=True).input_ids.to(device)
        # Use the Lance forward path; for un-mode (text-only) we just run
        # the underlying Qwen2Model directly.
        L = ids.shape[1]
        if L < 2:
            continue
        emb = inner.model.embed_tokens(ids)                   # [1, L, H]
        # mrope expects [3, B, L]
        pos = torch.arange(L, device=device).unsqueeze(0).unsqueeze(0).expand(3, 1, L)
        out = inner.model.forward_inference(
            packed_query_sequence=emb.squeeze(0),
            query_lens=torch.tensor([L], device=device),
            packed_query_position_ids=pos,                    # [3, 1, L]
            packed_query_indexes=torch.arange(L, device=device),
            mode="und",
            update_past_key_values=False,
        )
        hidden = out.packed_query_sequence                    # [L, H]
        logits = inner.lm_head(hidden)                        # [L, V]
        # next-token cross-entropy
        shift_logits = logits[:-1]
        shift_labels = ids[0, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.float(), shift_labels, reduction="sum")
        total_logp += -loss.item()
        total_tokens += shift_labels.shape[0]
        print(f"  [{i+1}/{len(corpus)}] tokens={shift_labels.shape[0]:4d}  "
              f"running_ppl={math.exp(-total_logp/total_tokens):.3f}")

    ppl = math.exp(-total_logp / total_tokens) if total_tokens else float("inf")
    return {"perplexity": ppl, "total_tokens": total_tokens,
            "avg_logp_per_token": total_logp / total_tokens if total_tokens else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--vit_path", default="downloads/Qwen2.5-VL-ViT")
    ap.add_argument("--awq_dir", default=None)
    ap.add_argument("--out_perplexity", type=Path, required=True)
    ap.add_argument("--max_seq", type=int, default=256)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("POSITION_EMBEDDING_3D_VERSION", "v2")
    os.environ.setdefault("EXP_HW_20250819", "False")

    print(f"[build] model_path={args.model_path}  awq={args.awq_dir}")
    t0 = time.time()
    model, tok = build_lance(Path(args.model_path), Path(args.vit_path),
                              Path(args.awq_dir) if args.awq_dir else None)
    print(f"[build] done in {time.time()-t0:.1f}s")

    print(f"[ppl] {len(CORPUS)} samples, max_seq={args.max_seq}")
    t0 = time.time()
    result = compute_perplexity(model, tok, CORPUS, args.max_seq)
    result["elapsed_s"] = time.time() - t0
    result["awq_dir"] = args.awq_dir
    result["model_path"] = args.model_path

    args.out_perplexity.parent.mkdir(parents=True, exist_ok=True)
    args.out_perplexity.write_text(json.dumps(result, indent=2))
    print(f"\n[done] perplexity = {result['perplexity']:.3f} on "
          f"{result['total_tokens']} tokens ({result['elapsed_s']:.1f}s)")
    print(f"[done] wrote {args.out_perplexity}")


if __name__ == "__main__":
    main()
