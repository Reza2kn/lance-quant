"""Perplexity benchmark for Lance variants.

For text completion only — runs the language_model on chunks of held-out
text, sums log-likelihood per token, returns perplexity = exp(-mean_logp).

Uses the same memory-frugal loader as our other scripts (meta-init +
streaming bf16). For a 4-bit AWQ run, the WQLinearINT4 swap happens via
the patches/quantized_linear.py path.

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


def build_lance(model_path: Path, vit_path: Path, awq_dir: Path | None):
    """Construct Lance via inference_lance.main and stop right before the
    validation loop. Returns the model handle."""
    src_root = Path(__file__).resolve().parent
    # Run inference_lance with our usual args, but install a side-channel that
    # captures the model and exits the validation loop after a single call.
    import inference_lance as IL
    from modeling.lance import Lance
    from modeling.lance.qwen2_navit import Qwen2ForCausalLM
    from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel

    _OQ, _OV, _OL = Qwen2ForCausalLM.__init__, \
        Qwen2_5_VisionTransformerPretrainedModel.__init__, Lance.__init__
    def _Q(self, c):
        with _meta_init(): _OQ(self, c)
    def _V(self, c):
        with _meta_init(): _OV(self, c)
    def _L(self, *a, **k):
        with _meta_init(): _OL(self, *a, **k)
    Qwen2ForCausalLM.__init__ = _Q
    Qwen2_5_VisionTransformerPretrainedModel.__init__ = _V
    Lance.__init__ = _L

    if awq_dir:
        sys.path.insert(0, str(src_root))
        from run_quant_eval import (swap_to_awq, stream_pass_through_weights,
                                      stream_awq_buffers, WQLinearINT4)
        WQLinearINT4.MODE = "ondemand"
        def _loader(model, model_args):
            mods = swap_to_awq(model, Path(awq_dir))
            stream_pass_through_weights(model, Path(awq_dir))
            stream_awq_buffers(mods, Path(awq_dir))
            class _M:
                missing_keys, unexpected_keys = [], []
            return _M()
        IL.init_from_model_path_if_needed = _loader
    else:
        from run_baseline import _streaming_bf16_loader
        IL.init_from_model_path_if_needed = _streaming_bf16_loader

    sys.argv = [
        "inference_lance.py",
        "--model_path", str(model_path), "--vit_path", str(vit_path),
        "--vit_type", "qwen_2_5_vl_original",
        "--llm_qk_norm", "true", "--llm_qk_norm_und", "true",
        "--llm_qk_norm_gen", "true", "--tie_word_embeddings", "false",
        "--validation_num_timesteps", "1", "--validation_timestep_shift", "3.5",
        "--copy_init_moe", "true", "--max_num_frames", "121",
        "--max_latent_size", "64", "--latent_patch_size", "1", "1", "1",
        "--visual_und", "true", "--visual_gen", "true",
        "--vae_model_type", "wan", "--apply_qwen_2_5_vl_pos_emb", "true",
        "--apply_chat_template", "false", "--cfg_type", "0",
        "--validation_data_seed", "42",
        "--video_height", "768", "--video_width", "768", "--num_frames", "1",
        "--task", "x2t_image", "--save_path_gen", "results/_ppl_bootstrap",
        "--resolution", "image_768res", "--text_template", "true",
        "--cfg_text_scale", "4.0", "--use_KVcache", "true",
        "--val_dataset_config_file", "config/examples/x2t_image_example.json",
    ]

    state = {}
    orig_validate = IL.validate_on_fixed_batch
    def _capture(*a, **kw):
        state["model"] = kw.get("fsdp_model") or a[0]
        state["tokenizer"] = kw.get("tokenizer")
        raise StopIteration                                # abort the run after model build
    IL.validate_on_fixed_batch = _capture
    try:
        IL.main()
    except StopIteration:
        pass
    IL.validate_on_fixed_batch = orig_validate
    return state["model"], state["tokenizer"]


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
            packed_query_position_ids=pos.squeeze(1),         # [3, L]
            packed_query_indexes=torch.arange(L, device=device),
            mode="und",
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
