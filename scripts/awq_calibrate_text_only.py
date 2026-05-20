"""Text-only AWQ calibration for Lance's understanding path.

Why: the standard multimodal calibrate (awq_calibrate_single.py) OOMs on
16 GB GPUs for image_edit / x2t_video / video_edit because of the VAE +
ViT memory footprint. But we want MORE activation diversity for long-form
generation (cases like "describe the chart") which the original 6 x2t_image
samples and 11 t2i prompts under-cover.

This script:
  * Builds Lance with our memory-frugal meta-init loader
  * BUT keeps VAE on CPU and skips ViT loading entirely (visual_und=false,
    visual_gen=false) — only the LLM (language_model.*) is on GPU
  * Tokenises a small text corpus (wiki samples) and runs language_model
    forward in a loop with activation hooks
  * Saves per-channel mean(|x|) for every und-path Linear

The gen-path (`_moe_gen`) Linears get NO activations from this run, so this
is purely complementary to the existing t2i calibration. Merge with awq_merge_stats.py.

Usage:
    python awq_calibrate_text_only.py \\
        --model_path downloads/Lance_3B_Video \\
        --out ../calib/v2_text_und_stats.pt \\
        --n_samples 64 --max_tokens 512
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open


QUANT_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "self_attn.q_proj_moe_gen", "self_attn.k_proj_moe_gen",
    "self_attn.v_proj_moe_gen", "self_attn.o_proj_moe_gen",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj",
)


class ActStats:
    __slots__ = ("sum_abs", "n_tokens", "n_calls")
    def __init__(self):
        self.sum_abs = None
        self.n_tokens = 0
        self.n_calls = 0
    def update(self, x):
        x = x.detach().to(torch.float32).reshape(-1, x.shape[-1])
        if x.numel() == 0: return
        s = x.abs().sum(dim=0).cpu()
        self.sum_abs = s if self.sum_abs is None else self.sum_abs + s
        self.n_tokens += x.shape[0]
        self.n_calls += 1


@contextmanager
def _meta_init():
    orig = torch.empty
    def _e(*s, **kw):
        kw.setdefault("device", "meta")
        return orig(*s, **kw)
    torch.empty = _e
    try: yield
    finally: torch.empty = orig


def install_hooks(language_model):
    targets = []
    for n, m in language_model.named_modules():
        if (isinstance(m, torch.nn.Linear)
                and any(n.endswith(s) for s in QUANT_SUFFIXES)):
            targets.append(n)
    print(f"[hooks] {len(targets)} target Linears")
    stats = {n: ActStats() for n in targets}
    handles = []
    mods = dict(language_model.named_modules())
    for n in targets:
        def make(name):
            def h(mod, inputs, out):
                if isinstance(inputs, tuple) and len(inputs) > 0:
                    stats[name].update(inputs[0])
            return h
        handles.append(mods[n].register_forward_hook(make(n)))
    return stats, handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance_src", default=os.path.expanduser("~/lance-quant/src"))
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=512)
    args = ap.parse_args()

    sys.path.insert(0, args.lance_src)
    from modeling.lance.qwen2_navit import Qwen2ForCausalLM
    from modeling.qwen2 import Qwen2Tokenizer
    from modeling.qwen2.modeling_qwen2 import Qwen2Config
    import json as _json

    # Load LLM config
    cfg_dict = _json.loads((Path(args.model_path) / "llm_config.json").read_text())
    cfg = Qwen2Config(**{k: v for k, v in cfg_dict.items() if not isinstance(v, dict)})
    cfg.layer_module = "Qwen2MoTDecoderLayer"
    cfg.qk_norm = True
    cfg.qk_norm_und = True
    cfg.qk_norm_gen = True
    cfg.freeze_und = False
    cfg.tie_word_embeddings = False
    cfg.apply_qwen_2_5_vl_pos_emb = True

    print("[build] meta-init Qwen2ForCausalLM (LLM only, no ViT/VAE)")
    with _meta_init():
        llm = Qwen2ForCausalLM(cfg)

    # Stream load LLM weights to GPU bf16
    print("[load] streaming weights")
    t0 = time.time()
    own = dict(llm.state_dict(keep_vars=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_loaded = 0
    with safe_open(str(Path(args.model_path) / "model.safetensors"),
                    framework="pt", device="cpu") as f:
        for k in f.keys():
            if not k.startswith("language_model."):
                continue
            local = k[len("language_model."):]
            if local not in own:
                continue
            t = f.get_tensor(k)
            if t.is_floating_point() and t.dtype != torch.bfloat16:
                t = t.to(torch.bfloat16)
            p = own[local]
            with torch.no_grad():
                if p.device.type == "meta":
                    p.data = t.to(device)
                else:
                    p.data.copy_(t.to(device))
            n_loaded += 1
    print(f"[load] {n_loaded} LLM tensors in {time.time()-t0:.1f}s")
    # Force everything to device + bf16 (catches Embedding params that
    # weren't on meta device after meta_init)
    llm = llm.to(device=device, dtype=torch.bfloat16)
    llm.eval()
    if torch.cuda.is_available():
        print(f"[load] GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Tokenizer
    tok = Qwen2Tokenizer.from_pretrained(args.model_path)

    # Calibration corpus: use wiki + ChartQA-like instructions for long-form coverage
    corpus = [
        "The Colosseum is an oval amphitheatre in the centre of the city of Rome, Italy. " * 8,
        "According to data from the National Cancer Institute, the five-year survival rate increased from 49% in 1975 to 68% in 2017, indicating substantial progress in early detection and treatment. " * 4,
        "Analyze the pie chart carefully. The blue segment representing technology firms accounts for 42%, the red segment for healthcare is 28%, the green for finance is 17%, and the remaining 13% is split between energy and retail. " * 4,
        "The Apollo 11 mission launched on July 16, 1969, with Neil Armstrong, Buzz Aldrin, and Michael Collins aboard. " * 6,
        "In the financial year ending December 2003, the company reported total revenue of approximately $2.1 billion, with a net profit margin of 12.3%. Operating expenses grew by 8% year-over-year. " * 4,
        "Quantum mechanics describes the behavior of matter and light at the atomic and subatomic scales. " * 8,
        "The Persian Gulf, also known as the Arabian Gulf, is a Mediterranean sea in West Asia. " * 8,
        "Solar eclipses occur when the Moon passes between the Sun and Earth, completely or partially blocking the Sun. " * 6,
    ]
    # Cycle to reach n_samples
    while len(corpus) < args.n_samples:
        corpus = corpus + corpus
    corpus = corpus[: args.n_samples]

    print(f"[hooks] installing on {sum(1 for n,_ in llm.named_modules() if isinstance(_, torch.nn.Linear))} Linears (only und-path will receive activations)")
    stats, handles = install_hooks(llm)

    print(f"[run] {len(corpus)} sequences, max_tokens={args.max_tokens}")
    t0 = time.time()
    # We can't call llm.forward(input_ids) directly — it's not a standard
    # transformers forward. But the underlying nn.Module hierarchy works:
    # tokenize, then run model.model (the Qwen2Model) with positional ids.
    inner = llm.model
    pad_id = tok.pad_token_id or tok.eos_token_id or 0
    for i, text in enumerate(corpus):
        ids = tok(text, return_tensors="pt", max_length=args.max_tokens,
                   truncation=True).input_ids.to(device)
        position_ids = torch.arange(ids.shape[1], device=device).unsqueeze(0)
        # The Lance Qwen2Model.forward expects a packed_query_sequence (embedded
        # tensor), not raw IDs. We embed manually then call forward_inference.
        with torch.no_grad():
            emb = inner.embed_tokens(ids)              # [1, L, H]
            # Lance forward expects packed_query_position_ids per qwen2.5_vl
            # mrope: [3, B, L]. Build a simple non-multimodal position id.
            if inner.apply_qwen_2_5_vl_pos_emb:
                pos = position_ids.unsqueeze(0).expand(3, -1, -1)
            else:
                pos = position_ids
            try:
                _ = inner.forward_inference(
                    packed_query_sequence=emb.squeeze(0),
                    query_lens=torch.tensor([emb.shape[1]], device=device),
                    packed_query_position_ids=pos.squeeze(1) if pos.dim() > 2 else pos,
                    packed_query_indexes=torch.arange(emb.shape[1], device=device),
                    mode="und",
                )
            except Exception as e:
                print(f"  sample {i}: {type(e).__name__}: {str(e)[:120]}")
                continue
        if (i + 1) % 8 == 0:
            print(f"  [{i+1}/{len(corpus)}] {time.time()-t0:.1f}s, "
                  f"GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")
    print(f"[run] done in {time.time()-t0:.1f}s")
    for h in handles: h.remove()

    # Save with the language_model. prefix (so it merges with our other stats)
    out = {"format": 1, "task": "text_und",
            "n_linears": len(stats),
            "stats": {"language_model." + n: {"sum_abs": s.sum_abs,
                                                "n_tokens": s.n_tokens,
                                                "n_calls": s.n_calls}
                      for n, s in stats.items() if s.sum_abs is not None}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    n_with = sum(1 for v in stats.values() if v.sum_abs is not None)
    total_tokens = sum(s.n_tokens for s in stats.values())
    print(f"[done] {n_with}/{len(stats)} linears with data, {total_tokens} total tokens")
    print(f"[done] saved {args.out}")


if __name__ == "__main__":
    main()
