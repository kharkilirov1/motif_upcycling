"""stage2_real_motif.py - Stage 2 REAL collector (pretrained Qwen + trained overlay).

Unlike the smoke test, this uses a genuinely pretrained causal LM (Qwen2.5-0.5B),
patches MLP layers with the repo's real `MotifSwiGLUMLP`, and BRIEFLY TRAINS ONLY
the overlay (contextual router + motif-LoRA + adapter-only SARC) with an LM loss on
real text. Training is required because a function-preserving overlay starts neutral
(alpha == 1, zero LoRA/SARC), which would make the router/SARC signals degenerate.

After training, per-(token, patched-layer) signals + decisions are extracted on
held-out text and saved as an npz for load_and_test.py.

Signals (signal_spec.md, transformer column):
  confidence    = 1 - H(next-token dist)/log V         (LM head; per token)
  cand_entropy  = H(router alpha over M motif slices)   (router; per token,layer)
  margin        = alpha_top1 - alpha_top2               (router; per token,layer)
  update_scale  = log( RMS(lora_delta) / RMS(x) )       (SARC relative scale r)
  (kappa: no transformer analog -> omitted, run with 4)

Decisions:
  motif  = argmax_m alpha           (categorical; per token,layer)
  budget = # slices with alpha > 1  (tier; neutral alpha == 1)
  adapt  = P0 probe: does the motif-LoRA delta lower this token's NLL?  (binary)
  temperature = predictive entropy  (continuous; per token)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from motif_upcycling import MotifUpcycleConfig, SARCConfig  # noqa: E402
from motif_upcycling.swiglu import MotifSwiGLUMLP  # noqa: E402

torch.set_num_threads(max(1, torch.get_num_threads()))

FALLBACK_TEXT = """
Machine learning models transform inputs through many layers of computation.
A transformer block mixes information across positions with attention and then
applies a position-wise feed-forward network. The feed-forward network expands the
hidden dimension, applies a nonlinearity, and projects back down. Researchers study
how individual components specialize during training. Some channels appear to act as
detectors for particular patterns, while others contribute to broad smoothing.
Understanding this structure may help build smaller and more efficient models.
The city woke slowly under a grey sky, and the river carried thin sheets of mist.
She opened the old book, and the smell of paper filled the quiet room at once.
Numbers, when arranged in the right order, can reveal surprising regularities.
He counted the coins twice, then a third time, just to be sure of the total.
Language is a sequence of choices, each one shaped by everything that came before.
Prime numbers grow sparse, yet they never run out, scattered along the line.
""".strip()


def get_texts():
    for ident in ("Salesforce/wikitext", "wikitext"):
        try:
            from datasets import load_dataset
            ds = load_dataset(ident, "wikitext-2-raw-v1", split="train")
            texts = [t for t in ds["text"][:8000] if len(t.strip()) > 80]
            if len(texts) > 50:
                return texts[:800], f"{ident}:wikitext-2-raw-v1"
        except Exception as e:
            print(f"[stage2] {ident} unavailable ({type(e).__name__}); trying next")
    paras = [p.strip() for p in FALLBACK_TEXT.split("\n") if p.strip()]
    return paras * 12, "embedded_fallback"


def patch_layers(model, layer_indices, num_motifs):
    cfg = MotifUpcycleConfig(
        num_motifs=num_motifs, router_hidden=64, router_type="contextual",
        lora_rank=8, lora_alpha=8.0, freeze_base=True,
        sarc=SARCConfig(enabled=True, mode="adapter_only", max_delta=0.10, log_stats=True),
    )
    wrappers = []
    for i in layer_indices:
        dense = model.model.layers[i].mlp
        w = MotifSwiGLUMLP(dense, cfg)
        model.model.layers[i].mlp = w
        wrappers.append(w)
    return wrappers


def overlay_params(wrappers):
    ps = []
    for w in wrappers:
        for n, p in w.named_parameters():
            if p.requires_grad:
                ps.append(p)
    return ps


def encode_batches(tok, texts, seq_len, max_batches, seed=0):
    rng = np.random.default_rng(seed)
    ids_all = []
    for t in texts:
        e = tok(t, return_tensors="pt").input_ids[0]
        if e.numel() >= 8:
            ids_all.append(e)
    if not ids_all:
        raise SystemExit("no usable text")
    cat = torch.cat(ids_all)
    n = cat.numel() // seq_len
    cat = cat[: n * seq_len].view(n, seq_len)
    idx = rng.permutation(n)[:max_batches]
    return cat[idx]


def train_overlay(model, wrappers, batches, steps, lr=5e-3):
    params = overlay_params(wrappers)
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()
    t0 = time.time()
    losses = []
    bi = 0
    for step in range(steps):
        ids = batches[bi % len(batches)].unsqueeze(0)
        bi += 1
        out = model(ids, labels=ids)
        loss = out.loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % 10 == 0 or step == steps - 1:
            print(f"[stage2] train step {step:3d}/{steps} loss={loss.item():.4f} "
                  f"({(time.time()-t0)/(step+1):.1f}s/step)")
    model.eval()
    return n_train, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="11,12")
    ap.add_argument("--num-motifs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--train-batches", type=int, default=160)
    ap.add_argument("--train-steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    layer_indices = [int(x) for x in args.layers.split(",")]

    print(f"[stage2] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    wrappers = patch_layers(model, layer_indices, args.num_motifs)
    M = wrappers[0].num_motifs

    texts, src = get_texts()
    print(f"[stage2] text source: {src} ({len(texts)} chunks)")
    all_batches = encode_batches(tok, texts, args.seq_len,
                                 args.train_batches + args.eval_batches, seed=args.seed)
    train_b = all_batches[: args.train_batches]
    eval_b = all_batches[args.train_batches: args.train_batches + args.eval_batches]
    if len(eval_b) == 0:
        eval_b = all_batches[-min(args.eval_batches, len(all_batches)):]

    n_train, _ = train_overlay(model, wrappers, train_b, args.train_steps, lr=args.lr)
    print(f"[stage2] trained {n_train} overlay params on {len(train_b)} batches")

    # capture per-layer x, alpha, lora_delta via forward hooks during eval
    captured = {}

    def make_hook(li, w):
        def hook(module, inp, out):
            x = inp[0].detach()
            with torch.no_grad():
                alpha = w._router_alpha(x)            # [1,T,M]
                delta = w.lora_delta(x)               # [1,T,d]
            captured[li] = (x[0], alpha[0], delta[0])
        return hook

    handles = [model.model.layers[i].mlp.register_forward_hook(make_hook(i, w))
               for i, w in zip(layer_indices, wrappers)]

    sig_names = ["confidence", "cand_entropy", "margin", "update_scale"]
    rows, motif, budget, adapt, temp = [], [], [], [], []
    logV = float(np.log(model.config.vocab_size))

    def set_lora(flag):
        for w in wrappers:
            w.use_lora = flag and (w.config.lora_rank > 0)

    with torch.no_grad():
        for b in eval_b:
            ids = b.unsqueeze(0)
            captured.clear()
            set_lora(True)
            out_with = model(ids)
            logits = out_with.logits[0]                       # [T, V]
            # per-token predictive entropy on positions 0..T-2 (predicting next token)
            p = F.softmax(logits.float(), dim=-1)
            ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)  # [T]
            tgt = ids[0]
            nll_with = F.cross_entropy(logits[:-1], tgt[1:], reduction="none")  # [T-1]
            cap_with = {li: captured[li] for li in layer_indices}

            set_lora(False)
            out_wo = model(ids)
            nll_wo = F.cross_entropy(out_wo.logits[0][:-1], tgt[1:], reduction="none")
            set_lora(True)

            T = ids.shape[1]
            for li in layer_indices:
                x_l, alpha_l, delta_l = cap_with[li]
                a = (alpha_l / alpha_l.sum(-1, keepdim=True)).numpy()  # [T,M] prob simplex
                srt = np.sort(a, axis=-1)[:, ::-1]
                rms_d = torch.sqrt((delta_l.float() ** 2).mean(-1) + 1e-8)
                rms_x = torch.sqrt((x_l.float() ** 2).mean(-1) + 1e-8)
                us = torch.log(rms_d / rms_x).numpy()             # [T]
                ent_np = ent.numpy()
                for t in range(T - 1):                            # last token has no next-token target
                    a_t = a[t]
                    rows.append([
                        float(1.0 - ent_np[t] / logV),            # confidence
                        float(-(np.clip(a_t, 1e-12, 1) * np.log(np.clip(a_t, 1e-12, 1))).sum()),  # cand_entropy
                        float(srt[t, 0] - srt[t, 1]),             # margin
                        float(us[t]),                             # update_scale (SARC r)
                    ])
                    motif.append(int(a_t.argmax()))
                    budget.append(int((alpha_l[t].numpy() > 1.0).sum()))   # neutral alpha==1
                    adapt.append(1 if float(nll_with[t]) < float(nll_wo[t]) else 0)
                    temp.append(float(ent_np[t]))

    for h in handles:
        h.remove()

    X = np.asarray(rows, float)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=X, signal_names=np.array(sig_names),
             dec__motif=np.array(motif), dec__budget=np.array(budget),
             dec__adapt=np.array(adapt), dec__temperature=np.array(temp, float))
    print(f"[stage2] wrote {args.out}: {X.shape[0]} (token,layer) sites x {X.shape[1]} signals")
    print(f"[stage2] alpha std per signal: " +
          " ".join(f"{n}={X[:,i].std():.3f}" for i, n in enumerate(sig_names)))
    print(f"[stage2] base-rates: adapt={np.mean(adapt):.3f} "
          f"budget classes={np.bincount(budget, minlength=M+1).tolist()} "
          f"motif classes={len(set(motif))}")


if __name__ == "__main__":
    main()
