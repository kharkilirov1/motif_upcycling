"""from_scratch_split.py - FOG Theorem 4 / Corollary 3 tested by TRAINING FROM SCRATCH.

The Qwen experiments only re-cut an already-uniformly-trained model (compress / adapt),
which cannot reveal whether a natively heterogeneous architecture is better -- the base's
representations are already shaped by uniform geometry. The honest test is to TRAIN tiny
GPTs from scratch at EQUAL compute and sweep the attention<->FFN budget split.

Setup
-----
Per layer the inner budget is fixed: T = 2*d_attn + d_ff  (so per-layer non-embedding
params = 2*d_model*T are CONSTANT for every split). We sweep d_attn in multiples of
head_dim; d_ff = T - 2*d_attn. The STANDARD transformer is exactly one point of the
sweep: d_attn = d_model, d_ff = 4*d_model (=> T = 6*d_model). Everything else (vocab,
d_model, depth, context, steps, data) is held fixed and all models are trained from
scratch with the same budget.

Prediction (Cor 3): if the val-loss minimum over the sweep is NOT at the standard
point, the conventional uniform attention/FFN ratio is suboptimal at equal compute --
and this can only be seen by training from scratch, not by cutting a finished model.
This is honest because it is a whole CURVE, not one hand-picked config.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(max(1, torch.get_num_threads()))
THIS = Path(__file__).resolve()


# ----------------------------------------------------------------- data (char-level)
def load_corpus(max_chars: int):
    text = None
    for ident in ("Salesforce/wikitext", "wikitext"):
        try:
            from datasets import load_dataset
            ds = load_dataset(ident, "wikitext-2-raw-v1", split="train")
            text = "\n".join(t for t in ds["text"] if t.strip())
            if len(text) > 10000:
                break
        except Exception:
            text = None
    if not text:
        text = ("the quick brown fox jumps over the lazy dog. " * 5000)
    text = text[:max_chars]
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    return data, len(chars)


def get_batch(data, block, bs, rng):
    ix = rng.integers(0, len(data) - block - 1, size=bs)
    x = np.stack([data[i:i + block] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block] for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)


# ----------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d_model, d_attn, d_ff, n_heads, block):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_attn = d_attn
        self.hd = d_attn // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_attn, bias=False)
        self.proj = nn.Linear(d_attn, d_model, bias=False)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(block, block)).view(1, 1, block, block))

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.split(self.d_attn, dim=2)
        q = q.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, self.d_attn)
        x = x + self.proj(y)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model, d_attn, d_ff, n_heads, n_layers, block):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, block, d_model))
        self.blocks = nn.ModuleList([Block(d_model, d_attn, d_ff, n_heads, block) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.block = block
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos[:, :T]
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------- train / eval
def train_one(vocab, d_model, d_attn, d_ff, n_heads, n_layers, block, data_tr, data_va,
              steps, bs, lr, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyGPT(vocab, d_model, d_attn, d_ff, n_heads, n_layers, block)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    model.train()
    for step in range(steps):
        x, y = get_batch(data_tr, block, bs, rng)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # cosine lr
        for g in opt.param_groups:
            g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps)) * min(1.0, (step + 1) / 50)
        opt.step()
    model.eval()
    vrng = np.random.default_rng(12345)
    losses = []
    with torch.no_grad():
        for _ in range(40):
            x, y = get_batch(data_va, block, bs, vrng)
            _, l = model(x, y)
            losses.append(float(l))
    return float(np.mean(losses)), n_params(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--head-dim", type=int, default=24)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--block", type=int, default=96)
    ap.add_argument("--d-attn-grid", default="24,48,72,96,120,144,168")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--max-chars", type=int, default=300000)
    ap.add_argument("--out", default="data/from_scratch_split_report.json")
    args = ap.parse_args()

    dm = args.d_model
    T = 6 * dm                                    # inner budget; standard point d_attn=dm, d_ff=4dm
    grid = [int(x) for x in args.d_attn_grid.split(",") if 0 < int(x) and 2 * int(x) < T]
    seeds = [int(s) for s in args.seeds.split(",")]

    data, vocab = load_corpus(args.max_chars)
    n_tr = int(0.9 * len(data))
    data_tr, data_va = data[:n_tr], data[n_tr:]
    print(f"[data] chars={len(data)} vocab={vocab} train={len(data_tr)} val={len(data_va)}")
    print(f"[setup] d_model={dm} T={T} standard d_attn={dm} d_ff={4*dm}  grid={grid}")

    rows = []
    for da in grid:
        d_ff = T - 2 * da
        nh = max(1, da // args.head_dim)
        da = nh * args.head_dim                  # snap so divisible by head_dim
        d_ff = T - 2 * da
        f = 2 * da / T
        vlosses, npar = [], 0
        for sd in seeds:
            vl, npar = train_one(vocab, dm, da, d_ff, nh, args.n_layers, args.block,
                                 data_tr, data_va, args.steps, args.bs, args.lr, sd)
            vlosses.append(vl)
        mean = float(np.mean(vlosses))
        is_std = (da == dm)
        rows.append({"d_attn": da, "d_ff": d_ff, "n_heads": nh, "attn_frac": f,
                     "val_loss_mean": mean, "val_loss_per_seed": vlosses,
                     "n_params": npar, "is_standard": is_std})
        print(f"[run] d_attn={da:3d} d_ff={d_ff:3d} heads={nh:2d} attn_frac={f:.2f} "
              f"params={npar} val_loss={mean:.4f} {'<-- STANDARD' if is_std else ''}")

    best = min(rows, key=lambda r: r["val_loss_mean"])
    std = next(r for r in rows if r["is_standard"])
    # noise check: is best meaningfully below standard?
    def se(xs):
        return (np.std(xs) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0
    gap = std["val_loss_mean"] - best["val_loss_mean"]
    pooled_se = math.sqrt(se(std["val_loss_per_seed"]) ** 2 + se(best["val_loss_per_seed"]) ** 2)
    shifted = (not best["is_standard"]) and (gap > 2 * pooled_se)
    if shifted:
        verdict = (f"UNIFORM SUBOPTIMAL: best split is d_attn={best['d_attn']} "
                   f"(attn_frac={best['attn_frac']:.2f}), val_loss={best['val_loss_mean']:.4f}, vs "
                   f"standard d_attn={std['d_attn']} val_loss={std['val_loss_mean']:.4f} "
                   f"(gap {gap:.4f} > 2*SE {2*pooled_se:.4f}). Trained from scratch at equal "
                   f"params => the conventional attention/FFN ratio is not compute-optimal here.")
    elif best["is_standard"]:
        verdict = ("STANDARD IS OPTIMAL: the conventional ratio minimises val loss on the sweep.")
    else:
        verdict = (f"INCONCLUSIVE: best (d_attn={best['d_attn']}) beats standard by {gap:.4f} "
                   f"but within noise (2*SE={2*pooled_se:.4f}).")

    report = {"d_model": dm, "T": T, "n_layers": args.n_layers, "block": args.block,
              "steps": args.steps, "seeds": seeds, "vocab": vocab,
              "standard_d_attn": dm, "rows": rows,
              "best": best, "standard": std, "gap_std_minus_best": gap,
              "pooled_se": pooled_se, "shifted": bool(shifted), "verdict": verdict}
    outp = THIS.parent / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(report, f, indent=2)

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fr = [r["attn_frac"] for r in rows]
        vl = [r["val_loss_mean"] for r in rows]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(fr, vl, "o-")
        sx = std["attn_frac"]; ax.axvline(sx, color="k", ls=":", label="standard ratio")
        bx = best["attn_frac"]; ax.scatter([bx], [best["val_loss_mean"]], color="red", zorder=5, label="best")
        ax.set_xlabel("attention share of inner budget  2*d_attn / T")
        ax.set_ylabel("val loss (from scratch, equal params)")
        ax.set_title("Attention<->FFN split swept from scratch\n(min off the standard point => uniform suboptimal)", fontsize=10)
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(str(outp.parent / "from_scratch_split.png"), dpi=130, bbox_inches="tight")
        print(f"[figure] {outp.parent / 'from_scratch_split.png'}")
    except Exception as e:
        print(f"[figure] skipped ({type(e).__name__})")

    print("\n================ SUMMARY ================")
    for r in rows:
        print(f"  attn_frac={r['attn_frac']:.2f} d_attn={r['d_attn']:3d} d_ff={r['d_ff']:3d}  "
              f"val_loss={r['val_loss_mean']:.4f}{'  (standard)' if r['is_standard'] else ''}")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
