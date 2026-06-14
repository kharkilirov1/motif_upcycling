"""collect_symbolic_aot_v8.py - Stage 1 collector (the cheap decisive test).

The original ``collect_symbolic_aot.py`` targets an ``hma_aot_v31`` reliability
package that is not in the provided uploads. The AOT substrate that *is*
available is ``aot_v8`` (Structured Operator Transformer). This collector wires
the same five identifiability signals + five decisions to the real ``aot_v8``
objects, so Stage 1 can actually run.

Substitution recorded for RESULTS.md
------------------------------------
- substrate: ``aot_v8.AOTStructuredComposer`` (group/marker composition task)
  instead of ``hma_aot_v31``.
- a **site** = one operator-bearing token in a held-out (long-chain) evaluation
  sequence. All five signals and five decisions are read at that same token.
- ``aot_v8`` has no built-in reliability estimator, so ``kappa`` is a small
  reliability head trained on FROZEN (noised) encoder states to predict per-token
  operator correctness (mirrors ``train_reliability_estimator`` / ``reliability_score``).
- ``aot_v8`` has no repair action set, so ``update_scale`` is the correction
  magnitude ``1 - p(true_op)`` (probability mass that must move to fix the token),
  the symbolic analog of SARC scale / repair confidence.
- The clean AOT model parses single tokens perfectly, which makes the decisions
  degenerate (trust==1 everywhere). The original collector handled this with a
  ``--noise`` knob feeding ``simulate_base_observation``. Here ``--noise`` injects
  Gaussian perception noise into the encoder states, applied UNIFORMLY to the op
  head, kind head and reliability head, so every signal reads the same noised
  site. This is what gives the decisions (trust/adapt/budget) real variance.

Signals come from three *distinct* mechanisms so the test is genuinely
cross-mechanism (not five views of one softmax):
  confidence, cand_entropy, update_scale  <- operator head
  margin                                  <- kind/router head (top1-top2)
  kappa                                   <- separately-trained reliability head

Decisions are grounded in ground truth (the true operator), never in the signal
vector, except ``temperature`` which is margin-derived by spec.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the vendored aot_v8 importable regardless of cwd.
THIS = Path(__file__).resolve()
EXP_ROOT = THIS.parent.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from aot_v8.data import (  # noqa: E402
    OP_NONE, batch_examples, build_examples, make_vocab_for_splits,
)
from aot_v8.ops import DOMAIN_SIZES, FLAT_OFFSETS, ID2DOMAIN  # noqa: E402
from aot_v8.train_eval import set_seed, train_structured  # noqa: E402


def _entropy(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


class ReliabilityHead(nn.Module):
    """Predicts P(operator parsed correctly) from a frozen encoder state."""

    def __init__(self, d_model: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


@torch.no_grad()
def _encode(model, input_ids, noise=0.0, generator=None):
    """Return per-token hidden states, kind_probs, op_probs from the AOT model.

    ``noise`` injects Gaussian perception noise into the encoder states (scaled by
    the per-feature std) BEFORE the heads, so all heads read the same noised site.
    """
    h = model.encoder(input_ids)
    if noise > 0:
        pad = input_ids.eq(0).unsqueeze(-1)
        std = h[~pad.expand_as(h)].std().clamp_min(1e-6) if (~pad).any() else 1.0
        eps = torch.randn(h.shape, generator=generator) * (noise * std)
        h = (h + eps).masked_fill(pad, 0.0)
    kind_probs = F.softmax(model.kind_head(h), dim=-1)
    op_probs = F.softmax(model.op_head(h), dim=-1)
    return h, kind_probs, op_probs


def train_reliability_head(model, examples, vocab, max_len, noise=0.0, seed=0,
                           epochs=80, lr=2e-3, resamples=4):
    """Train a reliability head on frozen (noised) encoder states (train split).

    To calibrate kappa to correctness *under the collection noise*, several noisy
    resamples of the train states are pooled.
    """
    set_seed(seed)
    b = batch_examples(examples, vocab, max_len)
    op_labels = b["op_labels"]
    mask = op_labels.ne(OP_NONE)
    gen = torch.Generator().manual_seed(seed + 7)

    feat_list, corr_list = [], []
    for _ in range(max(1, resamples) if noise > 0 else 1):
        with torch.no_grad():
            h, _, op_probs = _encode(model, b["input_ids"], noise=noise, generator=gen)
        op_pred = op_probs.argmax(-1)
        feat_list.append(h[mask].detach())
        corr_list.append((op_pred[mask] == op_labels[mask]).float())
    feats = torch.cat(feat_list, 0)
    correct = torch.cat(corr_list, 0)

    head = ReliabilityHead(feats.shape[-1])
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        logit = head(feats)
        loss = F.binary_cross_entropy_with_logits(logit, correct)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-lengths", default="1,2,3,4,5")
    ap.add_argument("--test-lengths", default="5,10,20,50")
    ap.add_argument("--train-per-len", type=int, default=160)
    ap.add_argument("--test-per-len", type=int, default=160)
    ap.add_argument("--aot-epochs", type=int, default=60)
    ap.add_argument("--noise", type=float, default=1.75,
                    help="Gaussian perception noise on encoder states (gives decisions variance)")
    ap.add_argument("--margin-source", choices=["kind", "op"], default="kind",
                    help="kind=router/kind-head top1-top2 (cross-mechanism); op=op-head top1-top2")
    ap.add_argument("--distractor-rate", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()

    train_lengths = [int(x) for x in args.train_lengths.split(",")]
    test_lengths = [int(x) for x in args.test_lengths.split(",")]

    set_seed(args.seed)
    train = build_examples(train_lengths, args.train_per_len, seed=args.seed,
                           distractor_rate=args.distractor_rate, alias_mode="mixed")
    test = build_examples(test_lengths, args.test_per_len, seed=args.seed + 99,
                          distractor_rate=args.distractor_rate, alias_mode="mixed")
    vocab = make_vocab_for_splits(train, test)
    max_len = min(256, max(len(vocab.tokens(e.text)) for e in train + test))

    print(f"training AOT ({args.aot_epochs} epochs) on {len(train)} short chains ...")
    model, _ = train_structured(train, vocab, max_len, epochs=args.aot_epochs, seed=args.seed)
    model.eval()
    print(f"training reliability head on frozen encoder states (noise={args.noise}) ...")
    kappa_head = train_reliability_head(model, train, vocab, max_len, noise=args.noise, seed=args.seed)

    # ---- log per-(operator token) signals + decisions on the held-out test set ----
    b = batch_examples(test, vocab, max_len)
    gen = torch.Generator().manual_seed(args.seed + 123)
    h, kind_probs, op_probs = _encode(model, b["input_ids"], noise=args.noise, generator=gen)
    with torch.no_grad():
        kappa_all = torch.sigmoid(kappa_head(h))  # [B, T]

    input_ids = b["input_ids"]
    kind_labels = b["kind_labels"]
    op_labels = b["op_labels"]
    B, T = input_ids.shape

    sig_names = ["confidence", "cand_entropy", "margin", "kappa", "update_scale"]
    rows, motif, budget, adapt, trust, temp = [], [], [], [], [], []

    for bi in range(B):
        for pos in range(T):
            k = int(kind_labels[bi, pos])
            o_true = int(op_labels[bi, pos])
            if k == 0 or o_true == OP_NONE:
                continue  # not an operator-bearing token -> not a site
            domain = ID2DOMAIN[k]
            n_ops = DOMAIN_SIZES[domain]

            op_p = op_probs[bi, pos, :n_ops].detach().numpy()
            op_p = op_p / max(op_p.sum(), 1e-9)
            srt = np.sort(op_p)[::-1]
            kind_p = kind_probs[bi, pos].detach().numpy()
            kind_srt = np.sort(kind_p)[::-1]

            confidence = float(op_p.max())                       # op-head certainty
            cand_entropy = _entropy(op_p)                        # op-head spread
            if args.margin_source == "op":
                margin = float(srt[0] - srt[1])                  # op-head margin (sensitivity)
            else:
                margin = float(kind_srt[0] - kind_srt[1])        # kind/router-head margin
            kappa = float(kappa_all[bi, pos])                    # reliability head
            update_scale = float(1.0 - op_p[o_true])             # correction magnitude
            rows.append([confidence, cand_entropy, margin, kappa, update_scale])

            # ground-truth decisions
            order = np.argsort(op_p)[::-1]
            rank_true = int(np.where(order == o_true)[0][0])     # 0 = top-1
            hard_correct = rank_true == 0
            in_top2 = rank_true <= 1

            # domain-unique operator role (d4:0-7, c8:8-15, bool:16-19), so the same
            # local id in different domains is not conflated into one class.
            motif.append(FLAT_OFFSETS[domain] + o_true)           # which operator role
            trust.append(1 if hard_correct else 0)               # trust the parse?
            # 3 cost tiers: correct(cheap) / repairable(medium) / lost(expensive)
            budget.append(0 if hard_correct else (1 if in_top2 else 2))
            # band-shaped: intervene only in the repairable middle zone
            adapt.append(1 if (not hard_correct and in_top2) else 0)
            temp.append(margin)                                   # continuous, margin-derived

    X = np.asarray(rows, float)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=X, signal_names=np.array(sig_names),
             dec__motif=np.array(motif), dec__budget=np.array(budget),
             dec__adapt=np.array(adapt), dec__trust=np.array(trust),
             dec__temperature=np.array(temp, float))
    print(f"wrote {args.out}: {X.shape[0]} operator-token sites x {X.shape[1]} signals")
    print(f"  decision base-rates: trust={np.mean(trust):.3f} adapt={np.mean(adapt):.3f} "
          f"budget(0/1/2)={np.bincount(budget, minlength=3)}")


if __name__ == "__main__":
    main()
