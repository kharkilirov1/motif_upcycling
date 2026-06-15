# BRIDGE RESULTS — does the measured margin predict the from-scratch optimum?

**No fabricated numbers**; verbatim from `causal_margin_bridge.py
--steps 1200 --grid 48,72,96,120,144 --seeds 0,1 --deltas 16,32`. Report:
`data/causal_margin_bridge_report.json`. Figure: `data/bridge_equal_margin.png`.

This is the protocol's **Thm 3.1 / T4** tested end-to-end on **from-scratch trained**
TinyGPTs: it connects the donor-style margin *measurement* to the *trained* optimum of
the attention↔FFN budget split (T = 2·d_attn + d_ff held constant, so every split has
equal params). The recipe's claim: equalize the measured causal margins (m_attn = m_ffn)
and you land on the loss-optimal split.

## Result — HONEST NEGATIVE for the one-shot estimator

| d_attn | attn_frac | val loss | margin gap m_ffn − m_attn (×1e-8) |
|---:|---:|---:|---:|
| 48 | 0.17 | 1.8816 | **−108.5** |
| **72** | **0.25** | **1.8674** ← min | −5.0 |
| 96 (standard) | 0.33 | 1.8725 | +0.2 |
| 120 | 0.42 | 1.8794 | +0.3 |
| 144 | 0.50 | 1.8891 | +1.2 |

- **val-loss minimum: d_attn = 72** (reproduces the from-scratch ground truth — uniform
  is suboptimal; the optimum has *less* attention than standard).
- **equal-margin point (gap = 0): d_attn ≈ 95** → nearest grid point **96 = the standard
  split**, a full grid step *away* from the true optimum (72). **MISS.**
- At the true optimum (72) the measured gap is still **negative** (m_attn > m_ffn → "grow
  attention"), which would push you *back toward standard*, away from the minimum.

So **equal-margin does NOT recover the trained optimum here.**

## Diagnosis — it's the estimator, not separability

This is precisely the **trainability ≠ representability** failure mode (protocol §7) and
the open problem on margin estimators (§8). σ is *small* at this granularity (established
separately: attention vs FFN σ≈0.02), so the closed-form is *licensed in principle*. What
fails is the cheap **one-shot margin probe**: rank-truncating a *trained* model measures
how much it leans on its *existing* capacity, not the *architectural value of more*
capacity. That estimator **systematically over-values attention** — it keeps saying "grow
attention" past the optimum, placing equal-margin at ≈95 instead of 72.

The probe is not useless: at extreme starvation (d_attn=48) it correctly *screams* "grow
attention" (gap −1.1e-6, and indeed loss is high there), so it captures the **gross**
direction. It just **overshoots near the optimum**, where the curve is flat and the
representational/trainable gap dominates.

## Where this leaves the protocol (capstone across the session)
1. **η ≈ 1.85×** — heterogeneity waste is real (worth chasing).
2. **σ ≈ 0.02 coarse (attn/FFN)** — roles are separable; the recipe is licensed *in
   principle* at the granularity practitioners use.
3. **But the one-shot margin estimator is biased** — equal-margin lands on the standard
   split, not the trained optimum. The closed-form recipe as stated does **not** recover
   the optimum from a single representational probe.

**Implication / fix (the protocol's own prescription).** Replace the representational
one-shot probe with the **fixed-point realize-and-adapt loop** (steps 6–7): re-instantiate
at the candidate budget and *retrain* before re-measuring, so the margin is *trainable*,
not representational. That is more expensive than O(m) one-shot probes — eroding part of
the efficiency claim — and quantifying that cost vs a blind sweep is the real open
question (§8: "statistical guarantees on the margin estimators").

## Caveats
- Tiny char-level TinyGPT, d_model=96, 3 layers, 2 seeds; the ground-truth curve is flat
  (min 1.8674 vs standard 1.8725, ~0.005), so this is a hard regime for any estimator.
- Single rank-truncation estimator; a gradient-based or retrained margin could do better
  (untested here).

## Reproduce
```bash
python causal_margin_bridge.py --steps 1200 --grid 48,72,96,120,144 --seeds 0,1 --deltas 16,32
```
