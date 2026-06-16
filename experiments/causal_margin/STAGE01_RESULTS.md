# STAGE 0 + 1 RESULTS — the prize evaporates under proper power

**No fabricated numbers**; verbatim from `stage01_trainable_margin.py --steps 1200
--grid 24,48,72,96,120,144,168 --seeds 0,1,2,3`. Report: `data/stage01_report.json`.
Figure: `data/stage01_trainable_margin.png`.

## Stage 0 — strengthened from-scratch ground truth (4 seeds, was 2)

| d_attn | attn_frac | val loss | ±SE |
|---:|---:|---:|---:|
| 24 | 0.08 | 1.8775 | 0.0037 |
| 48 | 0.17 | 1.8785 | 0.0023 |
| **72** | **0.25** | **1.8754** | 0.0052 |
| 96 (standard) | 0.33 | 1.8789 | 0.0055 |
| 120 | 0.42 | 1.8804 | 0.0051 |
| 144 | 0.50 | 1.8866 | 0.0027 |
| 168 | 0.58 | 1.8845 | 0.0026 |

**The val-loss is FLAT across d_attn ∈ [24, 120] (attn_frac 0.08–0.42).** The argmin (72)
beats the standard split (96) by only **0.0035, vs pooled 2·SE = 0.0150** → **NOT
significant**. Only attention-heavy splits (≥0.5) are clearly worse.

> **Correction of an earlier underpowered result.** The 2-seed run in
> `from_scratch_morphology` reported "uniform suboptimal" (gap 0.009 > 2·SE 0.006). With
> **4 seeds the SE grows and the gap vanishes into noise.** The split simply does not move
> trained val-loss within the normal range at this scale.

## Stage 1 — trainable-margin descent

Replacing the bridge's biased one-shot representational probe with the **trainable**
margin (finite difference of *retrained* val-loss), margin-guided coordinate descent from
the standard split:

- path `96 → 72`, **lands on d_attn=72** (the argmin), training **4/7 architectures**
  (16/28 trains) vs the full sweep.
- BUT improvement over standard is **+0.0035 vs 2·SE 0.0150 → within noise (hollow)**.

So the descent *navigates* to the lowest mean point and is cheaper than the sweep, but
there is **no significant prize to win** — the "hit" is on a flat plateau.

## What this means (honest, and it cuts against the idea)

1. **The η≈1.85× is a functional-error number, not a trained-val-loss number.** It was
   measured on SVD reconstruction error of a frozen donor. When you actually **train each
   split from scratch (adequately seeded)**, the attention/FFN budget split is **within
   noise** over the whole normal range. The trained-val-loss η here is ≈ **1.0**.
2. **At this scale the allocation question is moot** — not because margin-descent fails
   (it reaches the argmin), but because there is essentially nothing to allocate: the
   landscape is flat. The strong version can't be demonstrated where there's no prize.
3. This sharpens the earlier verdict: the deflation is now stronger and better-measured.
   The functional/representational signals (SVD curves, σ) are real but **do not translate
   into a from-scratch trained-loss advantage at d_model=96**.

## The new gating question (supersedes "Stage 2 with many roles")

Before testing the O(m)-vs-sweep efficiency claim (Stage 2), we must first find a regime
where **the budget split significantly moves trained val-loss** — otherwise every later
comparison is noise. Candidates, in order of cost:
- larger model / more data / longer training (does a significant optimum emerge as the
  model becomes capacity-bound?);
- a harder task where attention vs memory genuinely trade off;
- if **no reachable regime** shows a significant split-dependence → the practical prize is
  illusory and the strong version is effectively dead, leaving only the framework (σ as a
  diagnostic, function-preserving init).

## Caveats
- Tiny char-level TinyGPT, d_model=96, 3 layers, 1200 steps. The flatness may itself be a
  small-scale / under-training artifact — exactly why the next gate is **scale**.
- "uniform_suboptimal=False" here is a power statement at 4 seeds, not proof of exact
  equality; the honest claim is "no detectable advantage at this budget."

## Reproduce
```bash
python stage01_trainable_margin.py --steps 1200 --grid 24,48,72,96,120,144,168 --seeds 0,1,2,3
```
