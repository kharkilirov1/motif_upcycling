# Designing an architecture from error-vs-budget curves — a plan

A concrete procedure to turn FOG Theorem 4 (the function-morphology correspondence)
into an actual **architecture-design loop**: choose, per motif and per layer, both how
much budget to spend **and which implementation to use**, by equalizing the *marginal*
error reduction — then validate by training from scratch and track whether the advantage
survives scale.

This plan is written against what the experiments in `experiments/` already established,
so it does not repeat their mistakes.

## 0. Principle

Minimize total error at fixed total budget
`E(B_1..B_m) = Σ_v α_v · ε_{i_v}(B_v)`  s.t.  `Σ_v B_v = B_tot`,
whose optimum (Theorem 4) equalizes weighted marginal error reduction:
`α_v · ε'_{i_v}(B_v) = λ for all active v`. Uniform allocation is optimal *only* when
those marginals already match (Corollary 2); otherwise it is strictly suboptimal (Cor 3).

There are **two design levers**, not one:
1. **Allocation** — how much budget each motif/layer gets (`B_v`).
2. **Curve shaping** — *which implementation* realizes each motif (attention type, FFN
   gating, normalization, sparsity, low-rank), because each implementation has a
   *different* `ε(B)` curve (FOG Theorem 2', the finite-macro-library extension). Design =
   pick the implementation whose curve dominates in the budget region the allocation
   lands in, **and** the allocation, jointly.

## 1. Hard constraints learned from our own experiments (do not re-learn these)

- **`ε(B)` must be measured from scratch.** Static SVD compression of a uniformly-trained
  model (Qwen, `real_motif_budget.py`) measures *post-hoc compressibility*, not the curve
  a from-scratch design would see. It gave a clean 17% but on a confounded base.
- **One-shot allocation on a frozen base ties.** `trainable_het_rank.py` (gradient
  water-filling on frozen Qwen) was a statistical tie: the frozen uniform base does the
  heavy lifting, so adapter allocation is second-order. ⇒ allocation must be **co-trained
  and iterative**, not chosen once on a fixed model.
- **The effect is small at tiny scale.** `from_scratch_split.py` (from-scratch attn/FFN
  sweep) found the optimum off the standard ratio but only by ~0.009 nats. ⇒ the decisive
  question is **scale**, not a single small-scale win.
- **Always keep a homogeneous null control and a noise-aware verdict** (gap > 2·SE). The
  synthetic harness (`run_het_budget.py`) shows uniform IS optimal when curves match —
  any pipeline must reproduce that, or it is rigged.

## 2. The design loop (stages, with decision gates)

### Stage 1 — Budget space + cost model
Define the knobs and a closed-form cost (params *and* FLOPs/token; reuse `params.py`):
per-layer `d_attn` (= heads × head_dim), `d_ff`, optional per-layer residual width
bottleneck, attention family (MHA/GQA/MQA), FFN family (dense/gated/low-rank). Build a
generator: allocation vector → model at a **fixed total cost** (as in `from_scratch_split`,
where `T = 2·d_attn + d_ff` was held constant so every config has equal params).
*Deliverable:* `alloc_vector → model`, with `assert equal_cost`.

### Stage 2 — Measure from-scratch `ε_i(B)` curves (cheaply)
- **Method A (ground truth):** vary ONE motif's budget over a grid, train from scratch
  (short), record val loss → `ε_i(B)` and its discrete slope. Per motif/group.
- **Method B (cheap proxy to rank, then verify):** gradient/Hessian sensitivity,
  activation effective-rank. Faster, but **must be validated against Method A** — we
  already saw a one-shot gradient proxy mislead (the tie).
*Gate:* curves must be monotone-decreasing, roughly convex (project to convex envelope as
in `run_het_budget`), **and differ across motifs**. If they coincide → uniform is correct,
stop (this is the homogeneous-control case).

### Stage 3 — Solve the allocation
Marginal-equalization via the **exact DP allocator** already written
(`run_het_budget.alloc_optimal`) over the measured curves at fixed total cost. Extend to
**per-layer** allocation (depth-dependent), not just a global ratio.
*Deliverable:* a heterogeneous geometry + the predicted `E`.

### Stage 4 — Validate from scratch vs uniform
Train allocated geometry vs uniform baseline at equal cost, ≥4 seeds, noise-aware verdict
(reuse the `from_scratch_split` training/verdict code).
*Gate:* allocated < uniform by > 2·SE ⇒ proceed; else refine curves / stop.

### Stage 5 — Iterate to a fixed point (handles non-stationarity)
Curves move as you reallocate and train — this is *why* the one-shot frozen test tied.
Coordinate descent: measure → allocate → train → **re-measure at the new operating point**
→ re-allocate, until the allocation stabilizes. Track that `E` decreases monotonically.
*Gate:* allocation converges and beats uniform at the fixed point.

### Stage 6 — Scale study (the make-or-break)
Repeat Stages 2–5 at `d_model ∈ {96, 192, 384}` (and ≥2 depths). Plot the
uniform→allocated gap vs scale. **The bet is only worth pursuing if the gap is constant or
grows with scale.** If it shrinks toward noise by 384 ⇒ morphology is a small-scale
artifact — report honestly and stop.
*Gate:* gap non-decreasing across ≥2 scale points.

### Stage 7 — Curve shaping (beyond allocation)
For each motif, compare implementations (GQA vs MHA; gated vs dense vs low-rank FFN;
pre/post-norm; sparsity). Each yields a different `ε(B)` curve. Choose, per motif, the
implementation whose curve dominates in the allocated budget region. This is the joint
(implementation × budget) design — the full FOG macro-library program, and the part most
likely to produce a *large* (not 0.5%) effect.

### Stage 8 — Amortization / transfer (the economics)
The real risk: **search cost can eat the gain.** Test whether an allocation found at small
scale transfers (rank-only) to larger scale, so you don't re-derive it at frontier. If it
transfers ⇒ a practical recipe (cheap small-scale design → big-scale model). If not ⇒ the
allocation must be re-derived per scale, and you must show the derivation is cheap relative
to the training it saves.
*Gate:* small→large transfer beats the uniform large model, or the per-scale search is
provably cheap.

## 3. Risk register
| risk | mitigation / where handled |
|---|---|
| `ε` measured on confounded base | from-scratch curves only (Stage 2) |
| one-shot allocation ties | iterate to fixed point (Stage 5) |
| proxy rule misleads | validate Method B vs Method A (Stage 2) |
| effect vanishes at scale | scale study is an explicit kill gate (Stage 6) |
| search cost > gain | amortization/transfer test (Stage 8) |
| over-claiming from noise | ≥4 seeds, 2·SE verdict, homogeneous null (all stages) |

## 4. First concrete implementation step
Extend `experiments/from_scratch_morphology/from_scratch_split.py` into a pipeline:
`measure_curves_from_scratch()` (Stage 2, per-motif probes) → `alloc_optimal()` (Stage 3,
reuse `heterogeneous_budget`) → `train_and_compare()` (Stage 4) at `d_model=96`, producing
the allocated geometry and its head-to-head vs uniform. Then run the **scale sweep**
(Stage 6) — that single curve (gap vs scale) is the decision-grade result.

## 5. Success vs kill criteria
- **Advance** if: allocated beats uniform from scratch by > 2·SE, the allocation converges
  under iteration, and the gap is non-decreasing across ≥2 scales.
- **Kill / de-prioritize** if: the gap collapses to noise by `d_model=384`, or the search
  cost exceeds the training it saves and does not amortize across scale.

The honest north star: not "heterogeneity helps a 370k char model by 0.5%", but "a
curve-driven design procedure produces a model that beats the uniform design at equal
compute, and the advantage **grows** with scale and **amortizes** across it." Stages 6 and
8 are where this program lives or dies.
