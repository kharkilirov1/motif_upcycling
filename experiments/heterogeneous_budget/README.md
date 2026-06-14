# Heterogeneous Motif Budget — an empirical test of FOG Theorem 4

**Question.** The Finite-Operator-Grammar paper proves (Theorem 4 / Corollaries 2–3)
that when motif components have *different* error-vs-budget curves `eps_i(B)`, a
**uniform** budget allocation is **strictly suboptimal** — the optimum equalises the
*marginal* error reduction across components, which is generally non-uniform. Does
that hold for *real trained components*, or are the curves in practice too similar
for it to matter? (The same "are the margins actually there?" risk that the
identifiability-field experiment ran into.)

**Answer (this experiment):** on a controlled task it holds clearly — heterogeneous
allocation beats uniform by ~11% median (up to ~31%) at equal total budget, the gap
**vanishes on a homogeneous null control**, and **grows with heterogeneity**.

## Design (faithful to the theorem's separable convex objective)
- **Objective = the theorem's exact form** `E = sum_m eps_m(w_m)` at fixed total
  budget `sum_m w_m`: `M=4` independent sub-functions, expert `m` fits sub-function
  `m`, so the total error is literally separable.
- **Genuinely heterogeneous curves.** Each sub-function is a sigmoid-staircase ridge
  with `complexity` steps; a width-`w` ReLU FFN fits ~`w` steps, so `eps_m(w)`
  decreases with a knee near `w = complexity`. Heterogeneous task complexities
  `[1, 4, 12, 32]`; homogeneous control `[8, 8, 8, 8]`.
- **Measured, not assumed.** `eps_m(w)` is measured by training an expert at each grid
  width (averaged over seeds), then projected onto the monotone + **convex** lower
  envelope required by the paper's Assumption M1.
- **Budget = total hidden width** `sum_m w_m` (∝ params and FLOPs; per-motif in/out
  dims fixed). The **uniform** partition uses the repo's real `even_motif_sizes`;
  `motif_slices` validates any allocation tiles the budget — the same channel-partition
  machinery `MotifSwiGLUMLP` uses.
- **Allocator = exact** (DP over the width grid, `sum <= B`), so the optimum is
  *guaranteed* ≤ uniform (uniform is itself feasible). Baselines: uniform, exact
  optimum, and random allocations.
- **Null control.** On the homogeneous task all curves are identical, so the theorem
  predicts uniform is optimal (gap 0). That control is what separates a real effect
  from a method that always favours heterogeneity.

## Run
```bash
python run_het_budget.py --n 4000 --steps 600 --reps 3
```
Outputs `data/het_budget.png` and `data/het_budget_report.json`. See `RESULTS.md`.

## Connection to the program
This is the cheap decisive test for FOG Theorem 4 — the "morphology-mismatch" result
that says heterogeneous operators deserve heterogeneous resource geometry, and that a
uniform residual/interface width is provably wasteful. It is the resource-side twin of
the identifiability-field finding (signals are *separate* axes, not one field): both say
**heterogeneity is the point, not a nuisance.**
