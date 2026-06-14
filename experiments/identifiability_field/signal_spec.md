# signal_spec.md — what to measure, where it comes from, how to store it

All stages produce the SAME artifact: an `.npz` consumed by `load_and_test.py`.

## Data contract (npz)
- `X`            : float array, shape (n_sites, k_signals)  — one row per site
- `signal_names` : array of k strings, column order of X
- `dec__<name>`  : one array per decision target, length n_sites
                   (integer = categorical decision; float w/ >10 uniques = regression)
Example:
```python
import numpy as np
np.savez(out, X=X, signal_names=np.array(names),
         dec__motif=motif, dec__budget=budget, dec__adapt=adapt,
         dec__trust=trust, dec__temperature=temp)
```
A "site" = the smallest unit at which the field is supposed to live: one **step/event**
(symbolic) or one **(token, layer)** at a patched layer (transformer). Every signal and
every decision for a row must be read at that same site (no leakage across sites).

## The 5 signals
| signal            | symbolic (AOT) source                                  | transformer (Motif) source                                    |
|-------------------|--------------------------------------------------------|---------------------------------------------------------------|
| `confidence`      | `max(active_probs)` and/or `max(op_probs)`             | 1 − predictive entropy of next-token dist (or max softmax)    |
| `cand_entropy`    | `H(op_probs)` (entropy over operator candidates)       | entropy of router weights `α` over motif slices               |
| `margin`          | top1−top2 of `op_probs` (or `active_probs`)            | top1−top2 of router weights `α` at the site                   |
| `kappa`           | `reliability_score(model, obs)` ∈ (0,1)                | (optional) learned reliability head; else omit this column    |
| `update_scale`    | `repair_conf` as a proxy for intervention magnitude    | SARC relative scale `r = log( RMS(Δu)/RMS(x) )` at the site    |

You need ≥3 of these columns for a meaningful test; more is better. If `kappa` has no
transformer analog yet, drop it (set its column out) and run with 4 — the engine is
agnostic to k.

## The decisions (what the field is supposed to drive)
| decision      | symbolic (AOT)                                            | transformer (Motif)                                        |
|---------------|-----------------------------------------------------------|------------------------------------------------------------|
| `motif`       | optimal/active operator id (categorical)                  | argmax router slice at the site (categorical)              |
| `budget`      | optimal action cost-tier from `ACTION_COST` (3 levels)    | would top-k route >1 slice? (tiered) / activated-slice count|
| `adapt`       | is `repair`/`beam` the counterfactual-best action? (bin)  | does adding motif-LoRA at this site lower loss? (P0 probe)  |
| `trust`       | `repair_state == true_next_state` (repair correct, bin)   | (optional) reliability-head correct                        |
| `temperature` | margin-derived continuous target                          | predictive entropy (continuous)                            |

For the symbolic optimal action use `argmax(counterfactual_action_scores(...))` over
`ACTIONS = ['hard','repair','freeze','beam']`. The `adapt` decision is deliberately
non-monotone (band-shaped) — keep the degree-2 readout so it is representable.

## Reading the verdict
- **SHARED** needs all of: mean|corr| > 0.35, PC1 EVR > 0.55, min|PC1 loading| > 0.30,
  worst sufficiency ratio > 0.85.
- Report the full block regardless; if it says SEPARATE, that is the (valid) answer
  for that substrate.
