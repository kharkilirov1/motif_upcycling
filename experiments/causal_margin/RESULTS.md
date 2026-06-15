# RESULTS — Causal-Margin Allocation, core measurements on real Qwen2.5-0.5B

**No fabricated numbers**; verbatim from `causal_margin.py` (`--layers 8..15
--ranks 2..256 --calib-seq 12 --eval-seq 24 --sigma-rank 16`). Report:
`data/causal_margin_report.json`. Forward-only (no training).

This implements the protocol's pre-registered real-model tests (§6) on Qwen2.5-0.5B,
with roles = motif types {q, o (compare), gate (select), up (expand), down (memory)},
each role spanning layers 8–15.

## Headline (two numbers, honest split)
- **η ≈ 1.85× (the waste is real).** A uniform *equal-budget-per-role* allocation needs
  **~1.7–2.0× the parameters** to match the capped-water-filling optimum's error. The
  protocol's positive consequence holds on a real model: uniform interfaces waste a
  large, ~scale-stable fraction of budget. (Measured η even *exceeds* the clean
  saturating-role limit m/(m−|S|)=5/4=1.25, because the non-saturating roles also have
  very different curves.)
- **σ ≈ 0.90 (LARGE — the closed-form recipe is NOT licensed).** The separability index
  (offdiag/diag of the budget-space loss Hessian, via end-to-end LM-loss interactions
  under joint role compression) is large and *grows* with perturbation
  (0.82 / 0.89 / 0.98 at probe ranks 8/16/32). Per the protocol's own §6, this is the
  "Δ_i>0 but σ large" outcome: **roles are real but coupled through the residual stream,
  so margin water-filling degrades to guided joint search, not a closed form.**

## Role error-vs-budget curves εᵢ(r) (functional, on real activations)
```
q     r2=0.881  r8=0.747  r32=0.511  r128=0.182  r256=0.069   (steep; saturates near top -> S)
o     r2=0.948  r8=0.908  r32=0.791  r128=0.460  r256=0.201   (compare, slower)
gate  r2=0.448  r8=0.385  r32=0.290  r128=0.190  r256=0.130   (select, compressible: low throughout)
up    r2=0.979  r8=0.934  r32=0.835  r128=0.648  r256=0.486   (expand, incompressible)
down  r2=0.930  r8=0.905  r32=0.839  r128=0.645  r256=0.453   (memory, incompressible)
```
Saturating set **S = {q}** (only q is essentially flat at the top of the grid). The
curves genuinely differ by motif type — the precondition for any allocation gain.

## η (budget-efficiency, Def 2.1)
| B_tot | optimal error E* | B_uni to match E* | η = B_uni/B_tot |
|---:|---:|---:|---:|
| 1.34M | 3.611 | 2.42M | 1.81 |
| 2.00M | 3.432 | 3.96M | 1.98 |
| 2.67M | 3.261 | 4.66M | 1.75 |
| 4.01M | 2.998 | 7.61M | 1.90 |
| 5.34M | 2.758 | 9.73M | 1.82 |

η is stable at ~1.85× across budgets — a real, persistent efficiency gap. (Theory's
clean limit m/(m−|S|)=1.25 *under-predicts* it; the simple saturating-role model is not
the whole story because the non-saturating roles' curves also differ a lot.)

## σ (separability index, T2 — the verdict)
```
probe rank  8 : mean|diag dL|=0.960  mean|interaction|=0.786  sigma=0.819
probe rank 16 : mean|diag dL|=0.937  mean|interaction|=0.832  sigma=0.888
probe rank 32 : mean|diag dL|=0.992  mean|interaction|=0.973  sigma=0.981
sigma (mean) = 0.896   ->  LARGE
per-role dL_i (rank 32): gate=+2.63  up=+0.85  down=+0.70  o=+0.62  q=+0.17
strongest interaction: (gate, up) = -2.47   (strongly SUB-additive)
```
**σ ≈ 0.90 ≫ 0.15** → the fine roles are not separable on real Qwen. The coupling is
dominated by the FFN: `gate` is by far the most loss-sensitive module (ΔL=+2.63 alone),
and `gate`+`up` is strongly **sub-additive** (−2.47) — wrecking both together hurts much
less than the sum, i.e. they share error budget (are partly *substitutable*).

## Interpretation (honest)
1. **The prize is real (~1.85×), the closed-form shortcut is not.** Heterogeneous budget
   genuinely buys ~1.85× compute efficiency on a real model, but you cannot obtain it by
   independently measuring per-role margins and water-filling — σ is large, so the
   allocation must be searched jointly (guided coordinate search at best).
2. **The large σ is concentrated in FFN-internal substitutability.** `gate`/`up`/`down`
   behave like one coupled FFN reservoir, not three independent roles. By the protocol's
   own §7 ("if two roles are ε-substitutable, merge them"), they should be **one coarse
   `FFN` role**. This makes a sharp, testable prediction: **at coarse granularity
   (attention vs FFN), σ should drop** — the recipe may be licensed coarsely even though
   it fails for fine motifs. This is exactly the protocol's §6 fallback ("keep the coarse
   mixing-vs-tokenwise allocation; abandon fine motifs honestly").

## Caveats
- σ is measured by SVD-truncation perturbations on a **frozen** model (representational
  curvature), not the from-scratch trained operating point; the truncation-based σ is a
  proxy and large perturbations inflate it (σ grows with probe rank). A from-scratch /
  smaller-δ σ could differ.
- The σ metric conflates "coupling" with "damage saturation" when one role (gate)
  dominates; the sub-additive sign supports the substitutability reading.
- One model/size; coarse-role σ (the decisive follow-up) not yet run.

## Verdict vs the protocol's pre-registered outcomes (§6)
This is the **"Δ_i>0 but σ large ⇒ roles real but coupled; recipe becomes guided search,
not closed form"** branch — with the added, actionable finding that the coupling is
FFN-internal, predicting that **coarse (attention/FFN) roles may be separable**. Next
test: re-run σ with merged roles {attn = q,o; ffn = gate,up,down}.

## Reproduce
```bash
python causal_margin.py --layers 8,9,10,11,12,13,14,15 --ranks 2,4,8,16,32,64,128,192,256 \
    --calib-seq 12 --eval-seq 24 --sigma-rank 16
```
