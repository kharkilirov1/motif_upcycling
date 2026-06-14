# RESULTS — from-scratch attention/FFN split: the uniform ratio is mildly suboptimal

**No fabricated numbers**; every value is verbatim from `from_scratch_split.py`
(`--d-attn-grid 24,48,72,96,120,144,168 --steps 1500 --seeds 0,1,2,3 --max-chars 300000`).
Figure: `data/from_scratch_split.png`; full output: `data/from_scratch_split_report.json`.

## Why this experiment exists
The Qwen experiments only re-cut an already-uniformly-trained model (SVD compression,
LoRA adapters). That cannot test a *native* heterogeneous architecture — the base is
already shaped by uniform geometry (and indeed the trainable-adapter test tied). The
correct test is to **train from scratch** at equal compute and sweep the geometry. This
is the user's hypothesis: *a new architecture must be trained from scratch to show its
advantage.*

## Setup
Tiny GPT, char-level wikitext-2. d_model=96, 3 layers, context 96, ~373k params.
Per layer the inner budget `T = 2*d_attn + d_ff = 6*d_model = 576` is **fixed**, so every
split has the same parameter count (373,776 → 372,912 across the sweep, ±0.2% from
snapping d_attn to head_dim=24). The **standard transformer is one sweep point**:
`d_attn = d_model = 96`, `d_ff = 4*d_model = 384`, attention share `2*d_attn/T = 0.33`.
All models trained from scratch, same steps/data/lr, 4 seeds.

## Result
| attention share `2·d_attn/T` | d_attn | d_ff | heads | params | val loss (mean of 4 seeds) |
|---:|---:|---:|---:|---:|---:|
| 0.08 | 24 | 528 | 1 | 373,776 | 1.8494 |
| 0.17 | 48 | 480 | 2 | 373,632 | 1.8495 |
| **0.25** | **72** | **432** | **3** | **373,488** | **1.8395  ← best** |
| 0.33 (standard) | 96 | 384 | 4 | 373,344 | 1.8484 |
| 0.42 | 120 | 336 | 5 | 373,200 | 1.8488 |
| 0.50 | 144 | 288 | 6 | 373,056 | 1.8468 |
| 0.58 | 168 | 240 | 7 | 372,912 | 1.8596 |

**VERDICT: UNIFORM SUBOPTIMAL.** The minimum is at attention share **0.25**
(`d_attn=72`), not the standard **0.33**. Best beats standard by **0.0090** nats, and
**2·SE = 0.0057**, so the gap clears the noise threshold. The optimum location (0.25) is
reproducible: an earlier 2-seed run gave the identical minimum (gap 0.0093 > 2·SE 0.0058).
At equal compute, trained from scratch, the model prefers **slightly less attention width
and more FFN** than the conventional ratio.

## Interpretation
- This is the from-scratch confirmation of FOG Theorem 4 / Corollary 3: the
  attention and FFN motifs have different error-vs-budget curves, so the conventional
  uniform ratio is not the compute-optimal allocation.
- Crucially, it is visible **only** by training from scratch. The Qwen retrofit could not
  show it: static SVD only measured post-hoc compressibility of a uniform-trained model,
  and the trainable-adapter test tied because the frozen uniform base did the heavy
  lifting. The user's hypothesis — *native architecture, trained from scratch* — is the
  right lens, and it changes the verdict from "tie" to "uniform suboptimal".

## Honest caveats
- **The effect is small.** 0.0090 nats is ~0.5% of the loss; the curve is shallow
  (1.840–1.860) and not perfectly smooth (the 0.50 point dips below 0.42, i.e. residual
  noise even at 4 seeds). The claim is "the optimum is off the standard point and the gap
  clears 2·SE", not "a large win".
- **Tiny scale.** 373k params, char-level, 1500 CPU steps, one depth, one dataset. The
  *direction* (more FFN / less attention here) is specific to this regime; the optimal
  split is known to move with scale, data, and depth. The transferable claim is
  "uniform is not automatically optimal", not a universal ratio.
- **One axis.** This sweeps a single global attention/FFN ratio (uniform across layers).
  The stronger architecture claim — per-layer / per-motif heterogeneity — is the natural
  next step (option B: per-layer allocation from scratch).

## Reproduce
```bash
pip install torch numpy matplotlib datasets
python from_scratch_split.py --d-attn-grid 24,48,72,96,120,144,168 --steps 1500 --seeds 0,1,2,3
```

## Where this leaves the program
Three rungs, increasingly honest:
1. synthetic budget allocation (from scratch, controlled) — heterogeneous clearly wins;
2. real Qwen **static compression** — heterogeneous wins (but is post-hoc, on a uniform base);
3. real Qwen **trainable adapters** — tie (frozen uniform base dominates);
4. **this**: tiny GPT **from scratch** — the conventional uniform ratio is suboptimal
   (small but significant), which is the first evidence on the actual architecture-design
   question rather than on retrofitting.
The next step toward a real heterogeneous architecture is per-layer/per-motif allocation
trained from scratch, and testing whether the gap grows with scale.
