# identifiability-field (Stage 0–1 run)

Tests one hypothesis: that **margin (FOG), κ (AOT reliability), equilibrium
deviation (ABPT), SARC relative scale (Motif-Upcycling), and candidate-set entropy
(AOT)** are five readouts of a single per-site *identifiability field* — not five
separate mechanisms.

- **`GOAL.md`** — the hypothesis and falsifiable predictions (read first).
- **`CLAUDE.md`** — the staged run plan and hard rules.
- **`signal_spec.md`** — what to measure + the `.npz` data contract.
- **`RESULTS.md`** — **the verdict of this run** (Stage 0 ✅, Stage 1 → SEPARATE,
  Stage 2/3 gated off by protocol).

## Layout
```
field_test.py                         # validated analysis engine + Stage-0 power check
load_and_test.py                      # run the engine on a real signals npz
collectors/collect_symbolic_aot_v8.py # Stage 1: signals/decisions from the aot_v8 task (RUNS)
collectors/collect_transformer_motif.py # Stage 2: per-(token,layer) signals from Motif model (specced)
aot_v8/                               # vendored Stage-1 substrate (Structured Operator Transformer)
data/                                 # collectors write *.npz (gitignored) + figures here
```

## Quick start
```bash
pip install -r requirements.txt
python field_test.py                                              # Stage 0 sanity
python collectors/collect_symbolic_aot_v8.py --out data/aot_signals.npz --noise 1.75
python load_and_test.py --data data/aot_signals.npz --fig data/aot_field.png  # Stage 1
```

## Result (summary)
Every stage that ran returns **SEPARATE MECHANISMS**:
- **Stage 1 (symbolic AOT):** SEPARATE, robust across noise 1.0/1.75/2.5 and both
  margin sources. Operator-perception signals collapse to one factor; router `margin`
  and the reliability head `kappa` are separate axes.
- **Stage 2 (real Qwen2.5-0.5B + trained motif overlay, by user request):** SEPARATE.
  Router signals form PC1; the LM-head `confidence` barely loads (0.12) and SARC
  `update_scale` is off-axis.
- **Stage 3 (cross-substrate transfer):** AUC 0.506 ≈ chance for predicting transformer
  intervention sites. The transferred readout tracks predictive entropy (Spearman
  −0.73) but does not encode "where to intervene".

So the strong "one identifiability field" hypothesis fails on both a symbolic
substrate and a real transformer. See `RESULTS.md` for full verdict blocks.

## Non-negotiable
Never fabricate results. A stage that can't run is reported as blocked. The null
result (SEPARATE) is a valid finding.
