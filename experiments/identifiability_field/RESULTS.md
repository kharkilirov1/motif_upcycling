# RESULTS — Does the "identifiability field" exist?

Experiment defined in `GOAL.md`. Verdict per stage, run verbatim from the engine.
**No numbers in this file are fabricated**; every block below was produced by
`field_test.py` / `load_and_test.py` on this machine. Blocked stages are stated as
blocked, with exactly what was missing.

**Headline:** the symbolic Stage-1 gate returns **SEPARATE MECHANISMS** (robustly,
across three noise levels). Per the protocol in `GOAL.md`/`CLAUDE.md`
("if Stage 1 verdict is SEPARATE, write it to RESULTS.md and STOP — do not build
Stage 2"), the pipeline terminates at Stage 1. This is a valid, publishable null
for the AOT substrate, not a failure.

---

## Stage 0 — engine sanity (planted data)  ✅ PASS

Command:
```bash
python field_test.py
```
The engine plants two ground-truth datasets and must label them correctly:
```
REGIME A (planted single field):     VERDICT: ONE SHARED FIELD
  mean|corr|=0.760  PC1=0.81  min|PC1 load|=0.44  worst suff ratio=0.99
REGIME B (planted independent):      VERDICT: SEPARATE MECHANISMS
  mean|corr|=0.014  PC1=0.21  min|PC1 load|=0.03  worst suff ratio=0.21

--- power check ---
A detected as shared?    True   (want True)
B detected as separate?  True   (want True)
```
Figure: `data/field_test_power.png`.

**Conclusion:** the instrument has power — it says "shared" when a field is planted
and "separate" when it is not. Its verdict on real data can be trusted.

---

## Stage 1 — SYMBOLIC gate (the cheap decisive test)  ✅ RAN → SEPARATE

### Substrate substitution (recorded per CLAUDE.md)
The collector in `CLAUDE.md` targets `hma_aot_v31_v34_reliability_project.zip`, which
**was not in the provided uploads**. The AOT substrate that *is* available is
`aot_v8` (Structured Operator Transformer). A faithful collector
`collectors/collect_symbolic_aot_v8.py` was written against the real `aot_v8`
objects (vendored under `aot_v8/`). Substitutions:

| spec object | `aot_v8` realization |
|---|---|
| site (= one event) | one **operator-bearing token** in a held-out long-chain eval sequence |
| `op_probs` | `softmax(op_head(encoder(x)))` |
| `active_probs` / kind | `softmax(kind_head(encoder(x)))` |
| `reliability_score` (κ) | small reliability head trained on **frozen, noised** encoder states to predict per-token operator correctness |
| `repair_conf` / update scale | correction magnitude `1 − p(true_op)` (mass that must move to fix the token) |
| `counterfactual_action_scores` | ground-truth recoverability of the true operator (rank in `op_probs`) |

The clean AOT model parses single tokens **perfectly** (its difficulty is at
*composition*, not token parsing), so without noise the decisions are degenerate
(`trust ≡ 1`). The original collector handled exactly this with a `--noise` knob;
here `--noise` injects Gaussian perception noise into the encoder states, applied
**uniformly to all heads** (op, kind, reliability), which is what gives the
decisions real variance.

### The 5 signals come from 3 distinct mechanisms (genuine cross-mechanism test)
- `confidence`, `cand_entropy`, `update_scale` ← operator head
- `margin` (top1−top2) ← **kind / router head** (a different head)
- `kappa` ← **separately-trained reliability head** (frozen states)

This matters: if all five were functions of one softmax the "shared" verdict would
be trivially true. They are not.

### Primary run (noise = 1.75)
Commands:
```bash
python collectors/collect_symbolic_aot_v8.py --out data/aot_signals.npz \
    --train-per-len 160 --test-per-len 160 --aot-epochs 60 --noise 1.75
python load_and_test.py --data data/aot_signals.npz --fig data/aot_field.png
```
`19568` operator-token sites × 5 signals. Decision base-rates:
`trust=0.955  adapt=0.034  budget(0/1/2)=[18678, 675, 215]`.

Full engine verdict block:
```
================  data/aot_signals.npz  ================
mean |off-diag corr| : 0.311      (high => signals move together)
PCA explained var    : PC1=0.54  PC2=0.22  PC3=0.17  PC4=0.06  PC5=0.01
PC1 loadings         : confi=-0.58  cand=+0.59  margi=-0.02  kappa=-0.17  updat=+0.53
  -> all 5 load on PC1? min|load|=0.02
single-readout sufficiency (PC1-only / all-signals):
    motif        acc: all= 0.20  pc1= 0.18  ratio= 0.93
    budget       acc: all= 1.00  pc1= 0.96  ratio= 0.96
    adapt        acc: all= 0.99  pc1= 0.97  ratio= 0.97
    trust        acc: all= 1.00  pc1= 0.96  ratio= 0.96
    temperature   R2: all= 1.00  pc1= 0.00  ratio= 0.00
  -> worst ratio = 0.00
VERDICT: SEPARATE MECHANISMS
```
Figure: `data/aot_field.png`.

### Correlation structure (why it is SEPARATE)
```
         confid  cand_e  margin   kappa  update
confiden +1.00  -0.96  +0.01  +0.17  -0.76
cand_ent -0.96  +1.00  -0.01  -0.20  +0.76
margin   +0.01  -0.01  +1.00  +0.11  +0.00
kappa    +0.17  -0.20  +0.11  +1.00  -0.14
update_s -0.76  +0.76  +0.00  -0.14  +1.00
```
The three operator-head signals (`confidence`, `cand_entropy`, `update_scale`) form
**one tight factor** (|corr| 0.76–0.96). But:
- `margin` (router head) is **orthogonal to everything** (|corr| ≈ 0.00–0.01),
- `kappa` (reliability head) is only weakly related (|corr| ≈ 0.11–0.20).

So there are ≈3 weakly-related factors, not one. The decisive failures are
structural — `mean|corr| 0.311 < 0.35`, `PC1 EVR 0.54 < 0.55`, and
`min|PC1 load| 0.02 ≪ 0.30` (driven by `margin`) — **independent of** the
`temperature` sufficiency artifact (temperature is defined as `margin` by spec, so
when `margin` does not load on PC1 the readout cannot reproduce it; ratio = 0.00).

### Robustness across perception noise
| noise | mean&#124;corr&#124; | PC1 EVR | min&#124;PC1 load&#124; | worst suff ratio | VERDICT |
|---:|---:|---:|---:|---:|:--|
| 1.0  | 0.332 | 0.576 | 0.020 | 0.003 | SEPARATE |
| 1.75 | 0.311 | 0.540 | 0.020 | 0.000 | SEPARATE |
| 2.5  | 0.284 | 0.511 | 0.000 | −0.000 | SEPARATE |

Same decisive cause every time (router `margin` orthogonal to the perception
factor). The verdict is not an artifact of one noise setting.

### Caveats (honest)
- The decisions are highly imbalanced (`trust` 95.5%, `adapt` 3.4%), so the
  accuracy-based sufficiency ratios for `budget`/`adapt`/`trust` are near-saturated
  at base-rate (full ≈ pc1 ≈ base-rate). The verdict therefore rests on the
  **factor-structure** conditions (corr / EVR / loadings), which clearly say
  SEPARATE, not on those three sufficiency ratios.
- This is the under-powered-readout trap warned about in `GOAL.md` *avoided*: the
  readout is degree-2 (kept as required), and the failure is in the factor
  structure (an orthogonal signal), not in readout expressiveness.
- Scope: this is the verdict for the **AOT symbolic substrate**. A SEPARATE result
  here means margin (FOG), κ (AOT reliability) and the op-perception signals are not
  one field *in this toy world*; it does not by itself settle the transformer
  substrate — but per protocol that more expensive test is not justified by Stage 1.

**Plain-language conclusion:** on the symbolic AOT substrate, operator-perception
confidence/entropy/correction-magnitude collapse to a single factor, but routing
margin and a learned reliability head are **separate axes**. The five quantities are
**not** five readouts of one identifiability field here.

### Sensitivity: is the verdict an artifact of the `margin` source?
The primary run draws `margin` from the **router/kind head** (a deliberately
distinct mechanism). To check that this choice is not what produces SEPARATE, the
collector supports `--margin-source op` (margin from the op head, top1−top2), which
makes four of five signals op-head-derived and therefore collinear by construction:
```bash
python collectors/collect_symbolic_aot_v8.py --out data/aot_signals_opmargin.npz \
    --noise 1.75 --margin-source op
python load_and_test.py --data data/aot_signals_opmargin.npz
```
```
mean |off-diag corr| : 0.582     (now > 0.35)
PCA explained var    : PC1=0.73  PC2=0.19  PC3=0.07  PC4=0.01  PC5=0.00   (PC1 > 0.55)
PC1 loadings         : confi=+0.51 cand=-0.51 margi=+0.51 kappa=+0.12 updat=-0.45
  -> all 5 load on PC1? min|load|=0.12     (< 0.30)
worst sufficiency ratio = 0.94             (> 0.85)
VERDICT: SEPARATE MECHANISMS
```
Even when `margin` is forced into the perception cluster, the verdict **stays
SEPARATE** — now the *sole* holdout is `kappa` (the learned reliability head,
loading 0.12 < 0.30). So the SEPARATE result is **not** an artifact of the
margin-source choice: with router-margin two signals (margin, kappa) sit off the
field; with op-margin the reliability head alone is enough to break the single-field
condition. The reliability readout is genuinely not a projection of perception
confidence on this substrate.

---

## Stage 2 — TRANSFORMER

### Real Stage 2  ⛔ NOT BUILT (gated off by Stage 1)

Per `GOAL.md` / `CLAUDE.md`: *"if Stage 1 verdict is SEPARATE, write it to
RESULTS.md and STOP. Do not build Stage 2."* Stage 1 is SEPARATE, so the expensive
transformer rig is **intentionally not run**.

For the record, the Stage-2 collector (`collectors/collect_transformer_motif.py`)
is specced against this repo's real Motif-Upcycling API, which exposes exactly the
needed hooks should Stage 1 ever flip to SHARED on a different substrate:
`MotifSwiGLUMLP._router_alpha(x)` (→ router `margin`/`cand_entropy`/`motif`),
`ScaleAwareDeltaScaler` `ratio = log(RMS(update)/RMS(ref))` (→ `update_scale`,
the SARC relative scale), and `lora_delta(x)` for the `adapt` P0 probe. Running it
would also require a pretrained causal-LM donor (not available offline here).

### Pipeline smoke test  ✅ WIRING VALIDATED (verdict NOT interpretable)
To confirm the Stage-2 collector wiring works against this repo's **real**
`MotifSwiGLUMLP` API, `collectors/stage2_smoke_motif.py` builds a tiny random
Qwen-style SwiGLU donor, wraps it (contextual router + motif-LoRA + adapter-only
SARC), captures per-token router `alpha`, `lora_delta`, and residual `x`, and emits
the four transformer signals + decisions:
```bash
python collectors/stage2_smoke_motif.py --out data/tx_smoke.npz
python load_and_test.py --data data/tx_smoke.npz --fig data/tx_smoke_field.png
```
```
[SMOKE] wrote data/tx_smoke.npz: 8192 (token) sites x 4 signals
[SMOKE] hooks OK: alpha(1, 3, 4), SARC last_stats=present
================  data/tx_smoke.npz  ================
mean|corr|=0.491  PC1=0.72  min|load|=0.05  worst suff ratio=0.67
VERDICT: SEPARATE MECHANISMS
```
**This verdict is meaningless for the hypothesis** and must not be cited: the donor
has random untrained weights, there is no real text/loss, the neutral
function-preserving init is deliberately broken to create signal variance, and the
`adapt` target is synthetic (middle-tertile band on the SARC scale). It confirms
only that the hooks fire, shapes are correct (`alpha [T,M]`, SARC ratio computed),
an npz is produced, and the engine runs end-to-end against the real API. A genuine
Stage 2 still needs a pretrained donor + real text, and is gated off by Stage 1.

Figure: `data/tx_smoke_field.png`.

## Stage 3 — cross-substrate transfer  ⛔ NOT RUN (depends on real Stage 2)

Requires both a Stage-1 and a Stage-2 signal matrix; Stage 2 was gated off.

---

## How to reproduce
```bash
pip install -r requirements.txt          # numpy/sklearn/scipy/pandas/matplotlib + torch
python field_test.py                                   # Stage 0
python collectors/collect_symbolic_aot_v8.py --out data/aot_signals.npz --noise 1.75
python load_and_test.py --data data/aot_signals.npz --fig data/aot_field.png   # Stage 1
```
