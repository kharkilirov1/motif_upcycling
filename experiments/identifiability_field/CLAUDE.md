# CLAUDE.md — instructions for Claude Code

You are running an experiment defined in `GOAL.md`. Read it first. Your job is to
get a **verdict** ("ONE SHARED FIELD" vs "SEPARATE MECHANISMS") at each stage that
can run, and write `RESULTS.md`. Do not fabricate; honestly report blocked stages.

## 0. Setup
```bash
pip install -r requirements.txt          # numpy/sklearn/scipy/pandas/matplotlib (always)
# torch is required for Stage 1+. transformers/datasets only for Stage 2.
python field_test.py                     # STAGE 0 sanity: must print SHARED for A, SEPARATE for B
```
If Stage 0's power check does not print `A ... True` and `B ... True`, fix the
engine before continuing.

## 1. Stage 1 — symbolic gate (the cheap decisive test)
The AOT reliability project is the substrate. Obtain it (the user has
`hma_aot_v31_v34_reliability_project.zip`; unzip so `hma_aot_v31/` is importable).
Then complete and run the collector:
```bash
python collectors/collect_symbolic_aot.py --aot-root /path/to/hma_aot_v31_v34_project \
       --out data/aot_signals.npz --n-seq 4000 --noise 1.75
python load_and_test.py --data data/aot_signals.npz --fig data/aot_field.png
```
`collect_symbolic_aot.py` has the exact hook points marked `# >>> WIRE`. They
reference real objects in their code: `StepObservation.features`, `active_probs`,
`op_probs`, `reliability_score(...)`, `counterfactual_action_scores(...)`,
`ACTIONS`. Verify each against the actual module (signatures may have drifted) and
fill the assembly. Each row = one event/step; align all signals at that same step.

**Decision rule:** if Stage 1 verdict is SEPARATE, write it to `RESULTS.md` and
STOP. Do not build Stage 2.

## 2. Stage 2 — transformer (only if Stage 1 says SHARED)
Needs (a) a pretrained causal LM and (b) the user's Motif-Upcycling implementation
(NOT in the provided files — ask the user for the repo or the patched-layer module).
Complete `collectors/collect_transformer_motif.py` (hooks marked `# >>> WIRE`),
which extracts per-(token,layer) signals at patched layers from `signal_spec.md`.
```bash
python collectors/collect_transformer_motif.py --model <hf-or-local-path> \
       --motif-impl /path/to/motif_upcycling --out data/tx_signals.npz
python load_and_test.py --data data/tx_signals.npz --fig data/tx_field.png
```

## 3. Stage 3 — cross-substrate transfer (optional)
Fit a 1-D field readout on `data/aot_signals.npz`; apply (rank-only) to
`data/tx_signals.npz` to predict high-value intervention sites; report rank
correlation / AUC vs chance. Above chance ⇒ substrate-independent field.

## Output
Write `RESULTS.md` containing, for every stage attempted: the command, the engine's
full verdict block (mean|corr|, PCA EVR, PC1 loadings, sufficiency table, VERDICT),
the figure path, and a one-line plain-language conclusion. For blocked stages, state
exactly what was missing.

## Reminders
- Keep the degree-2 readout (see GOAL.md "method note"); never downgrade to linear.
- The data contract for `load_and_test.py` is in `signal_spec.md` (npz layout).
- If unsure whether a signal/decision exists in the user's code, inspect the module
  and adapt rather than guessing silently; note any substitution in `RESULTS.md`.
