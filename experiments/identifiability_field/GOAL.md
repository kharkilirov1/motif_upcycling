# GOAL — Does the "identifiability field" exist?

## The hypothesis (one sentence)
Across these research lines, five separately-named quantities —
**margin** (FOG motif separability), **κ** (AOT repair reliability),
**equilibrium deviation** (ABPT), **SARC relative scale** (Motif-Upcycling),
**candidate-set entropy** (AOT active selection) — are not five mechanisms but
**five readouts of one latent per-site field** that measures *how identified /
how settled* the computation is at a given (token, site). If true, then
confidence, routing, compute-budget, temperature, plasticity-trigger and
self-trust are all projections of this single field.

## What would make it TRUE (two conditions, both required)
1. **Shared factor.** Measured jointly per site, the signals collapse to one
   dominant latent: high mean |correlation|, PC1 explains most variance, and
   *every* signal loads on PC1.
2. **Single-readout sufficiency.** A 1-D readout of that latent can drive *all*
   the downstream decisions (which-motif / budget / adapt? / trust? / temperature)
   about as well as using all signals together. This is what separates a real
   shared cause from coincidental correlation.

## What would make it FALSE (the null — a valid, publishable result)
Signals are near-orthogonal, the scree is flat (every PC ~ equal), loadings are
scattered, or one readout cannot reproduce the decisions. Then they are genuinely
**separate contributions** and should be developed independently. Reaching this
verdict cleanly is a success, not a failure.

## Method note that already bit us (do not forget)
The single-readout test must use a readout head **as expressive as the decision
shape**. A linear head on a 1-D signal cannot represent a band-shaped decision
("adapt only in a *middle* zone of the field") even when the information is fully
present — and that produced a false negative in validation until a degree-2
readout was used. Under-powered readout ⇒ you *under-credit* the field. The engine
already uses a degree-2 readout for both arms; keep it for any new decisions.

## Staged plan (run in this order — earlier stages gate later ones)
- **Stage 0 — engine sanity (no model, seconds).** Run the engine on planted
  synthetic data. It must label a planted single-field "SHARED" and planted
  independent signals "SEPARATE". If it doesn't, fix the engine before anything.
- **Stage 1 — SYMBOLIC gate (CPU + torch, no model download).** Use the AOT code
  (group/marker tasks) to log the real signals + decisions per step, assemble the
  matrix, run the engine. This is the cheap decisive test. **If Stage 1 says
  SEPARATE, STOP and report** — the expensive transformer rig is then pointless.
- **Stage 2 — TRANSFORMER (needs a pretrained model + the Motif-Upcycling impl).**
  Extract per-(token,layer) signals from patched layers (SARC scale, router
  margin/entropy, update-norm ratio) + decisions; run the engine on real text.
- **Stage 3 — cross-substrate (optional, the ambitious one).** Calibrate the field
  on Stage-1 symbolic data, test whether its ranking transfers to predicting useful
  intervention sites in Stage 2. Substrate-independence = the strongest evidence.

## Hard rules
- **Never fabricate numbers.** If a stage cannot run (missing torch, no model, no
  data), say so explicitly in `RESULTS.md` and stop. A blocked stage reported
  honestly is worth more than an invented result.
- Report the verdict per stage verbatim from the engine, plus the figure.
- Correlation is not sufficiency: always report BOTH conditions, not just PCA.
