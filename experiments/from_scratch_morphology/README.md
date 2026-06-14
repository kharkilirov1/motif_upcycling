# From-scratch morphology — does the conventional attention/FFN ratio leave quality on the table?

This experiment answers a critique of the Qwen budget experiments: re-cutting an
already-uniformly-trained model (SVD compression / LoRA adapters) cannot reveal whether
a **natively heterogeneous** architecture is better, because the base's representations
are already shaped by uniform geometry. The honest test is to **train from scratch** at
equal compute and see whether the conventional design is the optimum.

## What it does
Trains tiny GPTs **from scratch** (char-level, wikitext-2) at **equal parameters**,
sweeping how the per-layer inner budget is split between attention and FFN. The budget
`T = 2*d_attn + d_ff` is held fixed, so every point has the same parameter count; the
**standard transformer is one point of the sweep** (`d_attn = d_model`, `d_ff = 4*d_model`
→ attention share `2*d_attn/T = 1/3`). If the val-loss minimum is off the standard point,
the uniform ratio is not compute-optimal — and this is honest because it is a whole
**curve**, not a hand-picked config.

```bash
python from_scratch_split.py --d-attn-grid 24,48,72,96,120,144,168 --steps 1500 --seeds 0,1,2,3
```
Outputs `data/from_scratch_split.png` and `data/from_scratch_split_report.json`.

## Result (4 seeds)
The minimum is at attention share **0.25** (`d_attn=72, d_ff=432`), **not** the standard
**0.33** (`d_attn=96, d_ff=384`); best beats standard by **0.0090** nats, just past
**2·SE = 0.0057**, and the location is reproducible across 2- and 4-seed runs. So at equal
compute, trained from scratch, the conventional attention/FFN ratio is **mildly
suboptimal** — the model wants slightly less attention width and more FFN here.

This is exactly what FOG Theorem 4 / Corollary 3 predicts, and it could **only** be seen
by training from scratch — the Qwen retrofit could not (it tied / only showed post-hoc
compressibility). See `RESULTS.md` for full numbers and honest caveats (the effect is
small at this tiny scale).
