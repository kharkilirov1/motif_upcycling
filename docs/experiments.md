# Experiment notes

The current repository includes preliminary result CSVs from the Qwen-family runs
reported in the paper. To analyze them:

```bash
python scripts/analyze_histories.py \
  --eval-history results/colab_qwen35_10000_serious_eval_history.csv \
  --router-summary results/colab_qwen35_10000_serious_router_summary.csv
```

Recommended SARC ablations:

1. baseline P4 motif/LoRA;
2. P4S adapter-only, `max_delta=0.05`;
3. P4S adapter-only, `max_delta=0.10`;
4. P4S adapter-only, `max_delta=0.25`;
5. full-update SARC as a riskier ablation;
6. motif-wise SARC if expert-level outputs are available.

Primary metrics:

- best eval mean delta;
- final eval mean delta;
- late degradation;
- per-domain deltas;
- router entropy/utilization;
- update RMS / state RMS;
- SARC scale distribution.
