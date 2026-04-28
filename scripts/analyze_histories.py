#!/usr/bin/env python
"""Analyze train/eval/router CSV histories from a motif-upcycling run.

This is intentionally lightweight and accepts the CSVs produced by the Colab
experiments referenced in the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-history", type=Path, required=True)
    parser.add_argument("--router-summary", type=Path, default=None)
    args = parser.parse_args()

    eval_df = pd.read_csv(args.eval_history)
    print("eval columns:", list(eval_df.columns))

    numeric = eval_df.select_dtypes(include="number")
    if "mean_delta_vs_baseline" in numeric.columns:
        best_idx = numeric["mean_delta_vs_baseline"].idxmax()
        print("best row:")
        print(eval_df.loc[best_idx].to_string())
    else:
        print(numeric.describe().to_string())

    if args.router_summary is not None and args.router_summary.exists():
        router = pd.read_csv(args.router_summary)
        print("\nrouter columns:", list(router.columns))
        print(router.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
