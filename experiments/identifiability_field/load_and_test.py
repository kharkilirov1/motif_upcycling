"""Run the field test on a real signals matrix produced by a collector.

Usage:
    python load_and_test.py --data data/aot_signals.npz --fig data/aot_field.png
"""
import argparse
import numpy as np
from field_test import run_field_test


def load(path):
    d = np.load(path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=float)
    names = [str(s) for s in d["signal_names"]]
    decisions = {k[len("dec__"):]: d[k] for k in d.files if k.startswith("dec__")}
    if not decisions:
        raise SystemExit("no decision targets (keys 'dec__*') found in npz — see signal_spec.md")
    return X, names, decisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="npz with X, signal_names, dec__* (see signal_spec.md)")
    ap.add_argument("--fig", default=None, help="optional path to save a correlation/scree figure")
    args = ap.parse_args()

    X, names, decisions = load(args.data)
    print(f"loaded {X.shape[0]} sites x {X.shape[1]} signals: {names}")
    print(f"decisions: {list(decisions)}")
    res = run_field_test(X, decisions, name=args.data, signal_names=names)

    if args.fig:
        # single-panel correlation + scree for real data (figure() in field_test is 2-regime;
        # here we draw a minimal standalone version)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        Xs = StandardScaler().fit_transform(X)
        C = np.corrcoef(Xs.T)
        evr = PCA().fit(Xs).explained_variance_ratio_
        short = [s.split("_")[0][:6] for s in names]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        im = ax[0].imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
        ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels(short, fontsize=8)
        ax[0].set_title(f"signal correlations\nmean|corr|={res['mean_abs_corr']:.2f}", fontsize=9)
        fig.colorbar(im, ax=ax[0], fraction=0.046)
        ax[1].plot(range(1, len(evr) + 1), evr, "o-")
        ax[1].set_ylim(0, 1); ax[1].set_xlabel("PC"); ax[1].set_ylabel("explained var")
        ax[1].set_title(f"scree (PC1={evr[0]:.2f})", fontsize=9)
        fig.tight_layout(); fig.savefig(args.fig, dpi=130, bbox_inches="tight")
        print(f"[figure] {args.fig}")

    verdict = "ONE SHARED FIELD" if res["shared"] else "SEPARATE MECHANISMS"
    print(f"\n>>> VERDICT for {args.data}: {verdict}")


if __name__ == "__main__":
    main()
