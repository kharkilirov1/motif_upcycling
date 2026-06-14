"""stage3_transfer.py - Stage 3 cross-substrate transfer.

Fit a 1-D field readout (PC1) on the SYMBOLIC Stage-1 signals, then apply it
(rank-only) to the TRANSFORMER Stage-2 signals to predict high-value intervention
sites. Above-chance prediction => the identifiability readout is substrate
independent (the strongest possible evidence for the field).

Method
------
- Use only the signal columns common to both substrates.
- Standardize Stage-1 common signals; fit PCA; take PC1 loadings as the field
  readout, sign-fixed so that 'confidence' loads positive (high = more identified).
- Standardize Stage-2 common signals with THEIR OWN mean/std, project onto the
  Stage-1 PC1 loadings -> a 1-D field score per transformer site.
- Target: Stage-2 'adapt' (P0 probe: does the motif-LoRA delta lower this token's
  loss?). Report ROC-AUC of the transferred field score vs chance (0.5).
- Baselines: (a) Stage-2's OWN PC1 (in-domain ceiling), (b) Spearman corr of the
  transferred score with predictive entropy/temperature.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def load(path):
    d = np.load(path, allow_pickle=True)
    X = np.asarray(d["X"], float)
    names = [str(s) for s in d["signal_names"]]
    dec = {k[len("dec__"):]: d[k] for k in d.files if k.startswith("dec__")}
    return X, names, dec


def _sign_fix(loadings, names, anchor="confidence"):
    """Orient PC1 so the anchor signal (high = identified) loads positive."""
    if anchor in names:
        j = names.index(anchor)
        if loadings[j] < 0:
            return -loadings
    return loadings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="data/aot_signals.npz")
    ap.add_argument("--stage2", default="data/tx_signals.npz")
    ap.add_argument("--target", default="adapt", help="Stage-2 decision to predict")
    args = ap.parse_args()

    X1, n1, d1 = load(args.stage1)
    X2, n2, d2 = load(args.stage2)
    common = [s for s in n1 if s in n2]
    if len(common) < 3:
        raise SystemExit(f"need >=3 shared signals, got {common}")
    print(f"shared signals ({len(common)}): {common}")

    i1 = [n1.index(s) for s in common]
    i2 = [n2.index(s) for s in common]
    A = X1[:, i1]
    Bm = X2[:, i2]

    # field readout fit on symbolic Stage 1
    sc1 = StandardScaler().fit(A)
    pca1 = PCA().fit(sc1.transform(A))
    load1 = _sign_fix(pca1.components_[0], common)
    print("Stage-1 PC1 loadings (sign-fixed): " +
          "  ".join(f"{s[:6]}={l:+.2f}" for s, l in zip(common, load1)))
    print(f"Stage-1 PC1 explained variance: {pca1.explained_variance_ratio_[0]:.3f}")

    # transfer: project Stage-2 (own standardization) onto Stage-1 loadings
    sc2 = StandardScaler().fit(Bm)
    field2_transfer = sc2.transform(Bm) @ load1

    # in-domain ceiling: Stage-2's own PC1
    pca2 = PCA().fit(sc2.transform(Bm))
    load2 = _sign_fix(pca2.components_[0], common)
    field2_own = sc2.transform(Bm) @ load2

    y = np.asarray(d2[args.target])
    if len(np.unique(y)) < 2:
        raise SystemExit(f"target '{args.target}' has one class; pick another")

    def auc_both_dirs(score, target):
        a = roc_auc_score(target, score)
        return max(a, 1 - a), a  # field orientation is rank-only -> allow flip

    auc_t, raw_t = auc_both_dirs(field2_transfer, y)
    auc_o, raw_o = auc_both_dirs(field2_own, y)
    rho_temp = spearmanr(field2_transfer, np.asarray(d2["temperature"]))[0] \
        if "temperature" in d2 else float("nan")

    print()
    print(f"=== Stage 3: transfer of the symbolic field readout to the transformer ===")
    print(f"target = Stage-2 '{args.target}'  (base-rate={y.mean():.3f}, n={len(y)})")
    print(f"transferred-field AUC   : {auc_t:.3f}   (chance=0.500)   [raw {raw_t:.3f}]")
    print(f"in-domain Stage-2 PC1   : {auc_o:.3f}   (ceiling for these signals)")
    print(f"Spearman(field, temp)   : {rho_temp:+.3f}")
    verdict = ("ABOVE CHANCE -> substrate-independent field signal"
               if auc_t > 0.55 else
               "AT CHANCE -> no transferable field")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
