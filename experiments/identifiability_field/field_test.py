"""
field_test.py  —  Does the "identifiability field" exist, or are these 5 separate mechanisms?

This implements the DECISIVE cheap test discussed:
  signals  = [equilibrium_dev, sarc_scale, margin, kappa, cand_entropy]  measured per (token, site)
  question = are they projections of ONE latent field, or 5 independent things?

Two parts:
  (1) factor structure  : corr matrix + PCA. One shared latent => high |corr|, PC1 dominates,
                          every signal loads on PC1.
  (2) single-readout test: can a 1-D readout (PC1) drive ALL the downstream decisions
                          (which-motif / budget / adapt? / trust? / temperature) about as well
                          as using all 5 signals? If yes => the field is sufficient, not just
                          correlated. This is the part that separates "shared cause" from
                          "coincidentally correlated".

IMPORTANT: the numbers this script prints are from PLANTED SYNTHETIC ground truth.
They prove the TEST has power (it says "shared" when there is one field, "separate" when there
isn't). They say NOTHING about your real system. To test your hypothesis, replace the synthetic
matrices with your real measured signals via run_field_test(X, decisions).
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIGNALS = ["equilibrium_dev", "sarc_scale", "margin", "kappa", "cand_entropy"]
DECISIONS_INFO = "motif(3-cls) budget(3-lvl) adapt?(bin) trust?(bin) temperature(reg)"


# ----------------------------------------------------------------------------- ground truth
def _decisions_from_latents(Z, shared):
    """5 decisions the separate mechanisms supposedly drive. shared=> all from one latent."""
    idx = [0, 0, 0, 0, 0] if shared else [0, 1, 2, 3, 4]
    return {
        "motif":       np.digitize(Z[:, idx[0]], [-0.6, 0.6]),     # which operator role
        "budget":      np.digitize(Z[:, idx[1]], [-0.4, 0.4]),     # compute budget level
        "adapt":      ((Z[:, idx[2]] > -0.3) & (Z[:, idx[2]] < 0.8)).astype(int),  # adapt?
        "trust":       (Z[:, idx[3]] > 0.0).astype(int),           # trust repair?
        "temperature": Z[:, idx[4]],                               # sampling temperature
    }

def make_one_field(n, rng, snr=2.0):
    """ONE latent field z drives all signals (different monotone maps) and all decisions."""
    z = rng.standard_normal(n)
    raw = np.column_stack([
        -1.0 * z,                                  # equilibrium_dev (high = unsettled)
        -0.9 * np.tanh(z),                         # sarc_scale (big correction when unsettled)
        1.2 * z,                                   # margin (high = identified)
        np.tanh(1.5 * z),                          # kappa (reliability)
        -0.8 * z + 0.2 * (z**2 - 1),               # cand_entropy (falls as settled; mild nonlin)
    ])
    X = raw + rng.standard_normal(raw.shape) * (raw.std(0, keepdims=True) / snr)
    return X, _decisions_from_latents(np.column_stack([z] * 5), shared=True)

def make_independent(n, rng, snr=2.0):
    """FIVE independent latents -> 5 unrelated signals and decisions tied to different latents."""
    Z = rng.standard_normal((n, 5))
    raw = np.column_stack([
        -1.0 * Z[:, 0],
        -0.9 * np.tanh(Z[:, 1]),
        1.2 * Z[:, 2],
        np.tanh(1.5 * Z[:, 3]),
        -0.8 * Z[:, 4],
    ])
    X = raw + rng.standard_normal(raw.shape) * (raw.std(0, keepdims=True) / snr)
    return X, _decisions_from_latents(Z, shared=False)


# ----------------------------------------------------------------------------- the test
def run_field_test(X, decisions, name="data", signal_names=None, verbose=True):
    """X: (n_sites, n_signals).  decisions: dict name->target array (categorical or continuous)."""
    names = list(signal_names) if signal_names is not None else SIGNALS
    Xs = StandardScaler().fit_transform(X)
    C = np.corrcoef(Xs.T)
    mean_abs_corr = float(np.mean(np.abs(C[~np.eye(len(C), dtype=bool)])))

    pca = PCA().fit(Xs)
    evr = pca.explained_variance_ratio_
    pc1_load = pca.components_[0]
    Z1 = pca.transform(Xs)[:, 0:1]            # the candidate 1-D field readout

    # readouts use a degree-2 expansion so non-monotone decisions (e.g. "adapt in a band")
    # are representable; SAME head class for all-signals and PC1-only => fair sufficiency ratio.
    def _clf(): return make_pipeline(PolynomialFeatures(2, include_bias=False),
                                     LogisticRegression(max_iter=2000))
    def _reg(): return make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge())
    suff = []  # sufficiency: PC1-only vs all-signals, per decision
    for k, y in decisions.items():
        if np.issubdtype(np.asarray(y).dtype, np.floating) and len(np.unique(y)) > 10:
            full = cross_val_score(_reg(), Xs, y, cv=5, scoring="r2").mean()
            one  = cross_val_score(_reg(), Z1, y, cv=5, scoring="r2").mean()
            m = "R2"
        else:
            full = cross_val_score(_clf(), Xs, y, cv=5).mean()
            one  = cross_val_score(_clf(), Z1, y, cv=5).mean()
            m = "acc"
        suff.append((k, m, full, one, (one / full) if full > 1e-6 else float("nan")))

    ratios = [r for *_, r in suff if not np.isnan(r)]
    suff_min = float(np.min(ratios))
    # verdict thresholds (deliberately simple / transparent)
    shared = (mean_abs_corr > 0.35) and (evr[0] > 0.55) and (np.min(np.abs(pc1_load)) > 0.30) and (suff_min > 0.85)

    if verbose:
        print(f"\n================  {name}  ================")
        print(f"mean |off-diag corr| : {mean_abs_corr:.3f}      (high => signals move together)")
        print(f"PCA explained var    : " + "  ".join(f"PC{i+1}={v:.2f}" for i, v in enumerate(evr)))
        print(f"PC1 loadings         : " + "  ".join(f"{s.split('_')[0][:5]}={l:+.2f}"
                                                      for s, l in zip(names, pc1_load)))
        print(f"  -> all 5 load on PC1? min|load|={np.min(np.abs(pc1_load)):.2f}")
        print(f"single-readout sufficiency (PC1-only / all-signals):")
        for k, m, full, one, r in suff:
            print(f"    {k:12s} {m:>3s}: all={full:5.2f}  pc1={one:5.2f}  ratio={r:5.2f}")
        print(f"  -> worst ratio = {suff_min:.2f}")
        print(f"VERDICT: {'ONE SHARED FIELD' if shared else 'SEPARATE MECHANISMS'}")
    return dict(name=name, mean_abs_corr=mean_abs_corr, evr=evr, pc1_load=pc1_load,
                sufficiency=suff, suff_min=suff_min, shared=shared, corr=C)


def figure(res_a, res_b, path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    short = [s.split("_")[0][:5] for s in SIGNALS]
    for a, res, ttl in [(ax[0], res_a, "A: one field (planted)"),
                        (ax[1], res_b, "B: independent (planted)")]:
        im = a.imshow(res["corr"], vmin=-1, vmax=1, cmap="RdBu_r")
        a.set_xticks(range(5)); a.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        a.set_yticks(range(5)); a.set_yticklabels(short, fontsize=8)
        a.set_title(f"{ttl}\nmean|corr|={res['mean_abs_corr']:.2f}", fontsize=9)
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].plot(range(1, 6), res_a["evr"], "o-", label="A: one field")
    ax[2].plot(range(1, 6), res_b["evr"], "s--", label="B: independent")
    ax[2].set_xlabel("principal component"); ax[2].set_ylabel("explained variance")
    ax[2].set_title("scree: PC1 dominates only\nif there is one field", fontsize=9)
    ax[2].legend(fontsize=8); ax[2].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\n[figure] {path}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 4000
    Xa, da = make_one_field(n, rng)
    Xb, db = make_independent(n, rng)
    ra = run_field_test(Xa, da, name="REGIME A: planted single field")
    rb = run_field_test(Xb, db, name="REGIME B: planted independent")
    figure(ra, rb, "data/field_test_power.png")
    print("\n--- power check ---")
    print(f"A detected as shared?      {ra['shared']}   (want True)")
    print(f"B detected as separate?    {not rb['shared']}   (want True)")
    print("\nTo test YOUR system: build X=(n_sites x 5) from your logged signals aligned at a")
    print("common (token,layer) index, build the 5 decision targets, then call run_field_test(X, decisions).")
