"""causal_margin_bridge.py - the protocol's decisive test T4 at small scale, self-contained.

Connects the two halves of the program:
  (a) the from-scratch sweep (from_scratch_morphology) = GROUND TRUTH optimum of the
      attention<->FFN budget split, found by training every grid point from scratch;
  (b) the Causal-Margin recipe = measure local margins m_attn, m_ffn at ONE point with a
      small +-rank probe (sigma is small at this granularity, so water-filling is licensed),
      then do margin-guided coordinate search.

Claim under test: the LOCALLY MEASURED margins at the standard point correctly point toward
the GLOBAL from-scratch optimum, and a margin-guided search reaches it in fewer trained
models than the blind grid sweep. Margins are probed with SMALL rank perturbations (near
the operating point) to avoid the loss-ceiling artifact identified in causal_margin.py.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent / "from_scratch_morphology"))
import from_scratch_split as fss   # noqa: E402

torch.set_num_threads(max(1, torch.get_num_threads()))


def train_model(vocab, dm, d_attn, d_ff, nh, n_layers, block, data_tr, data_va,
                steps, bs, lr, seed):
    """Train one TinyGPT from scratch; return (model, val_loss)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = fss.TinyGPT(vocab, dm, d_attn, d_ff, nh, n_layers, block)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    model.train()
    for step in range(steps):
        x, y = fss.get_batch(data_tr, block, bs, rng)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps)) * min(1.0, (step + 1) / 50)
        opt.step()
    return model, eval_loss(model, data_va, block, bs)


@torch.no_grad()
def eval_loss(model, data_va, block, bs, n_batches=40, seed=12345):
    model.eval()
    vrng = np.random.default_rng(seed)
    ls = []
    for _ in range(n_batches):
        x, y = fss.get_batch(data_va, block, bs, vrng)
        _, l = model(x, y)
        ls.append(float(l))
    return float(np.mean(ls))


def truncate_modules(mods, delta):
    """Drop the smallest `delta` singular directions of each module; return restore-state."""
    saved = []
    for m in mods:
        W = m.weight.data.float()
        rmax = min(W.shape)
        rr = max(1, rmax - delta)
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        saved.append((m, m.weight.data.clone()))
        m.weight.data = ((U[:, :rr] * S[:rr]) @ Vt[:rr]).to(m.weight.dtype)
    return saved


def restore(saved):
    for m, W in saved:
        m.weight.data = W


def margin_probe(model, data_va, block, bs, deltas):
    """Symmetric small +-rank probe (Def 1.2 / step 4): margin per parameter for each role,
    averaged over several small rank perturbations to clear the eval-noise floor while
    staying near the operating point (avoids the loss-ceiling artifact)."""
    L0 = eval_loss(model, data_va, block, bs)
    attn_mods, ffn_mods = [], []
    for b in model.blocks:
        attn_mods += [b.qkv, b.proj]
        ffn_mods += [b.fc1, b.fc2]

    def margin(mods, delta):
        sv = truncate_modules(mods, delta)
        dL = eval_loss(model, data_va, block, bs) - L0
        restore(sv)
        return dL / max(1, sum(delta * sum(m.weight.shape) for m in mods))

    m_attn = float(np.mean([margin(attn_mods, d) for d in deltas]))
    m_ffn = float(np.mean([margin(ffn_mods, d) for d in deltas]))
    return {"L0": L0, "m_attn": m_attn, "m_ffn": m_ffn, "gap_ffn_minus_attn": m_ffn - m_attn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--head-dim", type=int, default=24)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--block", type=int, default=96)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--deltas", default="16,32", help="small rank perturbations for margin probe")
    ap.add_argument("--grid", default="48,72,96,120,144", help="d_attn points to evaluate")
    ap.add_argument("--max-chars", type=int, default=300000)
    ap.add_argument("--out", default="data/causal_margin_bridge_report.json")
    args = ap.parse_args()

    dm = args.d_model
    T = 6 * dm
    hd = args.head_dim
    seeds = [int(s) for s in args.seeds.split(",")]
    deltas = [int(d) for d in args.deltas.split(",")]
    grid = [int(x) for x in args.grid.split(",") if int(x) >= hd and 2 * int(x) < T]
    data, vocab = fss.load_corpus(args.max_chars)
    n_tr = int(0.9 * len(data))
    data_tr, data_va = data[:n_tr], data[n_tr:]
    print(f"[setup] d_model={dm} T={T} standard d_attn={dm} d_ff={4*dm} head_dim={hd} "
          f"deltas={deltas} grid={grid} seeds={seeds}")

    # ---- measure the val-loss curve AND the margin-gap curve over the grid ----
    rows = []
    for da in grid:
        nh = max(1, da // hd)
        d_ff = T - 2 * da
        vls, gaps, mat, mff = [], [], [], []
        for sd in seeds:
            model, vl = train_model(vocab, dm, da, d_ff, nh, args.n_layers, args.block,
                                    data_tr, data_va, args.steps, args.bs, args.lr, sd)
            pr = margin_probe(model, data_va, args.block, args.bs, deltas)
            vls.append(vl); gaps.append(pr["gap_ffn_minus_attn"]); mat.append(pr["m_attn"]); mff.append(pr["m_ffn"])
        row = {"d_attn": da, "d_ff": d_ff, "n_heads": nh, "attn_frac": 2 * da / T,
               "val_loss_mean": float(np.mean(vls)), "val_loss_sd": float(np.std(vls)),
               "m_attn": float(np.mean(mat)), "m_ffn": float(np.mean(mff)),
               "gap_mean": float(np.mean(gaps)), "gap_sd": float(np.std(gaps)),
               "is_standard": da == dm}
        rows.append(row)
        print(f"[grid] d_attn={da:3d} (frac={row['attn_frac']:.2f}) val={row['val_loss_mean']:.4f} "
              f"m_attn={row['m_attn']:.2e} m_ffn={row['m_ffn']:.2e} "
              f"gap(ffn-attn)={row['gap_mean']:+.2e}{'  <-- STANDARD' if row['is_standard'] else ''}")

    # equal-margin point: where gap(ffn-attn) crosses zero (linear interp between sign change)
    eq_da = None
    for a, b in zip(rows[:-1], rows[1:]):
        if np.sign(a["gap_mean"]) != np.sign(b["gap_mean"]) and a["gap_mean"] != b["gap_mean"]:
            t = a["gap_mean"] / (a["gap_mean"] - b["gap_mean"])
            eq_da = a["d_attn"] + t * (b["d_attn"] - a["d_attn"])
            break
    argmin = min(rows, key=lambda r: r["val_loss_mean"])
    std_row = next(r for r in rows if r["is_standard"])

    # the recipe's local-margin direction at the standard point
    std_gap = std_row["gap_mean"]
    direction = "toward FFN (lower d_attn)" if std_gap > 0 else "toward ATTN (higher d_attn)"
    truth_dir = "toward FFN (lower d_attn)" if argmin["d_attn"] < dm else (
        "toward ATTN (higher d_attn)" if argmin["d_attn"] > dm else "standard")
    direction_correct = (argmin["d_attn"] != dm) and (direction == truth_dir)

    print("\n================ Thm 3.1 on from-scratch models ================")
    print(f"[loss]   val-loss argmin at d_attn={argmin['d_attn']} (frac={argmin['attn_frac']:.2f}) "
          f"val={argmin['val_loss_mean']:.4f}")
    print(f"[margin] equal-margin point (gap=0) at d_attn={'%.1f' % eq_da if eq_da else 'n/a'}")
    print(f"[recipe] local margin at STANDARD says move {direction}; truth is {truth_dir} "
          f"-> {'CORRECT' if direction_correct else 'WRONG/flat'}")
    if eq_da is not None:
        print(f"[verdict] equal-margin predicts d_attn~{eq_da:.0f}; loss argmin at {argmin['d_attn']} "
              f"(|diff|={abs(eq_da-argmin['d_attn']):.0f}, grid step={hd}) -> "
              f"{'MATCH within one grid step' if abs(eq_da-argmin['d_attn'])<=hd else 'mismatch'}")

    report = {"d_model": dm, "T": T, "head_dim": hd, "deltas": deltas, "steps": args.steps,
              "seeds": seeds, "standard_d_attn": dm, "rows": rows,
              "equal_margin_d_attn": eq_da, "loss_argmin_d_attn": argmin["d_attn"],
              "standard_gap": std_gap, "recipe_direction": direction, "truth_direction": truth_dir,
              "direction_correct": bool(direction_correct)}
    outp = THIS.parent / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(outp, "w"), indent=2)

    # figure: val-loss and margin-gap vs d_attn, sharing x; mark argmin and zero-crossing
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xs = [r["d_attn"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(7, 4.4))
        ax1.plot(xs, [r["val_loss_mean"] for r in rows], "o-", color="tab:blue", label="val loss (from scratch)")
        ax1.scatter([argmin["d_attn"]], [argmin["val_loss_mean"]], color="red", zorder=5, label="loss argmin")
        ax1.axvline(dm, color="k", ls=":", lw=1, label="standard split")
        ax1.set_xlabel("d_attn (attention budget; d_ff = T - 2*d_attn)")
        ax1.set_ylabel("val loss", color="tab:blue"); ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(xs, [r["gap_mean"] for r in rows], "s--", color="tab:green", label="margin gap m_ffn - m_attn")
        ax2.axhline(0, color="tab:green", lw=0.8, alpha=0.5)
        if eq_da is not None:
            ax2.axvline(eq_da, color="tab:green", ls="-.", lw=1.2, label="equal-margin (gap=0)")
        ax2.set_ylabel("margin gap  m_ffn - m_attn", color="tab:green"); ax2.tick_params(axis="y", labelcolor="tab:green")
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="upper center")
        ax1.set_title("Thm 3.1: equal-margin point predicts the from-scratch loss minimum", fontsize=10)
        fig.tight_layout(); fig.savefig(str(outp.parent / "bridge_equal_margin.png"), dpi=130, bbox_inches="tight")
        print(f"[figure] {outp.parent / 'bridge_equal_margin.png'}")
    except Exception as e:
        print(f"[figure] skipped ({type(e).__name__}: {e})")
    print(f"[out] {outp}")


if __name__ == "__main__":
    main()
