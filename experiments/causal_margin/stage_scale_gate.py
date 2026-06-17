"""stage_scale_gate.py - the scale gate of the causal-margin plan.

Stage 0 showed that at d_model=96 the trained val-loss is FLAT across the attention/FFN
split (no significant prize). Before any O(m)-vs-sweep efficiency test is meaningful we
must know whether a SIGNIFICANT split-optimum EMERGES WITH SCALE. This trains the split
sweep at several model sizes and asks one question:

    does the effect size  gap(best - standard) / pooled-SE  grow with d_model
    and cross significance?

If yes -> the prize is real once the model is capacity-bound -> proceed to Stage 2.
If it stays within noise at every reachable scale -> the practical prize is illusory and
the strong (cheap closed-form) version is effectively dead, leaving only the framework.

Same equal-params testbed as from_scratch_morphology: T = 6*d_model, sweep attn_frac =
2*d_attn/T at fixed T (every split has identical non-embedding params).
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

CACHE_PATH = THIS.parent / "data" / "scale_gate_cache.json"


def load_cache():
    if CACHE_PATH.exists():
        try:
            return json.load(open(CACHE_PATH))
        except Exception:
            return {}
    return {}


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    json.dump(cache, open(tmp, "w"))
    tmp.replace(CACHE_PATH)   # atomic: a reap mid-write cannot corrupt the cache


def train_val(vocab, dm, d_attn, d_ff, nh, n_layers, block, data_tr, data_va, steps, bs, lr, seed):
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
    model.eval()
    vrng = np.random.default_rng(12345)
    ls = []
    with torch.no_grad():
        for _ in range(50):
            x, y = fss.get_batch(data_va, block, bs, vrng)
            _, l = model(x, y)
            ls.append(float(l))
    return float(np.mean(ls))


def main():
    ap = argparse.ArgumentParser()
    # rungs: "d_model:head_dim:steps" comma-separated
    ap.add_argument("--rungs", default="96:24:1500,160:40:2000,224:56:2400")
    ap.add_argument("--fracs", default="0.167,0.25,0.333,0.417,0.5")
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-chars", type=int, default=1500000)
    ap.add_argument("--out", default="data/scale_gate_report.json")
    args = ap.parse_args()

    fracs = [float(x) for x in args.fracs.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    rungs = []
    for r in args.rungs.split(","):
        dm, hd, st = r.split(":"); rungs.append((int(dm), int(hd), int(st)))

    data, vocab = fss.load_corpus(args.max_chars)
    n_tr = int(0.9 * len(data)); data_tr, data_va = data[:n_tr], data[n_tr:]
    print(f"[data] chars={len(data)} vocab={vocab}  rungs={rungs} fracs={fracs} seeds={seeds}")

    cache = load_cache()

    def cached_train(da, d_ff, nh, steps, sd):
        # key pins everything that changes the trained val-loss, so reusing across reaps is safe
        key = f"mc{args.max_chars}|v{vocab}|dm{dm}|da{da}|dff{d_ff}|nh{nh}|nl{args.n_layers}|" \
              f"blk{args.block}|bs{args.bs}|lr{args.lr}|st{steps}|sd{sd}"
        if key in cache:
            return cache[key]
        v = train_val(vocab, dm, da, d_ff, nh, args.n_layers, args.block,
                      data_tr, data_va, steps, args.bs, args.lr, sd)
        cache[key] = v; save_cache(cache)   # persist after every train -> reap-proof
        return v

    summary = []
    for dm, hd, steps in rungs:
        T = 6 * dm
        std_da = dm                                       # standard: attn_frac = 1/3
        pts = {}
        for fr in fracs:
            da = int(round(fr * T / 2 / hd)) * hd
            da = max(hd, da)
            if 2 * da >= T:
                continue
            nh = max(1, da // hd); d_ff = T - 2 * da
            vls = [cached_train(da, d_ff, nh, steps, sd) for sd in seeds]
            pts[da] = {"attn_frac": 2 * da / T, "mean": float(np.mean(vls)),
                       "se": float(np.std(vls) / math.sqrt(len(vls))), "per_seed": vls}
            print(f"[d={dm} steps={steps}] d_attn={da:3d} frac={pts[da]['attn_frac']:.2f} "
                  f"val={pts[da]['mean']:.4f} +-{pts[da]['se']:.4f}")
        # nearest trained point to the standard split
        std_key = min(pts, key=lambda d: abs(d - std_da))
        best = min(pts, key=lambda d: pts[d]["mean"])
        gap = pts[std_key]["mean"] - pts[best]["mean"]
        pooled_se = math.sqrt(pts[std_key]["se"] ** 2 + pts[best]["se"] ** 2)
        eff = gap / pooled_se if pooled_se > 0 else 0.0
        sig = (best != std_key) and (gap > 2 * pooled_se)
        rung = {"d_model": dm, "head_dim": hd, "steps": steps, "T": T,
                "points": {str(k): v for k, v in pts.items()},
                "standard_d_attn": std_key, "best_d_attn": best,
                "best_frac": pts[best]["attn_frac"], "gap": gap, "pooled_se": pooled_se,
                "effect_size_gap_over_se": eff, "significant": bool(sig)}
        summary.append(rung)
        print(f"  -> d={dm}: best frac={pts[best]['attn_frac']:.2f} gap={gap:.4f} "
              f"effect={eff:.2f}*SE significant={sig}\n")

    effs = [r["effect_size_gap_over_se"] for r in summary]
    dms = [r["d_model"] for r in summary]
    rising = all(effs[i] <= effs[i + 1] + 1e-9 for i in range(len(effs) - 1))
    any_sig = any(r["significant"] for r in summary)
    if any_sig and rising:
        verdict = (f"PRIZE EMERGES WITH SCALE: effect size {['%.2f'%e for e in effs]} (in SE units) "
                   f"rises with d_model {dms} and crosses significance -> the split optimum is real "
                   f"once capacity-bound -> Stage 2 (O(m) vs sweep) is warranted.")
    elif any_sig:
        verdict = (f"SIGNIFICANT AT SOME SCALE but not monotone in d_model: effects {['%.2f'%e for e in effs]} "
                   f"for {dms}. Prize exists but scale-dependence is noisy; needs more seeds/scales.")
    else:
        verdict = (f"STILL FLAT AT ALL SCALES: effect size {['%.2f'%e for e in effs]} (SE units) for "
                   f"d_model {dms} never crosses significance. The attention/FFN split does not move "
                   f"trained val-loss within reach -> the practical prize is illusory; the strong "
                   f"cheap-recipe version is effectively dead at these scales (framework survives).")
    print(f"VERDICT: {verdict}")

    report = {"rungs": rungs, "fracs": fracs, "seeds": seeds, "max_chars": args.max_chars,
              "vocab": vocab, "summary": summary, "effect_sizes": effs, "d_models": dms,
              "rising": bool(rising), "any_significant": bool(any_sig), "verdict": verdict}
    outp = THIS.parent / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(outp, "w"), indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
        for r in summary:
            xs = sorted(int(k) for k in r["points"])
            fr = [r["points"][str(x)]["attn_frac"] for x in xs]
            ys = [r["points"][str(x)]["mean"] for x in xs]
            es = [r["points"][str(x)]["se"] for x in xs]
            a1.errorbar(fr, ys, yerr=es, fmt="o-", label=f"d_model={r['d_model']}")
        a1.axvline(1/3, color="k", ls=":", lw=1, label="standard (1/3)")
        a1.set_xlabel("attention share 2*d_attn/T"); a1.set_ylabel("val loss")
        a1.set_title("split sweep per scale"); a1.legend(fontsize=7); a1.grid(alpha=0.3)
        a2.plot(dms, effs, "o-", color="tab:purple")
        a2.axhline(2.0, color="red", ls="--", label="significance (2*SE)")
        a2.set_xlabel("d_model"); a2.set_ylabel("effect size: gap(best-standard) / SE")
        a2.set_title("does the prize grow with scale?"); a2.legend(fontsize=8); a2.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(str(outp.parent / "scale_gate.png"), dpi=130, bbox_inches="tight")
        print(f"[figure] {outp.parent / 'scale_gate.png'}")
    except Exception as e:
        print(f"[figure] skipped ({type(e).__name__}: {e})")
    print(f"[out] {outp}")


if __name__ == "__main__":
    main()
