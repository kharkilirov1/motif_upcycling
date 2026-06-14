"""real_motif_budget.py - FOG Theorem 4 on REAL Transformer motifs (Qwen2.5-0.5B).

The synthetic test (`run_het_budget.py`) showed heterogeneous budget allocation beats
uniform when motif error-vs-budget curves differ. This script asks the same question
for the ACTUAL motifs of a pretrained Transformer, with no synthetic data.

Method
------
* Motif = one projection module per layer: attention q_proj/o_proj ("compare"),
  MLP gate_proj ("select"), up_proj ("expand"), down_proj ("memory/compress").
* eps_i(B) is MEASURED, not assumed: take each weight W, SVD it, and for budget = rank
  r measure the *functional* reconstruction error on REAL token activations x:
      eps_i(r) = E_x|| (W - W_r) x ||^2 / E_x|| W x ||^2 .
  Different motif types have different singular spectra => different curves. (For SVD
  truncation in Frobenius norm the curve is exactly convex-decreasing; the activation-
  weighted version is measured directly.)
* Budget = total low-rank parameters  sum_i r_i*(out_i + in_i).
* Compare, at EQUAL total budget:
    - UNIFORM  : the same rank r0 on every module (a uniform bottleneck / interface),
    - OPTIMAL  : exact DP allocation minimising total functional error,
    - RANDOM   : random feasible allocations.
* Decisive end-metric: replace each module's weight by its rank-allocated reconstruction
  and measure end-to-end LM loss on held-out wikitext under UNIFORM vs OPTIMAL at the
  SAME total parameters. Lower loss for OPTIMAL => Theorem 4 holds for real motifs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(max(1, torch.get_num_threads()))
THIS = Path(__file__).resolve()


# --------------------------------------------------------------------- data
def get_eval_texts(n_chunks: int = 400):
    for ident in ("Salesforce/wikitext", "wikitext"):
        try:
            from datasets import load_dataset
            ds = load_dataset(ident, "wikitext-2-raw-v1", split="train")
            texts = [t for t in ds["text"][:8000] if len(t.strip()) > 80]
            if len(texts) > 50:
                return texts[:n_chunks], f"{ident}:wikitext-2-raw-v1"
        except Exception as e:
            print(f"[real] {ident} unavailable ({type(e).__name__})")
    return None, None


def encode(tok, texts, seq_len, n_seq, seed):
    ids = []
    for t in texts:
        e = tok(t, return_tensors="pt").input_ids[0]
        if e.numel() >= 8:
            ids.append(e)
    cat = torch.cat(ids)
    n = cat.numel() // seq_len
    cat = cat[: n * seq_len].view(n, seq_len)
    idx = np.random.default_rng(seed).permutation(n)[:n_seq]
    return cat[idx]


# --------------------------------------------------------------------- motifs
TYPE_PATHS = {
    "q": ("self_attn", "q_proj"), "o": ("self_attn", "o_proj"),
    "gate": ("mlp", "gate_proj"), "up": ("mlp", "up_proj"), "down": ("mlp", "down_proj"),
}


def get_module(model, layer, mtype):
    sub, name = TYPE_PATHS[mtype]
    return getattr(getattr(model.model.layers[layer], sub), name)


# --------------------------------------------------------------------- curves
def measure_curves(model, modules, calib_ids, ranks, max_rows=3000):
    """For each module: capture real input activations, SVD the weight, and return
    eps(r) (functional reconstruction error) and cost(r) (=r*(out+in))."""
    cache: Dict[str, List[torch.Tensor]] = {k: [] for k in modules}
    handles = []

    def mk(key):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            cache[key].append(x)
        return hook

    for key, mod in modules.items():
        handles.append(mod.register_forward_hook(mk(key)))
    with torch.no_grad():
        for i in range(calib_ids.shape[0]):
            model(calib_ids[i:i + 1])
    for h in handles:
        h.remove()

    curves = {}
    for key, mod in modules.items():
        X = torch.cat(cache[key], 0).float()
        if X.shape[0] > max_rows:
            sel = torch.randperm(X.shape[0])[:max_rows]
            X = X[sel]
        W = mod.weight.data.float()                      # [out, in]
        FO = X @ W.t()                                    # [N, out]
        fo_norm = float((FO ** 2).sum()) + 1e-9
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        out_f, in_f = W.shape
        rmax = min(out_f, in_f)
        eps, cost = {}, {}
        for r in ranks:
            rr = min(r, rmax)
            Wr = (U[:, :rr] * S[:rr]) @ Vt[:rr]
            err = float(((X @ (W - Wr).t()) ** 2).sum()) / fo_norm
            eps[r] = err
            cost[r] = rr * (out_f + in_f)
        curves[key] = {"eps": eps, "cost": cost, "shape": (out_f, in_f),
                       "type": key.split(":")[1]}
        print(f"[curve] {key:14s} shape={tuple(W.shape)}  " +
              "  ".join(f"r{r}={eps[r]:.3f}" for r in ranks))
    return curves


# --------------------------------------------------------------------- allocation
def alloc_uniform(curves, ranks, r0) -> Dict[str, int]:
    return {k: r0 for k in curves}


def total_cost(curves, alloc) -> int:
    return int(sum(curves[k]["cost"][alloc[k]] for k in alloc))


def total_eps(curves, alloc) -> float:
    return float(sum(curves[k]["eps"][alloc[k]] for k in alloc))


def alloc_optimal(curves, ranks, budget, buckets=400) -> Dict[str, int]:
    """Exact DP: minimise sum eps_i(r_i) s.t. sum cost_i(r_i) <= budget.
    Costs are quantised to `buckets` levels. Uniform is feasible, so optimal <= uniform."""
    keys = list(curves)
    unit = max(1, budget // buckets)
    INF = float("inf")
    dp = [INF] * (buckets + 1)
    dp[0] = 0.0
    back: List[Dict[int, Tuple[int, int]]] = [dict() for _ in range(len(keys))]
    # dp over modules
    cur = [INF] * (buckets + 1); cur[0] = 0.0
    choice = [[-1] * (buckets + 1) for _ in keys]
    for mi, k in enumerate(keys):
        nxt = [INF] * (buckets + 1)
        nchoice = [-1] * (buckets + 1)
        for b in range(buckets + 1):
            if cur[b] == INF:
                continue
            for r in ranks:
                cb = b + int(round(curves[k]["cost"][r] / unit))
                if cb > buckets:
                    continue
                val = cur[b] + curves[k]["eps"][r]
                if val < nxt[cb]:
                    nxt[cb] = val
                    nchoice[cb] = (b, r)
        cur = nxt
        choice[mi] = nchoice
    # best bucket <= budget
    bestb = min(range(buckets + 1), key=lambda b: cur[b])
    # backtrack
    alloc = {}
    b = bestb
    for mi in reversed(range(len(keys))):
        pb, r = choice[mi][b]
        alloc[keys[mi]] = r
        b = pb
    return alloc


def alloc_random(curves, ranks, budget, k, seed) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    keys = list(curves)
    errs = []
    for _ in range(k):
        a = {key: int(rng.choice(ranks)) for key in keys}
        while total_cost(curves, a) > budget:
            big = max(keys, key=lambda key: a[key])
            gi = ranks.index(a[big])
            if gi == 0:
                break
            a[big] = ranks[gi - 1]
        errs.append(total_eps(curves, a))
    return float(np.mean(errs)), float(np.min(errs))


# --------------------------------------------------------------------- end metric
def apply_alloc(model, curves, svd_cache, alloc):
    """Replace each module weight by its rank-r reconstruction (returns originals)."""
    saved = {}
    for key, r in alloc.items():
        mod = curves[key]["_mod"]
        U, S, Vt, rmax = svd_cache[key]
        rr = min(r, rmax)
        Wr = ((U[:, :rr] * S[:rr]) @ Vt[:rr]).to(mod.weight.dtype)
        saved[key] = mod.weight.data
        mod.weight.data = Wr
    return saved


def restore(curves, saved):
    for key, W in saved.items():
        curves[key]["_mod"].weight.data = W


@torch.no_grad()
def lm_loss(model, eval_ids) -> float:
    tot, ntok = 0.0, 0
    for i in range(eval_ids.shape[0]):
        ids = eval_ids[i:i + 1]
        out = model(ids)
        l = F.cross_entropy(out.logits[0, :-1], ids[0, 1:], reduction="sum")
        tot += float(l); ntok += ids.shape[1] - 1
    return tot / max(1, ntok)


def make_figure(curves, ranks, rows, types, full_loss, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # panel 1: eps(r) per motif type (averaged over layers)
    for t in types:
        ys = []
        for r in ranks:
            vals = [curves[k]["eps"][r] for k in curves if curves[k]["type"] == t]
            ys.append(float(np.mean(vals)))
        ax[0].plot(ranks, ys, "o-", label=t)
    ax[0].set_xscale("log"); ax[0].set_xlabel("rank (budget)"); ax[0].set_ylabel("functional error eps(r)")
    ax[0].set_title("real Qwen motifs have DIFFERENT\nerror-vs-budget curves", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # panel 2: end-to-end loss uniform vs optimal
    B = [r["budget_params"] / 1e6 for r in rows]
    ax[1].axhline(full_loss, color="k", ls=":", label="full model")
    ax[1].plot(B, [r["uniform_loss"] for r in rows], "o-", label="uniform bottleneck")
    ax[1].plot(B, [r["optimal_loss"] for r in rows], "s-", label="optimal (heterogeneous)")
    ax[1].set_xlabel("total budget (M params)"); ax[1].set_ylabel("held-out LM loss")
    ax[1].set_title("equal budget: optimal allocation\nloses less LM loss", fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    # panel 3: optimal rank by type (heterogeneity), largest budget
    last = rows[-1]["optimal_rank_by_type"]
    ax[2].bar(range(len(types)), [last[t] for t in types])
    ax[2].set_xticks(range(len(types))); ax[2].set_xticklabels(types)
    ax[2].set_ylabel("mean optimal rank"); ax[2].set_title("optimal budget is heterogeneous\nacross motif types", fontsize=10)
    ax[2].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"[figure] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="6,8,10,12,14,16")
    ap.add_argument("--types", default="q,o,gate,up,down")
    ap.add_argument("--ranks", default="4,8,16,32,64,128,192,256")
    ap.add_argument("--uniform-ranks", default="16,32,64,128")  # budget levels
    ap.add_argument("--calib-seq", type=int, default=16)
    ap.add_argument("--eval-seq", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--out", default="data/real_motif_budget_report.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    layers = [int(x) for x in args.layers.split(",")]
    types = args.types.split(",")
    ranks = [int(x) for x in args.ranks.split(",")]
    uniform_levels = [int(x) for x in args.uniform_ranks.split(",")]

    print(f"[real] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    texts, src = get_eval_texts()
    if texts is None:
        raise SystemExit("no wikitext available; cannot run real Stage")
    print(f"[real] text: {src}")
    all_ids = encode(tok, texts, args.seq_len, args.calib_seq + args.eval_seq, args.seed)
    calib_ids, eval_ids = all_ids[:args.calib_seq], all_ids[args.calib_seq:]

    modules = {f"L{L}:{t}": get_module(model, L, t) for L in layers for t in types}
    curves = measure_curves(model, modules, calib_ids, ranks)
    for key in curves:
        curves[key]["_mod"] = modules[key]

    # svd cache for end-metric reconstruction
    svd_cache = {}
    for key, mod in modules.items():
        W = mod.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        svd_cache[key] = (U, S, Vt, min(W.shape))

    full_loss = lm_loss(model, eval_ids)
    print(f"[real] full-model held-out LM loss = {full_loss:.4f}")

    rows = []
    for r0 in uniform_levels:
        uni = alloc_uniform(curves, ranks, r0)
        budget = total_cost(curves, uni)
        opt = alloc_optimal(curves, ranks, budget)
        eu, eo = total_eps(curves, uni), total_eps(curves, opt)
        rmean, rmin = alloc_random(curves, ranks, budget, 200, args.seed + r0)

        su = apply_alloc(model, curves, svd_cache, uni); loss_u = lm_loss(model, eval_ids); restore(curves, su)
        so = apply_alloc(model, curves, svd_cache, opt); loss_o = lm_loss(model, eval_ids); restore(curves, so)

        # per-type mean optimal rank (heterogeneity readout)
        by_type = {t: [] for t in types}
        for key, r in opt.items():
            by_type[key.split(":")[1]].append(r)
        type_rank = {t: float(np.mean(v)) for t, v in by_type.items()}

        rel_eps = (eu - eo) / max(eu, 1e-9)
        d_uni, d_opt = loss_u - full_loss, loss_o - full_loss
        rel_loss = (d_uni - d_opt) / max(d_uni, 1e-9)
        rows.append({"uniform_rank": r0, "budget_params": budget,
                     "uniform_eps": eu, "optimal_eps": eo, "random_mean_eps": rmean,
                     "eps_improvement": rel_eps,
                     "uniform_loss": loss_u, "optimal_loss": loss_o, "full_loss": full_loss,
                     "loss_gap_reduction": rel_loss, "optimal_rank_by_type": type_rank})
        print(f"[B r0={r0:3d}] eps uniform={eu:.3f} optimal={eo:.3f} (improve {rel_eps*100:4.1f}%)  "
              f"| loss full={full_loss:.3f} uni={loss_u:.3f} opt={loss_o:.3f} "
              f"(extra-loss reduced {rel_loss*100:4.1f}%)  | opt ranks {type_rank}")

    eps_med = float(np.median([r["eps_improvement"] for r in rows]))
    loss_med = float(np.median([r["loss_gap_reduction"] for r in rows]))
    supported = (eps_med > 0.05) and (loss_med > 0.05)
    verdict = ("THEOREM 4 SUPPORTED on real Transformer motifs: at equal parameter budget, "
               "marginal-optimal rank allocation beats a uniform bottleneck in both functional "
               "error and end-to-end LM loss."
               if supported else
               "WEAK/NOT SUPPORTED at this scale: real motif curves too similar or loss noisy.")
    report = {"model": args.model, "text_source": src, "layers": layers, "types": types,
              "ranks": ranks, "full_loss": full_loss, "rows": rows,
              "eps_median_improvement": eps_med, "loss_median_gap_reduction": loss_med,
              "verdict": verdict, "supported": bool(supported),
              "curves": {k: {"eps": curves[k]["eps"], "cost": curves[k]["cost"],
                             "type": curves[k]["type"]} for k in curves}}
    outp = THIS.parent / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(report, f, indent=2)
    make_figure(curves, ranks, rows, types, full_loss, str(outp.parent / "real_motif_budget.png"))

    print("\n================ SUMMARY (real Transformer motifs) ================")
    print(f"median functional-error improvement (uniform->optimal): {eps_med*100:.1f}%")
    print(f"median end-to-end extra-loss reduction (uniform->optimal): {loss_med*100:.1f}%")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
