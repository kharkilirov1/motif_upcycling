"""causal_margin.py - core measurements of the Causal-Margin Allocation protocol on
a REAL pretrained model (Qwen2.5-0.5B), forward-only (no training).

Implements, on real Qwen motifs:
  * role error-vs-budget curves eps_i(B)  (functional SVD-truncation error on real acts),
  * saturating set S and caps B_i^sat                              (Def 1.3),
  * capped water-filling allocation                                (Thm 3.1),
  * budget-efficiency factor eta(B_tot) = B_tot^uni / B_tot        (Def 2.1, Thm 2.2),
    and its large-budget limit vs the theoretical m/(m-|S|),
  * the SEPARABILITY INDEX sigma via END-TO-END LM-loss interactions when roles are
    jointly compressed -- the real-model test (T2) that catches residual-stream coupling,
    which the per-module functional error CANNOT see (it is additive by construction).

Roles = motif types {q,o (compare), gate (select), up (expand), down (memory)}, each role
= all its projection modules across a band of layers (so a role spans depth).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(max(1, torch.get_num_threads()))
THIS = Path(__file__).resolve()

TYPE_PATHS = {"q": ("self_attn", "q_proj"), "o": ("self_attn", "o_proj"),
              "gate": ("mlp", "gate_proj"), "up": ("mlp", "up_proj"), "down": ("mlp", "down_proj")}


def get_module(model, layer, mtype):
    sub, name = TYPE_PATHS[mtype]
    return getattr(getattr(model.model.layers[layer], sub), name)


def get_eval(tok, n_chunks, seq_len, n_seq, seed):
    text = None
    for ident in ("Salesforce/wikitext", "wikitext"):
        try:
            from datasets import load_dataset
            ds = load_dataset(ident, "wikitext-2-raw-v1", split="train")
            text = [t for t in ds["text"][:8000] if len(t.strip()) > 80][:n_chunks]
            break
        except Exception:
            text = None
    if not text:
        raise SystemExit("wikitext unavailable")
    ids = [tok(t, return_tensors="pt").input_ids[0] for t in text]
    cat = torch.cat([e for e in ids if e.numel() >= 8])
    n = cat.numel() // seq_len
    cat = cat[: n * seq_len].view(n, seq_len)
    idx = np.random.default_rng(seed).permutation(n)[:n_seq]
    return cat[idx]


@torch.no_grad()
def lm_loss(model, eval_ids):
    tot, ntok = 0.0, 0
    for i in range(eval_ids.shape[0]):
        ids = eval_ids[i:i + 1]
        l = F.cross_entropy(model(ids).logits[0, :-1], ids[0, 1:], reduction="sum")
        tot += float(l); ntok += ids.shape[1] - 1
    return tot / max(1, ntok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="8,9,10,11,12,13,14,15")
    ap.add_argument("--types", default="q,o,gate,up,down")
    ap.add_argument("--groups", default="", help="coarse roles, e.g. 'attn=q+o;ffn=gate+up+down'")
    ap.add_argument("--ranks", default="2,4,8,16,32,64,128,192,256")
    ap.add_argument("--sigma-rank", type=int, default=16, help="probe rank for sigma interactions")
    ap.add_argument("--sigma-ranks", default="", help="explicit probe ranks for sigma (small "
                    "perturbations near the operating point avoid the loss-ceiling artifact)")
    ap.add_argument("--calib-seq", type=int, default=12)
    ap.add_argument("--eval-seq", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--sat-tol", type=float, default=0.01, help="rel slope below which a role is 'saturated'")
    ap.add_argument("--out", default="data/causal_margin_report.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    layers = [int(x) for x in args.layers.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]
    # roles: either fine (each type its own role) or coarse groups via --groups
    if args.groups:
        roles = {}
        for grp in args.groups.split(";"):
            name, mem = grp.split("="); roles[name] = mem.split("+")
    else:
        roles = {t: [t] for t in args.types.split(",")}
    roles_list = list(roles)
    base_types = sorted({bt for mem in roles.values() for bt in mem})
    members = {name: [(L, bt) for L in layers for bt in roles[name]] for name in roles_list}
    M = len(roles_list)
    types = base_types  # base modules to instrument

    print(f"[cm] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    calib = get_eval(tok, 600, args.seq_len, args.calib_seq, 0)
    ev = get_eval(tok, 600, args.seq_len, args.eval_seq, 1)

    # ---- capture activations, SVD each module ----
    modules = {(L, t): get_module(model, L, t) for L in layers for t in types}
    cache = {k: [] for k in modules}
    handles = []
    for k, mod in modules.items():
        handles.append(mod.register_forward_hook(lambda mod, inp, out, kk=k: cache[kk].append(
            inp[0].detach().reshape(-1, inp[0].shape[-1]))))
    with torch.no_grad():
        for i in range(calib.shape[0]):
            model(calib[i:i + 1])
    for h in handles:
        h.remove()

    svd = {}; eps_mod = {}; cost_mod = {}
    for k, mod in modules.items():
        X = torch.cat(cache[k], 0).float()
        if X.shape[0] > 2000:
            X = X[torch.randperm(X.shape[0])[:2000]]
        W = mod.weight.data.float()
        FO = X @ W.t(); fon = float((FO ** 2).sum()) + 1e-9
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        svd[k] = (U, S, Vt, min(W.shape))
        out_f, in_f = W.shape
        eps_mod[k] = {}; cost_mod[k] = {}
        for r in ranks:
            rr = min(r, min(W.shape))
            Wr = (U[:, :rr] * S[:rr]) @ Vt[:rr]
            eps_mod[k][r] = float(((X @ (W - Wr).t()) ** 2).sum()) / fon
            cost_mod[k][r] = rr * (out_f + in_f)

    # ---- role-level curves (aggregate member modules across layers) ----
    role_eps = {nm: {r: float(np.mean([eps_mod[k][r] for k in members[nm]])) for r in ranks} for nm in roles_list}
    role_cost = {nm: {r: int(sum(cost_mod[k][r] for k in members[nm])) for r in ranks} for nm in roles_list}
    print(f"[curves] roles={roles}")
    print("[curves] role functional error eps_i(r):")
    for t in roles_list:
        print(f"  {t:5s} " + "  ".join(f"r{r}={role_eps[t][r]:.3f}" for r in ranks))

    # ---- saturating set S: relative slope of eps over the top of the grid ----
    S = set()
    caps = {}
    for t in roles_list:
        e = [role_eps[t][r] for r in ranks]
        # relative drop over the last two grid steps
        rel = (e[-2] - e[-1]) / max(e[0] - e[-1], 1e-9)
        # cap = smallest rank whose eps is within sat-tol of the floor (best on grid)
        floor = e[-1]
        cap = ranks[-1]
        for r in ranks:
            if role_eps[t][r] - floor <= args.sat_tol * max(e[0] - floor, 1e-9):
                cap = r; break
        caps[t] = cap
        if rel < 0.05:                      # essentially flat at the top => saturating
            S.add(t)
    print(f"[roles] saturating set S={sorted(S)}  caps={caps}")

    # ---- eta(B_tot): capped water-filling optimal vs uniform, matching error ----
    def cost_of(alloc):
        return sum(role_cost[t][alloc[t]] for t in roles_list)

    def err_of(alloc):
        return sum(role_eps[t][alloc[t]] for t in roles_list)

    def optimal_alloc(budget):
        # exact DP over rank grid (capped), min total eps s.t. total cost <= budget
        keys = roles_list
        cur = {0: (0.0, {})}
        for t in keys:
            nxt = {}
            for used, (er, al) in cur.items():
                for r in ranks:
                    if t in S and r > caps[t]:
                        continue                # respect cap for saturating roles
                    c = used + role_cost[t][r]
                    if c > budget:
                        continue
                    ne = er + role_eps[t][r]
                    if c not in nxt or ne < nxt[c][0]:
                        d = dict(al); d[t] = r; nxt[c] = (ne, d)
            cur = nxt
        return min(cur.values(), key=lambda v: v[0])[1]

    def uniform_alloc(budget):
        # protocol Def 2.1: equal BUDGET per role (B_tot/m), realized as the largest
        # in-grid rank each role can afford within its share.
        per = budget / M
        a = {}
        for t in roles_list:
            r_sel = ranks[0]
            for r in ranks:
                if role_cost[t][r] <= per:
                    r_sel = r
            a[t] = r_sel
        return a

    # sweep B_tot, compute eta = B_uni/B_tot where B_uni is the smallest equal-budget
    # uniform total needed to reach optimal's error E_opt.
    base = cost_of({t: ranks[2] for t in roles_list})
    full_cost = cost_of({t: ranks[-1] for t in roles_list})
    # fine geometric budget grid for the uniform-match search
    uni_grid = sorted(set(int(b) for b in np.geomspace(base * 0.3, full_cost * 6, 80)))
    etas = []
    for mult in [1.0, 1.5, 2.0, 3.0, 4.0]:
        Btot = int(min(full_cost, base * mult))
        bopt = optimal_alloc(Btot)
        Eopt = err_of(bopt)
        Buni = None
        for B in uni_grid:
            if err_of(uniform_alloc(B)) <= Eopt + 1e-9:
                Buni = B; break
        eta = (Buni / Btot) if Buni else float("inf")
        etas.append({"B_tot": Btot, "E_opt": Eopt, "B_uni_to_match": Buni, "eta": eta,
                     "opt_alloc": bopt, "uniform_alloc_at_Btot": uniform_alloc(Btot)})
        es = "inf" if eta == float("inf") else round(eta, 3)
        print(f"[eta] B_tot={Btot} E_opt={Eopt:.3f} B_uni={Buni} eta={es}")
    eta_theory = M / max(1, (M - len(S)))
    print(f"[eta] theoretical limit m/(m-|S|) = {M}/{M-len(S)} = {eta_theory:.3f}")

    # ---- sigma: end-to-end LM-loss interactions under JOINT role compression ----
    def truncate_role(nm, r):
        saved = {}
        for k in members[nm]:
            mod = modules[k]; U, Sv, Vt, rmax = svd[k]
            rr = min(r, rmax)
            saved[k] = mod.weight.data
            mod.weight.data = ((U[:, :rr] * Sv[:rr]) @ Vt[:rr]).to(mod.weight.dtype)
        return saved

    def restore(saved):
        for k, W in saved.items():
            modules[k].weight.data = W

    L0 = lm_loss(model, ev)
    if args.sigma_ranks:
        probe_ranks = [int(x) for x in args.sigma_ranks.split(",")]
    else:
        probe_ranks = sorted(set([max(2, args.sigma_rank // 2), args.sigma_rank, args.sigma_rank * 2]))
    sigma_by_rank = {}
    dL_last = {}; inter_last = {}
    for rp in probe_ranks:
        dL = {}
        for t in roles_list:
            sv = truncate_role(t, rp); dL[t] = lm_loss(model, ev) - L0; restore(sv)
        inter = {}
        for a, b in itertools.combinations(roles_list, 2):
            sa = truncate_role(a, rp); sb = truncate_role(b, rp)
            dLab = lm_loss(model, ev) - L0
            restore(sb); restore(sa)
            inter[(a, b)] = dLab - (dL[a] + dL[b])     # off-diagonal interaction
        diag = np.mean([abs(dL[t]) for t in roles_list])
        offd = np.mean([abs(v) for v in inter.values()])
        sigma_by_rank[rp] = float(offd / max(diag, 1e-9))
        dL_last, inter_last = dL, inter
        print(f"[sigma] probe rank={rp:3d}  mean|diag|={diag:.4f}  mean|inter|={offd:.4f}  "
              f"sigma={sigma_by_rank[rp]:.3f}")
    sigma = float(np.mean(list(sigma_by_rank.values())))
    rp = probe_ranks[-1]
    print(f"[sigma] full L0={L0:.4f}  sigma (mean over probe ranks {probe_ranks}) = {sigma:.3f}")
    print("[sigma] per-role dL_i (last rank): " + "  ".join(f"{t}={dL_last[t]:+.3f}" for t in roles_list))
    worst = max(inter_last.items(), key=lambda kv: abs(kv[1]))
    print(f"[sigma] strongest interaction (last rank): {worst[0]} = {worst[1]:+.4f}")
    dL = dL_last; inter = inter_last

    # verdict
    if sigma < 0.15:
        verdict = (f"sigma={sigma:.2f} SMALL: roles ~separable on real Qwen -> capped "
                   f"water-filling licensed; recipe (not alchemy). eta limit ~{eta_theory:.2f}.")
    elif sigma < 0.5:
        verdict = (f"sigma={sigma:.2f} MODERATE: partial coupling (residual stream) -> recipe "
                   f"becomes guided coordinate search, water-filling local-only.")
    else:
        verdict = (f"sigma={sigma:.2f} LARGE: roles strongly coupled -> closed-form recipe not "
                   f"licensed; budgets must be searched jointly.")
    print(f"\nVERDICT: {verdict}")

    report = {"model": args.model, "layers": layers, "roles": roles, "ranks": ranks,
              "role_eps": role_eps, "role_cost": role_cost, "saturating_set": sorted(S),
              "caps": caps, "eta_sweep": etas, "eta_theory_limit": eta_theory,
              "L0": L0, "sigma_by_rank": sigma_by_rank, "dL_role": dL,
              "interactions": {f"{a}+{b}": v for (a, b), v in inter.items()},
              "sigma": sigma, "verdict": verdict}
    outp = THIS.parent / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[out] {outp}")


if __name__ == "__main__":
    main()
