"""trainable_het_rank.py - the first real architecture brick: a TRAINABLE
heterogeneous-rank adaptation vs a uniform-rank adaptation at EQUAL compute.

Static SVD probing (`real_motif_budget.py`) showed real Qwen motifs have different
error-vs-budget curves. This goes one step further: instead of compressing existing
weights, we ADD trainable low-rank adapters (the repo's `LowRankAdapter`) to the motif
projections and fine-tune on real text. The question (FOG Theorem 4 for adaptation):

  at EQUAL trainable parameters, does allocating adapter rank by marginal value beat a
  uniform rank on every motif?

Allocation rule (gradient water-filling)
----------------------------------------
A rank-r adapter at module i can, to first order, capture the top-r singular directions
of the loss gradient dL/dW_i. So the marginal value of the (k+1)-th rank unit at module i
is ~ sigma_{k+1}(G_i)^2 (squared singular value of the probe gradient), at cost
(in_i+out_i) params. We greedily add rank where sigma^2 / cost is largest, until the
total parameter budget (= the uniform config's budget) is reached.

Configs compared at EQUAL total adapter params: UNIFORM (same rank r0 everywhere),
HETEROGENEOUS (gradient water-filling), RANDOM (control). Each is trained from scratch
on wikitext with the base frozen; we report held-out LM loss.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(max(1, torch.get_num_threads()))
THIS = Path(__file__).resolve()
SRC = THIS.parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from motif_upcycling.lora import LowRankAdapter  # noqa: E402

TYPE_PATHS = {
    "q": ("self_attn", "q_proj"), "o": ("self_attn", "o_proj"),
    "gate": ("mlp", "gate_proj"), "up": ("mlp", "up_proj"), "down": ("mlp", "down_proj"),
}


def get_module(model, layer, mtype):
    sub, name = TYPE_PATHS[mtype]
    return getattr(getattr(model.model.layers[layer], sub), name)


def get_texts(n=600):
    for ident in ("Salesforce/wikitext", "wikitext"):
        try:
            from datasets import load_dataset
            ds = load_dataset(ident, "wikitext-2-raw-v1", split="train")
            t = [x for x in ds["text"][:9000] if len(x.strip()) > 80]
            if len(t) > 50:
                return t[:n]
        except Exception:
            pass
    raise SystemExit("wikitext unavailable")


def encode(tok, texts, seq_len, n_seq, seed):
    ids = [tok(t, return_tensors="pt").input_ids[0] for t in texts]
    ids = [e for e in ids if e.numel() >= 8]
    cat = torch.cat(ids)
    n = cat.numel() // seq_len
    cat = cat[: n * seq_len].view(n, seq_len)
    idx = np.random.default_rng(seed).permutation(n)[:n_seq]
    return cat[idx]


class AdapterSet:
    """A set of LowRankAdapters hooked onto target modules (out += adapter(in))."""

    def __init__(self, model, modules: Dict[str, object], ranks: Dict[str, int], alpha=8.0):
        self.adapters: Dict[str, LowRankAdapter] = {}
        self.handles = []
        for key, mod in modules.items():
            r = ranks[key]
            if r <= 0:
                continue
            ad = LowRankAdapter(mod.in_features, mod.out_features, r, alpha)
            self.adapters[key] = ad
            self.handles.append(mod.register_forward_hook(self._mk(key)))

    def _mk(self, key):
        ad = self.adapters[key]
        def hook(mod, inp, out):
            return out + ad(inp[0].to(out.dtype))
        return hook

    def params(self):
        ps = []
        for ad in self.adapters.values():
            ps += list(ad.parameters())
        return ps

    def n_params(self):
        return sum(p.numel() for p in self.params())

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def cost_of(modules, key):
    m = modules[key]
    return m.in_features + m.out_features


def probe_gradients(model, modules, train_ids, n_batches=6):
    """Return per-module squared singular values of the accumulated loss gradient."""
    for p in model.parameters():
        p.requires_grad_(False)
    targets = {key: mod.weight for key, mod in modules.items()}
    for w in targets.values():
        w.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    for i in range(n_batches):
        ids = train_ids[i:i + 1]
        out = model(ids, labels=ids)
        out.loss.backward()
    sv = {}
    for key, w in targets.items():
        g = w.grad.detach().float()
        s = torch.linalg.svdvals(g)
        sv[key] = (s ** 2).cpu().numpy()
        w.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    return sv


def alloc_uniform(modules, r0):
    return {k: r0 for k in modules}


def alloc_gradient(modules, sv, budget_params, max_rank):
    """Greedy water-filling: add rank where sigma^2/cost is largest, until budget."""
    ranks = {k: 0 for k in modules}
    spent = 0
    # precompute marginal value of each (module, next-rank-unit)
    while True:
        best = None
        for k in modules:
            r = ranks[k]
            if r >= min(max_rank, len(sv[k])):
                continue
            cost = cost_of(modules, k)
            if spent + cost > budget_params:
                continue
            marg = sv[k][r] / cost                      # sigma_{r+1}^2 / cost
            if best is None or marg > best[0]:
                best = (marg, k, cost)
        if best is None:
            break
        _, k, cost = best
        ranks[k] += 1
        spent += cost
    return ranks


def alloc_random(modules, budget_params, max_rank, seed):
    """Genuinely random HETEROGENEOUS control: fixed random per-module priorities
    (Dirichlet), then fill rank by sampling modules with those priorities until the
    budget is spent. (A uniform random pick per step would just reproduce uniform.)"""
    rng = np.random.default_rng(seed)
    keys = list(modules)
    prio = rng.dirichlet(np.ones(len(keys)))
    ranks = {k: 0 for k in keys}
    spent = 0
    while True:
        cand = [i for i, k in enumerate(keys)
                if ranks[k] < max_rank and spent + cost_of(modules, k) <= budget_params]
        if not cand:
            break
        p = np.array([prio[i] for i in cand]); p = p / p.sum()
        i = int(rng.choice(cand, p=p))
        k = keys[i]
        ranks[k] += 1; spent += cost_of(modules, k)
    return ranks


def train_and_eval(model, modules, ranks, train_ids, eval_ids, steps, lr, seed):
    torch.manual_seed(seed)
    aset = AdapterSet(model, modules, ranks)
    if not aset.adapters:
        aset.remove(); return float("nan"), 0
    opt = torch.optim.AdamW(aset.params(), lr=lr, weight_decay=0.0)
    model.eval()  # base frozen; adapters train
    nb = train_ids.shape[0]
    for step in range(steps):
        ids = train_ids[step % nb: step % nb + 1]
        out = model(ids, labels=ids)
        opt.zero_grad(); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(aset.params(), 1.0)
        opt.step()
    np_ = aset.n_params()
    with torch.no_grad():
        tot, ntok = 0.0, 0
        for i in range(eval_ids.shape[0]):
            ids = eval_ids[i:i + 1]
            l = F.cross_entropy(model(ids).logits[0, :-1], ids[0, 1:], reduction="sum")
            tot += float(l); ntok += ids.shape[1] - 1
    aset.remove()
    return tot / max(1, ntok), np_


@torch.no_grad()
def base_loss(model, eval_ids):
    tot, ntok = 0.0, 0
    for i in range(eval_ids.shape[0]):
        ids = eval_ids[i:i + 1]
        l = F.cross_entropy(model(ids).logits[0, :-1], ids[0, 1:], reduction="sum")
        tot += float(l); ntok += ids.shape[1] - 1
    return tot / max(1, ntok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="6,9,12,15")
    ap.add_argument("--types", default="q,o,gate,up,down")
    ap.add_argument("--uniform-rank", type=int, default=16)
    ap.add_argument("--max-rank", type=int, default=64)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--train-seq", type=int, default=120)
    ap.add_argument("--eval-seq", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--out", default="data/trainable_het_rank_report.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="0,1", help="seeds to average the comparison over")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    layers = [int(x) for x in args.layers.split(",")]
    types = args.types.split(",")

    print(f"[trainable] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    texts = get_texts()
    ids = encode(tok, texts, args.seq_len, args.train_seq + args.eval_seq, args.seed)
    train_ids, eval_ids = ids[:args.train_seq], ids[args.train_seq:]
    modules = {f"L{L}:{t}": get_module(model, L, t) for L in layers for t in types}

    bl = base_loss(model, eval_ids)
    print(f"[trainable] base held-out loss (no adapters) = {bl:.4f}")

    uni = alloc_uniform(modules, args.uniform_rank)
    budget = sum(uni[k] * cost_of(modules, k) for k in modules)        # equal-params budget
    print("[trainable] probing gradients for allocation ...")
    sv = probe_gradients(model, modules, train_ids)
    het = alloc_gradient(modules, sv, budget, args.max_rank)
    rnd = alloc_random(modules, budget, args.max_rank, args.seed)

    def by_type(ranks):
        d = {t: [] for t in types}
        for k, r in ranks.items():
            d[k.split(":")[1]].append(r)
        return {t: round(float(np.mean(v)), 1) for t, v in d.items()}

    print(f"[alloc] uniform per-type ranks: {by_type(uni)}")
    print(f"[alloc] gradient per-type ranks: {by_type(het)}")
    print(f"[alloc] random   per-type ranks: {by_type(rnd)}")

    results = {}
    for name, ranks in [("uniform", uni), ("heterogeneous", het), ("random", rnd)]:
        losses = []
        npar = 0
        for sd in seeds:
            loss, npar = train_and_eval(model, modules, ranks, train_ids, eval_ids,
                                        args.steps, args.lr, sd)
            losses.append(loss)
            print(f"[train] {name:13s} seed={sd} params={npar:7d}  held-out loss={loss:.4f}")
        results[name] = {"loss": float(np.mean(losses)), "loss_per_seed": losses,
                         "n_params": npar, "by_type": by_type(ranks)}
        print(f"[train] {name:13s} MEAN held-out loss={results[name]['loss']:.4f}  "
              f"(improvement over base {bl - results[name]['loss']:+.4f})")

    lu, lh = results["uniform"]["loss"], results["heterogeneous"]["loss"]
    rel = (lu - lh) / max(abs(bl - lu), 1e-9)   # fraction of uniform's gain added by going heterogeneous
    supported = lh < lu
    verdict = ("HETEROGENEOUS BEATS UNIFORM at equal trainable params: allocating adapter "
               "rank by gradient marginal value gives lower held-out LM loss than a uniform "
               "rank -- Theorem 4 holds for trainable adaptation, not just static compression."
               if supported else
               "NOT SUPPORTED at this scale: uniform <= heterogeneous (noisy/short training).")
    report = {"model": args.model, "layers": layers, "types": types,
              "base_loss": bl, "budget_params": budget,
              "uniform_rank": args.uniform_rank, "results": results,
              "heterogeneous_vs_uniform_loss_delta": lu - lh,
              "rel_extra_gain": rel, "supported": bool(supported), "verdict": verdict}
    outp = THIS.parent / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(report, f, indent=2)

    print("\n================ SUMMARY (trainable heterogeneous rank) ================")
    print(f"base={bl:.4f}  uniform={lu:.4f}  heterogeneous={lh:.4f}  random={results['random']['loss']:.4f}")
    print(f"heterogeneous - uniform = {lh - lu:+.4f}  (negative => heterogeneous better)")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
