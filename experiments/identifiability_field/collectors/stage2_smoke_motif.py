"""stage2_smoke_motif.py - Stage 2 PIPELINE SMOKE TEST (not a real result).

This is NOT the gated real Stage 2. Per GOAL.md the real transformer test is only
justified if Stage 1 says SHARED (it said SEPARATE). This script only validates that
the Stage-2 collector wiring works against this repo's REAL Motif-Upcycling API
(`MotifSwiGLUMLP`): that the per-(token) hooks fire, the signals (router entropy /
margin, SARC relative scale, motif argmax) have the right shapes, an npz is written,
and the engine runs end-to-end.

IMPORTANT — why the verdict here is meaningless:
  * the donor SwiGLU MLP has RANDOM untrained weights (no pretrained LM),
  * there is no real text / next-token loss,
  * to get any signal variance at all we deliberately break the function-preserving
    neutral initialization (randomize the router fc2, LoRA B, and SARC head),
  * the `adapt` decision uses a SYNTHETIC target, not a real loss-lowering probe.
So any "SHARED/SEPARATE" it prints says nothing about the hypothesis. It is a wiring
check only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# make the repo's src importable
THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from motif_upcycling import MotifUpcycleConfig, SARCConfig  # noqa: E402
from motif_upcycling.swiglu import MotifSwiGLUMLP  # noqa: E402


class TinyQwenMLP(nn.Module):
    """Minimal Qwen-style SwiGLU MLP donor (gate/up/down), random weights."""

    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.act_fn = torch.nn.functional.silu


def _break_neutral_init(mlp: MotifSwiGLUMLP, gen: torch.Generator):
    """Randomize the zero-initialized parts so router alpha and SARC scale vary.

    This destroys the function-preserving guarantee on purpose; it exists only so
    the smoke test produces non-degenerate signals.
    """
    with torch.no_grad():
        if mlp.router is not None and hasattr(mlp.router, "fc2"):
            mlp.router.fc2.weight.normal_(0, 0.5, generator=gen)
            mlp.router.fc2.bias.normal_(0, 0.5, generator=gen)
        for name, p in mlp.named_parameters():
            if name.endswith(".B"):  # LoRA B (zero-init)
                p.normal_(0, 0.1, generator=gen)
        if mlp.sarc_scaler is not None and hasattr(mlp.sarc_scaler, "mlp"):
            mlp.sarc_scaler.mlp[-1].weight.normal_(0, 1.0, generator=gen)
            mlp.sarc_scaler.mlp[-1].bias.normal_(0, 1.0, generator=gen)


def _entropy_rows(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(-1, keepdims=True)
    return -(P * np.log(P)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--inter", type=int, default=256)
    ap.add_argument("--num-motifs", type=int, default=4)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gen = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)

    donor = TinyQwenMLP(args.hidden, args.inter)
    cfg = MotifUpcycleConfig(
        num_motifs=args.num_motifs, router_hidden=32, router_type="contextual",
        lora_rank=8, lora_alpha=8.0,
        sarc=SARCConfig(enabled=True, mode="adapter_only", max_delta=0.10, log_stats=True),
    )
    mlp = MotifSwiGLUMLP(donor, cfg).eval()
    _break_neutral_init(mlp, gen)

    M = mlp.num_motifs
    rows, motif, budget, temp, us_all = [], [], [], [], []
    fwd_shape_ok = True

    with torch.no_grad():
        for _ in range(args.seq):
            x = torch.randn(1, args.tokens, args.hidden, generator=gen)  # [1,T,d] residual input
            # Exercise the FULL patched module (base_update + lora_delta + router + SARC).
            # In adapter_only mode this invokes the SARC scaler internally, validating it
            # and populating last_stats (log_stats=True).
            out = mlp(x)
            fwd_shape_ok = fwd_shape_ok and tuple(out.shape) == (1, args.tokens, args.hidden)
            alpha = mlp._router_alpha(x)[0]               # [T, M]   router weights (M*softmax)
            delta = mlp.lora_delta(x)[0]                  # [T, d]   trainable motif/LoRA update
            xt = x[0]                                     # [T, d]
            a = (alpha / alpha.sum(-1, keepdim=True)).numpy()  # normalize to a prob simplex
            srt = np.sort(a, axis=-1)[:, ::-1]
            rms_d = torch.sqrt((delta.float() ** 2).mean(-1) + 1e-8)
            rms_x = torch.sqrt((xt.float() ** 2).mean(-1) + 1e-8)
            update_scale = torch.log(rms_d / rms_x).numpy()    # SARC relative scale r

            confidence = srt[:, 0]                             # max router weight
            cand_entropy = _entropy_rows(a)                    # router entropy
            margin = srt[:, 0] - srt[:, 1]                     # router top1-top2
            for t in range(args.tokens):
                rows.append([float(confidence[t]), float(cand_entropy[t]),
                             float(margin[t]), float(update_scale[t])])
                motif.append(int(a[t].argmax()))                          # argmax slice
                budget.append(int((a[t] > (1.0 / M)).sum()))              # #active slices (tier)
                temp.append(float(cand_entropy[t]))
                us_all.append(float(update_scale[t]))

    X = np.asarray(rows, float)
    # SYNTHETIC adapt target (no real loss): band-shaped on the middle tertile of
    # the SARC relative scale -> non-monotone, 2-class by construction.
    us = np.asarray(us_all)
    lo, hi = np.quantile(us, [1 / 3, 2 / 3])
    adapt = ((us > lo) & (us < hi)).astype(int)
    sig_names = ["confidence", "cand_entropy", "margin", "update_scale"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=X, signal_names=np.array(sig_names),
             dec__motif=np.array(motif), dec__budget=np.array(budget),
             dec__adapt=np.array(adapt), dec__temperature=np.array(temp, float))
    print(f"[SMOKE] wrote {args.out}: {X.shape[0]} (token) sites x {X.shape[1]} signals")
    print(f"[SMOKE] hooks OK: alpha{tuple(mlp._router_alpha(torch.randn(1,3,args.hidden)).shape)}, "
          f"forward_shape_ok={fwd_shape_ok}, "
          f"SARC last_stats={'present' if mlp.sarc_scaler.last_stats is not None else 'none'}")
    print("[SMOKE] WARNING: random untrained weights + synthetic targets -> verdict is NOT interpretable")


if __name__ == "__main__":
    main()
