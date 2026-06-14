"""
collect_transformer_motif.py — Stage 2 collector (run only if Stage 1 says SHARED).

Extracts per-(token, layer) signals at patched layers of a Motif-Upcycled model and
saves an npz for load_and_test.py.

NOTE: your Motif-Upcycling implementation was NOT in the provided files, so the hooks
below are specced against the paper's equations. Point `--motif-impl` at your code and
complete each `# >>> WIRE`. Requires torch + transformers + the model weights.

Signals per site (token t, patched layer L), see signal_spec.md:
  confidence    = 1 - normalized predictive entropy of next-token dist at t (or max softmax)
  cand_entropy  = entropy of router weights alpha over the M motif slices at (t, L)
  margin        = alpha_top1 - alpha_top2 at (t, L)
  update_scale  = SARC relative scale r = log( RMS(delta_u) / RMS(x) ) at (t, L)
  (kappa: omit unless you have a reliability head)

Decisions per site:
  motif  = argmax_m alpha at (t, L)                      (categorical)
  budget = number of slices with alpha above a mass threshold (tier)
  adapt  = P0 causal probe: does motif-LoRA at (t,L) lower token loss?  (binary)
           -> compute loss with vs without the motif delta at that site.
  temperature = predictive entropy (continuous)
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path to the causal LM donor")
    ap.add_argument("--motif-impl", required=True, help="path to your Motif-Upcycling implementation")
    ap.add_argument("--text", default=None, help="eval text file; else uses a small built-in sample")
    ap.add_argument("--layers", default="", help="comma-sep patched layer indices, e.g. 11,12,13,14")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-sites", type=int, default=8000)
    args = ap.parse_args()

    import sys, torch
    sys.path.insert(0, args.motif_impl)

    # >>> WIRE 1: load donor + wrap patched layers with your exact-sum factorization,
    #     contextual router, motif-LoRA deltas, and SARC. Register forward hooks on each
    #     patched layer's FFN wrapper to capture, per token:
    #         alpha            (router weights over M slices)  shape [T, M]
    #         delta_u          (trainable motif/LoRA update)   shape [T, d]
    #         x_resid          (residual input to the block)   shape [T, d]
    #     and at the model head: next-token logits             shape [T, V]
    # from motif_upcycling import load_motif_model, patched_layer_indices
    # model, tok = load_motif_model(args.model)
    raise SystemExit(
        "Stage 2 stub: wire your Motif-Upcycling model + hooks here (see header + signal_spec.md). "
        "Capture per-token alpha, delta_u, residual x, and head logits at patched layers, then "
        "fill the assembly below."
    )

    # ---- assembly template (delete the SystemExit above once hooks are in) ----
    def rms(v, axis=-1):
        return torch.sqrt((v.float() ** 2).mean(axis) + 1e-8)

    def H_rows(P):                                  # entropy per row of a prob matrix
        P = P.clamp_min(1e-12)
        return float  # replaced per-row below

    sig_names = ["confidence", "cand_entropy", "margin", "update_scale"]
    rows, motif, budget, adapt, temp = [], [], [], [], []

    # for each captured (token t, patched layer L):
    #   alpha_tL : [M]      delta_tL : [d]      x_tL : [d]      logits_t : [V]
    #   p = softmax(logits_t); ent = -(p*log p).sum(); maxsm = p.max()
    #   srt = sort(alpha_tL, desc)
    #   confidence   = 1 - ent / log(V)
    #   cand_entropy = -(alpha_tL * log alpha_tL).sum()
    #   margin       = srt[0] - srt[1]
    #   update_scale = log( rms(delta_tL) / rms(x_tL) )         # == SARC r
    #   motif        = argmax(alpha_tL)
    #   budget       = (cumulative-sorted-alpha reaching 0.9 -> #slices) tier
    #   adapt        = 1 if token-loss(with delta) < token-loss(without delta) else 0   # P0 probe
    #   temp         = ent
    #   rows.append([confidence, cand_entropy, margin, update_scale]); ...

    X = np.asarray(rows, float)[: args.max_sites]
    np.savez(args.out, X=X, signal_names=np.array(sig_names),
             dec__motif=np.array(motif), dec__budget=np.array(budget),
             dec__adapt=np.array(adapt), dec__temperature=np.array(temp, float))
    print(f"wrote {args.out}: {X.shape[0]} sites x {X.shape[1]} signals")


if __name__ == "__main__":
    main()
