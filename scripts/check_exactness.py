#!/usr/bin/env python
"""Local exactness smoke test for MotifSwiGLUMLP."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from motif_upcycling import MotifSwiGLUMLP, MotifUpcycleConfig, SARCConfig, max_abs_diff, trainable_parameter_count


class TinySwiGLU(nn.Module):
    def __init__(self, d: int, i: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(d, i, bias=bias)
        self.up_proj = nn.Linear(d, i, bias=bias)
        self.down_proj = nn.Linear(i, d, bias=bias)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--i", type=int, default=192)
    parser.add_argument("--num-motifs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--sarc-mode", choices=["off", "adapter_only", "motif_wise", "full_update"], default="off")
    args = parser.parse_args()

    torch.manual_seed(0)
    dense = TinySwiGLU(args.d, args.i)
    x = torch.randn(2, 11, args.d)
    y_dense = dense(x)

    sarc = SARCConfig(enabled=args.sarc_mode != "off", mode=args.sarc_mode)
    cfg = MotifUpcycleConfig(num_motifs=args.num_motifs, router_hidden=16, lora_rank=args.rank, sarc=sarc)
    wrapped = MotifSwiGLUMLP(dense, cfg)
    y_wrap = wrapped(x)

    print(f"max_abs_diff={max_abs_diff(y_dense, y_wrap):.6e}")
    print(f"trainable_params={trainable_parameter_count(wrapped):,}")


if __name__ == "__main__":
    main()
