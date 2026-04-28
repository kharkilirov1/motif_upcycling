#!/usr/bin/env python
"""Closed-form parameter budgets for motif-upcycling and SARC."""

from __future__ import annotations

import argparse

from motif_upcycling.params import p4_layer_param_count, router_param_count, motif_lora_param_count, sarc_param_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--num-motifs", type=int, default=3)
    parser.add_argument("--router-hidden", type=int, default=128)
    parser.add_argument("--sarc-hidden", type=int, default=16)
    parser.add_argument("--sarc-gates", type=int, default=0)
    args = parser.parse_args()

    router = router_param_count(args.hidden_size, args.router_hidden, args.num_motifs)
    lora = motif_lora_param_count(args.hidden_size, args.intermediate_size, args.rank)
    p4_layer = p4_layer_param_count(
        args.hidden_size, args.intermediate_size, args.rank, args.router_hidden, args.num_motifs
    )
    sarc = sarc_param_count(args.sarc_hidden, args.sarc_gates)
    total = args.layers * p4_layer + sarc

    print(f"router/layer:      {router:,}")
    print(f"motif LoRA/layer:  {lora:,}")
    print(f"P4/layer:          {p4_layer:,}")
    print(f"P4 total:          {args.layers * p4_layer:,}")
    print(f"SARC extra:        {sarc:,}")
    print(f"total trainable:   {total:,}")


if __name__ == "__main__":
    main()
