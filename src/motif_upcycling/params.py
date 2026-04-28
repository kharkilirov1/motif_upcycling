"""Closed-form parameter budgets for motif-upcycling experiments."""

from __future__ import annotations


def router_param_count(
    hidden_size: int,
    router_hidden: int = 128,
    num_motifs: int = 3,
    router_type: str = "contextual",
) -> int:
    if router_type == "contextual":
        return int(hidden_size) * int(router_hidden) + int(router_hidden) + int(router_hidden) * int(num_motifs) + int(num_motifs)
    if router_type == "static":
        return int(num_motifs)
    if router_type in ("none", None):
        return 0
    raise ValueError("router_type must be 'contextual', 'static', or 'none'")


def motif_lora_param_count(hidden_size: int, intermediate_size: int, rank: int = 8, placements: int = 6) -> int:
    return int(placements) * int(rank) * (int(hidden_size) + int(intermediate_size))


def sarc_param_count(hidden: int = 16, num_gates: int = 1) -> int:
    # MLP 1 -> H -> 1: (H weights + H bias) + (H weights + 1 bias) = 3H+1.
    return int(num_gates) * (3 * int(hidden) + 1)


def p4_layer_param_count(
    hidden_size: int,
    intermediate_size: int,
    rank: int = 8,
    router_hidden: int = 128,
    num_motifs: int = 3,
    router_type: str = "contextual",
    lora_placements: int = 6,
) -> int:
    return router_param_count(hidden_size, router_hidden, num_motifs, router_type) + motif_lora_param_count(
        hidden_size, intermediate_size, rank, placements=lora_placements
    )
