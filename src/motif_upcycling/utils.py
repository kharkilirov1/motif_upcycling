"""Utility functions for channel partitions and parameter counting."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


def even_motif_sizes(intermediate_size: int, num_motifs: int) -> List[int]:
    """Split intermediate channels as evenly as possible."""

    if num_motifs <= 0:
        raise ValueError("num_motifs must be positive")
    base = int(intermediate_size) // int(num_motifs)
    rem = int(intermediate_size) % int(num_motifs)
    return [base + (1 if i < rem else 0) for i in range(int(num_motifs))]


def motif_slices(intermediate_size: int, sizes: Sequence[int]) -> List[slice]:
    """Convert motif sizes into non-overlapping slices that cover the hidden axis."""

    sizes = [int(s) for s in sizes]
    if any(s <= 0 for s in sizes):
        raise ValueError(f"all motif sizes must be positive, got {sizes}")
    if sum(sizes) != int(intermediate_size):
        raise ValueError(
            f"motif sizes must sum to intermediate_size={intermediate_size}, got {sum(sizes)}"
        )
    out: List[slice] = []
    start = 0
    for size in sizes:
        out.append(slice(start, start + size))
        start += size
    return out


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def total_parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().detach().cpu())


def set_trainable_only(module: nn.Module, trainable_names: Iterable[str]) -> None:
    """Set ``requires_grad`` true only for parameters whose name contains one token."""

    tokens = tuple(trainable_names)
    for name, p in module.named_parameters():
        p.requires_grad_(any(token in name for token in tokens))
