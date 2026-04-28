"""Small low-rank adapters used by motif-upcycling."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankAdapter(nn.Module):
    """LoRA delta for a bias-free linear map ``y = x W^T``.

    Parameter count is ``rank * (in_features + out_features)``. The B matrix is
    initialized to zero, so the adapter output is exactly zero at initialization.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(rank)

        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.A), self.B) * self.scale
