"""Motif routers with neutral initialization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticRouter(nn.Module):
    """A per-layer trainable router with ``M`` logits.

    Initial logits are zero. Used together with ``M * softmax(logits)`` this gives
    a neutral coefficient of one for every expert.
    """

    def __init__(self, num_motifs: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(int(num_motifs)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1] + (self.logits.numel(),)
        return self.logits.view(*([1] * (x.ndim - 1)), -1).expand(shape)


class ContextualRouter(nn.Module):
    """Token-wise contextual router ``r(x)``.

    The second projection is zero-initialized, so ``r(x)=0`` at initialization.
    Parameter count: ``d * hidden + hidden + hidden * M + M``.
    """

    def __init__(self, hidden_size: int, router_hidden: int, num_motifs: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.router_hidden = int(router_hidden)
        self.num_motifs = int(num_motifs)
        self.fc1 = nn.Linear(self.hidden_size, self.router_hidden, bias=True)
        self.fc2 = nn.Linear(self.router_hidden, self.num_motifs, bias=True)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))
