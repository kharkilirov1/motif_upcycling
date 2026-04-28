"""Scale-Aware Residual Control (SARC).

SARC modulates residual or adapter updates using their RMS scale relative to a
reference tensor. The identity-preserving correction form starts with scale==1,
so inserting it into a pretrained model does not change the current computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class SARCStats:
    ratio_mean: float
    ratio_std: float
    ratio_min: float
    ratio_max: float
    scale_mean: float
    scale_std: float
    scale_min: float
    scale_max: float
    rms_ref_mean: float
    rms_update_mean: float
    delta_abs_mean: float
    delta_max_abs: float


class ScaleAwareDeltaScaler(nn.Module):
    """Returns a token-wise scalar scale for a delta/update tensor.

    Formula:
        ratio = log((RMS(update)+eps) / (RMS(ref)+eps))
        scale = 1 + max_delta * tanh(MLP(ratio))

    The last MLP layer is zero-initialized, hence ``scale == 1`` at
    initialization. The perturbation is bounded by ``max_delta``.
    """

    def __init__(
        self,
        hidden: int = 16,
        eps: float = 1e-6,
        max_delta: float = 0.10,
        detach_ratio: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.eps = float(eps)
        self.max_delta = float(max_delta)
        self.detach_ratio = bool(detach_ratio)
        self.log_stats = bool(log_stats)
        self.mlp = nn.Sequential(
            nn.Linear(1, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.last_stats: Optional[SARCStats] = None

    @staticmethod
    def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.sqrt(torch.mean(x.float().pow(2), dim=-1, keepdim=True) + eps)

    def forward(self, ref: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        rms_ref = self.rms(ref, self.eps)
        rms_update = self.rms(update, self.eps)
        ratio = torch.log((rms_update + self.eps) / (rms_ref + self.eps))
        if self.detach_ratio:
            ratio = ratio.detach()

        delta = self.max_delta * torch.tanh(self.mlp(ratio.float()))
        scale = 1.0 + delta

        if self.log_stats:
            with torch.no_grad():
                self.last_stats = SARCStats(
                    ratio_mean=float(ratio.mean().detach().cpu()),
                    ratio_std=float(ratio.std(unbiased=False).detach().cpu()),
                    ratio_min=float(ratio.min().detach().cpu()),
                    ratio_max=float(ratio.max().detach().cpu()),
                    scale_mean=float(scale.mean().detach().cpu()),
                    scale_std=float(scale.std(unbiased=False).detach().cpu()),
                    scale_min=float(scale.min().detach().cpu()),
                    scale_max=float(scale.max().detach().cpu()),
                    rms_ref_mean=float(rms_ref.mean().detach().cpu()),
                    rms_update_mean=float(rms_update.mean().detach().cpu()),
                    delta_abs_mean=float(delta.abs().mean().detach().cpu()),
                    delta_max_abs=float(delta.abs().max().detach().cpu()),
                )
        return scale.to(update.dtype)


class ScaleAwareResidualCorrection(nn.Module):
    """Wraps residual addition as ``out = x + scale(ref, update) * update``."""

    def __init__(
        self,
        hidden: int = 16,
        eps: float = 1e-6,
        max_delta: float = 0.10,
        detach_ratio: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__()
        self.scaler = ScaleAwareDeltaScaler(hidden, eps, max_delta, detach_ratio, log_stats)

    def forward(self, x: torch.Tensor, update: torch.Tensor, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        if ref is None:
            ref = x
        scale = self.scaler(ref=ref, update=update)
        return x + scale * update


class BoundedRelativeGate(nn.Module):
    """A bounded-update ablation: ``g = sigmoid(log_tau - ratio)``.

    This is not identity-preserving for a full pretrained residual branch unless
    tau is initialized very large. It is mainly useful as an adapter-only
    stability ablation.
    """

    def __init__(self, tau_init: float = 1.0, eps: float = 1e-6, log_stats: bool = False) -> None:
        super().__init__()
        if tau_init <= 0:
            raise ValueError("tau_init must be positive")
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())
        self.eps = float(eps)
        self.log_stats = bool(log_stats)
        self.last_stats: Optional[Dict[str, float]] = None

    @staticmethod
    def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.sqrt(torch.mean(x.float().pow(2), dim=-1, keepdim=True) + eps)

    def forward(self, ref: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        rms_ref = self.rms(ref, self.eps)
        rms_update = self.rms(update, self.eps)
        ratio = torch.log((rms_update + self.eps) / (rms_ref + self.eps))
        scale = torch.sigmoid(self.log_tau - ratio)
        if self.log_stats:
            with torch.no_grad():
                self.last_stats = {
                    "ratio_mean": float(ratio.mean().detach().cpu()),
                    "scale_mean": float(scale.mean().detach().cpu()),
                    "scale_min": float(scale.min().detach().cpu()),
                    "scale_max": float(scale.max().detach().cpu()),
                    "tau": float(self.log_tau.exp().detach().cpu()),
                }
        return scale.to(update.dtype)
