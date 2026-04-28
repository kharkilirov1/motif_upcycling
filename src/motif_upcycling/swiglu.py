"""Function-preserving motif-upcycling wrapper for Qwen-style SwiGLU MLPs."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MotifUpcycleConfig
from .lora import LowRankAdapter
from .routers import ContextualRouter, StaticRouter
from .sarc import BoundedRelativeGate, ScaleAwareDeltaScaler
from .utils import even_motif_sizes, motif_slices


def _linear(x: torch.Tensor, linear: nn.Linear, rows: Optional[slice] = None) -> torch.Tensor:
    weight = linear.weight if rows is None else linear.weight[rows, :]
    if linear.bias is None:
        bias = None
    else:
        bias = linear.bias if rows is None else linear.bias[rows]
    return F.linear(x, weight, bias)


class MotifSwiGLUMLP(nn.Module):
    """Exact-sum motif wrapper around a Qwen-style SwiGLU MLP.

    Expected dense MLP attributes: ``gate_proj``, ``up_proj``, ``down_proj`` and
    optionally ``act_fn``. This matches Hugging Face Qwen2/Qwen3/Qwen-style MLPs.

    With neutral router and zero-init LoRA/SARC, the wrapper is function-preserving
    up to normal floating-point summation differences.
    """

    def __init__(self, dense_mlp: nn.Module, config: MotifUpcycleConfig):
        super().__init__()
        self.dense_mlp = dense_mlp
        self.config = config

        for name in ("gate_proj", "up_proj", "down_proj"):
            if not hasattr(dense_mlp, name):
                raise AttributeError(f"dense_mlp must expose {name}")

        self.gate_proj: nn.Linear = dense_mlp.gate_proj
        self.up_proj: nn.Linear = dense_mlp.up_proj
        self.down_proj: nn.Linear = dense_mlp.down_proj
        self.act_fn = getattr(dense_mlp, "act_fn", F.silu)

        self.hidden_size = int(self.gate_proj.in_features)
        self.intermediate_size = int(self.gate_proj.out_features)
        if self.up_proj.in_features != self.hidden_size or self.up_proj.out_features != self.intermediate_size:
            raise ValueError("gate_proj and up_proj shapes do not match")
        if self.down_proj.in_features != self.intermediate_size or self.down_proj.out_features != self.hidden_size:
            raise ValueError("down_proj shape does not match SwiGLU dimensions")

        sizes = list(config.motif_sizes) if config.motif_sizes is not None else even_motif_sizes(
            self.intermediate_size, config.num_motifs
        )
        self.motif_sizes = sizes
        self.slices = motif_slices(self.intermediate_size, sizes)
        self.num_motifs = len(sizes)

        if config.freeze_base:
            for p in dense_mlp.parameters():
                p.requires_grad_(False)

        if config.router_type == "contextual":
            self.router: Optional[nn.Module] = ContextualRouter(
                self.hidden_size, config.router_hidden, self.num_motifs
            )
        elif config.router_type == "static":
            self.router = StaticRouter(self.num_motifs)
        elif config.router_type in ("none", None):
            self.router = None
        else:
            raise ValueError("router_type must be 'contextual', 'static', or 'none'")

        r = int(config.lora_rank)
        self.use_lora = r > 0
        if self.use_lora:
            alpha = float(config.lora_alpha)
            # Six trainable placements used in the paper's P4 variant:
            # select: gate; expand: up/down; memory: gate/up/down.
            self.lora_gate_select = LowRankAdapter(self.hidden_size, self.intermediate_size, r, alpha)
            self.lora_gate_memory = LowRankAdapter(self.hidden_size, self.intermediate_size, r, alpha)
            self.lora_up_expand = LowRankAdapter(self.hidden_size, self.intermediate_size, r, alpha)
            self.lora_up_memory = LowRankAdapter(self.hidden_size, self.intermediate_size, r, alpha)
            self.lora_down_expand = LowRankAdapter(self.intermediate_size, self.hidden_size, r, alpha)
            self.lora_down_memory = LowRankAdapter(self.intermediate_size, self.hidden_size, r, alpha)

        self.sarc_mode = config.sarc.mode if config.sarc.enabled else "off"
        if config.sarc.enabled and self.sarc_mode in {"adapter_only", "full_update"}:
            self.sarc_scaler: Optional[nn.Module] = ScaleAwareDeltaScaler(
                hidden=config.sarc.hidden,
                eps=config.sarc.eps,
                max_delta=config.sarc.max_delta,
                detach_ratio=config.sarc.detach_ratio,
                log_stats=config.sarc.log_stats,
            )
        elif config.sarc.enabled and self.sarc_mode == "bounded_clip":
            self.sarc_scaler = BoundedRelativeGate(
                tau_init=config.sarc.bounded_tau_init,
                eps=config.sarc.eps,
                log_stats=config.sarc.log_stats,
            )
        else:
            self.sarc_scaler = None

        if config.sarc.enabled and self.sarc_mode == "motif_wise":
            self.motif_sarc_scalers = nn.ModuleList(
                [
                    ScaleAwareDeltaScaler(
                        hidden=config.sarc.hidden,
                        eps=config.sarc.eps,
                        max_delta=config.sarc.max_delta,
                        detach_ratio=config.sarc.detach_ratio,
                        log_stats=config.sarc.log_stats,
                    )
                    for _ in range(self.num_motifs)
                ]
            )
        else:
            self.motif_sarc_scalers = None

    def _expert_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        for s in self.slices:
            gate_s = _linear(x, self.gate_proj, s)
            up_s = _linear(x, self.up_proj, s)
            h_s = self.act_fn(gate_s) * up_s
            # The down bias belongs to the whole summed output, not to every slice.
            out_s = F.linear(h_s, self.down_proj.weight[:, s], None)
            outs.append(out_s)
        return outs

    def _router_alpha(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.router is None:
            return None
        logits = self.router(x)
        return self.num_motifs * torch.softmax(logits, dim=-1)

    def base_update(self, x: torch.Tensor) -> torch.Tensor:
        """Routed frozen exact-split update."""

        expert_outs = self._expert_outputs(x)

        if self.motif_sarc_scalers is not None:
            expert_outs = [
                scaler(ref=x, update=expert) * expert
                for scaler, expert in zip(self.motif_sarc_scalers, expert_outs)
            ]

        alpha = self._router_alpha(x)
        if alpha is None:
            out = torch.stack(expert_outs, dim=0).sum(dim=0)
        else:
            out = torch.zeros_like(expert_outs[0])
            for m, expert in enumerate(expert_outs):
                out = out + alpha[..., m : m + 1] * expert

        if self.down_proj.bias is not None:
            out = out + self.down_proj.bias
        return out

    def lora_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Trainable motif-local LoRA delta; exactly zero at initialization."""

        if not self.use_lora:
            return torch.zeros_like(x)

        gate_base = self.gate_proj(x)
        up_base = self.up_proj(x)
        h_base = self.act_fn(gate_base) * up_base

        gate_delta = self.lora_gate_select(x) + self.lora_gate_memory(x)
        up_delta = self.lora_up_expand(x) + self.lora_up_memory(x)
        h_adapted = self.act_fn(gate_base + gate_delta) * (up_base + up_delta)
        h_delta = h_adapted - h_base

        out_delta = F.linear(h_delta, self.down_proj.weight, None)
        out_delta = out_delta + self.lora_down_expand(h_adapted) + self.lora_down_memory(h_adapted)
        return out_delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_update(x)
        delta = self.lora_delta(x)

        if self.sarc_scaler is None or self.sarc_mode == "off":
            return base + delta

        if self.sarc_mode == "adapter_only":
            # Safe default: donor/base update is left unchanged.
            ref = base if self.config.sarc.ratio_mode == "delta_over_base" else x
            scale = self.sarc_scaler(ref=ref, update=delta)
            return base + scale * delta

        if self.sarc_mode == "full_update":
            full = base + delta
            scale = self.sarc_scaler(ref=x, update=full)
            return scale * full

        if self.sarc_mode == "bounded_clip":
            # Intended as adapter-only ablation.
            scale = self.sarc_scaler(ref=x, update=delta)
            return base + scale * delta

        if self.sarc_mode == "motif_wise":
            # Motif-wise SARC is applied inside base_update(); LoRA delta remains normal.
            return base + delta

        raise ValueError(f"unknown SARC mode: {self.sarc_mode}")
