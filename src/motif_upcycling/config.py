"""Configuration objects for motif-upcycling and SARC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

RouterType = Literal["contextual", "static", "none"]
SARCMode = Literal["off", "adapter_only", "full_update", "motif_wise", "bounded_clip"]
SARCRatioMode = Literal["update_over_x", "delta_over_x", "delta_over_base"]


@dataclass
class SARCConfig:
    """Scale-Aware Residual Control configuration.

    The safest pretrained-model mode is ``adapter_only``: the frozen/base update is
    left unchanged and SARC scales only the trainable delta. Because the scale
    controller is zero-initialized, the module starts as an exact identity on the
    current baseline.
    """

    enabled: bool = False
    mode: SARCMode = "off"
    hidden: int = 16
    eps: float = 1e-6
    max_delta: float = 0.10
    ratio_mode: SARCRatioMode = "delta_over_x"
    detach_ratio: bool = False
    # bounded_clip is an ablation, not the default pretrained-safe path.
    bounded_tau_init: float = 1.0
    log_stats: bool = False


@dataclass
class MotifUpcycleConfig:
    """Configuration for a motif-upcycled Qwen-style SwiGLU MLP.

    Args:
        num_motifs: Number of channel-slice experts.
        router_hidden: Hidden width of the contextual router MLP.
        router_type: ``contextual``, ``static``, or ``none``.
        lora_rank: Rank for each motif-local LoRA module. Set 0 to disable LoRA.
        lora_alpha: Scaling factor. Effective LoRA scale is ``alpha / rank``.
        motif_sizes: Optional explicit intermediate-channel partition. If omitted,
            channels are split evenly.
        freeze_base: Freeze all parameters in the wrapped dense MLP.
        sarc: Optional Scale-Aware Residual Control configuration.
    """

    num_motifs: int = 3
    router_hidden: int = 128
    router_type: RouterType = "contextual"
    lora_rank: int = 8
    lora_alpha: float = 8.0
    motif_sizes: Optional[Sequence[int]] = None
    freeze_base: bool = True
    sarc: SARCConfig = field(default_factory=SARCConfig)
