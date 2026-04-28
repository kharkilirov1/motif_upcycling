"""Motif-Upcycling: structure-preserving adaptation of Transformer models."""

from .config import MotifUpcycleConfig, SARCConfig
from .lora import LowRankAdapter
from .params import motif_lora_param_count, p4_layer_param_count, router_param_count, sarc_param_count
from .patch import get_decoder_layers, patch_qwen_mlp_layers
from .routers import ContextualRouter, StaticRouter
from .sarc import BoundedRelativeGate, ScaleAwareDeltaScaler, ScaleAwareResidualCorrection
from .swiglu import MotifSwiGLUMLP
from .utils import even_motif_sizes, max_abs_diff, motif_slices, total_parameter_count, trainable_parameter_count

__all__ = [
    "MotifUpcycleConfig",
    "SARCConfig",
    "LowRankAdapter",
    "router_param_count",
    "motif_lora_param_count",
    "p4_layer_param_count",
    "sarc_param_count",
    "get_decoder_layers",
    "patch_qwen_mlp_layers",
    "ContextualRouter",
    "StaticRouter",
    "BoundedRelativeGate",
    "ScaleAwareDeltaScaler",
    "ScaleAwareResidualCorrection",
    "MotifSwiGLUMLP",
    "even_motif_sizes",
    "motif_slices",
    "max_abs_diff",
    "trainable_parameter_count",
    "total_parameter_count",
]
