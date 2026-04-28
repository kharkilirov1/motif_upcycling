"""Helpers for patching Hugging Face Qwen/Qwen-like models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch.nn as nn

from .config import MotifUpcycleConfig
from .swiglu import MotifSwiGLUMLP


def get_decoder_layers(model: nn.Module) -> Any:
    """Return a decoder-layer list for common HF causal LM layouts."""

    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for parent_name, layers_name in candidates:
        parent = getattr(model, parent_name, None)
        if parent is not None and hasattr(parent, layers_name):
            return getattr(parent, layers_name)
    if hasattr(model, "layers"):
        return getattr(model, "layers")
    raise AttributeError("could not locate decoder layers; expected model.model.layers or similar")


def patch_qwen_mlp_layers(
    model: nn.Module,
    layer_indices: Iterable[int],
    config: MotifUpcycleConfig | None = None,
    freeze_all: bool = True,
) -> nn.Module:
    """Patch selected Qwen/Qwen-like decoder MLPs in-place.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> from motif_upcycling import MotifUpcycleConfig, patch_qwen_mlp_layers
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype="auto")
        >>> cfg = MotifUpcycleConfig(lora_rank=8, router_hidden=128, router_type="contextual")
        >>> patch_qwen_mlp_layers(model, [15, 16, 17, 18], cfg)
    """

    if config is None:
        config = MotifUpcycleConfig()

    if freeze_all:
        for p in model.parameters():
            p.requires_grad_(False)

    layers = get_decoder_layers(model)
    for idx in layer_indices:
        old_mlp = layers[int(idx)].mlp
        layers[int(idx)].mlp = MotifSwiGLUMLP(old_mlp, config)
    return model
