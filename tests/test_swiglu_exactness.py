import torch
import torch.nn as nn
import torch.nn.functional as F

from motif_upcycling import MotifSwiGLUMLP, MotifUpcycleConfig, SARCConfig
from motif_upcycling.utils import max_abs_diff, trainable_parameter_count


class TinySwiGLU(nn.Module):
    def __init__(self, d=16, i=30, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(d, i, bias=bias)
        self.up_proj = nn.Linear(d, i, bias=bias)
        self.down_proj = nn.Linear(i, d, bias=bias)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_neutral_exactness_no_sarc():
    torch.manual_seed(0)
    dense = TinySwiGLU()
    x = torch.randn(2, 7, 16)
    y_dense = dense(x)
    cfg = MotifUpcycleConfig(num_motifs=3, router_hidden=8, router_type="contextual", lora_rank=2)
    wrapped = MotifSwiGLUMLP(dense, cfg)
    y_wrap = wrapped(x)
    assert max_abs_diff(y_dense, y_wrap) < 1e-6


def test_neutral_exactness_adapter_only_sarc():
    torch.manual_seed(0)
    dense = TinySwiGLU()
    x = torch.randn(2, 7, 16)
    y_dense = dense(x)
    cfg = MotifUpcycleConfig(
        num_motifs=3,
        router_hidden=8,
        router_type="contextual",
        lora_rank=2,
        sarc=SARCConfig(enabled=True, mode="adapter_only"),
    )
    wrapped = MotifSwiGLUMLP(dense, cfg)
    y_wrap = wrapped(x)
    assert max_abs_diff(y_dense, y_wrap) < 1e-6


def test_neutral_exactness_motif_wise_sarc():
    torch.manual_seed(0)
    dense = TinySwiGLU()
    x = torch.randn(2, 7, 16)
    y_dense = dense(x)
    cfg = MotifUpcycleConfig(
        num_motifs=3,
        router_hidden=8,
        router_type="contextual",
        lora_rank=2,
        sarc=SARCConfig(enabled=True, mode="motif_wise"),
    )
    wrapped = MotifSwiGLUMLP(dense, cfg)
    y_wrap = wrapped(x)
    assert max_abs_diff(y_dense, y_wrap) < 1e-6


def test_trainable_count_positive():
    dense = TinySwiGLU()
    cfg = MotifUpcycleConfig(num_motifs=3, router_hidden=8, router_type="contextual", lora_rank=2)
    wrapped = MotifSwiGLUMLP(dense, cfg)
    assert trainable_parameter_count(wrapped) > 0
