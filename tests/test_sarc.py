import torch

from motif_upcycling.sarc import ScaleAwareDeltaScaler, ScaleAwareResidualCorrection
from motif_upcycling.params import sarc_param_count
from motif_upcycling.utils import trainable_parameter_count


def test_sarc_residual_identity_init_fp32():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    u = torch.randn(2, 5, 16)
    gate = ScaleAwareResidualCorrection(hidden=16, max_delta=0.10)
    y = gate(x, u)
    y_ref = x + u
    assert (y - y_ref).abs().max().item() < 1e-6


def test_sarc_zero_update():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    u = torch.zeros_like(x)
    gate = ScaleAwareResidualCorrection(hidden=16)
    y = gate(x, u)
    assert (y - x).abs().max().item() < 1e-6


def test_sarc_no_nan_with_zero_ref():
    x = torch.zeros(2, 5, 16)
    u = torch.randn(2, 5, 16)
    gate = ScaleAwareResidualCorrection(hidden=16)
    y = gate(x, u)
    assert torch.isfinite(y).all()


def test_sarc_param_count():
    scaler = ScaleAwareDeltaScaler(hidden=16)
    assert trainable_parameter_count(scaler) == sarc_param_count(hidden=16)
