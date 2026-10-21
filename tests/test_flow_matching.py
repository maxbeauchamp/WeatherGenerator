# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the flow-matching decoders (weathergen.model.flow_matching)."""

import pytest
import torch
from omegaconf import OmegaConf

from weathergen.datasets.stream_data import StreamData
from weathergen.model import flow_matching as fm
from weathergen.model.model import Model

C, DIM = 3, 32


def _cfg(**kw):
    base = {"enabled": True, "streams": ["CICE"]}
    base.update(kw)
    return fm.get_flow_matching_config(OmegaConf.create({"flow_matching": base}))


def test_config_disabled_by_default():
    assert fm.get_flow_matching_config(OmegaConf.create({}))["enabled"] is False
    assert _cfg(streams=[])["enabled"] is False


def test_config_values_and_validation():
    c = _cfg(inference={"num_members": 4})
    assert c["enabled"] and c["streams"] == ["CICE"]
    assert c["inference"]["num_members"] == 4 and c["inference"]["solver"] == "heun"
    with pytest.raises(AssertionError):
        _cfg(target="velocity")
    with pytest.raises(AssertionError):
        _cfg(inference={"solver": "rk4"})


def test_conditioner_zero_at_init_and_after_reset():
    cond = fm.FlowMatchingConditioner(C, DIM, "CICE")
    x = torch.randn(10, C)
    out = cond(x, torch.full((10, C), float("nan")), torch.rand(10))
    assert out.shape == (10, DIM) and torch.count_nonzero(out) == 0
    with torch.no_grad():
        cond.w2.fill_(1.0)
    container = torch.nn.Module()
    container.c = cond
    Model.reset_parameters(container)  # used for modules missing from a checkpoint
    assert torch.count_nonzero(cond.w2) == 0 and torch.count_nonzero(cond.b2) == 0


def test_conditioner_gradient_reaches_output_layer():
    cond = fm.FlowMatchingConditioner(C, DIM, "CICE")
    cond(torch.randn(8, C), torch.randn(8, C), torch.rand(8)).sum().backward()
    assert cond.w2.grad.abs().sum() > 0


def test_training_interpolant_end_points():
    c = _cfg(target="increment")
    x_tgt, x_prev = torch.randn(50, C), torch.randn(50, C)
    g = torch.Generator().manual_seed(0)
    x_t, x_prev_c, base = fm.training_inputs(x_tgt, x_prev, torch.ones(50), c, generator=g)
    assert torch.allclose(x_t, x_tgt - x_prev)  # t = 1: data (increment)
    assert torch.equal(base, x_prev_c)
    g = torch.Generator().manual_seed(1)
    x_t0, _, _ = fm.training_inputs(x_tgt, x_prev, torch.zeros(50), c, generator=g)
    eps = torch.randn((50, C), generator=torch.Generator().manual_seed(1))
    assert torch.allclose(x_t0, eps)  # t = 0: noise


def test_loss_weighting():
    pred, tgt, t = torch.randn(1, 20, C), torch.randn(20, C), torch.rand(20) * 0.9
    assert torch.equal(fm.apply_loss_weighting(pred, tgt, t, _cfg(loss_weighting="x")), pred)
    pv = fm.apply_loss_weighting(pred, tgt, t, _cfg(loss_weighting="v"))
    w = 1.0 / (1.0 - t) ** 2
    assert torch.allclose(((pv[0] - tgt) ** 2), w[:, None] * (pred[0] - tgt) ** 2, rtol=1e-4)


@pytest.mark.parametrize("target", ["state", "increment"])
def test_sampler_recovers_gaussian(target):
    """With the optimal x-prediction denoiser of y ~ N(mu, s^2), Heun sampling gives N(mu, s^2)."""
    torch.manual_seed(0)
    n, mu, s = 100_000, 0.5, 0.7
    c = _cfg(target=target, inference={"num_steps": 16, "solver": "heun"})
    x_prev = 0.3 + 0.2 * torch.randn(n, 1)

    def denoise(x_t, xp, t):
        y_hat = mu + t * s**2 / ((1 - t) ** 2 + t**2 * s**2) * (x_t - t * mu)
        return y_hat + (xp if target == "increment" else 0.0)  # decoder returns a state

    x = fm.sample_member(denoise, x_prev, torch.randn(n, 1), c)
    y = x - (x_prev if target == "increment" else 0.0)
    assert abs(y.mean().item() - mu) < 0.02
    assert abs(y.std().item() - s) < 0.02


def test_sampler_constant_denoiser_is_exact_identity():
    """Zero-initialised conditioning: every member equals the deterministic prediction."""
    d = torch.randn(30, C)
    x = fm.sample_member(lambda x_t, xp, t: d, torch.randn(30, C), torch.randn(30, C), _cfg())
    assert torch.equal(x, d)


def test_noise_correlation():
    torch.manual_seed(0)
    e0 = torch.randn(200_000)
    e1 = fm.next_noise(e0, e0.shape, "cpu", 0.8)
    assert abs(torch.corrcoef(torch.stack([e0, e1]))[0, 1].item() - 0.8) < 0.01
    assert abs(e1.std().item() - 1.0) < 0.01
    assert not torch.equal(fm.next_noise(e0, e0.shape, "cpu", 0.0), e0)


def test_stream_data_fm_fields_device():
    sd = StreamData(0, 1, 3, 12)
    assert sd.fm_target == [None, None, None]
    sd.add_fm_values(1, torch.zeros(5, C), torch.ones(5, C))
    sd.to_device("cpu")
    assert sd.fm_prev[1].shape == (5, C) and sd.fm_target[0] is None
