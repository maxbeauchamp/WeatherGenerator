# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the flow-matching decoders (weathergen.model.flow_matching)."""

from types import SimpleNamespace

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


def test_config_validation_sampling():
    assert _cfg()["validation"] == {"sampling": False, "num_members": 2, "num_steps": 4}
    assert _cfg(validation={"sampling": True})["validation"]["sampling"] is True
    with pytest.raises(AssertionError):
        _cfg(validation={"num_steps": 0})


class _FakeModel:
    """The attributes of Model that predict_flow_matching reads."""

    _fm_gather = Model._fm_gather
    predict_flow_matching = Model.predict_flow_matching
    set_flow_matching_sampling = Model.set_flow_matching_sampling

    def __init__(self, cfg):
        self.fm_cfg = self.fm_sampling_cfg = cfg
        self.fm_sampling, self.training = False, False
        self.fm_conditioners = {"CICE": fm.FlowMatchingConditioner(C, DIM, "CICE")}


def _fm_batch(n_per_step):
    """One sample of the stream CICE with n_per_step[k] target points at step k."""
    sd = StreamData(0, 1, len(n_per_step), 12)
    for k, n in enumerate(n_per_step):
        sd.add_fm_values(k, torch.randn(n, C), torch.randn(n, C))
    return SimpleNamespace(samples=[SimpleNamespace(streams_data={"CICE": sd})])


def _predict(model, batch, step, n, fm_state):
    def decode_points(cond):
        return cond[:, :C].unsqueeze(0)  # (1, N, C)

    return model.predict_flow_matching(
        "CICE", step, batch, torch.zeros(n, DIM), decode_points, [n], fm_state
    )


def test_sampling_settings_of_the_validation():
    model = _FakeModel(_cfg())
    model.set_flow_matching_sampling(True, {"num_members": 2, "num_steps": 2})
    assert model.fm_sampling and model.fm_sampling_cfg["inference"]["num_members"] == 2
    assert model.fm_cfg["inference"]["num_members"] == 8  # the inference settings stay
    assert _predict(model, _fm_batch([5]), 0, 5, {}).shape == (2, 5, C)


def test_warning_when_the_members_restart_from_the_truth(monkeypatch):
    import weathergen.model.model as model_module  # noqa: PLC0415

    seen = []
    monkeypatch.setattr(model_module.logger, "warning", seen.append)
    model = _FakeModel(_cfg())
    model.set_flow_matching_sampling(True, {"num_members": 2, "num_steps": 1})
    batch, state = _fm_batch([5, 5, 4]), {}
    for step, n in enumerate([5, 5]):
        _predict(model, batch, step, n, state)
    assert seen == []  # same points: every member continues from its own state
    _predict(model, batch, 2, 4, state)
    assert len(seen) == 1 and "TRUE previous state" in seen[0]


def test_validation_switches_only_models_not_sampling():
    from weathergen.train.trainer import Trainer  # noqa: PLC0415

    cf = OmegaConf.create({"flow_matching": {"enabled": True, "streams": ["CICE"],
                                             "validation": {"sampling": True}}})  # fmt: skip
    model, ema = _FakeModel(_cfg()), _FakeModel(_cfg())
    ema.fm_sampling = True  # already sampling (inference): left as is
    trainer = SimpleNamespace(cf=cf, model=model, ema_model=SimpleNamespace(ema_model=ema))
    cores = Trainer._set_fm_validation_sampling(trainer)
    assert cores == [model] and model.fm_sampling_cfg["inference"]["num_steps"] == 4
    cf.flow_matching.validation.sampling = False
    model.fm_sampling = False
    assert Trainer._set_fm_validation_sampling(trainer) == [] and not model.fm_sampling
