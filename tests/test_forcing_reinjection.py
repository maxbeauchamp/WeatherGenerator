# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for forcing re-injection (weathergen.model.forcing_injection)."""

import torch
from omegaconf import OmegaConf

from weathergen.datasets.batch import BatchSamples
from weathergen.datasets.stream_data import StreamData
from weathergen.model.forcing_injection import ForcingInjection, get_forcing_reinjection_config
from weathergen.model.model import Model

DIM = 16


def test_config_disabled_by_default():
    assert get_forcing_reinjection_config(OmegaConf.create({}))["enabled"] is False
    cf = OmegaConf.create({"forcing_reinjection": {"enabled": True, "streams": []}})
    assert get_forcing_reinjection_config(cf)["enabled"] is False  # no stream -> disabled
    cf = OmegaConf.create({"forcing_reinjection": {"enabled": False, "streams": ["ERA5"]}})
    assert get_forcing_reinjection_config(cf)["enabled"] is False


def test_config_enabled():
    cf = OmegaConf.create({"forcing_reinjection": {"enabled": True, "streams": ["ERA5"]}})
    fr = get_forcing_reinjection_config(cf)
    assert fr == {"enabled": True, "streams": ["ERA5"], "train_encoder": False}


def test_zero_init_is_exact_identity():
    inj = ForcingInjection(DIM)
    tokens = torch.randn(2, 7, DIM)
    forcing = torch.randn(2, 7, DIM)
    forcing[0, 3] = float("nan")  # must not leak into the latent state
    out = inj(tokens, forcing)
    assert torch.equal(out, tokens)


def test_gradient_reaches_projection():
    inj = ForcingInjection(DIM)
    tokens = torch.randn(2, 5, DIM)
    forcing = torch.randn(2, 5, DIM)
    inj(tokens, forcing).pow(2).sum().backward()
    # d/dW of (tokens + W f + b)^2 at W = 0 is 2 * tokens^T f: non zero
    assert inj.weight.grad is not None and inj.weight.grad.abs().sum() > 0
    assert inj.bias.grad is not None and inj.bias.grad.abs().sum() > 0


def test_model_reset_parameters_keeps_zero_init():
    container = torch.nn.Module()
    container.forcing_injection = ForcingInjection(DIM)
    with torch.no_grad():
        container.forcing_injection.weight.fill_(1.0)
    # Model.reset_parameters is used for new modules / fresh FSDP init: must zero the injection
    Model.reset_parameters(container)
    assert torch.count_nonzero(container.forcing_injection.weight) == 0
    assert torch.count_nonzero(container.forcing_injection.bias) == 0


def _batch_samples(n_samples=3, n_cells=12):
    names = ["A", "B"]
    bs = BatchSamples(names, n_samples, 1, [0])
    for s in bs.samples:
        for name in names:
            s.add_stream_data(name, StreamData(0, 1, 1, n_cells))
    bs.tokens_lens = torch.zeros((1, n_samples, len(names), n_cells), dtype=torch.int32)
    return bs


def test_batch_samples_forcing_default_none():
    assert _batch_samples().get_forcing_samples() is None


def test_batch_samples_subset_and_device_with_forcing():
    bs = _batch_samples()
    bs.forcing_samples = {1: _batch_samples(), 2: _batch_samples()}
    sub = bs.get_subset([0, 2])
    assert len(sub) == 2
    assert set(sub.get_forcing_samples()) == {1, 2}
    assert all(len(v) == 2 for v in sub.get_forcing_samples().values())
    assert all(v.tokens_lens.shape[1] == 2 for v in sub.get_forcing_samples().values())
    sub.to_device("cpu")
    assert all(v.get_device() == "cpu" for v in sub.get_forcing_samples().values())
