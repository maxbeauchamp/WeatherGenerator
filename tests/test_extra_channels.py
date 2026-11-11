# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for input channels added after pre-training (weathergen.model.extra_channels)."""

import re

import pytest
import torch
from omegaconf import OmegaConf

from weathergen.model.engines import EmbeddingEngine
from weathergen.model.extra_channels import (
    ExtraChannelEmbed,
    extra_channel_columns,
    source_channel_names,
)
from weathergen.model.model import Model

OLD = ["10u", "10v", "2d", "2t", "sp", "tp"]
# anemoi order (dataset index, alphabetical here): the new channels are NOT at the end
NEW = ["10u", "10v", "2d", "2t", "sp", "ssrd", "strd", "tp"]
EXTRA = ["ssrd", "strd"]
META = 7  # stream id, time, local coords, geoinfo columns before the data (any number works)
TOKEN_SIZE, NUM_TOKENS, DIM_OUT = 4, 1, 32


def _cf(extra, names):
    era5 = {
        "type": "anemoi",
        "token_size": TOKEN_SIZE,
        "embed": {"net": "transformer", "num_tokens": NUM_TOKENS, "num_heads": 2,
                  "dim_embed": 16, "num_blocks": 1},
        "val_source_channels": names,
    }
    if extra is not None:
        era5["embed_extra_channels"] = extra
    return OmegaConf.create(
        {
            "mixed_precision_dtype": "fp32",
            "ae_local_dim_embed": DIM_OUT,
            "embed_dropout_rate": 0.0,
            "norm_type": "LayerNorm",
            "embed_unembed_mode": "full",
            "streams": {"ERA5": era5},
        }
    )


def test_source_channel_names_prefers_val():
    si = {"train_source_channels": OLD, "val_source_channels": NEW}
    assert source_channel_names(si) == NEW  # inference: only 'val' is current
    assert source_channel_names({"train_source_channels": NEW}) == NEW
    with pytest.raises(ValueError):
        source_channel_names({})


def test_columns():
    base, extra = extra_channel_columns(NEW, META + len(NEW), EXTRA)
    assert extra == [META + 5, META + 6]
    assert base == list(range(META + 5)) + [META + 7]
    # the old data columns stay in the pre-trained order
    assert [NEW[c - META] for c in base if c >= META] == OLD
    with pytest.raises(ValueError):
        extra_channel_columns(NEW, META + len(NEW), ["msl"])
    with pytest.raises(ValueError):
        extra_channel_columns(NEW, META + len(NEW), ["ssrd", "ssrd"])


def test_zero_init_output_is_zero_and_gradient_reaches_output_layer():
    emb = ExtraChannelEmbed(2, TOKEN_SIZE, NUM_TOKENS, DIM_OUT, "ERA5")
    x = torch.randn(5, TOKEN_SIZE, 2)
    out = emb(x)
    assert out.shape == (5, NUM_TOKENS, DIM_OUT)
    assert torch.count_nonzero(out) == 0
    out.sum().backward()
    assert emb.w2.grad.abs().sum() > 0 and emb.b2.grad.abs().sum() > 0


def test_nan_does_not_leak():
    emb = ExtraChannelEmbed(2, TOKEN_SIZE, NUM_TOKENS, DIM_OUT, "ERA5")
    with torch.no_grad():
        emb.w2.normal_()
    x = torch.randn(3, TOKEN_SIZE, 2)
    x[0, 1, 0] = float("nan")
    assert torch.isfinite(emb(x)).all()


def test_engine_is_exactly_the_pretrained_embedder_at_start():
    torch.manual_seed(0)
    old = EmbeddingEngine(_cf(None, OLD), [META + len(OLD)])
    new = EmbeddingEngine(_cf(EXTRA, NEW), [META + len(NEW)])
    # the pre-trained checkpoint loads into the new engine: same keys and shapes for the embedder
    sd_old = old.state_dict()
    missing, unexpected = new.load_state_dict(sd_old, strict=False)
    assert unexpected == []
    assert all(k.startswith("extra_embeds.ERA5.") for k in missing) and missing
    old.eval()
    new.eval()

    x_new = torch.randn(9, TOKEN_SIZE, META + len(NEW))
    base_cols, _ = new.extra_cols["ERA5"]
    x_old = x_new[..., base_cols]
    # the stream embedder casts q, k, v to bf16 and relies on autocast, as in training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    old, new, x_new, x_old = old.to(device), new.to(device), x_new.to(device), x_old.to(device)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        assert torch.equal(new.embed_stream("ERA5", x_new), old.embed_stream("ERA5", x_old))
        # once trained, the new channels change the tokens
        new.extra_embeds["ERA5"].w2.normal_()
        assert not torch.equal(new.embed_stream("ERA5", x_new), old.embed_stream("ERA5", x_old))


def test_disabled_path_unchanged():
    eng = EmbeddingEngine(_cf(None, OLD), [META + len(OLD)])
    assert eng.extra_cols == {} and len(eng.extra_embeds) == 0
    assert not any(k.startswith("extra_embeds") for k in eng.state_dict())


def test_model_reset_parameters_keeps_zero_init():
    container = torch.nn.Module()
    container.extra = ExtraChannelEmbed(2, TOKEN_SIZE, NUM_TOKENS, DIM_OUT, "ERA5")
    with torch.no_grad():
        container.extra.w2.fill_(1.0)
    Model.reset_parameters(container)
    assert torch.count_nonzero(container.extra.w2) == 0
    assert torch.count_nonzero(container.extra.b2) == 0


def test_freeze_regex_of_the_phase1_config():
    """Phase 1 trains only the extra-channel embedder (config finetune_ice_era5_rad_phase1.yml)."""
    freeze = r"^(?!(encoder|encoder\.embed_engine|encoder\.embed_engine\.extra_embeds)$)(?!.*ExtraChannelEmbed_ERA5$).*$"  # noqa: E501
    frozen = lambda n: re.fullmatch(freeze, n) is not None  # noqa: E731
    assert not frozen("ExtraChannelEmbed_ERA5")
    for parent in ["encoder", "encoder.embed_engine", "encoder.embed_engine.extra_embeds"]:
        assert not frozen(parent)
    for n in ["StreamEmbedder_ERA5", "StreamEmbedder_CICE", "encoder.embed_engine.embeds",
              "forecast_engine", "TargetPredictionEngine_CICE", "ForcingInjection",
              "encoder.ae_local_engine"]:
        assert frozen(n)
