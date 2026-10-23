# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the decoder neighbourhood (weathergen.model.decoder_neighbourhood)."""

import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.datasets.utils import healpix_verts_rots, locs_to_cell_coords_ctrs
from weathergen.model import decoder_neighbourhood as dn
from weathergen.model.model import Model

HL = 3  # small HEALPix level for the tests (768 cells)
H = 4


def _cfg(**kw):
    base = {"enabled": True}
    base.update(kw)
    return dn.get_decoder_neighbourhood_config(OmegaConf.create({"decoder_neighbourhood": base}))


def test_config():
    assert dn.get_decoder_neighbourhood_config(OmegaConf.create({}))["enabled"] is False
    c = _cfg(rings=2, streams=["CICE"])
    assert c["enabled"] and c["rings"] == 2 and c["streams"] == ["CICE"]
    assert dn.uses_neighbourhood(c, "CICE") and not dn.uses_neighbourhood(c, "ERA5")
    assert dn.uses_neighbourhood(_cfg(), "ANY")  # [] = all decoders
    with pytest.raises(AssertionError):
        _cfg(rings=3)


def test_neighbourhood_ring1_is_the_original_one():
    import astropy_healpix as hp

    idx, pad, ring = dn.build_neighbourhood(HL, 1)
    n = 12 * 4**HL
    temp = hp.neighbours(np.arange(n), 2**HL, order="nested").transpose()
    for i, row in enumerate(temp):
        row[row == -1] = i
    assert idx.shape == (n, 9) and not pad.any()
    assert (idx[:, 0] == np.arange(n)).all() and (idx[:, 1:] == temp).all()
    assert ring.tolist() == [0] + [1] * 8


def test_neighbourhood_ring2():
    idx1, _, _ = dn.build_neighbourhood(HL, 1)
    idx, pad, ring = dn.build_neighbourhood(HL, 2)
    assert idx.shape[1] == 25 and (idx[:, :9] == idx1).all() and not pad[:, :9].any()
    assert (ring[9:] == 2).all()
    for i in range(len(idx)):
        r1, r2 = set(idx[i, :9].tolist()), idx[i, 9:][~pad[i, 9:]].tolist()
        assert len(set(r2)) == len(r2) and not (set(r2) & r1)
    n2 = (~pad[:, 9:]).sum(1)
    assert np.mean(n2 == 16) > 0.8 and n2.min() >= 10  # 16 except near the 8 HEALPix vertices


def _points_and_coords(geo, n_per_cell=3, jitter=0.3, coord_dim=120, seed=0):
    """Random points inside each cell and the coordinate encoding as built by the tokenizer."""
    g = torch.Generator().manual_seed(seed)
    ctrs, rots = healpix_verts_rots(HL, 0.5, 0.5)
    ctrs = ctrs.to(torch.float32)
    n = len(ctrs)
    p = ctrs.repeat_interleave(n_per_cell, 0)
    p = p + jitter * geo.d_cell * torch.randn(p.shape, generator=g)
    p = p / p.norm(dim=-1, keepdim=True)
    locs = list(torch.split(p, n_per_cell))
    a_self = torch.tensor([1.0, 0.0, 0.0]) - locs_to_cell_coords_ctrs(rots.to(torch.float32), locs)
    coords = torch.zeros((len(p), coord_dim))
    coords[:, -39:-36] = a_self
    cell_idx = torch.arange(n).repeat_interleave(n_per_cell)
    return p, coords, cell_idx, ctrs


@pytest.mark.parametrize("rings", [1, 2])
def test_relative_features_match_true_distances(rings):
    geo = dn.DecoderGeometry(HL, rings)
    # jitter 0.1 cell: the points stay (almost always) inside their cell
    p, coords, cell_idx, ctrs = _points_and_coords(geo, jitter=0.1)
    feats = geo.relative_features(coords, cell_idx)
    assert feats.shape == (len(p), geo.num_slots, 3)
    d2_true = ((p[:, None, :] - ctrs[geo.nbours[cell_idx]]) ** 2).sum(-1) / geo.d_cell**2
    assert torch.allclose(feats[..., 2], d2_true, atol=1e-3, rtol=1e-3)
    # the own cell is (almost always) the closest slot
    assert (d2_true.argmin(1) == 0).float().mean() > 0.9


def test_relative_features_zero_at_cell_centre():
    geo = dn.DecoderGeometry(HL, 1)
    _, coords, cell_idx, _ = _points_and_coords(geo, n_per_cell=1, jitter=0.0)
    feats = geo.relative_features(coords, cell_idx)
    assert feats[:, 0].abs().max() < 1e-4
    assert 0.5 < feats[:, 1:, 2].mean() < 2.5  # first ring ~1 cell away


def test_bias_zero_at_init_and_after_reset():
    b = dn.RelativePositionBias(H, 16, "RelativePositionBias_X_0", ring2_init_logit=-50.0)
    geo = dn.DecoderGeometry(HL, 2)
    _, coords, cell_idx, _ = _points_and_coords(geo)
    feats = geo.relative_features(coords, cell_idx)
    out = b(feats, geo.pad[cell_idx], geo.slot_ring)
    assert out.shape == (len(cell_idx), H, 25)
    assert torch.count_nonzero(out[..., :9]) == 0  # first ring: exactly no bias
    open2 = ~geo.pad[cell_idx][:, None, 9:].expand(-1, H, -1)
    assert torch.all(out[..., 9:][open2] == -50.0)
    assert torch.all(out[..., 9:][~open2] == dn.PAD_LOGIT)
    with torch.no_grad():
        b.w2.fill_(1.0)
    container = torch.nn.Module()
    container.b = b
    Model.reset_parameters(container)
    assert torch.count_nonzero(b.w2) == 0 and torch.all(b.ring2_logit == -50.0)


def test_bias_gradient():
    b = dn.RelativePositionBias(H, 16, "RelativePositionBias_X_0")
    feats = torch.randn(10, 9, 3)
    pad, ring = torch.zeros(10, 9, dtype=torch.bool), torch.zeros(9, dtype=torch.long)
    b(feats, pad, ring).sum().backward()
    assert b.w2.grad.abs().sum() > 0


def test_augment_qkv_is_an_exact_additive_bias():
    torch.manual_seed(0)
    nq, nk, d, k = 6, 9, 32, 9
    qs, ks, vs = torch.randn(nq, H, d), torch.randn(nk, H, d), torch.randn(nk, H, d)
    bias = 3.0 * torch.randn(nq, H, k)
    onehot = torch.eye(k)[torch.arange(nk) % k]
    qa, ka, va, scale = dn.augment_qkv(qs, ks, vs, bias, onehot)
    assert qa.shape[-1] % 8 == 0 and scale == pytest.approx(1 / math.sqrt(d))
    att_aug = torch.softmax(torch.einsum("qhd,khd->hqk", qa, ka) * scale, -1)
    ref = torch.einsum("qhd,khd->hqk", qs, ks) / math.sqrt(d) + bias.permute(1, 0, 2)
    att_ref = torch.softmax(ref, -1)
    assert torch.allclose(att_aug, att_ref, atol=1e-5)
    out = torch.einsum("hqk,khd->qhd", att_aug, va)[..., :d]
    assert torch.allclose(out, torch.einsum("hqk,khd->qhd", att_ref, vs), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flash attention needs a GPU")
def test_cross_attention_zero_bias_matches_original():
    from weathergen.model.attention import MultiCrossAttentionHeadVarlen

    torch.manual_seed(0)
    dev, dim_q, dim_kv, n_cells, n_q = "cuda", 64, 128, 7, 5
    att = MultiCrossAttentionHeadVarlen(dim_q, dim_kv, H, dim_head_proj=32).to(dev).eval()
    x_q = torch.randn(n_cells * n_q, dim_q, device=dev)
    x_kv = torch.randn(n_cells * 9, dim_kv, device=dev)
    q_lens = torch.tensor([0] + [n_q] * n_cells, dtype=torch.int32, device=dev)
    kv_lens = torch.tensor([0] + [9] * n_cells, dtype=torch.int32, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = att(x_q, x_kv, q_lens, kv_lens)
        bias = torch.zeros(n_cells * n_q, H, 9, device=dev)
        onehot = torch.eye(9, device=dev).repeat(n_cells, 1)
        out = att(x_q, x_kv, q_lens, kv_lens, attn_bias=(bias, onehot))
    assert torch.allclose(out.float(), ref.float(), atol=2e-2)  # bf16 kernels, other head dim
