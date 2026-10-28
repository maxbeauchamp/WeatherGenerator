# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests of the partition of unity of the decoders (model/partition_of_unity.py)."""

import numpy as np
import pytest
import torch
from astropy_healpix.healpy import ang2pix
from omegaconf import OmegaConf

from weathergen.datasets.utils import s2tor3
from weathergen.model.partition_of_unity import (
    PUGeometry,
    blend,
    get_partition_of_unity_config,
    uses_pu,
    wendland,
)

HL = 5  # level of the models


@pytest.fixture(scope="module")
def geo():
    return PUGeometry(HL, 1.0)


def random_points(n, seed=0, hl=HL):
    """Uniform points on the sphere, in the tokenizer convention (theta, phi) -> xyz, cell."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, 3, generator=g, dtype=torch.float64)
    v = v / v.norm(dim=1, keepdim=True)
    th = torch.arccos(v[:, 2])
    ph = torch.atan2(v[:, 1], v[:, 0]) % (2 * np.pi)
    cell = torch.from_numpy(ang2pix(2**hl, th.numpy(), ph.numpy(), nest=True))
    return th, ph, s2tor3(th, ph), cell


def cell_of(xyz, hl=HL):
    th = torch.arccos(xyz[:, 2].clamp(-1, 1))
    ph = torch.atan2(xyz[:, 1], xyz[:, 0]) % (2 * np.pi)
    return torch.from_numpy(ang2pix(2**hl, th.numpy(), ph.numpy(), nest=True))


def boundary_pairs(geo, n, seed=0, gap=0.0):
    """Pairs of points at distance ~eps on either side of a cell boundary: bisection along the
    great circle from a point near the centre of a random cell to a point near a neighbour
    centre. gap: extra offset of both points from the boundary (in segment units, 1 ~ d_cell).
    Returns (x_minus, cell_minus, x_plus, cell_plus)."""
    g = torch.Generator().manual_seed(seed)
    c = torch.randint(0, geo.ctrs.shape[0], (n,), generator=g)
    j = torch.randint(1, 9, (n,), generator=g)
    nb = geo.cands[c, j]
    c, nb = c[nb >= 0], nb[nb >= 0]
    jitter = 0.3 * geo.d_cell * torch.randn(len(c), 3, generator=g, dtype=torch.float64)
    p0 = geo.ctrs[c] + jitter * 0.5
    p1 = geo.ctrs[nb] + jitter * 0.5
    lo, hi = torch.zeros(len(c), dtype=torch.float64), torch.ones(len(c), dtype=torch.float64)

    def at(t):
        x = p0 + t[:, None] * (p1 - p0)
        return x / x.norm(dim=1, keepdim=True)

    hl = geo.healpix_level
    c0 = cell_of(at(lo), hl)
    keep = c0 != cell_of(at(hi), hl)
    lo, hi, c0 = lo[keep], hi[keep], c0[keep]
    p0, p1 = p0[keep], p1[keep]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        same = cell_of(at(mid), hl) == c0
        lo, hi = torch.where(same, mid, lo), torch.where(same, hi, mid)
    xm, xp = at(lo - gap), at(hi + gap)
    assert (xm - xp).norm(dim=1).max() < 1e-6 + 4 * gap * geo.d_cell
    return xm, cell_of(xm, hl), xp, cell_of(xp, hl)


def dense_weights(geo, x, c):
    cands, w = geo.weights(x, c)
    d = torch.zeros(len(x), geo.ctrs.shape[0], dtype=w.dtype)
    return d.scatter_add_(1, cands.clamp(min=0), w * (cands >= 0))


def brute_force_weights(geo, xyz):
    """Weights over ALL cells (no candidate set), dense (Nq, N)."""
    psi = wendland(torch.cdist(xyz.to(torch.float64), geo.ctrs) / geo.h)
    return psi / psi.sum(-1, keepdim=True)


def test_config():
    assert not get_partition_of_unity_config(None)["enabled"]
    assert not get_partition_of_unity_config(OmegaConf.create({}))["enabled"]
    cf = OmegaConf.create({"partition_of_unity": {"enabled": True, "streams": ["CICE"]}})
    pu = get_partition_of_unity_config(cf)
    assert pu["enabled"] and pu["support"] == 1.0
    assert uses_pu(pu, "CICE") and not uses_pu(pu, "ERA5")
    pu_all = get_partition_of_unity_config(OmegaConf.create({"partition_of_unity": {"enabled": 1}}))
    assert uses_pu(pu_all, "ERA5")
    with pytest.raises(AssertionError):
        get_partition_of_unity_config(
            OmegaConf.create({"partition_of_unity": {"enabled": True, "support": 0.5}})
        )


def test_wendland():
    s = torch.linspace(0, 1.5, 151, dtype=torch.float64)
    p = wendland(s)
    assert p[0] == 1.0 and (p[s >= 1] == 0).all() and (p[s < 1] > 0).all()
    assert (p.diff() <= 0).all()


def test_candidates(geo):
    c = geo.cands
    assert c.shape == (12 * 4**HL, 25)
    assert (c[:, 0] == torch.arange(c.shape[0])).all()
    for row in c[::997]:
        r = row[row >= 0]
        assert len(r) == len(set(r.tolist())), "duplicate candidate"
    # 8 HEALPix vertices: some cells have 7 neighbours
    assert (c[:, 1:9] < 0).any()


def test_weights_match_all_cells(geo):
    """Weights from the 25 candidates == weights over all cells: the candidate set never cuts
    the support, so the weights do not depend on the cell of x (continuity)."""
    _, _, xyz, cell = random_points(20000)
    cands, w = geo.weights(xyz, cell)
    assert torch.allclose(w.sum(-1), torch.ones(len(w), dtype=w.dtype))
    dense = torch.zeros(len(w), geo.ctrs.shape[0], dtype=w.dtype)
    dense.scatter_add_(1, cands.clamp(min=0), w * (cands >= 0))
    assert torch.allclose(dense, brute_force_weights(geo, xyz), atol=1e-12)


def test_weights_near_cell_corners(geo):
    """Worst case for the denominator: points at the cell corners (incl. the 8 vertices)."""
    import astropy_healpix as hp  # noqa: PLC0415

    n = geo.ctrs.shape[0]
    for dx, dy in [(1e-9, 1e-9), (1 - 1e-9, 1e-9), (1e-9, 1 - 1e-9), (1 - 1e-9, 1 - 1e-9)]:
        lon, lat = hp.healpix_to_lonlat(np.arange(n), 2**HL, dx=dx, dy=dy, order="nested")
        xyz = s2tor3(torch.from_numpy(np.pi / 2 - lat.value), torch.from_numpy(lon.value))
        cands, w = geo.weights(xyz, torch.arange(n))
        assert torch.isfinite(w).all()
        assert torch.allclose(w.sum(-1), torch.ones(n, dtype=w.dtype))
        dense = torch.zeros(n, n, dtype=w.dtype)
        dense.scatter_add_(1, cands.clamp(min=0), w * (cands >= 0))
        assert torch.allclose(dense, brute_force_weights(geo, xyz), atol=1e-12)


def test_weights_equal_across_boundaries(geo):
    """Points just on either side of a cell boundary get the same weights (up to O(eps))."""
    xm, cm, xp, cp = boundary_pairs(geo, 5000, seed=1)
    assert len(xm) > 1000 and (cm != cp).all()
    assert (dense_weights(geo, xm, cm) - dense_weights(geo, xp, cp)).abs().max() < 1e-5


def test_pairs(geo):
    _, _, xyz, cell = random_points(5000, seed=2)
    point_idx, pair_cell, weight, per_cell = geo.pairs(xyz, cell)
    n = geo.ctrs.shape[0]
    assert per_cell.shape == (n,) and int(per_cell.sum()) == len(point_idx)
    # sorted by cell, then by point
    key = pair_cell * len(xyz) + point_idx
    assert (key.diff() > 0).all()
    # every point keeps its own cell, weights sum to 1 per point
    own = torch.zeros(len(xyz), dtype=torch.bool)
    own[point_idx[pair_cell == cell[point_idx]]] = True
    assert own.all()
    s = torch.zeros(len(xyz), dtype=torch.float64).index_add_(0, point_idx, weight.double())
    assert torch.allclose(s, torch.ones_like(s), atol=1e-6)
    # cost: ~3.2 pairs per point with support 1.0
    assert 2.5 < len(point_idx) / len(xyz) < 4.0


def test_blend_is_continuous_on_toy_decoder(geo):
    """Toy check (not the model): with cell-wise decoders D_c(x) = a_c + <b_c, x> (a different
    smooth function per cell), the PU blend is continuous while the original (own cell only)
    jumps at the boundaries."""
    n = geo.ctrs.shape[0]
    g = torch.Generator().manual_seed(3)
    a, b = torch.randn(n, 2, generator=g), torch.randn(n, 3, 2, generator=g) * 0.1

    def decode(x, c):  # D_c(x), 2 channels
        return a[c] + torch.einsum("pi,pic->pc", x.float(), b[c])

    xyz, cell, xyz2, cell2 = boundary_pairs(geo, 5000, seed=4)

    def pu(x, c):
        q, pc, w, _ = geo.pairs(x, c)
        return blend(decode(x[q], pc), q, w, len(x))

    jump_orig = (decode(xyz, cell) - decode(xyz2, cell2)).abs().max()
    jump_pu = (pu(xyz, cell) - pu(xyz2, cell2)).abs().max()
    assert jump_orig > 0.1
    assert jump_pu < 1e-3


def test_blend_shapes():
    pred = torch.randn(3, 7, 2)  # (members, pairs, channels)
    point_idx = torch.tensor([0, 0, 1, 2, 2, 2, 3])
    weight = torch.tensor([0.25, 0.75, 1.0, 0.2, 0.3, 0.5, 1.0])
    out = blend(pred, point_idx, weight, 4)
    assert out.shape == (3, 4, 2)
    assert torch.allclose(out[:, 0], 0.25 * pred[:, 0] + 0.75 * pred[:, 1])
    assert torch.allclose(out[:, 3], pred[:, 6])


# ---------------------------------------------------------------- data loader: pair encodings


@pytest.fixture(scope="module")
def tok():
    from weathergen.datasets.tokenizer import Tokenizer  # noqa: PLC0415

    return Tokenizer(HL)


def encode_points(tok, xyz, cell, n_time=6, n_geo=2, seed=0, attrs_period=None):
    """Original target encoding of points (sorted by cell), as tokenize_apply_mask_target does.
    attrs_period: point i gets the random time / geoinfo of point i % attrs_period (e.g. the
    same attributes on both sides of a boundary pair). Returns lat/lon coords, encoding, counts
    per cell (in cell order) and the sorting order."""
    from weathergen.datasets.tokenizer_utils import (  # noqa: PLC0415
        get_target_coords_local,
        theta_phi_to_standard_coords,
    )
    from weathergen.datasets.utils import r3tos2  # noqa: PLC0415

    order = torch.from_numpy(np.argsort(cell.numpy(), kind="stable"))
    xyz = xyz[order]
    sph = r3tos2(xyz.to(torch.float32))  # (lat, lon) in [-pi/2, pi/2] x [-pi, pi] (r3tos2)
    lat = torch.rad2deg(sph[:, 0])
    lon = torch.rad2deg(sph[:, 1]) - 180.0  # tokenizer: phi = lon + 180
    coords = torch.stack([lat, lon], 1)  # float32, as the readers
    # cells from the (rounded) coords, as the tokenizer does
    th, ph = theta_phi_to_standard_coords(coords)
    hl = tok.healpix_level
    cell = torch.from_numpy(ang2pix(2**hl, th.numpy(), ph.numpy(), nest=True))
    g = torch.Generator().manual_seed(seed)
    times = torch.randn(len(xyz), n_time, generator=g)
    geoinfos = torch.randn(len(xyz), n_geo, generator=g)
    if attrs_period is not None:
        src = order % attrs_period
        times, geoinfos = times[src], geoinfos[src]
    per_cell = torch.bincount(cell, minlength=12 * 4**hl).to(torch.int32)
    enc = get_target_coords_local(
        3,
        hl,
        per_cell,
        coords,
        geoinfos,
        times,
        tok.hpy_verts_rots_target,
        tok.hpy_verts_local_target,
        tok.hpy_nctrs_target,
    )
    return coords, enc, per_cell, cell, order


def pair_encodings(geo, tok, coords, enc, per_cell):
    """Pairs from the data loader (pu_pairs) and their encodings as the model builds them."""
    from weathergen.model.partition_of_unity import PUEncoding, pu_pairs  # noqa: PLC0415

    xyz, pu_lens, point_idx, pair_cell, weight = pu_pairs(geo, coords, per_cell)
    point_idx, pair_cell = point_idx.to(torch.int64), pair_cell.to(torch.int64)
    pu_enc = PUEncoding(tok.hl_target)
    coords_pu = pu_enc.encode(enc[point_idx, :-99], xyz[point_idx], pair_cell)
    return coords_pu, pu_lens, point_idx, weight, pair_cell, xyz


def test_coords_roundtrip_cells(geo, tok):
    """The lat/lon convention of the test helper gives back the cells of the points."""
    from weathergen.datasets.tokenizer_utils import hpy_cell_splits  # noqa: PLC0415

    _, _, xyz, cell = random_points(3000, seed=5)
    coords, _, per_cell, _, _ = encode_points(tok, xyz, cell)
    splits, _, _ = hpy_cell_splits(coords.numpy(), HL)
    counts = torch.tensor([len(s) for s in splits], dtype=torch.int32)
    assert (counts == per_cell).all()


def test_pair_encoding_equals_the_tokenizer(geo, tok):
    """The vectorised encoding of the model (PUEncoding) equals get_target_coords_local, the
    tokenizer function, applied to the pairs grouped by cell."""
    from weathergen.datasets.tokenizer_utils import get_target_coords_local  # noqa: PLC0415

    _, _, xyz, cell = random_points(3000, seed=8)
    coords, enc, per_cell, _, _ = encode_points(tok, xyz, cell)
    coords_pu, pu_lens, point_idx, _, pair_cell, pts = pair_encodings(
        geo, tok, coords, enc, per_cell
    )
    attrs = enc[point_idx, :-99]
    ref = get_target_coords_local(
        attrs[:, 0], HL, pu_lens, coords[point_idx], attrs[:, 1:], attrs[:, :0],
        tok.hpy_verts_rots_target, tok.hpy_verts_local_target, tok.hpy_nctrs_target,
    )  # fmt: skip
    assert coords_pu.shape == ref.shape
    assert (coords_pu - ref).abs().max() < 1e-6
    # pairs sorted by cell, lengths per cell
    assert torch.equal(
        pair_cell, torch.repeat_interleave(torch.arange(len(pu_lens)), pu_lens.long())
    )
    assert pts.dtype == torch.float32 and pts.shape == (len(coords), 3)


def test_pu_pairs_own_cell_is_original(geo, tok):
    _, _, xyz, cell = random_points(3000, seed=6)
    coords, enc, per_cell, cell_sorted, _ = encode_points(tok, xyz, cell)
    coords_pu, pu_lens, point_idx, weight, pair_cell, _ = pair_encodings(
        geo, tok, coords, enc, per_cell
    )
    assert coords_pu.shape == (len(point_idx), enc.shape[1]) and coords_pu.dtype == enc.dtype
    assert int(pu_lens.sum()) == len(point_idx)
    # point attributes (stream id, time, geoinfo) copied for every pair
    assert torch.equal(coords_pu[:, :-99], enc[point_idx, :-99])
    # a pair in the point's own cell is the original encoding
    own = pair_cell == cell_sorted[point_idx]
    assert own.sum() == len(enc)
    assert torch.allclose(coords_pu[own], enc[point_idx[own]], atol=1e-6)
    # other pairs: the local coordinates differ
    assert (coords_pu[~own, -99:] - enc[point_idx[~own], -99:]).abs().amax(1).min() > 1e-4


def test_pu_pairs_continuous_for_a_fixed_cell(geo, tok):
    """x just on either side of a boundary, both encoded relative to the SAME cell c: the
    encodings are equal up to O(eps). (Relative to their own cells, they jump.)"""
    xm, cm, xp, cp = boundary_pairs(geo, 2000, seed=7, gap=1e-4)
    n = len(xm)
    xyz = torch.cat([xm, xp])
    cell = torch.cat([cm, cp])
    coords, enc, per_cell, cell_sorted, order = encode_points(tok, xyz, cell)
    coords_pu, pu_lens, point_idx, _, pair_cell, _ = pair_encodings(geo, tok, coords, enc, per_cell)
    # map sorted rows back to (side, boundary pair id)
    pid, side = order % n, order // n
    # encoding of every (boundary pair, side, cell): keep cells seen from both sides
    seen = {}
    for r in range(len(point_idx)):
        p = int(point_idx[r])
        seen.setdefault((int(pid[p]), int(pair_cell[r])), {})[int(side[p])] = coords_pu[r, -99:]
    both = [v for v in seen.values() if len(v) == 2]
    assert len(both) > n  # every boundary pair shares at least its two cells
    jump_fixed_cell = max(float((v[0] - v[1]).abs().max()) for v in both)
    inv = torch.argsort(order)  # row of every input point
    crossed = cell_sorted[inv[:n]] != cell_sorted[inv[n:]]
    assert crossed.float().mean() > 0.99
    jump_own_cell = float((enc[inv[:n]][crossed, -99:] - enc[inv[n:]][crossed, -99:]).abs().max())
    assert jump_fixed_cell < 1e-4
    assert jump_own_cell > 1e-2


def test_stream_data_pu_fields_device():
    from weathergen.datasets.stream_data import StreamData  # noqa: PLC0415

    sd = StreamData(0, 1, 3, 12)
    assert sd.pu_xyz == [None, None, None]
    i32 = torch.arange(7, dtype=torch.int32)
    sd.add_pu_pairs(
        1, torch.zeros(5, 3), torch.ones(12, dtype=torch.int32), i32, i32, torch.ones(7)
    )
    sd.to_device("cpu")
    assert sd.pu_xyz[1].shape == (5, 3) and sd.pu_weight[0] is None
    assert sd.target_coords_pu_lens[1].dtype == torch.int32 and sd.pu_cell[1].dtype == torch.int32


# ---------------------------------------------------------------- model: decoders end to end


HL_SMALL = 3  # 768 cells: fast on CPU (checked: weights == brute force at this level)


class _Batch:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)


def _small_decoder(width, dim=32, channels=2, seed=0):
    """Real decoder blocks of the model (PerceiverIOCoordConditioning, self-attention, AdaLN),
    small sizes, random weights. flash_attn is the CPU fallback of conftest.py locally."""
    from weathergen.model.engines import (  # noqa: PLC0415
        EnsPredictionHead,
        TargetPredictionEngineClassic,
    )
    from weathergen.model.layers import MLP  # noqa: PLC0415

    torch.manual_seed(seed)
    cf = OmegaConf.create(
        {
            "ae_global_dim_embed": dim,
            "ae_local_num_queries": 1,
            "decoder_type": "PerceiverIOCoordConditioning",
            "pred_self_attention": True,
            "pred_mlp_adaln": True,
            "with_flash_attention": True,
            "attention_dtype": "fp32",
            "norm_type": "LayerNorm",
            "qk_norm_type": None,
            "norm_eps": 1e-4,
            "mlp_norm_eps": 1e-5,
        }
    )
    si = {"name": "S", "target_readout": {"num_heads": 2}}
    engine = TargetPredictionEngineClassic(cf, [dim, dim, dim], width, 16, 2, 0.0, si)
    embed = MLP(width, dim, hidden_factor=2, with_residual=False, name="embed_S")
    head = EnsPredictionHead(dim, channels, 1, 1, stream_name="S")
    for m in (engine, embed, head):
        m.eval()
    return cf, engine, embed, head


def _fake_model(cf, engine, embed, head, pu_cfg, n_cells):
    """The attributes of Model used by predict_decoders (no encoder / forecast engine)."""
    from types import SimpleNamespace  # noqa: PLC0415

    from weathergen.model.model import Model  # noqa: PLC0415
    from weathergen.model.partition_of_unity import PUEncoding  # noqa: PLC0415

    m = SimpleNamespace(
        cf=cf,
        pred_heads={"S": head},
        num_aux_tokens=0,
        num_healpix_cells=n_cells,
        streams={"S": {}},
        embed_target_coords={"S": embed},
        target_token_engines={"S": engine},
        fm_conditioners=None,
        pu_cfg=pu_cfg,
        pu_enc=PUEncoding(HL_SMALL),
    )
    m._pu_gather = lambda *a: Model._pu_gather(m, *a)
    m._fm_gather = lambda *a: Model._fm_gather(m, *a)
    m.predict_flow_matching = lambda *a: Model.predict_flow_matching(m, *a)
    m.predict = lambda *a, **k: Model.predict_decoders(m, *a, **k)
    return m


def _add_fm(m, channels, dim, sampling, members=2):
    """Flow-matching decoder on stream S (zero-initialised conditioner)."""
    from weathergen.model.flow_matching import (  # noqa: PLC0415
        FlowMatchingConditioner,
        get_flow_matching_config,
    )

    fm = {"enabled": True, "streams": ["S"], "inference": {"num_members": members, "num_steps": 2}}
    m.fm_cfg = get_flow_matching_config(OmegaConf.create({"flow_matching": fm}))
    m.fm_conditioners = {"S": FlowMatchingConditioner(channels, dim, "S")}
    m.fm_sampling = sampling
    m.training = False
    return m.fm_conditioners["S"]


def _sample(tok, geo, coords, enc, per_cell):
    from weathergen.datasets.stream_data import StreamData  # noqa: PLC0415
    from weathergen.model.partition_of_unity import pu_pairs  # noqa: PLC0415

    sd = StreamData(0, 1, 1, len(per_cell))
    sd.target_coords[0] = enc
    sd.target_coords_lens[0] = per_cell
    sd.add_pu_pairs(0, *pu_pairs(geo, coords, per_cell))
    from types import SimpleNamespace  # noqa: PLC0415

    return SimpleNamespace(streams_data={"S": sd})


@pytest.fixture(scope="module")
def e2e():
    """Boundary pairs at level 3, their encodings and pairs, the real decoder blocks."""
    from weathergen.datasets.tokenizer import Tokenizer  # noqa: PLC0415
    from weathergen.model.model import ModelOutput  # noqa: PLC0415

    geo3 = PUGeometry(HL_SMALL, 1.0)
    tok3 = Tokenizer(HL_SMALL)
    xm, cm, xp, cp = boundary_pairs(geo3, 400, seed=11, gap=1e-4)
    n = len(xm)
    coords, enc, per_cell, cell_sorted, order = encode_points(
        tok3, torch.cat([xm, xp]), torch.cat([cm, cp]), attrs_period=n
    )
    inv = torch.argsort(order)
    crossed = cell_sorted[inv[:n]] != cell_sorted[inv[n:]]
    cf, engine, embed, head = _small_decoder(enc.shape[1])
    n_cells = 12 * 4**HL_SMALL
    nb = geo3.cands[:, :9].clone()
    nb[nb < 0] = torch.arange(n_cells)[:, None].expand(-1, 9)[nb < 0]  # as ModelParams.hp_nbours
    from types import SimpleNamespace  # noqa: PLC0415

    params = SimpleNamespace(hp_nbours=nb)
    g = torch.Generator().manual_seed(12)

    def run(pu_enabled, n_samples=1, same_tokens=True, fm=None, perturb_fm=False):
        pu_cfg = get_partition_of_unity_config(
            OmegaConf.create({"partition_of_unity": {"enabled": pu_enabled}})
        )
        m = _fake_model(cf, engine, embed, head, pu_cfg, n_cells)
        if fm is not None:
            cond = _add_fm(m, 2, cf.ae_global_dim_embed, sampling=(fm == "sampling"))
            if perturb_fm:
                with torch.no_grad():
                    for prm in cond.parameters():
                        prm.add_(0.1 * torch.randn(prm.shape, generator=g))
        tokens = torch.randn(1, n_cells, cf.ae_global_dim_embed, generator=g)
        tokens = (
            tokens.expand(n_samples, -1, -1)
            if same_tokens
            else torch.randn(n_samples, n_cells, cf.ae_global_dim_embed, generator=g)
        )
        batch = _Batch([_sample(tok3, geo3, coords, enc, per_cell) for _ in range(n_samples)])
        for smp in batch.samples:  # x_k, x_(k-1) for flow matching
            vals = torch.randn(len(enc), 2, generator=g)
            smp.streams_data["S"].add_fm_values(0, vals, vals + 0.1)
        with torch.no_grad():
            out = m.predict(params, 0, tokens, batch, ModelOutput(1), fm_state={})
        return [p.float() for p in out.physical[0]["S"]]  # per sample (ens, points, C)

    return SimpleNamespace(run=run, inv=inv, n=n, crossed=crossed, g=g)


def test_decoders_end_to_end_continuity(e2e):
    """Real decoder blocks, random weights and latent tokens: y(x-) vs y(x+) for x+- at ~3e-6
    rad on either side of a boundary. Original decoders jump; the PU is continuous."""
    torch.manual_seed(0)
    inv, n, crossed = e2e.inv, e2e.n, e2e.crossed
    assert crossed.float().mean() > 0.99
    out = {}
    for pu_enabled in (False, True):
        e2e.g.manual_seed(12)  # same latent tokens for both
        y = e2e.run(pu_enabled)[0][0]  # (points, C)
        out[pu_enabled] = (y[inv[:n]] - y[inv[n:]])[crossed].abs()
    jump_orig, jump_pu = float(out[False].max()), float(out[True].max())
    scale = float(out[False].median())
    assert jump_orig > 1e-2, jump_orig
    assert jump_pu < 1e-2 * jump_orig, (jump_pu, jump_orig)
    assert scale > 0


def test_decoders_pu_batch_offsets(e2e):
    """Two samples with the same latent tokens give the same PU output (pair offsets)."""
    y = e2e.run(True, n_samples=2, same_tokens=True)
    assert torch.allclose(y[0], y[1], atol=1e-6)


def test_decoders_pu_with_flow_matching(e2e):
    """FM + PU: the zero-initialised FM decoder is exactly the PU decoder; a non-zero
    conditioner changes it (the conditioning reaches the pairs); sampling gives members."""
    torch.manual_seed(0)
    e2e.g.manual_seed(12)
    y_pu = e2e.run(True)[0]
    e2e.g.manual_seed(12)
    y_fm0 = e2e.run(True, fm="train")[0]
    assert torch.equal(y_fm0, y_pu)
    e2e.g.manual_seed(12)
    y_fm = e2e.run(True, fm="train", perturb_fm=True)[0]
    assert (y_fm - y_pu).abs().max() > 1e-4
    e2e.g.manual_seed(12)
    y_s = e2e.run(True, fm="sampling")[0]
    assert y_s.shape == (2, *y_pu.shape[1:]) and torch.isfinite(y_s).all()
    # without PU (decode_points of the original decoder): zero-init FM == original decoder
    e2e.g.manual_seed(12)
    y_orig = e2e.run(False)[0]
    e2e.g.manual_seed(12)
    assert torch.equal(e2e.run(False, fm="train")[0], y_orig)
