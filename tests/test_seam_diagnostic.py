# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests of scripts/seam_diagnostic.py on synthetic fields (not model output)."""

import pathlib
import sys

import numpy as np
import pytest
import torch
from astropy_healpix.healpy import ang2pix

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import seam_diagnostic as S  # noqa: E402

from weathergen.datasets.tokenizer_utils import theta_phi_to_standard_coords  # noqa: E402

LEVEL = 5


@pytest.fixture(scope="module")
def field():
    """Arctic patch at 0.25 deg, smooth truth t (large scales)."""
    lat, lon = np.meshgrid(np.arange(60, 80, 0.25), np.arange(-60, 60, 0.25), indexing="ij")
    lat, lon = lat.ravel(), lon.ravel()
    t = np.sin(np.deg2rad(lat) * 7) + np.cos(np.deg2rad(lon) * 3)
    return lat, lon, t, S.Geometry(lat, lon, LEVEL)


def test_cells_match_the_tokenizer(field):
    lat, lon, _, geo = field
    th, ph = theta_phi_to_standard_coords(torch.tensor(np.stack([lat, lon], 1)))
    cell_tok = ang2pix(2**LEVEL, th.numpy(), ph.numpy(), nest=True)
    assert (geo.cell == cell_tok).all()
    assert geo.bdist.min() >= 0 and geo.bdist.max() <= 0.5
    assert 0.1 < geo.cross.mean() < 0.9


def test_seam_index_smooth_vs_blocky(field):
    lat, lon, t, geo = field
    rng = np.random.default_rng(0)
    smooth = t + 0.01 * np.sin(np.deg2rad(lon) * 11)  # smooth error
    offsets = 0.05 * rng.standard_normal(12 * 4**LEVEL)
    blocky = t + offsets[geo.cell]  # piecewise constant error per HEALPix cell (toy seams)
    m_smooth = S.seam_measures(geo, smooth, t)
    m_blocky = S.seam_measures(geo, blocky, t)
    assert abs(m_smooth["seam_index"] - 1.0) < 0.05
    assert m_blocky["seam_index"] > 2.0
    assert len(m_blocky["structure_ratio"]) == len(S.R_BINS) - 1


def test_missing_values(field):
    lat, lon, t, geo = field
    y = t.copy()
    y[::7] = np.nan
    m = S.seam_measures(geo, y, t)
    assert np.isfinite(m["seam_index"]) and np.isfinite(m["rmse"])


def test_gradient_shows_the_seams(field):
    """|grad| of a per-cell offset field is larger on cross-cell pairs (toy seams)."""
    lat, lon, t, geo = field
    rng = np.random.default_rng(1)
    blocky = t + 0.05 * rng.standard_normal(12 * 4**LEVEL)[geo.cell]
    g_t, g_b = S.point_gradient(geo, t), S.point_gradient(geo, blocky)
    near = geo.bdist < 0.05
    assert np.isfinite(g_t).all()
    assert np.median(g_b[near]) > 2 * np.median(g_t[near])
    assert np.median(g_b[~near]) < 2 * np.median(g_t[~near])


def test_map_figure(field, tmp_path):
    lat, lon, t, geo = field
    blocky = t + 0.05 * np.random.default_rng(2).standard_normal(12 * 4**LEVEL)[geo.cell]
    preds = {"base": blocky, "pu": t + 0.01}
    for name, box, cell_edges in [
        ("full", S.data_box(lat, lon), False),
        ("zoom", (0, 20, 65, 75), True),
    ]:
        path = tmp_path / f"m_{name}.png"
        S.map_figure(geo, lat, lon, t, preds, "toy", path, box, 0.25, LEVEL, cell_edges)
        assert path.stat().st_size > 10_000
    grid = S.ImageGrid(lat, lon, (0, 20, 65, 75), 0.25, LEVEL)
    assert grid.valid.mean() > 0.95 and grid.edges.any()


def test_polar_figure_and_animation(field, tmp_path):
    import plot_seaice as P

    lat, lon, t, geo = field
    view = S.PolarView(lat, lon, 50.0, 50.0, LEVEL)
    assert view.edges.any() and (view.grid.map >= 0).any()
    blocky = t + 0.05 * np.random.default_rng(2).standard_normal(12 * 4**LEVEL)[geo.cell]
    preds = {"base": blocky, "pu": t + 0.01}
    for name, box in [("polar", None), ("zoom", (0, 20, 65, 75))]:
        pf = S.PolarFigure(view, list(preds), "hi", box, dpi=60)
        pf.draw(geo, t, preds, "toy")
        path = tmp_path / f"p_{name}.png"
        pf.fig.savefig(path)
        assert path.stat().st_size > 10_000

        def draw(i, pf=pf):
            pf.draw(geo, t + 0.1 * i, {k: v + 0.1 * i for k, v in preds.items()}, f"frame {i}")

        out = P.save_animation(pf.fig, draw, 3, str(tmp_path / f"a_{name}"), 2.0, "gif")
        assert out.endswith(".gif") and pathlib.Path(out).stat().st_size > 10_000


def test_texture_smooth_vs_textured(field, tmp_path):
    """Toy: the target has small scales; a prediction without them has texture ratios < 1."""
    lat, lon, t, geo = field
    fine = 0.2 * np.sin(np.deg2rad(lat) * 400) * np.cos(np.deg2rad(lon) * 200)
    m_same = S.seam_measures(geo, t + fine, t + fine)
    m_smooth = S.seam_measures(geo, t, t + fine)
    assert abs(m_same["grad_ratio"] - 1) < 1e-9 and abs(m_same["small_scale_ratio"] - 1) < 1e-9
    assert m_smooth["grad_ratio"] < 0.7 and m_smooth["small_scale_ratio"] < 0.7
    tv = {"fsteps": [0, 1], "grad_ratio": [1.0, 0.5], "small_scale_ratio": [1.0, 0.4]}
    summary = {"fm": {"hi/mean": {"texture_vs_lead": tv}, "hi/member0": {"texture_vs_lead": tv}}}
    import matplotlib.pyplot as plt  # noqa: PLC0415

    S.plot_texture(summary, "hi", tmp_path, plt)
    assert (tmp_path / "texture_hi.png").exists()
    assert len((tmp_path / "texture_hi.csv").read_text().splitlines()) == 5
