# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Partition of unity (PU) of the decoders, against HEALPix seams.

Why: a target point x of cell c is decoded from the 9 latent cells around c, with a coordinate
encoding relative to c (query and AdaLN of every decoder block). Both change when x crosses a
cell boundary, so the decoded field jumps there.

What: decode x as if it belonged to every cell c near it, and blend with continuous weights:

    y(x) = sum_c w_c(x) D_c(x),   w_c(x) = psi(|x - y_c| / h) / sum_c' psi(|x - y_c'| / h)

- D_c(x): the unchanged decoder for cell c (keys of the 9 cells around c, encoding of x relative
  to c). For a fixed c it is continuous in x.
- psi(s) = (1 - s)_+^4 (4 s + 1): Wendland C2 function with compact support; y_c: centre of c;
  h = support * d_cell (d_cell: mean distance between centres of cells sharing an edge).
- Candidates: the 25 cells of rings 1 and 2 around the cell of x. Every cell with a non-zero
  weight is a candidate (checked against all cells in the tests), from either side of a
  boundary, so the weights are continuous. Each pair (x, c) is decoded with the 9 keys of c.
- The self-attention group of cell c holds all target points in the support of w_c (its own
  points and those of its neighbours that are close enough): the same group seen from both sides.

No parameters: the model is not the identity at start (the identity is the discontinuous
limit), a decoder fine-tuning is needed. Here: config, geometry and pairs (data loader, CPU),
encoding of the pairs and blend (model, GPU). The data loader only ships the pairs (point,
cell, weight) and the points on the sphere: the pair encodings are ~3.2x the target encoding
and are built on the GPU (shipping them from the workers overflows the shared memory on large
grids).

Config (main config):

    partition_of_unity:
      enabled: False
      streams: []      # decoders using it ([] = all decoders)
      support: 1.0     # h in units of d_cell
"""

import logging
import warnings

import numpy as np
import torch

logger = logging.getLogger(__name__)

_DEFAULTS = {"enabled": False, "streams": [], "support": 1.0}


def get_partition_of_unity_config(cf) -> dict:
    pu = cf.get("partition_of_unity", None) if cf is not None else None
    out = dict(_DEFAULTS)
    if not pu or not pu.get("enabled", False):
        return out
    for k in _DEFAULTS:
        if pu.get(k, None) is not None:
            out[k] = pu.get(k)
    out["streams"] = list(out["streams"] or [])
    out["support"] = float(out["support"])
    # lower bound: the nearest centre is at most 0.79 d_cell away (level 5, cell corners), so
    # the denominator stays > 0 with a margin; upper bound: the support stays inside the
    # ring-2 candidates
    assert 0.9 <= out["support"] <= 1.2, (
        f"partition_of_unity.support must be in [0.9, 1.2]: {out['support']}"
    )
    out["enabled"] = True
    return out


def uses_pu(pu_cfg: dict, stream_name: str) -> bool:
    return pu_cfg["enabled"] and (not pu_cfg["streams"] or stream_name in pu_cfg["streams"])


def wendland(s: torch.Tensor) -> torch.Tensor:
    """psi(s) = (1 - s)_+^4 (4 s + 1): C2, psi(0) = 1, psi(s >= 1) = 0."""
    t = (1.0 - s).clamp(min=0.0)
    return t**4 * (4.0 * s + 1.0)


def build_candidates(healpix_level: int) -> torch.Tensor:
    """
    Rings 1 and 2 of every cell, nested ordering: (N, 25) int64, self first, then the 8
    neighbours (astropy order), then the second ring sorted by index; -1 where missing.
    """
    import astropy_healpix as hp  # noqa: PLC0415

    n = 12 * 4**healpix_level
    with warnings.catch_warnings(action="ignore"):
        nb = hp.neighbours(np.arange(n), 2**healpix_level, order="nested").transpose()
    out = np.full((n, 25), -1, dtype=np.int64)
    out[:, 0] = np.arange(n)
    out[:, 1:9] = nb
    for i in range(n):
        r1 = set(nb[i][nb[i] >= 0].tolist()) | {i}
        r2 = set()
        for c in r1:
            r2 |= set(nb[c][nb[c] >= 0].tolist())
        r2 = sorted(r2 - r1)
        out[i, 9 : 9 + len(r2)] = r2
    return torch.from_numpy(out)


class PUGeometry:
    """
    HEALPix geometry for the PU weights (CPU, float64):
      ctrs (N, 3)      cell centres (same convention as the tokenizer: healpix_verts_rots)
      cands (N, 25)    rings 1 and 2 of every cell, -1 where missing
      d_cell           mean distance between centres of cells sharing an edge (unit sphere)
      h                support radius, support * d_cell
    """

    def __init__(self, healpix_level: int, support: float = 1.0):
        from weathergen.datasets.utils import healpix_verts_rots  # noqa: PLC0415

        ctrs, _ = healpix_verts_rots(healpix_level, 0.5, 0.5)
        self.healpix_level = healpix_level
        self.ctrs = ctrs.to(torch.float64)
        self.cands = build_candidates(healpix_level)
        nb = self.cands[:, 1:9]
        d = (self.ctrs[nb.clamp(min=0)] - self.ctrs[:, None, :]).norm(dim=-1)
        d[nb < 0] = float("inf")
        # the 4 nearest of the 8 neighbours share an edge with the cell
        self.d_cell = float(d.sort(-1).values[:, :4].mean())
        self.h = support * self.d_cell

    def weights(self, xyz: torch.Tensor, cell: torch.Tensor):
        """
        xyz (Nq, 3) points on the unit sphere, cell (Nq,) their HEALPix cell.
        Returns cands (Nq, 25) and normalised weights w (Nq, 25) float64 (0 for missing cells).
        """
        cands = self.cands[cell]
        valid = cands >= 0
        dist = (xyz.to(torch.float64)[:, None, :] - self.ctrs[cands.clamp(min=0)]).norm(dim=-1)
        psi = wendland(dist / self.h) * valid
        return cands, psi / psi.sum(-1, keepdim=True)

    def pairs(self, xyz: torch.Tensor, cell: torch.Tensor):
        """
        (point, cell) pairs with a non-zero weight, sorted by cell (then by point).
        Returns point_idx (P,) int64, pair_cell (P,) int64, weight (P,) float32,
        pairs_per_cell (N,) int32 (lengths of the self-attention groups).
        """
        cands, w = self.weights(xyz, cell)
        q, j = torch.nonzero(w > 0, as_tuple=True)
        pair_cell = cands[q, j]
        order = torch.from_numpy(np.lexsort((q.numpy(), pair_cell.numpy())))
        q, j, pair_cell = q[order], j[order], pair_cell[order]
        n = self.ctrs.shape[0]
        pairs_per_cell = torch.bincount(pair_cell, minlength=n).to(torch.int32)
        return q, pair_cell, w[q, j].to(torch.float32), pairs_per_cell


# the target coordinate encoding (tokenizer_utils.get_target_coords_local) is
# [stream_id, T time, G geoinfo, 99 local coords]: only the last 99 depend on the cell
_NUM_LOCAL_COORDS = 99


def pu_pairs(geo: PUGeometry, coords: torch.Tensor, points_per_cell: torch.Tensor):
    """
    Data loader: (point, cell) pairs of one stream and step.

    coords (Nq, 2) lat/lon of the target points and points_per_cell (N,) their counts per cell,
    as returned by tokenize_apply_mask_target (points sorted by cell).
    Returns xyz (Nq, 3) float32 (the points as the target encoding sees them),
    pairs_per_cell (N,) int32, point_idx (P,) int32, pair_cell (P,) int32, weight (P,) float32.
    """
    from weathergen.datasets.tokenizer_utils import theta_phi_to_standard_coords  # noqa: PLC0415
    from weathergen.datasets.utils import s2tor3  # noqa: PLC0415

    n_cells = geo.ctrs.shape[0]
    coords = torch.as_tensor(coords)
    if coords.numel() == 0:
        empty = torch.zeros(0, dtype=torch.int32)
        return (torch.zeros((0, 3)), torch.zeros(n_cells, dtype=torch.int32), empty, empty,
                torch.zeros(0, dtype=torch.float32))  # fmt: skip
    xyz64 = s2tor3(*theta_phi_to_standard_coords(coords.to(torch.float64)))
    cell = torch.repeat_interleave(torch.arange(n_cells), points_per_cell.to(torch.int64))
    point_idx, pair_cell, weight, pairs_per_cell = geo.pairs(xyz64, cell)
    # same computation as in get_target_coords_local (dtype of coords)
    xyz = s2tor3(*theta_phi_to_standard_coords(coords)).to(torch.float32)
    return (xyz, pairs_per_cell, point_idx.to(torch.int32), pair_cell.to(torch.int32),
            weight)  # fmt: skip


class PUEncoding:
    """
    Model: target encoding of the (point, cell) pairs, on the device of the points. Same columns
    and same operations as tokenizer_utils.get_target_coords_local, vectorised over the pairs:
    [point attributes (stream id, time, geoinfo), 99 local coordinates relative to the cell]:
      5 x (ref - R_v(c) x, 12 vertex coordinates of c)   v = 00, 10, 11, 01, centre
      8 x (ref - R_n(c) x)                               n = the 8 neighbour centres of c
    Tables from the tokenizer (float32, ~6 MB at level 5); plain tensors, not module buffers
    (not in the checkpoints).
    """

    def __init__(self, healpix_level: int):
        # tables built at first use: the model is created under torch.device("meta") with FSDP
        self.healpix_level = healpix_level
        self._cpu = None
        self._on = {}

    def _build(self):
        from weathergen.datasets.tokenizer import Tokenizer  # noqa: PLC0415
        from weathergen.datasets.utils import vecs_to_rots  # noqa: PLC0415

        with torch.device("cpu"):
            tok = Tokenizer(self.healpix_level)
            rots = torch.stack(tok.hpy_verts_rots_target)  # (5, N, 3, 3)
            verts_local = tok.hpy_verts_local_target.to(torch.float32)  # (N, 5, 12)
            # as in locs_to_ctr_coords
            nrots = torch.stack(
                [vecs_to_rots(c).to(torch.float32) for c in tok.hpy_nctrs_target]
            )  # (8, N, 3, 3)
        return rots, verts_local, nrots

    def _tables(self, device):
        if self._cpu is None:
            self._cpu = self._build()
        key = str(device)
        if key not in self._on:
            self._on[key] = tuple(t.to(device) for t in self._cpu)
        return self._on[key]

    def encode(self, attrs: torch.Tensor, xyz: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """attrs (P, W - 99), xyz (P, 3), cell (P,) int64 -> (P, W) in the dtype of attrs."""
        rots, verts_local, nrots = self._tables(xyz.device)
        x = xyz.to(torch.float32).unsqueeze(-1)
        ref = torch.tensor([1.0, 0.0, 0.0], device=xyz.device)
        cols = []
        for k in range(rots.shape[0]):
            cols.append(ref - torch.bmm(rots[k][cell], x).squeeze(-1))
            cols.append(verts_local[cell, k])
        for k in range(nrots.shape[0]):
            cols.append(ref - torch.bmm(nrots[k][cell], x).squeeze(-1))
        local = torch.cat(cols, -1)
        assert local.shape[-1] == _NUM_LOCAL_COORDS
        return torch.cat([attrs, local.to(attrs.dtype)], -1)


def blend(pred_pairs: torch.Tensor, point_idx: torch.Tensor, weight: torch.Tensor, n_points: int):
    """
    pred_pairs (..., P, C) decoder output for every pair -> (..., n_points, C):
    y(x) = sum_c w_c(x) D_c(x), accumulated in float32, returned in the dtype of pred_pairs.
    """
    w = weight.to(device=pred_pairs.device, dtype=torch.float32)[:, None]
    out = torch.zeros(
        (*pred_pairs.shape[:-2], n_points, pred_pairs.shape[-1]),
        dtype=torch.float32,
        device=pred_pairs.device,
    )
    out = out.index_add(-2, point_idx, pred_pairs.float() * w)
    return out.to(pred_pairs.dtype)
