# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Decoder neighbourhood: relative-position attention bias and a configurable HEALPix neighbourhood
for the cross-attention of the target prediction engines (decoders), to remove HEALPix seams.

Why: a target point of cell c is decoded from the latent tokens of c and its 8 neighbours, but
the keys carry no position. The attention weights cannot depend on where the point lies in the
cell, so the prediction jumps at cell boundaries (the key set changes, nothing makes the two
sides agree).

What: for every (target point q, neighbour slot j) a small MLP of the position of q relative to
the centre of the slot cell gives one additive logit per head:

    logit(q, j) = <q, k_j> / sqrt(d) + b_h(r_qj),   r_qj = (x_q - y_j) / d_cell

in the tangent frame of the cell of q (x_q: point, y_j: centre of slot cell j, d_cell: cell
size). The MLP output layer is zero-initialised: at start the decoder is exactly the original
one. With `rings: 2` the 16 cells of the second ring are added as keys, with a learnable
per-head logit initialised to -50 (closed at start, exact identity up to ~1e-15); the MLP can
then open them where useful. A cell that enters or leaves the neighbourhood across a boundary
then has a weight already close to zero: no jump.

Flash attention has no additive bias. The bias is injected EXACTLY by extending the head
dimension: q gets the per-slot biases (times sqrt(d)), k gets a one-hot of its slot, v gets
zeros; the softmax scale is kept at 1/sqrt(d).

Config (main config):

    decoder_neighbourhood:
      enabled: False
      streams: []          # decoders using it ([] = all decoders)
      rings: 1             # 1: 9 cells (as now) | 2: 25 cells
      hidden: 16           # hidden size of the bias MLP
      ring2_init_logit: -50.0
"""

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "enabled": False,
    "streams": [],
    "rings": 1,
    "hidden": 16,
    "ring2_init_logit": -50.0,
}
PAD_LOGIT = -1.0e4
# layout of the target coordinate encoding (tokenizer_utils.get_target_coords_local):
# [stream_id, 5 time, G geoinfo, 99 local coords]; the point relative to its own cell centre
# ("ref - R_c p", 3 values) sits at offset 60 of the 99 local coords, i.e. 39 from the end.
_SELF_OFFSET_FROM_END = 39


def get_decoder_neighbourhood_config(cf) -> dict:
    dn = cf.get("decoder_neighbourhood", None) if cf is not None else None
    out = dict(_DEFAULTS)
    if not dn or not dn.get("enabled", False):
        return out
    for k in _DEFAULTS:
        if dn.get(k, None) is not None:
            out[k] = dn.get(k)
    out["streams"] = list(out["streams"] or [])
    out["rings"] = int(out["rings"])
    out["hidden"] = int(out["hidden"])
    out["ring2_init_logit"] = float(out["ring2_init_logit"])
    assert out["rings"] in (1, 2), f"decoder_neighbourhood.rings must be 1 or 2: {out['rings']}"
    assert out["hidden"] >= 1
    out["enabled"] = True
    return out


def uses_neighbourhood(dn_cfg: dict, stream_name: str) -> bool:
    return dn_cfg["enabled"] and (not dn_cfg["streams"] or stream_name in dn_cfg["streams"])


def build_neighbourhood(healpix_level: int, rings: int):
    """
    Neighbour slots of every cell, nested ordering.
    Returns (idx (N, K) int64, pad (N, K) bool, slot_ring (K,) int64) with K = 9 (rings 1) or 25.
    Slots 0..8 are EXACTLY the original decoder neighbourhood (self + 8 neighbours in astropy
    order, missing ones replaced by self and NOT padded, as in ModelParams.hp_nbours).
    Slots 9..24 (rings 2): second ring, sorted by cell index; missing ones padded (masked).
    """
    import astropy_healpix as hp  # noqa: PLC0415

    nside = 2**healpix_level
    n = 12 * 4**healpix_level
    import warnings  # noqa: PLC0415

    with warnings.catch_warnings(action="ignore"):
        nb = hp.neighbours(np.arange(n), nside, order="nested").transpose()  # (n, 8)
    ring1 = nb.copy()
    for i, row in enumerate(ring1):
        row[row == -1] = i
    k = 9 if rings == 1 else 25
    idx = np.zeros((n, k), dtype=np.int64)
    pad = np.zeros((n, k), dtype=bool)
    idx[:, 0] = np.arange(n)
    idx[:, 1:9] = ring1
    slot_ring = np.zeros(k, dtype=np.int64)
    slot_ring[1:9] = 1
    if rings == 2:
        slot_ring[9:] = 2
        for i in range(n):
            r1 = set(nb[i][nb[i] >= 0].tolist()) | {i}
            r2 = set()
            for c in r1:
                cn = nb[c]
                r2 |= set(cn[cn >= 0].tolist())
            r2 = sorted(r2 - r1)[:16]
            idx[i, 9 : 9 + len(r2)] = r2
            idx[i, 9 + len(r2) :] = i
            pad[i, 9 + len(r2) :] = True
    return idx, pad, slot_ring


class DecoderGeometry(torch.nn.Module):
    """
    Precomputed geometry (non-persistent buffers, not in checkpoints):
      nbours (N, K)       slot cells of each cell
      pad (N, K)          padded slots
      slot_ring (K,)      0 self, 1 first ring, 2 second ring
      ctr_local (N, K, 3) centre of each slot cell in the frame of the cell (R_c c_k)
      d_cell              mean distance between neighbouring centres (unit sphere)
    """

    def __init__(self, healpix_level: int, rings: int):
        super().__init__()
        from weathergen.datasets.utils import healpix_verts_rots  # noqa: PLC0415

        idx, pad, slot_ring = build_neighbourhood(healpix_level, rings)
        ctrs, rots = healpix_verts_rots(healpix_level, 0.5, 0.5)  # same as the tokenizer
        ctrs = ctrs.to(torch.float64)
        rots = rots.to(torch.float64)
        idx_t = torch.from_numpy(idx)
        # R_c c_k for every slot: (N, K, 3)
        ctr_local = torch.einsum("nij,nkj->nki", rots, ctrs[idx_t])
        d = (ctrs[idx_t[:, 1:9]] - ctrs[:, None, :]).norm(dim=-1)
        d_cell = float(d[d > 0].mean())
        self.num_slots = idx.shape[1]
        self.d_cell = d_cell
        self.register_buffer("nbours", idx_t, persistent=False)
        self.register_buffer("pad", torch.from_numpy(pad), persistent=False)
        self.register_buffer("slot_ring", torch.from_numpy(slot_ring), persistent=False)
        self.register_buffer("ctr_local", ctr_local.to(torch.float32), persistent=False)

    def relative_features(self, coords: torch.Tensor, cell_idx: torch.Tensor) -> torch.Tensor:
        """
        coords: (Nq, C) target coordinate encodings (as fed to the decoder);
        cell_idx: (Nq,) HEALPix cell of each target point.
        Returns (Nq, K, 3) features [r_y, r_z, |r|^2] with r = (x_q - y_slot) / d_cell in the
        tangent frame of the point's own cell (r_x ~ |r|^2 / 2 is redundant).
        """
        ref = torch.tensor([1.0, 0.0, 0.0], device=coords.device, dtype=torch.float32)
        a_self = coords[:, -_SELF_OFFSET_FROM_END : -_SELF_OFFSET_FROM_END + 3].float()
        x = ref - a_self  # R_c p
        r = (x[:, None, :] - self.ctr_local[cell_idx]) / self.d_cell
        return torch.cat([r[..., 1:], (r * r).sum(-1, keepdim=True)], dim=-1)

    def key_slots_onehot(self, num_cells_total: int, tokens_per_cell: int, device, dtype):
        """(num_cells_total * K * tokens_per_cell, K) one-hot of the slot of every key token."""
        k = self.num_slots
        eye = torch.eye(k, device=device, dtype=dtype).repeat_interleave(tokens_per_cell, 0)
        return eye.repeat(num_cells_total, 1)


class RelativePositionBias(torch.nn.Module):
    """
    Per-head attention logit bias from the relative position of a target point and a neighbour
    cell: b_h(r) = MLP(r), 2 layers, output layer zero-initialised. Second-ring slots get an
    extra learnable per-head logit (init `ring2_init_logit`, closed), padded slots PAD_LOGIT.
    Parameters live on the module (no nn.Linear) so that `reset_parameters()` keeps the zeros.
    """

    def __init__(self, num_heads: int, hidden: int, name: str, ring2_init_logit: float = -50.0):
        super().__init__()
        self.name = name
        self.num_heads = num_heads
        self.ring2_init_logit = ring2_init_logit
        self.w1 = torch.nn.Parameter(torch.empty(hidden, 3))
        self.b1 = torch.nn.Parameter(torch.empty(hidden))
        self.w2 = torch.nn.Parameter(torch.empty(num_heads, hidden))
        self.b2 = torch.nn.Parameter(torch.empty(num_heads))
        self.ring2_logit = torch.nn.Parameter(torch.empty(num_heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.zeros_(self.b1)
        torch.nn.init.zeros_(self.w2)  # zero output: exact identity of the decoder at start
        torch.nn.init.zeros_(self.b2)
        torch.nn.init.constant_(self.ring2_logit, self.ring2_init_logit)

    def forward(self, feats, pad_q, slot_ring):
        """feats (Nq, K, 3), pad_q (Nq, K) bool, slot_ring (K,) -> bias (Nq, H, K) float32."""
        h = F.gelu(F.linear(feats.float(), self.w1, self.b1))
        b = F.linear(h, self.w2, self.b2)  # (Nq, K, H)
        ring2 = (slot_ring == 2).float()  # (K,)
        b = b + ring2[None, :, None] * self.ring2_logit[None, None, :]
        b = b.masked_fill(pad_q[..., None], PAD_LOGIT)
        return b.transpose(1, 2)


def augment_qkv(qs, ks, vs, q_bias, k_onehot):
    """
    Exact additive attention bias for flash attention.
    qs (Nq, H, D), ks/vs (Nk, H, D), q_bias (Nq, H, K), k_onehot (Nk, K).
    Returns (qs', ks', vs', scale) such that softmax(q'.k' * scale) = softmax(q.k/sqrt(D) + bias)
    with scale = 1/sqrt(D); the extra dims are padded to a multiple of 8 (flash requirement).
    The output must be sliced back to [..., :D].
    """
    d = qs.shape[-1]
    k = q_bias.shape[-1]
    kp = ((k + 7) // 8) * 8
    scale = 1.0 / math.sqrt(d)
    qe = torch.zeros((*qs.shape[:2], kp), device=qs.device, dtype=qs.dtype)
    qe[..., :k] = (q_bias * math.sqrt(d)).to(qs.dtype)
    ke = torch.zeros((*ks.shape[:2], kp), device=ks.device, dtype=ks.dtype)
    ke[..., :k] = k_onehot[:, None, :].to(ks.dtype)
    ve = torch.zeros((*vs.shape[:2], kp), device=vs.device, dtype=vs.dtype)
    return (
        torch.cat([qs, qe], -1),
        torch.cat([ks, ke], -1),
        torch.cat([vs, ve], -1),
        scale,
    )
