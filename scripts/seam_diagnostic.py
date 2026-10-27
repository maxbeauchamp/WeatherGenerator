#!/usr/bin/env python3
"""
HEALPix seam diagnostic of WeatherGenerator predictions (inference zip outputs).

Seams are jumps of the decoded field at the boundaries of the HEALPix cells of the model
(level 5 by default). Three measures, on the native points of a stream (no regridding):

1. Seam index: for pairs of neighbouring points (k nearest),
       R_cross  = E|y(a) - y(b)|^2 / E|t(a) - t(b)|^2   over pairs in different cells,
       R_within = same over pairs in the same cell,
       seam index = R_cross / R_within
   (y: prediction, t: target). ~1 without seams, > 1 with seams (the prediction varies more
   across cell boundaries than inside cells, relative to the truth).
2. Error vs position in the cell: RMSE binned by the distance to the nearest cell boundary,
   b = min(dx, 1 - dx, dy, 1 - dy) in [0, 0.5] (HEALPix offsets, cell units).
3. Structure function ratio S_y(r) / S_t(r), S(r) = E|f(x + r) - f(x)|^2 binned by the
   separation r (in cell sizes): the real-space counterpart of the power spectrum (no
   regridding of the curvilinear grid). < 1 at small r: too smooth; bumps near r ~ 1: blocks.

Ensembles (flow matching): the measures are computed on the ensemble mean and on each of the
first --members members (seams in the members).

  uv run --offline python scripts/seam_diagnostic.py \\
      $WD/results/inf_ice_era5_base/validation_chkpt*_rank0000.zip \\
      $WD/results/inf_ice_era5_pu/validation_chkpt*_rank0000.zip \\
      --labels base pu --stream CICE --channels aice hi --out-dir $WD/results/seams_pu
"""

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

NUM_B_BINS = 10
R_BINS = np.linspace(0.0, 3.0, 31)  # separation in cell sizes


def cell_and_offsets(lat, lon, level):
    """HEALPix cell (nested) and offsets in the convention of the tokenizer (phi = lon + 180)."""
    import astropy.units as u  # noqa: PLC0415
    import astropy_healpix as hp  # noqa: PLC0415

    cell, dx, dy = hp.lonlat_to_healpix(
        (np.asarray(lon, np.float64) + 180.0) * u.deg,
        np.asarray(lat, np.float64) * u.deg,
        2**level,
        order="nested",
        return_offsets=True,
    )
    return np.asarray(cell), np.minimum.reduce([dx, 1 - dx, dy, 1 - dy])


def unit_xyz(lat, lon):
    la, lo = np.deg2rad(np.asarray(lat, np.float64)), np.deg2rad(np.asarray(lon, np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def cell_size(level):
    """Mean distance between the centres of cells sharing an edge (unit sphere), approx."""
    return np.sqrt(4 * np.pi / (12 * 4**level))


class Geometry:
    """Neighbour pairs and separations of one set of points, computed once per point set."""

    def __init__(self, lat, lon, level, k=6, n_struct=4000, seed=0):
        from scipy.spatial import cKDTree  # noqa: PLC0415

        self.xyz = unit_xyz(lat, lon)
        self.cell, self.bdist = cell_and_offsets(lat, lon, level)
        self.size = cell_size(level)
        tree = cKDTree(self.xyz)
        _, nb = tree.query(self.xyz, k=k + 1)
        a = np.repeat(np.arange(len(self.xyz)), k)
        b = nb[:, 1:].ravel()
        keep = (b < len(self.xyz)) & (a < b)
        self.pa, self.pb = a[keep], b[keep]
        self.cross = self.cell[self.pa] != self.cell[self.pb]
        # structure function pairs: random subset against all points within 3 cell sizes
        rng = np.random.default_rng(seed)
        sub = rng.choice(len(self.xyz), size=min(n_struct, len(self.xyz)), replace=False)
        sub_tree = cKDTree(self.xyz[sub])
        sdm = sub_tree.sparse_distance_matrix(
            tree, R_BINS[-1] * self.size, output_type="coo_matrix"
        )
        self.sa, self.sb = sub[sdm.row], sdm.col
        self.sr = sdm.data / self.size
        keep = self.sr > 0
        self.sa, self.sb, self.sr = self.sa[keep], self.sb[keep], self.sr[keep]


def seam_measures(geo: Geometry, y, t):
    """y, t (points,) prediction and target (NaN = missing). Returns a dict of measures."""
    ok = np.isfinite(y) & np.isfinite(t)

    def ratio(mask):
        m = mask & ok[geo.pa] & ok[geo.pb]
        dt2 = np.mean((t[geo.pa[m]] - t[geo.pb[m]]) ** 2)
        dy2 = np.mean((y[geo.pa[m]] - y[geo.pb[m]]) ** 2)
        return float(dy2 / dt2) if dt2 > 0 else float("nan"), int(m.sum())

    r_cross, n_cross = ratio(geo.cross)
    r_within, n_within = ratio(~geo.cross)

    edges = np.linspace(0, 0.5, NUM_B_BINS + 1)
    ib = np.clip(np.digitize(geo.bdist, edges) - 1, 0, NUM_B_BINS - 1)
    err2 = (y - t) ** 2
    rmse_b = [
        float(np.sqrt(np.mean(err2[ok & (ib == i)]))) if (ok & (ib == i)).any() else float("nan")
        for i in range(NUM_B_BINS)
    ]

    m = ok[geo.sa] & ok[geo.sb]
    ir = np.digitize(geo.sr[m], R_BINS) - 1
    dy2 = (y[geo.sa[m]] - y[geo.sb[m]]) ** 2
    dt2 = (t[geo.sa[m]] - t[geo.sb[m]]) ** 2
    n_r = len(R_BINS) - 1
    s_y = np.bincount(ir, dy2, n_r) / np.maximum(np.bincount(ir, minlength=n_r), 1)
    s_t = np.bincount(ir, dt2, n_r) / np.maximum(np.bincount(ir, minlength=n_r), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s_ratio = np.where(s_t > 0, s_y / s_t, np.nan)

    return {
        "seam_index": r_cross / r_within if r_within > 0 else float("nan"),
        "ratio_cross": r_cross,
        "ratio_within": r_within,
        "pairs_cross": n_cross,
        "pairs_within": n_within,
        "rmse": float(np.sqrt(np.mean(err2[ok]))),
        "rmse_vs_boundary_distance": rmse_b,
        "structure_ratio": s_ratio.tolist(),
    }


def _mean(dicts, key):
    """Mean over samples / steps, ignoring NaN (empty bins stay NaN, without warnings)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.array([d[key] for d in dicts], dtype=np.float64), axis=0)


def run(zips, labels, stream, channels, samples, fsteps, level, members, out_dir):
    import wg_zip2netcdf as W  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    geos = {}
    for zp, label in zip(zips, labels, strict=True):
        store = W.Store(zp)
        sel = [s for s in store.samples if not samples or s in samples]
        res = {}  # (channel, kind) -> list of measures
        for sample in sel:
            for f in store.fsteps(sample, stream):
                if fsteps and f not in fsteps:
                    continue
                p = store.dataset(sample, stream, f, "prediction")
                t = store.dataset(sample, stream, f, "target")
                if p is None or t is None:
                    continue
                j = W.match_points(p["coords"][:, 0], p["coords"][:, 1], t["coords"])
                key = (len(p["coords"]), float(p["coords"][:, 0].sum()))
                if key not in geos:
                    geos[key] = Geometry(p["coords"][:, 0], p["coords"][:, 1], level)
                geo = geos[key]
                for ch in channels:
                    if ch not in p["channels"] or ch not in t["channels"]:
                        continue
                    ip, it = p["channels"].index(ch), t["channels"].index(ch)
                    tv = np.full(len(p["coords"]), np.nan)
                    ok = j >= 0
                    tv[j[ok]] = t["data"][ok, it, 0]
                    pred = p["data"][:, ip, :].astype(np.float64)  # (points, ens)
                    kinds = {"mean": pred.mean(-1)}
                    if pred.shape[-1] > 1:
                        for m in range(min(members, pred.shape[-1])):
                            kinds[f"member{m}"] = pred[:, m]
                    for kind, y in kinds.items():
                        res.setdefault((ch, kind), []).append(seam_measures(geo, y, tv))
        store.close()
        summary[label] = {
            f"{ch}/{kind}": {
                "n": len(v),
                "seam_index": float(_mean(v, "seam_index")),
                "rmse": float(_mean(v, "rmse")),
                "rmse_vs_boundary_distance": _mean(v, "rmse_vs_boundary_distance").tolist(),
                "structure_ratio": _mean(v, "structure_ratio").tolist(),
            }
            for (ch, kind), v in res.items()
        }
    (out_dir / "seam_summary.json").write_text(json.dumps(summary, indent=1))
    plot(summary, channels, out_dir)
    for label, s in summary.items():
        for k, v in s.items():
            si, rmse = v["seam_index"], v["rmse"]
            print(f"{label:>12s} {k:>14s}: seam index {si:.3f}  rmse {rmse:.4f}")  # noqa: T201
    print(f">> {out_dir / 'seam_summary.json'} and figures")  # noqa: T201
    return summary


def plot(summary, channels, out_dir):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    b_mid = (np.arange(NUM_B_BINS) + 0.5) * 0.5 / NUM_B_BINS
    r_mid = 0.5 * (R_BINS[1:] + R_BINS[:-1])
    for ch in channels:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for label, s in summary.items():
            v = s.get(f"{ch}/mean")
            if v is None:
                continue
            si = v["seam_index"]
            ax[0].plot(b_mid, v["rmse_vs_boundary_distance"], "o-", label=f"{label} ({si:.2f})")
            ax[1].plot(r_mid, v["structure_ratio"], "-", label=label)
        ax[0].set_xlabel("distance to the nearest cell boundary (cell units)")
        ax[0].set_ylabel("RMSE")
        ax[0].set_title(f"{ch}: error vs position in the cell (seam index)")
        ax[1].axhline(1.0, color="0.5", lw=0.8)
        ax[1].set_xlabel("separation r (cell sizes)")
        ax[1].set_ylabel("S_pred(r) / S_target(r)")
        ax[1].set_title(f"{ch}: structure function ratio")
        for a in ax:
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"seams_{ch}.png", dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("zips", nargs="+", type=pathlib.Path)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--stream", default="CICE")
    ap.add_argument("--channels", nargs="+", default=["aice", "hi"])
    ap.add_argument("--samples", nargs="*")
    ap.add_argument("--fsteps", nargs="*", type=int)
    ap.add_argument("--healpix-level", type=int, default=5)
    ap.add_argument("--members", type=int, default=2, help="members analysed one by one")
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("seams"))
    a = ap.parse_args()
    labels = a.labels or [z.parent.name for z in a.zips]
    run(a.zips, labels, a.stream, a.channels, a.samples, a.fsteps, a.healpix_level, a.members,
        a.out_dir)  # fmt: skip


if __name__ == "__main__":
    main()
