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

Side-by-side maps (--maps): for one sample and a few lead times, one column per run next to
the target. Rows: the field, the error (prediction - target), and the local gradient |grad y|
(mean |y(a) - y(b)| / |a - b| over the neighbour pairs), where seams show up as lines along
the cell boundaries. Same colour scale in a row. Views: the extent of the data, and the
--zoom boxes with the HEALPix cell boundaries drawn. Nearest-neighbour images, for display only
(the measures above use the native points).

  uv run --offline python scripts/seam_diagnostic.py \\
      $WD/results/inf_ice_era5_base/validation_chkpt*_rank0000.zip \\
      $WD/results/inf_ice_era5_pu/validation_chkpt*_rank0000.zip \\
      --labels base pu --stream CICE --channels aice hi --out-dir $WD/results/seams_pu \\
      --maps --zoom -80 -30 25 55
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


def point_gradient(geo: Geometry, y):
    """Local gradient magnitude per point: mean |y(a) - y(b)| / |a - b| over its neighbour pairs."""
    d = np.linalg.norm(geo.xyz[geo.pa] - geo.xyz[geo.pb], axis=-1) / geo.size  # cell sizes
    g = np.abs(y[geo.pa] - y[geo.pb]) / np.maximum(d, 1e-12)
    ok = np.isfinite(g)
    n = len(y)
    acc = np.bincount(geo.pa[ok], g[ok], n) + np.bincount(geo.pb[ok], g[ok], n)
    cnt = np.bincount(geo.pa[ok], minlength=n) + np.bincount(geo.pb[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, acc / cnt, np.nan)


def _wrap(lon):
    return (np.asarray(lon, np.float64) + 180.0) % 360.0 - 180.0


class ImageGrid:
    """Regular lon-lat image of a box; nearest native point within max_dist (display only)."""

    def __init__(self, lat, lon, box, res, level):
        from scipy.spatial import cKDTree  # noqa: PLC0415

        lon0, lon1, lat0, lat1 = box
        self.extent = (lon0, lon1, lat0, lat1)
        glon = np.arange(lon0 + res / 2, lon1, res)
        glat = np.arange(lat0 + res / 2, lat1, res)
        glat2, glon2 = np.meshgrid(glat, glon, indexing="ij")
        self.shape = glat2.shape
        # max distance: 1.5 x the larger of the image pixel and the median point spacing
        tree = cKDTree(unit_xyz(lat, _wrap(lon)))
        sub = np.linspace(0, len(lat) - 1, min(len(lat), 2000)).astype(int)
        spacing = np.median(tree.query(tree.data[sub], k=2)[0][:, 1])
        dmax = 1.5 * max(np.deg2rad(res), spacing)
        dist, self.idx = tree.query(
            unit_xyz(glat2.ravel(), glon2.ravel()), distance_upper_bound=dmax
        )
        self.valid = np.isfinite(dist)
        self.idx[~self.valid] = 0
        cells = cell_and_offsets(glat2.ravel(), glon2.ravel(), level)[0].reshape(self.shape)
        edge = np.zeros(self.shape, bool)
        edge[:, 1:] |= cells[:, 1:] != cells[:, :-1]
        edge[1:, :] |= cells[1:, :] != cells[:-1, :]
        self.edges = edge

    def image(self, values):
        img = np.asarray(values, np.float64)[self.idx]
        img[~self.valid] = np.nan
        return img.reshape(self.shape)


def data_box(lat, lon, margin=1.0):
    lon = _wrap(lon)
    return (max(-180.0, lon.min() - margin), min(180.0, lon.max() + margin),
            max(-90.0, lat.min() - margin), min(90.0, lat.max() + margin))  # fmt: skip


def map_figure(geo, lat, lon, target, preds, title, path, box, res, level, cell_edges):
    """One figure: columns = target + runs; rows = field, error, |grad|. preds: {label: y}."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    grid = ImageGrid(lat, lon, box, res, level)
    cols = {"target": target, **preds}
    imgs = {k: grid.image(v) for k, v in cols.items()}
    errs = {k: grid.image(v - target) for k, v in preds.items()}
    grads = {k: grid.image(point_gradient(geo, v)) for k, v in cols.items()}

    def lim(arrs, lo, hi):
        v = np.concatenate([a[np.isfinite(a)] for a in arrs] + [np.zeros(0)])
        return (np.percentile(v, lo), np.percentile(v, hi)) if v.size else (0.0, 1.0)

    fmin, fmax = lim([imgs["target"]], 1, 99)
    emax = max(lim([np.abs(e) for e in errs.values()], 0, 99)[1], 1e-12)
    gmax = max(lim(list(grads.values()), 0, 99)[1], 1e-12)
    n = len(cols)
    aspect = (box[3] - box[2]) / max(box[1] - box[0], 1e-6)
    fig, ax = plt.subplots(
        3, n, figsize=(3.4 * n + 0.8, 3 * 3.4 * aspect + 0.8), squeeze=False, layout="constrained"
    )
    rows = [
        ("field", imgs, {"cmap": "viridis", "vmin": fmin, "vmax": fmax}),
        ("prediction - target", errs, {"cmap": "RdBu_r", "vmin": -emax, "vmax": emax}),
        ("|grad| (per cell size)", grads, {"cmap": "magma", "vmin": 0.0, "vmax": gmax}),
    ]
    edges = np.ma.masked_where(~grid.edges, grid.edges.astype(float))
    for r, (name, data, kw) in enumerate(rows):
        im = None
        for c, label in enumerate(cols):
            a = ax[r, c]
            a.set_xticks([])
            a.set_yticks([])
            if label not in data:  # no error map for the target: empty panel, row label kept
                a.set_frame_on(False)
                continue
            im = a.imshow(
                data[label], origin="lower", extent=grid.extent, interpolation="nearest", **kw
            )
            if cell_edges:
                a.imshow(edges, origin="lower", extent=grid.extent, cmap="Greys", vmin=0, vmax=1,
                         alpha=0.35, interpolation="nearest")  # fmt: skip
            if r == 0:
                a.set_title(label, fontsize=10)
        ax[r, 0].set_ylabel(name, fontsize=9)
        fig.colorbar(im, ax=list(ax[r]), shrink=0.85, pad=0.01)
    where = f"lon {box[0]:g}..{box[1]:g}, lat {box[2]:g}..{box[3]:g}"
    edges_txt = ", grey: HEALPix cell boundaries" if cell_edges else ""
    fig.suptitle(f"{title} ({where}{edges_txt})", fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def maps(
    zips, labels, stream, channels, samples, fsteps, level, zooms, out_dir, res_full, res_zoom
):
    """Side-by-side maps of all runs for one sample (the first common one) and a few lead times."""
    import wg_zip2netcdf as W  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    stores = [W.Store(z) for z in zips]
    common = [s for s in stores[0].samples if all(s in st.samples for st in stores[1:])]
    sel = [s for s in common if not samples or s in samples][:1]
    if not sel:
        print("maps: no sample common to all runs")  # noqa: T201
        return
    sample = sel[0]
    steps = stores[0].fsteps(sample, stream)
    if fsteps:
        steps = [f for f in steps if f in fsteps]
    elif steps:
        steps = sorted({steps[0], steps[len(steps) // 2], steps[-1]})
    for f in steps:
        ref = stores[0].dataset(sample, stream, f, "target")
        if ref is None:
            continue
        lat, lon = ref["coords"][:, 0].astype(np.float64), ref["coords"][:, 1].astype(np.float64)
        geo = Geometry(lat, lon, level)
        for ch in channels:
            if ch not in ref["channels"]:
                continue
            target = ref["data"][:, ref["channels"].index(ch), 0].astype(np.float64)
            preds = {}
            for st, label in zip(stores, labels, strict=True):
                p = st.dataset(sample, stream, f, "prediction")
                if p is None or ch not in p["channels"]:
                    continue
                j = W.match_points(p["coords"][:, 0], p["coords"][:, 1], ref["coords"])
                y = np.full(len(lat), np.nan)
                ok = j >= 0
                y[ok] = p["data"][j[ok], p["channels"].index(ch), :].astype(np.float64).mean(-1)
                preds[label] = y
            views = [("full", data_box(lat, lon), res_full, False)]
            views += [(f"zoom{i}", tuple(z), res_zoom, True) for i, z in enumerate(zooms)]
            for name, box, res, cell_edges in views:
                path = out_dir / f"maps_{ch}_s{sample}_f{f}_{name}.png"
                title = f"{stream} {ch}, sample {sample}, step {f}, {name}"
                map_figure(geo, lat, lon, target, preds, title, path, box, res, level, cell_edges)
                print(f">> {path}")  # noqa: T201
    for st in stores:
        st.close()


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
    ap.add_argument("--maps", action="store_true", help="side-by-side maps of the runs")
    ap.add_argument("--map-fsteps", nargs="*", type=int, help="default: first, middle, last")
    ap.add_argument(
        "--zoom",
        nargs=4,
        type=float,
        action="append",
        default=[],
        metavar=("LON0", "LON1", "LAT0", "LAT1"),
        help="zoom box (repeatable)",
    )
    ap.add_argument("--res-full", type=float, default=0.25, help="pixel of the full view (deg)")
    ap.add_argument("--res-zoom", type=float, default=0.1, help="pixel of the zoom views (deg)")
    a = ap.parse_args()
    labels = a.labels or [z.parent.name for z in a.zips]
    run(a.zips, labels, a.stream, a.channels, a.samples, a.fsteps, a.healpix_level, a.members,
        a.out_dir)  # fmt: skip
    if a.maps:
        maps(a.zips, labels, a.stream, a.channels, a.samples, a.map_fsteps, a.healpix_level,
             a.zoom, a.out_dir, a.res_full, a.res_zoom)  # fmt: skip


if __name__ == "__main__":
    main()
