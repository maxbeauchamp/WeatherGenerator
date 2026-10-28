#!/usr/bin/env python3
"""
Publication-ready sea-ice figures and animations: WeatherGenerator forecast vs CICE truth.

Input (format detected from the extension, or forced with --format):
  zarr   : the raw inference output  results/<run>/validation_chkpt*_rank*.zip (or .zarr),
           read with weathergen.common.io; all samples, or --samples; stream --stream (CICE)
  netcdf : files written by scripts/wg_zip2netcdf.py (<var> = prediction, <var>_target = truth)
Output: for each sample, per variable
          <prefix>_<var>_s<NNNN>_d<DD>.png/.pdf   still frames (chosen lead days)
          <prefix>_<var>_s<NNNN>.mp4 (or .gif)     animation over the forecast lead times
Layout: [WeatherGenerator] [CICE] [WG - CICE], North Polar Stereographic, circular boundary,
        15 % ice edges, metrics (RMSE, sea-ice extent, IIEE), optional extent time series.

The native points are remapped once (nearest neighbour, max distance) onto a regular polar
stereographic grid: no seam / pole artefacts of curvilinear grids, fast animations.
Coastlines and land come from cartopy when available (WeatherGenerator env), else a light
fallback (no coastlines) is used.

Examples (from the WeatherGenerator root):
  # directly from the inference zip
  uv run --offline python scripts/plot_seaice.py \
      /e/scratch/weatherai/shared_work/results/inf_cice01/validation_chkpt00000_rank0000.zip \
      --samples 0 3 --vars aice hi speed --days 1 5 10 --animate --timeseries
  # from the NetCDF export
  uv run --offline python scripts/plot_seaice.py \
      /e/scratch/weatherai/shared_work/results/inf_cice01/netcdf/*CICE_s0000.nc \
      --vars aice hi speed --days 1 5 10 --animate --timeseries --out-dir figs_cice
  # all samples, PDF stills for a paper:
  uv run --offline python scripts/plot_seaice.py .../netcdf/*CICE_s*.nc --vars aice --days 10 --pdf
"""

from __future__ import annotations

import argparse
import contextlib
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:  # pragma: no cover
    HAS_CARTOPY = False

R_EARTH = 6371.0e3

# ------------------------------------------------------------------------------------------
# Colour maps (no cmocean dependency)
# ------------------------------------------------------------------------------------------
ICE = mcolors.LinearSegmentedColormap.from_list(
    "wg_ice", ["#03071e", "#0b2545", "#13406b", "#2f6690", "#5e9cc7", "#a9cfe8", "#e3f0f9", "#ffffff"])
THICK = mcolors.LinearSegmentedColormap.from_list(
    "wg_thick", ["#03071e", "#1d3557", "#2a6f97", "#61a5c2", "#a9d6e5", "#f1faee", "#ffd6a5", "#ff9f1c"])
DIFF = plt.get_cmap("RdBu_r")
LAND = "#d9d4c7"
OCEAN_BG = "#03071e"

VARS = {
    "aice": dict(label="Sea-ice concentration", units="", vmin=0, vmax=1, cmap=ICE, dmax=0.5,
                 fmt="{:.0%}"),
    "hi": dict(label="Sea-ice thickness (cell mean)", units="m", vmin=0, vmax=4, cmap=THICK, dmax=1.0),
    "hs": dict(label="Snow thickness (cell mean)", units="m", vmin=0, vmax=0.4, cmap=THICK, dmax=0.15),
    "speed": dict(label="Sea-ice drift speed", units="m s$^{-1}$", vmin=0, vmax=0.3, cmap="magma", dmax=0.15),
    "strength": dict(label="Ice strength", units="kN m$^{-1}$", vmin=0, vmax=60, cmap="cividis",
                     dmax=20, scale=1e-3),
    "uvel": dict(label="Ice velocity, grid x", units="m s$^{-1}$", vmin=-0.3, vmax=0.3, cmap="RdBu_r", dmax=0.15),
    "vvel": dict(label="Ice velocity, grid y", units="m s$^{-1}$", vmin=-0.3, vmax=0.3, cmap="RdBu_r", dmax=0.15),
}

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": None,  # fixed layout: every PNG / frame has the same size and colour bars
    "savefig.pad_inches": 0.04,
}


# ------------------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------------------
def load_netcdf(path: pathlib.Path, var: str) -> dict:
    """Read one wg_zip2netcdf file. Returns flattened points and (nstep, npts) arrays."""
    import xarray as xr  # noqa: PLC0415

    ds = xr.open_dataset(path)

    def get(name):
        if name not in ds:
            raise KeyError(f"{name} not in {path.name}: {list(ds.data_vars)}")
        a = ds[name]
        if "ens" in a.dims:
            a = a.mean("ens")
        return a.values.reshape(a.sizes["forecast_step"], -1).astype(np.float64)

    if var == "speed":
        pred = np.hypot(get("uvel"), get("vvel"))
        truth = np.hypot(get("uvel_target"), get("vvel_target"))
    else:
        pred, truth = get(var), get(f"{var}_target")
    scale = VARS.get(var, {}).get("scale", 1.0)
    return {
        "lat": ds["lat"].values.ravel().astype(np.float64),
        "lon": ds["lon"].values.ravel().astype(np.float64),
        "pred": pred * scale,
        "truth": truth * scale,
        "aice_pred": get("aice") if "aice" in ds else None,
        "aice_truth": get("aice_target") if "aice_target" in ds else None,
        "valid_time": ds["valid_time"].values,
        "lead_h": ds["lead_time_hours"].values if "lead_time_hours" in ds else
        np.arange(1, ds.sizes["forecast_step"] + 1) * 24.0,
        "ref_time": ds["forecast_reference_time"].values if "forecast_reference_time" in ds else None,
        "sample": int(ds.attrs.get("sample", 0)),
    }


@contextlib.contextmanager
def _quiet():
    """Silence numpy RuntimeWarnings (all-NaN slices on land / outside the domain)."""
    import warnings  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


class ZarrSource:
    """Raw inference output (.zip / .zarr). Points of every step / kind / sample are matched
    (nearest neighbour, as in wg_zip2netcdf) onto one reference point list, so that the polar
    remapping is built once and shared by all samples."""

    def __init__(self, path: pathlib.Path, stream: str, member=None, keep_ens: bool = False):
        self.member, self.keep_ens = member, keep_ens
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        import wg_zip2netcdf as W  # noqa: PLC0415  (same scripts/ folder)

        self.W, self.stream = W, stream
        self.store = W.Store(path)
        if stream not in self.store.streams:
            raise SystemExit(f"stream {stream!r} not in {path.name}: {self.store.streams}")
        self.ref = None  # (lat, lon) of the reference point list
        self._cache = {}

    @property
    def samples(self):
        return [int(x) for x in self.store.samples]

    def _sample(self, sample: int):
        if sample in self._cache:
            return self._cache[sample]
        st, W = self.store, self.W
        steps, per = [], []
        for f in st.fsteps(str(sample), self.stream):
            p = st.dataset(str(sample), self.stream, f, "prediction")
            t = st.dataset(str(sample), self.stream, f, "target")
            if p is not None and t is not None:  # fstep 0 has no target/prediction
                steps.append(f)
                per.append((p, t))
        if not steps:
            raise SystemExit(f"no prediction/target for sample {sample} / {self.stream}")
        if self.ref is None:
            c = per[0][1]["coords"]
            self.ref = (c[:, 0].astype(np.float64), c[:, 1].astype(np.float64))
        lat, lon = self.ref
        npts, chans = len(lat), per[0][0]["channels"]
        nens = int(per[0][0]["data"].shape[2])
        if self.member is not None and self.member >= nens:
            raise SystemExit(f"--member {self.member}: only {nens} member(s) in the output")
        ens = (np.full((len(steps), len(chans), npts, nens), np.nan, dtype=np.float32)
               if self.keep_ens else None)
        fields = {}
        for kind, k in (("pred", 0), ("truth", 1)):
            arr = np.full((len(steps), len(chans), npts), np.nan)
            for s, pt in enumerate(per):
                d = pt[k]
                j = W.match_points(lat, lon, d["coords"])
                ok = j >= 0
                dd = d["data"][ok]  # (points, channels, ens)
                if kind == "pred" and self.member is not None:
                    arr[s][:, j[ok]] = dd[:, :, self.member].T
                else:
                    with np.errstate(all="ignore"), _quiet():
                        arr[s][:, j[ok]] = np.nanmean(dd, axis=2).T  # mean over ens (NaN: land)
                if kind == "pred" and ens is not None:
                    ens[s][:, j[ok], :] = dd.transpose(1, 0, 2)
            fields[kind] = arr
        fields["ens"] = ens
        vt = np.array([pt[1]["times"].max() for pt in per], dtype="datetime64[ns]")
        ref_t = W.forecast_reference_time(st, str(sample), self.stream, vt)
        out = dict(chans=chans, fields=fields, valid_time=vt, ref_time=ref_t, nens=nens,
                   lead_h=((vt - ref_t) / np.timedelta64(1, "h")).astype(np.float64))
        self._cache = {sample: out}  # keep one sample in memory
        return out

    def load(self, sample: int, var: str) -> dict:
        d = self._sample(sample)
        ch = d["chans"]

        def get(kind, name):
            if name not in ch:
                raise KeyError(f"{name} not in stream {self.stream}: {ch}")
            return d["fields"][kind][:, ch.index(name), :]

        if var == "speed":
            pred = np.hypot(get("pred", "uvel"), get("pred", "vvel"))
            truth = np.hypot(get("truth", "uvel"), get("truth", "vvel"))
        else:
            pred, truth = get("pred", var), get("truth", var)
        scale = VARS.get(var, {}).get("scale", 1.0)
        has_a = "aice" in ch
        ens = d["fields"]["ens"]
        pred_ens = None
        if ens is not None:
            if var == "speed":
                pred_ens = np.hypot(ens[:, ch.index("uvel")], ens[:, ch.index("vvel")]) * scale
            else:
                pred_ens = ens[:, ch.index(var)] * scale
        return {
            "lat": self.ref[0], "lon": self.ref[1],
            "pred": pred * scale, "truth": truth * scale,
            "aice_pred": get("pred", "aice") if has_a else None,
            "aice_truth": get("truth", "aice") if has_a else None,
            "valid_time": d["valid_time"], "lead_h": d["lead_h"], "ref_time": d["ref_time"],
            "sample": sample, "nens": d["nens"], "pred_ens": pred_ens,
            "aice_ens": (d["fields"]["ens"][:, ch.index("aice")]
                         if has_a and d["fields"]["ens"] is not None else None),
        }


def detect_format(path: pathlib.Path, forced: str) -> str:
    if forced != "auto":
        return forced
    if path.suffix in (".zip", ".zarr"):
        return "zarr"
    if path.suffix in (".nc", ".nc4", ".netcdf"):
        return "netcdf"
    raise SystemExit(f"cannot detect the format of {path.name}: use --format zarr|netcdf")


# ------------------------------------------------------------------------------------------
# Polar stereographic remapping
# ------------------------------------------------------------------------------------------
class PolarGrid:
    """Regular North Polar Stereographic grid covering lat >= lat_min, with a nearest-neighbour
    mapping from native points. Uses the cartopy projection itself when available, so that the
    image is exactly aligned with cartopy coastlines."""

    def __init__(self, lat, lon, lat_min=45.0, res_km=12.5, max_dist_km=30.0):
        from scipy.spatial import cKDTree  # noqa: PLC0415

        self.lat_min = lat_min
        self.crs = ccrs.NorthPolarStereo() if HAS_CARTOPY else None
        ok = np.isfinite(lat) & np.isfinite(lon) & (lat >= lat_min - 2.0)
        self.src_idx = np.flatnonzero(ok)
        xs, ys = self.project(lon[ok], lat[ok])
        # radius of the lat_min circle (projected), and the output grid
        rx, ry = self.project(np.array([0.0]), np.array([lat_min]))
        self.radius = float(np.hypot(rx, ry)[0])
        r = self.radius
        n = int(np.ceil(2 * r / (res_km * 1e3)))
        c = np.linspace(-r, r, n)
        self.extent = (-r, r, -r, r)
        gx, gy = np.meshgrid(c, c[::-1])  # row 0 = top
        self.inside = np.hypot(gx, gy) <= r
        tree = cKDTree(np.column_stack([xs, ys]))
        d, j = tree.query(np.column_stack([gx[self.inside], gy[self.inside]]),
                          distance_upper_bound=max_dist_km * 1e3)
        self.shape = gx.shape
        self.map = np.full(gx.size, -1, dtype=np.int64)
        flat_inside = np.flatnonzero(self.inside.ravel())
        hit = np.isfinite(d)
        self.map[flat_inside[hit]] = self.src_idx[j[hit]]
        # true area of each pixel (spherical polar stereographic, scale factor k = 2 / (1 + sin(lat)))
        rho = np.hypot(gx, gy)
        plat = np.pi / 2 - 2 * np.arctan(rho / (2 * R_EARTH))
        k = 2.0 / (1.0 + np.sin(plat))
        self.area = (c[1] - c[0]) ** 2 / k**2  # m2

    def project(self, lon, lat):
        if self.crs is not None:
            p = self.crs.transform_points(ccrs.PlateCarree(), lon, lat)
            return p[:, 0], p[:, 1]
        phi, lam = np.deg2rad(lat), np.deg2rad(lon)
        rho = 2 * R_EARTH * np.tan(np.pi / 4 - phi / 2)
        return rho * np.sin(lam), -rho * np.cos(lam)

    def remap(self, values: np.ndarray) -> np.ndarray:
        out = np.full(self.map.size, np.nan)
        ok = self.map >= 0
        out[ok] = values[self.map[ok]]
        return out.reshape(self.shape)


# ------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------
def ice_metrics(pred_img, truth_img, aice_p, aice_t, area, var):
    ok = np.isfinite(pred_img) & np.isfinite(truth_img)
    rmse = float(np.sqrt(np.average((pred_img[ok] - truth_img[ok]) ** 2, weights=area[ok]))) if ok.any() else np.nan
    m = {"rmse": rmse}
    if aice_p is not None and aice_t is not None:
        okc = np.isfinite(aice_p) & np.isfinite(aice_t)
        ep, et = (aice_p >= 0.15) & okc, (aice_t >= 0.15) & okc
        m["sie_wg"] = float(area[ep].sum()) / 1e12  # 10^6 km2
        m["sie_cice"] = float(area[et].sum()) / 1e12
        m["iiee"] = float(area[ep ^ et].sum()) / 1e12
    return m


# ------------------------------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------------------------------
def fixed_ticks(cb, vmin, vmax, n=5):
    """Same ticks on every colour bar of a quantity, in every figure."""
    cb.set_ticks(np.linspace(vmin, vmax, n))


def save_animation(fig, draw, n, stem, fps):
    """mp4 with ffmpeg; otherwise a GIF with ONE palette shared by all frames.
    (matplotlib's PillowWriter quantises every frame to its own 256-colour palette: the same
    value then gets a different colour from frame to frame, colour bars included.)"""
    if animation.writers.is_available("ffmpeg"):
        out = f"{stem}.mp4"
        anim = animation.FuncAnimation(fig, draw, frames=range(n), blit=False)
        anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=6000,
                                                     extra_args=["-pix_fmt", "yuv420p"]), dpi=200)
        return out
    from PIL import Image  # noqa: PLC0415

    frames = []
    for s in range(n):
        draw(s)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    # global palette from (a subsample of) all frames, stacked
    step = max(1, n // 12)
    sub = [f.reduce(2) for f in frames[::step]]
    w, h = sub[0].size
    mosaic = Image.new("RGB", (w, h * len(sub)))
    for i, f in enumerate(sub):
        mosaic.paste(f, (0, i * h))
    # palette = the colour maps of the figure (sampled finely, so maps and colour bars keep
    # their true colours) + the most frequent other colours (land, text, edges, curves)
    cmaps = {im.cmap.name: im.cmap for ax in fig.axes for im in ax.get_images()}
    per = max(24, min(64, 150 // max(1, len(cmaps))))
    cols = [tuple(int(round(255 * c)) for c in cm(x)[:3]) for cm in cmaps.values()
            for x in np.linspace(0, 1, per)]
    fixed = ["#ffffff", "#000000", "#222222", "#111111", "#333333", "#4a4a4a", "#555555", "#888888",
             "#bbbbbb", "#e6e6e6", "#ff8c00", "#ffaa4c", "#ffd1a3", "#ffe8cc", LAND, OCEAN_BG]
    cols = [tuple(int(round(255 * c)) for c in mcolors.to_rgb(h)) for h in fixed] + cols
    rest = 255 - len(cols)
    extra = mosaic.quantize(colors=rest, method=Image.Quantize.MEDIANCUT).getpalette()[: 3 * rest]
    flat = [v for c in cols for v in c] + extra
    pal = Image.new("P", (1, 1))
    pal.putpalette(flat + [0] * (768 - len(flat)))
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    out = f"{stem}.gif"
    q[0].save(out, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0,
              optimize=False, disposal=1)
    return out


def make_axes(fig, rect, grid: PolarGrid):
    if HAS_CARTOPY:
        ax = fig.add_axes(rect, projection=grid.crs)
        ax.set_extent(grid.extent, crs=grid.crs)
        import matplotlib.path as mpath  # noqa: PLC0415

        theta = np.linspace(0, 2 * np.pi, 200)
        circle = mpath.Path(np.column_stack([np.sin(theta), np.cos(theta)]) * 0.5 + 0.5)
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND, edgecolor="none", zorder=0)
        ax.set_facecolor(LAND)
    else:
        ax = fig.add_axes(rect)
        ax.set_xlim(grid.extent[:2]); ax.set_ylim(grid.extent[2:])
        ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(plt.Circle((0, 0), grid.radius, facecolor=LAND, edgecolor="none", zorder=0))
    return ax


def decorate(ax, grid: PolarGrid):
    if HAS_CARTOPY:
        ax.coastlines(resolution="50m", linewidth=0.35, color="#4a4a4a", zorder=4)
        gl = ax.gridlines(linewidth=0.3, color="white", alpha=0.35, linestyle="-",
                          xlocs=np.arange(-180, 180, 30), ylocs=[50, 60, 70, 80], zorder=5)
        gl.n_steps = 90
    else:
        for la in (50, 60, 70, 80):
            r = 2 * R_EARTH * np.tan(np.pi / 4 - np.deg2rad(la) / 2)
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color="white", alpha=0.3, lw=0.3, zorder=5))
        ax.add_patch(plt.Circle((0, 0), grid.radius, fill=False, color="#4a4a4a", lw=0.6, zorder=6))


def imshow(ax, img, grid, **kw):
    return ax.imshow(np.where(grid.inside, img, np.nan), extent=grid.extent, origin="upper",
                     interpolation="nearest", zorder=2,
                     **({"transform": grid.crs} if HAS_CARTOPY else {}), **kw)


def edge(ax, aice_img, grid, color, ls):
    if aice_img is None or not np.isfinite(aice_img).any():
        return None
    x = np.linspace(grid.extent[0], grid.extent[1], grid.shape[1])
    y = np.linspace(grid.extent[3], grid.extent[2], grid.shape[0])
    # masked (not zero-filled) land: no spurious "ice edge" drawn along the coastline
    return ax.contour(x, y, np.ma.masked_invalid(aice_img), levels=[0.15], colors=[color],
                      linewidths=0.8, linestyles=ls, zorder=7,
                      **({"transform": grid.crs} if HAS_CARTOPY else {}))


class Frame:
    """One figure, updated in place for each lead time (fast animations)."""

    def __init__(self, data, var, grid, timeseries, title_extra=""):
        self.d, self.var, self.g, self.ts = data, var, grid, timeseries
        spec = VARS.get(var, dict(label=var, units="", vmin=np.nanpercentile(data["truth"], 1),
                                  vmax=np.nanpercentile(data["truth"], 99), cmap="viridis",
                                  dmax=np.nanstd(data["truth"])))
        self.spec = spec
        # ---- fixed layout in inches (no overlaps, identical for every frame) ----------
        W, M, GAP = 7.6, 2.25, 0.15           # figure width, map size, gap between maps
        HEAD, MET, CB, TS = 0.62, 0.22, 0.50, (1.45 if timeseries else 0.0)
        H = HEAD + 0.22 + M + MET + CB + TS + 0.05
        self.fig = plt.figure(figsize=(W, H), facecolor="white")

        def box(x, y, w, h):  # inches from bottom-left -> figure fraction
            return [x / W, y / H, w / W, h / H]

        x0 = (W - 3 * M - 2 * GAP) / 2
        y_map = TS + CB + MET
        self.axes = [make_axes(self.fig, box(x0 + i * (M + GAP), y_map, M, M), grid) for i in range(3)]
        for ax, t in zip(self.axes, ["WeatherGenerator", "CICE (truth)", "WG − CICE"]):
            ax.set_title(t, pad=4)
        self.suptitle = self.fig.text(x0 / W, 1 - 0.08 / H, "", ha="left", va="top", fontsize=10.5,
                                      weight="bold")
        self.subtitle = self.fig.text(x0 / W, 1 - 0.36 / H, "", ha="left", va="top", fontsize=7.5,
                                      color="#555")
        self.fig.text(1 - x0 / W, 1 - 0.08 / H, title_extra, ha="right", va="top", fontsize=7,
                      color="#888")
        nan = np.full(grid.shape, np.nan)
        self.im = []
        for i in range(3):
            if i < 2:
                kw = dict(cmap=spec["cmap"], vmin=spec["vmin"], vmax=spec["vmax"])
            else:
                kw = dict(cmap=DIFF, vmin=-spec["dmax"], vmax=spec["dmax"])
            self.im.append(imshow(self.axes[i], nan, grid, **kw))
            decorate(self.axes[i], grid)
        # metrics just under each map, colorbars below them
        self.metric_txt = [self.fig.text((x0 + i * (M + GAP) + M / 2) / W, (y_map - 0.04) / H, "",
                                         ha="center", va="top", fontsize=7, color="#333")
                           for i in range(3)]
        cb_y = TS + CB - 0.16
        cax1 = self.fig.add_axes(box(x0 + 0.25, cb_y, 2 * M + GAP - 0.5, 0.075))
        cax2 = self.fig.add_axes(box(x0 + 2 * (M + GAP) + 0.25, cb_y, M - 0.5, 0.075))
        u = f" ({spec['units']})" if spec["units"] else ""
        c1 = self.fig.colorbar(self.im[0], cax=cax1, orientation="horizontal",
                               extend="neither" if var == "aice" else "max")
        c1.set_label(spec["label"] + u, labelpad=1, fontsize=7.5)
        c2 = self.fig.colorbar(self.im[2], cax=cax2, orientation="horizontal", extend="both")
        c2.set_label("Difference" + u, labelpad=1, fontsize=7.5)
        fixed_ticks(c1, spec["vmin"], spec["vmax"])
        fixed_ticks(c2, -spec["dmax"], spec["dmax"])
        for c in (c1, c2):
            c.outline.set_linewidth(0.4)
            c.ax.tick_params(length=2, width=0.4, labelsize=6.5, pad=1)
        if var == "aice":
            c1.ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        self._ts_box = box(x0 + 0.45, 0.62, 3 * M + 2 * GAP - 0.55, TS - 0.72) if timeseries else None
        self.contours = []
        # pre-compute remapped fields and metrics for all steps
        n = data["pred"].shape[0]
        self.P = [grid.remap(data["pred"][s]) for s in range(n)]
        self.T = [grid.remap(data["truth"][s]) for s in range(n)]
        has_aice = data["aice_pred"] is not None
        self.AP = [grid.remap(data["aice_pred"][s]) for s in range(n)] if has_aice else [None] * n
        self.AT = [grid.remap(data["aice_truth"][s]) for s in range(n)] if has_aice else [None] * n
        self.M = [ice_metrics(self.P[s], self.T[s], self.AP[s], self.AT[s], grid.area, var) for s in range(n)]
        if timeseries:
            self.tax = self.fig.add_axes(self._ts_box)
            self._init_ts()

    def _init_ts(self):
        ax, lead = self.tax, self.d["lead_h"] / 24.0
        key = "sie_wg" in self.M[0]
        ys = ([m["sie_cice"] for m in self.M], [m["sie_wg"] for m in self.M]) if key else \
             ([np.nan] * len(lead), [m["rmse"] for m in self.M])
        self.ts_lines = [ax.plot([], [], color=c, lw=1.4, marker="o", ms=2.5, label=l)[0]
                         for c, l in (("#222222", "CICE"), ("#ff8c00", "WeatherGenerator"))]
        allv = np.array([v for y in ys for v in y if np.isfinite(v)])
        ax.set_xlim(lead.min() - 0.3, lead.max() + 0.3)
        if allv.size:
            pad = 0.08 * (allv.max() - allv.min() + 1e-9)
            ax.set_ylim(allv.min() - pad, allv.max() + pad)
        ax.set_xlabel("Lead time (days)", labelpad=1)
        ax.set_ylabel("Extent (10$^6$ km$^2$)" if key else "RMSE", labelpad=2)
        ax.spines[["top", "right"]].set_visible(False)
        for s in ax.spines.values():
            s.set_linewidth(0.5)
        ax.tick_params(length=2, width=0.4, labelsize=7)
        ax.grid(alpha=0.25, lw=0.4)
        # legend at a fixed place under the curve (does not move during the animation)
        self.fig.legend(handles=self.ts_lines, frameon=False, fontsize=7, loc="lower center",
                        bbox_to_anchor=(0.5, 0.0), ncol=2)
        self.ts_y, self.ts_x = ys, lead

    def draw(self, s):
        P, T = self.P[s], self.T[s]
        self.im[0].set_data(np.where(self.g.inside, P, np.nan))
        self.im[1].set_data(np.where(self.g.inside, T, np.nan))
        self.im[2].set_data(np.where(self.g.inside, P - T, np.nan))
        for c in self.contours:
            c.remove()
        self.contours = [c for c in (
            edge(self.axes[0], self.AP[s], self.g, "#ff8c00", "solid"),
            edge(self.axes[1], self.AT[s], self.g, "#ffffff", "dashed"),
            edge(self.axes[2], self.AP[s], self.g, "#ff8c00", "solid"),
            edge(self.axes[2], self.AT[s], self.g, "#111111", "dashed"),
        ) if c is not None]
        vt = np.datetime_as_string(self.d["valid_time"][s], unit="D")
        lead = self.d["lead_h"][s] / 24.0
        rt = self.d["ref_time"]
        ref = np.datetime_as_string(rt, unit="D") if rt is not None and not np.isnat(rt) else "?"
        self.suptitle.set_text(f"{self.spec['label']}  ·  valid {vt}")
        self.subtitle.set_text(f"forecast from {ref}, lead time +{lead:.0f} d  ·  "
                               "15 % ice edge — WG: orange, CICE: dashed")
        m = self.M[s]
        fmt = "{:.3f}" if self.spec["vmax"] <= 1.5 else "{:.2f}"
        self.metric_txt[2].set_text(f"RMSE {fmt.format(m['rmse'])}"
                                    + (f"   IIEE {m['iiee']:.2f}×10$^6$ km$^2$" if "iiee" in m else ""))
        if "sie_wg" in m:
            self.metric_txt[0].set_text(f"extent {m['sie_wg']:.2f}×10$^6$ km$^2$")
            self.metric_txt[1].set_text(f"extent {m['sie_cice']:.2f}×10$^6$ km$^2$")
        if self.ts:
            self.ts_lines[0].set_data(self.ts_x[: s + 1], self.ts_y[0][: s + 1])
            self.ts_lines[1].set_data(self.ts_x[: s + 1], self.ts_y[1][: s + 1])
        return self.im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=pathlib.Path, help=".zip/.zarr inference output or .nc files")
    ap.add_argument("--format", choices=["auto", "zarr", "netcdf"], default="auto",
                    help="input format (default: from the file extension)")
    ap.add_argument("--stream", default="CICE", help="stream to plot (zarr input)")
    ap.add_argument("--samples", nargs="*", type=int, help="samples to plot (zarr input; default all)")
    ap.add_argument("--vars", nargs="+", default=["aice", "hi", "speed"])
    ap.add_argument("--days", nargs="*", type=int, default=[1, 5, 10], help="lead days for still frames")
    ap.add_argument("--animate", action="store_true", help="animation over all lead times")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--timeseries", action="store_true", help="add the extent (or RMSE) curve")
    ap.add_argument("--pdf", action="store_true", help="also save still frames as PDF")
    ap.add_argument("--lat-min", type=float, default=45.0)
    ap.add_argument("--res-km", type=float, default=12.5)
    ap.add_argument("--label", default="", help="text in the top-right corner (e.g. run id)")
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("figs_seaice"))
    ap.add_argument("--member", type=int, default=None,
                    help="plot this ensemble member instead of the ensemble mean (zarr input)")
    ap.add_argument("--ens", action="store_true",
                    help="ensemble figures (zarr input, >= 2 members): for every --vars, a 2x3 figure "
                         "(mean, member 0, member 1 / spread, CICE, mean - CICE), also animated with "
                         "--animate; for aice also ice-edge probability and spread/RMSE per lead time")
    a = ap.parse_args()
    plt.rcParams.update(STYLE)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if not HAS_CARTOPY:
        print("cartopy not found: maps without coastlines (run in the WeatherGenerator env)")

    grid = None
    import re  # noqa: PLC0415

    for f in sorted(a.files):
        fmt = detect_format(f, a.format)
        # several rank files (one per GPU): their samples have the same local index -> tag files
        m_rank = re.search(r"rank(\d+)", f.name)
        tag = f"_r{int(m_rank.group(1)):02d}" if (len(a.files) > 1 and m_rank) else ""
        if fmt == "zarr":
            src = ZarrSource(f, a.stream, member=a.member, keep_ens=a.ens)
            samples = a.samples if a.samples else src.samples
            print(f"{f.name}: zarr input, stream {a.stream}, samples {samples}")
            jobs = [(lambda var, s=s: src.load(s, var)) for s in samples]
        else:
            jobs = [(lambda var, f=f: load_netcdf(f, var))]
        for job in jobs:
            for var in a.vars:
                d = job(var)
                d["tag"] = tag
                if grid is None:  # same native points for all inputs: build the mapping once
                    grid = PolarGrid(d["lat"], d["lon"], a.lat_min, a.res_km)
                render(d, var, grid, a, f)
                if a.ens:
                    if var == "aice":
                        render_ens(d, grid, a)
                    render_ens_anim(d, var, grid, a)

def render(d, var, grid, a, f):
    fr = Frame(d, var, grid, a.timeseries, a.label)
    nens = d.get("nens", 1)
    mem = getattr(a, "member", None)
    if mem is not None:
        fr.axes[0].set_title(f"WG, member {mem}", pad=4)
    elif nens > 1:
        fr.axes[0].set_title(f"WG, mean of {nens}", pad=4)
    stem = a.out_dir / (f"{a.label + '_' if a.label else ''}{var}{d.get('tag', '')}_s{d['sample']:04d}"
                        + (f"_m{mem:02d}" if mem is not None else ""))
    lead_days = np.rint(d["lead_h"] / 24.0).astype(int)
    for day in a.days:
        s = np.flatnonzero(lead_days == day)
        if not s.size:
            continue
        fr.draw(int(s[0]))
        fr.fig.savefig(f"{stem}_d{day:02d}.png")
        if a.pdf:
            fr.fig.savefig(f"{stem}_d{day:02d}.pdf")
        print("saved", f"{stem}_d{day:02d}.png")
    if a.animate:
        out = save_animation(fr.fig, fr.draw, d["pred"].shape[0], stem, a.fps)
        print("saved", out)
    plt.close(fr.fig)


def render_ens(d, grid, a):
    """Ensemble figure for aice: P(aice >= 15 %), spread, |mean - truth|; spread/RMSE per lead."""
    import warnings  # noqa: PLC0415

    E, T = d.get("aice_ens"), d.get("aice_truth")
    if E is None or T is None or d.get("nens", 1) < 2:
        print("--ens: skipped (needs aice and >= 2 members)")
        return
    nm = E.shape[-1]
    stem = a.out_dir / f"{a.label + '_' if a.label else ''}ens_aice{d.get('tag', '')}_s{d['sample']:04d}"
    lead = d["lead_h"] / 24.0
    lead_days = np.rint(lead).astype(int)
    rows = []
    for s in range(E.shape[0]):
        e = E[s].astype(np.float64)  # (points, members)
        nval = np.isfinite(e).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prob = np.where(nval > 0, (e >= 0.15).sum(axis=1) / np.maximum(nval, 1), np.nan)
            mean = np.nanmean(e, axis=1)
            spread = np.nanstd(e, axis=1, ddof=1)
        err = np.abs(mean - T[s])
        Pi, Si, Ei, Ti, Mi = (grid.remap(v) for v in (prob, spread, err, T[s], mean))
        # scores over the ice-related area (truth or mean >= 1 %), area weighted
        ok = np.isfinite(Si) & np.isfinite(Ei) & ((Ti >= 0.01) | (Mi >= 0.01))
        w = grid.area[ok]
        sp = float(np.sqrt(np.average(Si[ok] ** 2, weights=w))) if ok.any() else np.nan
        rm = float(np.sqrt(np.average(Ei[ok] ** 2, weights=w))) if ok.any() else np.nan
        ratio = np.sqrt((nm + 1) / nm) * sp / rm if rm > 0 else np.nan
        rows.append((lead[s], sp, rm, ratio))
        if lead_days[s] not in a.days:
            continue
        W, H, M, GAP = 7.6, 3.75, 2.25, 0.15
        x0 = (W - 3 * M - 2 * GAP) / 2
        fig = plt.figure(figsize=(W, H), facecolor="white")
        panels = [("P(aice ≥ 15 %)", Pi, dict(cmap=ICE, vmin=0, vmax=1), "#ff8c00"),
                  ("Ensemble spread (std)", Si, dict(cmap="viridis", vmin=0, vmax=0.3), "#ffffff"),
                  ("|ensemble mean − CICE|", Ei, dict(cmap="Reds", vmin=0, vmax=0.5), "#111111")]
        for i, (title, img, kw, col) in enumerate(panels):
            ax = make_axes(fig, [(x0 + i * (M + GAP)) / W, 0.62 / H, M / W, M / H], grid)
            ax.set_title(title, pad=4)
            im = imshow(ax, img, grid, **kw)
            decorate(ax, grid)
            edge(ax, Ti, grid, col, "dashed")
            cax = fig.add_axes([(x0 + i * (M + GAP) + 0.3) / W, 0.36 / H, (M - 0.6) / W, 0.07 / H])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            fixed_ticks(cb, kw["vmin"], kw["vmax"])
            cb.outline.set_linewidth(0.4)
            cb.ax.tick_params(length=2, width=0.4, labelsize=6.5, pad=1)
        vt = np.datetime_as_string(d["valid_time"][s], unit="D")
        fig.text(x0 / W, 1 - 0.06 / H, f"Sea-ice ensemble ({nm} members)  ·  valid {vt}, lead +{lead[s]:.0f} d",
                 ha="left", va="top", fontsize=10, weight="bold")
        fig.text(x0 / W, 1 - 0.30 / H, f"CICE 15 % edge dashed  ·  ice area: spread {sp:.3f}, RMSE of the "
                 f"mean {rm:.3f}, spread/RMSE {ratio:.2f} (1 = calibrated)", ha="left", va="top",
                 fontsize=7.5, color="#555")
        fig.text(1 - x0 / W, 1 - 0.06 / H, a.label, ha="right", va="top", fontsize=7, color="#888")
        out = f"{stem}_d{lead_days[s]:02d}.png"
        fig.savefig(out)
        plt.close(fig)
        print("saved", out)
    arr = np.array(rows)
    np.savetxt(f"{stem}_spread_skill.csv", arr, delimiter=",", fmt="%.6g",
               header="lead_days,spread,rmse_of_mean,spread_rmse_ratio", comments="")
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.plot(arr[:, 0], arr[:, 2], "o-", ms=2.5, color="#c0392b", label="RMSE of the ensemble mean")
    ax.plot(arr[:, 0], arr[:, 1], "o-", ms=2.5, color="#0369a1", label="ensemble spread")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("aice (ice area)")
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"Spread vs skill, sample {d['sample']} ({nm} members)", fontsize=8)
    fig.savefig(f"{stem}_spread_skill.png")
    plt.close(fig)
    print("saved", f"{stem}_spread_skill.csv", f"{stem}_spread_skill.png")


class EnsFrame:
    """2 x 3 ensemble figure for one variable, updated in place for each lead time:
    WG mean | member 0 | member 1  /  spread | CICE | mean - CICE  (+ time series)."""

    def __init__(self, d, var, grid, a):
        E, T = d["pred_ens"], d["truth"]
        AE, AT = d.get("aice_ens"), d.get("aice_truth")
        self.d, self.g, self.var = d, grid, var
        self.n, self.nm = E.shape[0], E.shape[-1]
        m1 = 1 if self.nm > 1 else 0
        area = grid.area
        spec = VARS.get(var, dict(label=var, units="", vmin=np.nanpercentile(T, 1),
                                  vmax=np.nanpercentile(T, 99), cmap="viridis", dmax=np.nanstd(T)))
        self.spec = spec
        # time series: extent for aice, volume for thicknesses, area mean otherwise
        if var == "aice":
            self.ts_label = "Extent (10$^6$ km$^2$)"

            def tsval(img, aimg):
                return float(area[np.isfinite(img) & (img >= 0.15)].sum()) / 1e12
        elif var in ("hi", "hs"):
            self.ts_label = ("Ice" if var == "hi" else "Snow") + " volume (10$^3$ km$^3$)"

            def tsval(img, aimg):
                ok = np.isfinite(img)
                return float((img[ok] * area[ok]).sum()) / 1e12
        else:
            self.ts_label = f"Mean {spec['label'].lower()} (ice area)"

            def tsval(img, aimg):
                ok = np.isfinite(img) & (np.nan_to_num(aimg) >= 0.15 if aimg is not None else True)
                return float(np.average(img[ok], weights=area[ok])) if ok.any() else np.nan

        self.F, self.TSV, self.MET = [], [], []
        for s in range(self.n):
            e = E[s].astype(np.float64)
            with np.errstate(all="ignore"), _quiet():
                mean = np.nanmean(e, axis=1)
                spread = np.nanstd(e, axis=1, ddof=1)
            f = {k: grid.remap(v).astype(np.float32) for k, v in
                 (("mean", mean), ("m0", e[:, 0]), ("m1", e[:, m1]), ("spread", spread), ("truth", T[s]))}
            f["dev"] = f["mean"] - f["truth"]
            if AE is not None and AT is not None:  # 15 % ice edges, from aice
                ae = AE[s].astype(np.float64)
                with np.errstate(all="ignore"), _quiet():
                    f["a_mean"] = grid.remap(np.nanmean(ae, axis=1))
                f["a_m0"], f["a_m1"], f["a_truth"] = (grid.remap(v) for v in (ae[:, 0], ae[:, m1], AT[s]))
            else:
                f["a_mean"] = f["a_m0"] = f["a_m1"] = f["a_truth"] = None
            self.F.append(f)
            mem_a = ([grid.remap(AE[s][:, k]) for k in range(self.nm)] if AE is not None else [None] * self.nm)
            self.TSV.append((tsval(f["truth"], f["a_truth"]), tsval(f["mean"], f["a_mean"]),
                             [tsval(grid.remap(e[:, k]), mem_a[k]) for k in range(self.nm)]))
            self.MET.append(ice_metrics(f["mean"], f["truth"], f["a_mean"], f["a_truth"], area, var))

        # ---- layout in inches ------------------------------------------------------------
        W, M, GAP, HEAD, TT, CB = 7.6, 2.25, 0.15, 0.80, 0.24, 0.45
        TS = 1.65 if a.timeseries else 0.0
        H = HEAD + 2 * (TT + M + CB) + TS + 0.05
        self.fig = plt.figure(figsize=(W, H), facecolor="white")

        def box(x, y, w, h):
            return [x / W, y / H, w / W, h / H]

        x0 = (W - 3 * M - 2 * GAP) / 2
        y2 = TS + CB
        y1 = y2 + M + TT + CB
        main = dict(cmap=spec["cmap"], vmin=spec["vmin"], vmax=spec["vmax"])
        layout = [  # key, title, imshow kwargs, row, col
            ("mean", f"WG, mean of {self.nm}", main, 0, 0),
            ("m0", "WG, member 0", main, 0, 1),
            ("m1", f"WG, member {m1}", main, 0, 2),
            ("spread", "Ensemble spread (std)", dict(cmap="viridis", vmin=0, vmax=0.6 * spec["dmax"]), 1, 0),
            ("truth", "CICE (truth)", main, 1, 1),
            ("dev", "WG mean − CICE", dict(cmap=DIFF, vmin=-spec["dmax"], vmax=spec["dmax"]), 1, 2),
        ]
        nan = np.full(grid.shape, np.nan)
        self.ax, self.im = {}, {}
        for key, title, kw, r, c in layout:
            ax = make_axes(self.fig, box(x0 + c * (M + GAP), y1 if r == 0 else y2, M, M), grid)
            ax.set_title(title, pad=4)
            self.im[key] = imshow(ax, nan, grid, **kw)
            decorate(ax, grid)
            self.ax[key] = ax

        pct = var == "aice"
        u = f" ({spec['units']})" if spec.get("units") else ""

        def cbar(key, x, y, w, label, extend):
            cax = self.fig.add_axes(box(x, y, w, 0.075))
            cb = self.fig.colorbar(self.im[key], cax=cax, orientation="horizontal", extend=extend)
            norm = self.im[key].norm
            fixed_ticks(cb, norm.vmin, norm.vmax)
            cb.set_label(label, labelpad=1, fontsize=7.5)
            cb.outline.set_linewidth(0.4)
            cb.ax.tick_params(length=2, width=0.4, labelsize=6.5, pad=1)
            if pct and key in ("mean", "truth"):
                cb.ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

        ext_main = "neither" if var == "aice" else "max"
        cbar("mean", x0 + 0.25, y1 - 0.2, 3 * M + 2 * GAP - 0.5, spec["label"] + u, ext_main)
        cbar("spread", x0 + 0.25, y2 - 0.2, M - 0.5, "Spread" + u, "max")
        cbar("truth", x0 + (M + GAP) + 0.25, y2 - 0.2, M - 0.5, "CICE" + u, ext_main)
        cbar("dev", x0 + 2 * (M + GAP) + 0.25, y2 - 0.2, M - 0.5, "Difference" + u, "both")

        self.suptitle = self.fig.text(x0 / W, 1 - 0.08 / H, "", ha="left", va="top", fontsize=10.5,
                                      weight="bold")
        self.subtitle = self.fig.text(x0 / W, 1 - 0.36 / H, "", ha="left", va="top", fontsize=7.5,
                                      color="#555")
        self.fig.text(1 - x0 / W, 1 - 0.08 / H, a.label + d.get("tag", ""), ha="right", va="top",
                      fontsize=7, color="#888")
        self.contours = []

        self.ts = a.timeseries
        if self.ts:
            lead = d["lead_h"] / 24.0
            tax = self.fig.add_axes(box(x0 + 0.45, 0.62, 3 * M + 2 * GAP - 0.55, TS - 0.92))
            from matplotlib.patches import Patch  # noqa: PLC0415

            self.tax, self.band = tax, None
            # member spread around the ensemble mean: envelope (min - max) of the member values
            self.lo = np.array([np.nanmin(ms) for _, _, ms in self.TSV])
            self.hi = np.array([np.nanmax(ms) for _, _, ms in self.TSV])
            self.mean_line = tax.plot([], [], color="#ff8c00", lw=1.4, marker="o", ms=2.5, label="WG mean")[0]
            self.true_line = tax.plot([], [], color="#222222", lw=1.4, marker="o", ms=2.5, label="CICE")[0]
            allv = np.array([v for (t, m, _), lo, hi in zip(self.TSV, self.lo, self.hi)
                             for v in (t, m, lo, hi) if np.isfinite(v)])
            tax.set_xlim(lead.min() - 0.3, lead.max() + 0.3)
            if allv.size:
                pad = 0.08 * (allv.max() - allv.min() + 1e-9)
                tax.set_ylim(allv.min() - pad, allv.max() + pad)
            tax.set_xlabel("Lead time (days)", labelpad=1)
            tax.set_ylabel(self.ts_label, labelpad=2)
            tax.spines[["top", "right"]].set_visible(False)
            tax.tick_params(length=2, width=0.4, labelsize=7)
            tax.grid(alpha=0.25, lw=0.4)
            handles = [self.true_line, self.mean_line,
                       Patch(facecolor="#ff8c00", alpha=0.3, edgecolor="none", label=f"spread of the {self.nm} members (min–max)")]
            # legend at a fixed place under the curve (does not move during the animation)
            self.fig.legend(handles=handles, frameon=False, fontsize=7, loc="lower center",
                            bbox_to_anchor=(0.5, 0.0), ncol=3)
            self.lead = lead

    def draw(self, s):
        f, g = self.F[s], self.g
        for k, im in self.im.items():
            im.set_data(np.where(g.inside, f[k], np.nan))
        for c in self.contours:
            c.remove()
        cs = []
        for k in ("mean", "m0", "m1"):
            cs += [edge(self.ax[k], f["a_" + k], g, "#ff8c00", "solid"),
                   edge(self.ax[k], f["a_truth"], g, "#ffffff", "dashed")]
        cs += [edge(self.ax["spread"], f["a_truth"], g, "#ffffff", "dashed"),
               edge(self.ax["truth"], f["a_truth"], g, "#ffffff", "dashed"),
               edge(self.ax["dev"], f["a_mean"], g, "#ff8c00", "solid"),
               edge(self.ax["dev"], f["a_truth"], g, "#111111", "dashed")]
        self.contours = [c for c in cs if c is not None]
        vt = np.datetime_as_string(self.d["valid_time"][s], unit="D")
        lead = self.d["lead_h"][s] / 24.0
        rt = self.d["ref_time"]
        ref = np.datetime_as_string(rt, unit="D") if rt is not None and not np.isnat(rt) else "?"
        t, m, ms = self.TSV[s]
        met = self.MET[s]
        fmt = "{:.3f}" if self.spec["vmax"] <= 1.5 else "{:.2f}"
        self.suptitle.set_text(f"{self.spec['label']}, {self.nm}-member ensemble  ·  valid {vt}")
        self.subtitle.set_text(
            f"forecast from {ref}, lead +{lead:.0f} d  ·  15 % ice edge: WG orange, CICE dashed\n"
            f"{self.ts_label}: CICE {t:.2f}, WG mean {m:.2f}, members {np.nanmin(ms):.2f}–{np.nanmax(ms):.2f}  ·  "
            f"mean: RMSE {fmt.format(met['rmse'])}"
            + (f", IIEE {met['iiee']:.2f}×10$^6$ km$^2$" if "iiee" in met else ""))
        if self.ts:
            x = self.lead[: s + 1]
            mean = np.array([e[1] for e in self.TSV[: s + 1]])
            if self.band is not None:
                self.band.remove()
            self.band = self.tax.fill_between(x, self.lo[: s + 1], self.hi[: s + 1], color="#ff8c00",
                                              alpha=0.3, lw=0, zorder=1)
            self.mean_line.set_data(x, mean)
            self.true_line.set_data(x, [e[0] for e in self.TSV[: s + 1]])
        return list(self.im.values())


def render_ens_anim(d, var, grid, a):
    if d.get("pred_ens") is None or d.get("nens", 1) < 2:
        print(f"--ens: {var} skipped (needs >= 2 members)")
        return
    fr = EnsFrame(d, var, grid, a)
    stem = a.out_dir / f"{a.label + '_' if a.label else ''}ens6_{var}{d.get('tag', '')}_s{d['sample']:04d}"
    lead_days = np.rint(d["lead_h"] / 24.0).astype(int)
    for day in a.days:
        idx = np.flatnonzero(lead_days == day)
        if idx.size:
            fr.draw(int(idx[0]))
            fr.fig.savefig(f"{stem}_d{day:02d}.png")
            print("saved", f"{stem}_d{day:02d}.png")
    if a.animate:
        out = save_animation(fr.fig, fr.draw, fr.n, stem, a.fps)
        print("saved", out)
    plt.close(fr.fig)

if __name__ == "__main__":
    main()
