#!/usr/bin/env python3
"""
Convert a WeatherGenerator inference output (validation_chkpt*_rank*.zip, zarr v3 ZipStore)
into NetCDF files, one file per (stream, sample).

Store layout (weathergen.common.io):  <sample>/<stream>/<fstep>/{prediction,target,source}/
    data (npoints, nchannels[, ens]), coords (npoints, 2 = lat, lon), times (npoints,)
    attrs: channels, geoinfo_channels, source_interval {start, end}

Output file  <out_dir>/<run_id>_<stream>_s<sample>.nc :
    dims   : forecast_step, [ens], ncells          (default, unstructured points)
             forecast_step, [ens], y, x            (with --grid-zarr, native 2D grid)
    coords : forecast_step, valid_time, lead_time_hours, forecast_reference_time, lat, lon
    vars   : <channel>          = prediction
             <channel>_target   = target (ground truth)

Run from the WeatherGenerator repo root, with its venv:
    uv run --offline python wg_zip2netcdf.py --run-id m1j5o7bp \
        --stream CGLORS-OCEAN --channels thetao_0.5 so_0.5 zos \
        --grid-zarr /e/data1/slmet/ml_training/nemo_cglorsv8_subset_1993_2023.zarr
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import numpy as np

log = logging.getLogger("wg_zip2netcdf")

DEFAULT_RESULTS = "/e/scratch/weatherai/shared_work/results"


# --------------------------------------------------------------------------------------
# Pure-numpy helpers (point matching across forecast steps and onto the native grid)
# --------------------------------------------------------------------------------------
def coord_keys(lat: np.ndarray, lon: np.ndarray, decimals: int = 4) -> np.ndarray:
    """Integer key per (lat, lon), robust to float noise. Applies the same normalisation as
    the GREP reader (lat clipped to [-90, 90], lon wrapped to [-180, 180))."""
    lat = np.clip(np.asarray(lat, dtype=np.float64), -90.0, 90.0)
    lon = (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0 - 180.0
    s = 10.0**decimals
    ilat = np.rint((lat + 90.0) * s).astype(np.int64)
    ilon = np.rint((lon + 180.0) * s).astype(np.int64)
    return ilat * np.int64(round(361 * s)) + ilon


def build_point_index(coords_per_step: list[np.ndarray], tol_km: float = 2.0):
    """
    Reference point list = points of the first step; points of the other steps are matched
    to it by nearest neighbour (tolerant to float noise and to a different point order).
    Returns (lat, lon, [index_into_reference for each step, -1 if unmatched]).
    """
    ref = coords_per_step[0]
    lat, lon = ref[:, 0].astype(np.float32), ref[:, 1].astype(np.float32)
    idx = [np.arange(len(ref))] + [match_points(lat, lon, c, tol_km) for c in coords_per_step[1:]]
    return lat, lon, idx


def match_points(ref_lat, ref_lon, coords: np.ndarray, tol_km: float = 2.0) -> np.ndarray:
    """Index of the nearest reference point for each row of coords (-1 if > tol_km)."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    tree = cKDTree(_unit_xyz(ref_lat, ref_lon))
    dist, pos = tree.query(_unit_xyz(coords[:, 0], coords[:, 1]), distance_upper_bound=tol_km / 6371.0)
    return np.where(np.isfinite(dist), pos, -1)


def _unit_xyz(lat, lon) -> np.ndarray:
    la, lo = np.deg2rad(np.asarray(lat, np.float64)), np.deg2rad(np.asarray(lon, np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def grid_mapping(pt_lat, pt_lon, grid_lat, grid_lon, tol_km: float = 2.0):
    """
    For each cell of a 2D grid, index of the nearest output point (or -1 if farther than
    tol_km). Nearest-neighbour instead of exact key matching: the coordinates written by the
    model carry float noise (~1e-5 deg), which made exact matching drop random points.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415

    tree = cKDTree(_unit_xyz(pt_lat, pt_lon))
    g = _unit_xyz(grid_lat.ravel(), grid_lon.ravel())
    valid = np.isfinite(g).all(axis=1)
    out = np.full(len(g), -1, dtype=np.int64)
    dist, pos = tree.query(g[valid], distance_upper_bound=tol_km / 6371.0)
    found = np.isfinite(dist)
    out[np.flatnonzero(valid)[found]] = pos[found]
    return out


def repeated_coords(lat, lon, max_repeat: int = 2):
    """Mask of points whose (lat, lon) appears more than max_repeat times, plus the most
    frequent such coordinate. A physical grid never has more than 2 cells at the same place
    (tripolar fold / periodic column), so heavier repeats are fill values."""
    shape = np.shape(lat)
    lat, lon = np.ravel(lat), np.ravel(lon)
    finite = np.isfinite(lat) & np.isfinite(lon)
    keys = np.where(finite, coord_keys(np.nan_to_num(lat), np.nan_to_num(lon)), -1)
    _, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    n_same = cnt[np.ravel(inv)]
    rep = (n_same > max_repeat) & finite
    top = None
    if rep.any():
        i = np.flatnonzero(rep)[np.argmax(n_same[rep])]
        top = (float(lat[i]), float(lon[i]), int(n_same[i]))
    return rep.reshape(shape), top


def dead_cells(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """NEMO land-only MPI subdomains are removed from the run and their nav_lat/nav_lon are
    filled with a constant (0/0 in most files). Mask: lat=lon=0, non-finite, or any coordinate
    shared by more than 2 cells."""
    rep, top = repeated_coords(lat, lon)
    if top is not None:
        log.info(f"grid: {np.count_nonzero(rep)} cells share repeated coordinates, "
                 f"most frequent = ({top[0]:.4f}, {top[1]:.4f}) x{top[2]}")
    return rep | ((lat == 0.0) & (lon == 0.0)) | ~np.isfinite(lat) | ~np.isfinite(lon)


def as_3d(arr: np.ndarray) -> np.ndarray:
    """(npoints, nchannels[, ens]) -> (npoints, nchannels, ens)."""
    arr = np.asarray(arr)
    return arr[..., None] if arr.ndim == 2 else arr


def to_dt64(v) -> np.datetime64:
    return np.datetime64(str(v), "ns") if v is not None else np.datetime64("NaT", "ns")


# --------------------------------------------------------------------------------------
# Reading the zip store
# --------------------------------------------------------------------------------------
class Store:
    """Thin wrapper: official weathergen reader if importable, plain zarr otherwise."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        self._wg = None
        try:
            from weathergen.common.io import zarrio_reader  # noqa: PLC0415

            self._wg = zarrio_reader(path).__enter__()
            self.samples = [str(s) for s in self._wg.samples]
            self.streams = list(self._wg.streams)
            log.info("Opened with weathergen.common.io")
        except ImportError:
            import zarr  # noqa: PLC0415
            from zarr.storage import ZipStore  # noqa: PLC0415

            self._root = zarr.open_group(ZipStore(path, mode="r", read_only=True), mode="r")
            self.samples = sorted(self._root.group_keys(), key=int)
            first = self._root[self.samples[0]]
            self.streams = list(first.group_keys())
            log.info("Opened with plain zarr (weathergen not importable)")

    def fsteps(self, sample: str, stream: str) -> list[int]:
        grp = (self._wg.data_root if self._wg else self._root)[f"{sample}/{stream}"]
        return sorted(int(k) for k in grp.group_keys())

    def dataset(self, sample: str, stream: str, fstep: int, kind: str):
        """Return dict(data, coords, times, channels, ref_time) or None."""
        root = self._wg.data_root if self._wg else self._root
        path = f"{sample}/{stream}/{fstep}"
        try:
            grp = root[path]
        except KeyError:
            return None
        if kind not in grp.group_keys():
            return None
        g = grp[kind]
        attrs = dict(g.attrs)
        si = attrs.get("source_interval", {}) or {}
        return {
            "data": as_3d(g["data"][:]),
            "coords": np.asarray(g["coords"][:], dtype=np.float32),
            "times": np.asarray(g["times"][:]).astype("datetime64[ns]"),
            "channels": list(attrs["channels"]),
            "ref_time": to_dt64(si.get("end")),
        }

    def close(self):
        if self._wg is not None:
            self._wg.__exit__(None, None, None)


# --------------------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------------------
def load_grid(grid_zarr: pathlib.Path):
    import xarray as xr  # noqa: PLC0415

    kw = {"zarr_format": 3} if (grid_zarr / "zarr.json").exists() else {"zarr_format": 2}
    ds = xr.open_zarr(grid_zarr, consolidated=True, chunks=None, **kw)
    for la, lo in (("nav_lat", "nav_lon"), ("TLAT", "TLON"), ("latitude", "longitude")):
        if la in ds.coords or la in ds.variables:
            lat, lon = ds[la].values, ds[lo].values
            if lat.ndim == 1:  # regular grid
                lon, lat = np.meshgrid(lon, lat)
            lat, lon = lat.astype(np.float32), lon.astype(np.float32)
            dead = dead_cells(lat, lon)
            lat[dead], lon[dead] = np.nan, np.nan
            log.info(f"Native grid from {grid_zarr.name}: {la}/{lo} {lat.shape}, "
                     f"{np.count_nonzero(dead)} dead cells masked")
            return lat, lon
    raise KeyError(f"No nav_lat/nav_lon, TLAT/TLON or latitude/longitude in {grid_zarr}")


def convert_one(store, sample, stream, kinds, channels_sel, fsteps_sel, grid, out_path):
    import xarray as xr  # noqa: PLC0415

    fsteps = store.fsteps(sample, stream)
    if fsteps_sel:
        fsteps = [f for f in fsteps if f in fsteps_sel]

    # collect datasets per step (skip fstep 0 if it holds no target/prediction)
    per_step = {}
    for f in fsteps:
        dsets = {k: store.dataset(sample, stream, f, k) for k in kinds}
        dsets = {k: v for k, v in dsets.items() if v is not None}
        if dsets:
            per_step[f] = dsets
    if not per_step:
        log.warning(f"sample {sample} / {stream}: no {kinds} data, skipped")
        return False
    steps = sorted(per_step)

    # common point index (coords of the first available kind at each step)
    ref_kind = next(k for k in kinds if all(k in per_step[f] for f in steps))
    coords = [per_step[f][ref_kind]["coords"] for f in steps]
    lat, lon, idx = build_point_index(coords)
    npts = len(lat)
    rep, top = repeated_coords(lat, lon)
    if top is not None:
        log.warning(
            f"{stream}: {np.count_nonzero(rep)} output points share repeated coordinates "
            f"(most frequent ({top[0]:.4f}, {top[1]:.4f}) x{top[2]}): fake points from removed "
            "NEMO land subdomains reach the model. They are not placed on the grid."
        )

    channels = per_step[steps[0]][ref_kind]["channels"]
    chan_idx = [i for i, c in enumerate(channels) if not channels_sel or c in channels_sel]
    if not chan_idx:
        log.warning(f"{stream}: none of {channels_sel} in {channels}, skipped")
        return False

    if grid is not None:
        gmap = grid_mapping(lat, lon, *grid)
        ny, nx = grid[0].shape
        n_used = len(np.unique(gmap[gmap >= 0]))
        log.info(
            f"{stream}: {np.count_nonzero(gmap >= 0)} grid cells filled, "
            f"{n_used}/{npts} output points placed on the grid"
        )
        spatial_dims = ("y", "x")
    else:
        spatial_dims = ("ncells",)

    valid_time = np.array(
        [per_step[f][ref_kind]["times"].max() for f in steps], dtype="datetime64[ns]"
    )
    ref_time = per_step[steps[0]][ref_kind]["ref_time"]
    lead_h = ((valid_time - ref_time) / np.timedelta64(1, "h")).astype(np.float32)

    data_vars = {}
    for kind in kinds:
        if not all(kind in per_step[f] for f in steps):
            log.warning(f"{stream}: '{kind}' missing at some steps, not written")
            continue
        ens = per_step[steps[0]][kind]["data"].shape[2]
        # index of each point of this kind into the common point list (own coords per kind)
        kidx = []
        for s, f in enumerate(steps):
            c = per_step[f][kind]["coords"]
            if kind == ref_kind:
                kidx.append(idx[s])
                if (idx[s] < 0).any():
                    log.warning(f"{stream} step {f}: {np.count_nonzero(idx[s] < 0)} points unmatched")
                continue
            m = match_points(lat, lon, c)
            if (m < 0).any():
                log.warning(f"{stream} {kind} step {f}: {np.count_nonzero(m < 0)} points unmatched")
            kidx.append(m)
        for ci in chan_idx:
            ch = channels[ci]
            buf = np.full((len(steps), ens, npts), np.nan, dtype=np.float32)
            for s, f in enumerate(steps):
                d = per_step[f][kind]["data"]  # (npoints, nch, ens)
                ok = kidx[s] >= 0
                buf[s][:, kidx[s][ok]] = d[ok, ci, :].T.astype(np.float32)
            if grid is not None:
                g = np.full((len(steps), ens, ny * nx), np.nan, dtype=np.float32)
                ok = gmap >= 0
                g[:, :, ok] = buf[:, :, gmap[ok]]
                buf = g.reshape(len(steps), ens, ny, nx)
            dims = ("forecast_step", "ens", *spatial_dims)
            if ens == 1:
                buf, dims = buf[:, 0], ("forecast_step", *spatial_dims)
            name = ch if kind == "prediction" else f"{ch}_{kind}"
            data_vars[name] = (dims, buf, {"long_name": f"{ch} ({kind})", "wg_kind": kind})

    coords_out = {
        "forecast_step": ("forecast_step", np.array(steps, dtype=np.int32)),
        "valid_time": ("forecast_step", valid_time),
        "lead_time_hours": ("forecast_step", lead_h, {"units": "hours"}),
        "forecast_reference_time": ((), ref_time),
    }
    if grid is not None:
        coords_out["lat"] = (("y", "x"), grid[0], {"units": "degrees_north"})
        coords_out["lon"] = (("y", "x"), grid[1], {"units": "degrees_east"})
    else:
        coords_out["lat"] = ("ncells", lat, {"units": "degrees_north"})
        coords_out["lon"] = ("ncells", lon, {"units": "degrees_east"})

    ds = xr.Dataset(data_vars, coords=coords_out)
    ds.attrs.update(
        {"source": str(store.path), "stream": stream, "sample": int(sample),
         "note": "<var> = prediction, <var>_target = target"}
    )
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in data_vars}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path, encoding=enc)
    log.info(f"wrote {out_path}  ({len(steps)} steps, {len(chan_idx)} channels)")
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-id", help="inference run id (looks in --results-dir/<run-id>/)")
    src.add_argument("--zip", type=pathlib.Path, help="explicit path to the .zip store")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS)
    p.add_argument("--out-dir", type=pathlib.Path, help="default: <results>/<run-id>/netcdf")
    p.add_argument("--stream", nargs="*", help="streams to export (default: all)")
    p.add_argument("--channels", nargs="*", help="channels to export (default: all)")
    p.add_argument("--samples", nargs="*", help="sample ids (default: all)")
    p.add_argument("--fsteps", nargs="*", type=int, help="forecast steps (default: all)")
    p.add_argument("--type", nargs="+", default=["prediction", "target"],
                   choices=["prediction", "target", "source"])
    p.add_argument("--grid-zarr", type=pathlib.Path,
                   help="input zarr of the stream (e.g. nemo_cglorsv8...zarr) to rebuild 2D (y, x) maps")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if a.zip:
        zips = [a.zip]
        run_id = a.zip.parent.name
    else:
        run_id = a.run_id
        zips = sorted(pathlib.Path(a.results_dir, run_id).glob("*.zip"))
    if not zips:
        sys.exit(f"No .zip store found for {a.run_id or a.zip}")
    out_dir = a.out_dir or pathlib.Path(a.results_dir, run_id, "netcdf")
    grid = load_grid(a.grid_zarr) if a.grid_zarr else None

    n = 0
    for zp in zips:
        store = Store(zp)
        tag = zp.stem.replace("validation_", "")
        streams = [s for s in store.streams if s != "latent" and (not a.stream or s in a.stream)]
        samples = [s for s in store.samples if not a.samples or s in a.samples]
        log.info(f"{zp.name}: {len(samples)} samples, streams {streams}")
        for stream in streams:
            for sample in samples:
                out = out_dir / f"{run_id}_{tag}_{stream}_s{int(sample):04d}.nc"
                n += convert_one(store, sample, stream, a.type, a.channels, a.fsteps, grid, out)
        store.close()
    log.info(f"done: {n} file(s) in {out_dir}")


if __name__ == "__main__":
    main()
