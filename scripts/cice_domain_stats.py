#!/usr/bin/env python3
"""
Compute normalisation statistics (mean, std) of selected variables of a curvilinear zarr
(CICE / NEMO, GREP-reader layout) on a lat/lon sub-domain and a time period.
Nothing is written to the zarr: results are printed and saved as YAML/JSON, ready to be
pasted into the stream config.

Points are selected like the WeatherGenerator GREP reader does:
  - longitudes wrapped to [-180, 180)
  - filler coordinates of removed NEMO land subdomains dropped (lat=lon=-1, lat=lon=0, NaN)
  - land / missing values dropped (NaN, |x| > 1e20)
  - zeros KEPT (stream uses zero_as_nan: False)

Example (cassandra, zarr>=3 environment):
  python cice_domain_stats.py /data/inputs/WEATHERGENERATOR/cice_cglorsv8_1993_2023.zarr \
      --vars aice hi hs strength sig1 sig2 uvel vvel \
      --lat-min 45 --period 1994-01-01 2022-12-31 --stride 5 --out cice_stats_45N
"""

import argparse
import json
import time

import dask
import numpy as np
import xarray as xr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zarr")
    ap.add_argument("--vars", nargs="+", required=True)
    ap.add_argument("--lat-min", type=float, default=-90.0)
    ap.add_argument("--lat-max", type=float, default=90.0)
    ap.add_argument("--lon-min", type=float, default=-180.0)
    ap.add_argument("--lon-max", type=float, default=180.0,
                    help="lon_min > lon_max = box crossing the dateline")
    ap.add_argument("--lat", default="TLAT", help="latitude variable (TLAT, nav_lat, ...)")
    ap.add_argument("--lon", default="TLON", help="longitude variable (TLON, nav_lon, ...)")
    ap.add_argument("--period", nargs=2, default=["1994-01-01", "2022-12-31"],
                    help="time period (default: training period of earthgen3)")
    ap.add_argument("--stride", type=int, default=5, help="use one time step out of N")
    ap.add_argument("--out", default="domain_stats", help="output prefix (.yml and .json)")
    a = ap.parse_args()

    ds = xr.open_zarr(a.zarr, consolidated=None)
    missing = [v for v in a.vars if v not in ds.data_vars]
    if missing:
        raise SystemExit(f"variables not found: {missing}")
    tname = next(c for c in ("time", "time_centered", "time_counter") if c in ds.coords)

    # ---- spatial selection (same rules as the GREP reader + `domain` option) ------------
    lat = ds[a.lat].values.astype(np.float64)
    lon = ds[a.lon].values.astype(np.float64)
    filler = ((lat == -1) & (lon == -1)) | ((lat == 0) & (lon == 0)) | ~np.isfinite(lat) | ~np.isfinite(lon)
    lat = np.clip(lat, -90, 90)
    lon = (lon + 180.0) % 360.0 - 180.0
    inside = (lat >= a.lat_min) & (lat <= a.lat_max)
    if a.lon_min <= a.lon_max:
        inside &= (lon >= a.lon_min) & (lon <= a.lon_max)
    else:
        inside &= (lon >= a.lon_min) | (lon <= a.lon_max)
    keep = inside & ~filler

    ydim, xdim = ds[a.lat].dims
    rows = np.flatnonzero(keep.any(axis=1))
    cols = np.flatnonzero(keep.any(axis=0))
    j0, j1, i0, i1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    keep_da = xr.DataArray(keep[j0:j1, i0:i1], dims=(ydim, xdim))
    print(f"domain lat [{a.lat_min}, {a.lat_max}] lon [{a.lon_min}, {a.lon_max}]: "
          f"{keep.sum()} grid cells (bbox {ydim} {j0}:{j1}, {xdim} {i0}:{i1})")

    # ---- temporal selection ---------------------------------------------------------------
    sub = ds[a.vars].isel({ydim: slice(j0, j1), xdim: slice(i0, i1)})
    sub = sub.sel({tname: slice(*a.period)}).isel({tname: slice(None, None, a.stride)})
    nt = sub.sizes[tname]
    print(f"period {a.period[0]} -> {a.period[1]}, stride {a.stride}: {nt} time steps")
    for v in a.vars:
        ch = ds[v].encoding.get("chunks")
        if ch and ch[0] > 1 and a.stride > 1:
            print(f"  note: {v} chunked {ch} along time -> stride does not reduce reads much")
            break

    # ---- one pass over the data for all variables ----------------------------------------
    lazy = []
    for v in a.vars:
        x = sub[v].astype("float64").where(keep_da)
        x = x.where(np.abs(x) < 1e20)
        lazy += [x.mean(), x.std(), x.count(), (x == 0).sum(), x.min(), x.max()]
    t0 = time.time()
    res = dask.compute(*lazy)
    print(f"computed in {(time.time() - t0) / 60:.1f} min\n")

    stats = {}
    print(f"{'var':10s} {'mean':>12s} {'std':>12s} {'min':>11s} {'max':>11s} {'zeros%':>7s}  "
          f"{'global mean/std in zarr':>26s}")
    for i, v in enumerate(a.vars):
        m, s, n, nz, mn, mx = (float(r) for r in res[6 * i: 6 * i + 6])
        stats[v] = [m, s]
        gm = float(ds[f"{v}_mean"].values) if f"{v}_mean" in ds else None
        gs = float(ds[f"{v}_std"].values) if f"{v}_std" in ds else None
        g = f"{gm:.4g} / {gs:.4g}" if gm is not None and gs is not None else "-"
        print(f"{v:10s} {m:12.5g} {s:12.5g} {mn:11.4g} {mx:11.4g} {100 * nz / max(n, 1):7.1f}  {g:>26s}")
        if s <= 1e-5:
            print(f"  WARNING: std of {v} ~ 0 -> the reader would replace it by 1")

    meta = {"zarr": a.zarr, "lat": [a.lat_min, a.lat_max], "lon": [a.lon_min, a.lon_max],
            "period": a.period, "stride": a.stride, "n_time": nt, "n_cells": int(keep.sum())}
    with open(f"{a.out}.json", "w") as f:
        json.dump({"meta": meta, "stats": stats}, f, indent=2)
    with open(f"{a.out}.yml", "w") as f:
        f.write(f"# mean/std on lat [{a.lat_min}, {a.lat_max}], lon [{a.lon_min}, {a.lon_max}], "
                f"{a.period[0]}..{a.period[1]} (1/{a.stride} steps, {nt} steps)\n")
        f.write("stats_override :\n")
        for v, (m, s) in stats.items():
            f.write(f"  {v:10s}: [{m:.6g}, {s:.6g}]\n")
    print(f"\nwritten {a.out}.yml and {a.out}.json")


if __name__ == "__main__":
    main()
