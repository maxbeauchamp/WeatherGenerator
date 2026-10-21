#!/usr/bin/env python3
"""
Compare the predictions of two WeatherGenerator inference runs (zip outputs), sample by sample,
stream by stream, forecast step by forecast step. Points are matched by coordinates.

Main use: non-regression of forcing re-injection. Run the SAME model on the SAME dates
(test_config.shuffle=False) without and with `forcing_reinjection.enabled=True`: with the
zero-initialised injection the predictions must be identical (max |diff| = 0, or ~1e-6 with
non-deterministic GPU kernels).

  uv run --offline python scripts/compare_inference_zips.py \
      /e/scratch/weatherai/shared_work/results/inf_noreinj/validation_chkpt00000_rank0000.zip \
      /e/scratch/weatherai/shared_work/results/inf_reinj0/validation_chkpt00000_rank0000.zip
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import wg_zip2netcdf as W  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_a", type=pathlib.Path)
    ap.add_argument("zip_b", type=pathlib.Path)
    ap.add_argument("--streams", nargs="*")
    ap.add_argument("--samples", nargs="*")
    ap.add_argument("--tol", type=float, default=1e-5, help="max |diff| considered identical")
    a = ap.parse_args()

    A, B = W.Store(a.zip_a), W.Store(a.zip_b)
    streams = [s for s in A.streams if s in B.streams and s != "latent"
               and (not a.streams or s in a.streams)]
    samples = [s for s in A.samples if s in B.samples and (not a.samples or s in a.samples)]
    print(f"{len(samples)} common samples, streams {streams}")

    worst = 0.0
    for stream in streams:
        for sample in samples:
            for f in A.fsteps(sample, stream):
                pa = A.dataset(sample, stream, f, "prediction")
                pb = B.dataset(sample, stream, f, "prediction")
                if pa is None or pb is None:
                    continue
                if pa["ref_time"] != pb["ref_time"]:
                    sys.exit(f"sample {sample}: different start dates ({pa['ref_time']} vs "
                             f"{pb['ref_time']}): rerun both with test_config.shuffle=False")
                j = W.match_points(pa["coords"][:, 0], pa["coords"][:, 1], pb["coords"])
                ok = j >= 0
                db = np.full_like(pa["data"], np.nan)
                db[j[ok]] = pb["data"][ok]
                d = np.abs(pa["data"] - db)
                m = float(np.nanmax(d)) if np.isfinite(d).any() else np.nan
                worst = max(worst, m if np.isfinite(m) else 0.0)
                rel = m / (float(np.nanstd(pa["data"])) + 1e-12)
                flag = "OK " if m <= a.tol else "DIFF"
                print(f"{flag} {stream:24s} s{int(sample):04d} f{f:02d}  max|diff| {m:.3e}  "
                      f"(rel {rel:.1e})  unmatched {np.count_nonzero(~ok)}")
    print(f"\n=> worst max|diff| = {worst:.3e}: "
          + ("IDENTICAL" if worst <= a.tol else "DIFFERENT (check the run options)"))
    sys.exit(0 if worst <= a.tol else 1)


if __name__ == "__main__":
    main()
