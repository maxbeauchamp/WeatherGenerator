#!/usr/bin/env python3
"""
Convert WeatherGenerator training metrics (results/<run_id>/<run_id>_train_metrics.json, one json
line per logging step) to TensorBoard event files.

  uv run --offline python scripts/metrics_to_tensorboard.py ice_era5_s1 ice_era5_s2 --chain
  uvx --with "setuptools<81" tensorboard --logdir /e/scratch/weatherai/shared_work/tensorboard \
      --host localhost --port 16006

Layout (default): one sub-run per stage, with the SAME tags,
  tensorboard/<run_id>/train/   tags CICE/mse.avg, loss_avg_mean, ...
  tensorboard/<run_id>/val/     same tags
so TensorBoard draws train and validation on the same chart (runs "<run_id>/train" and
"<run_id>/val"), and several run_ids on the same chart too. Use the run filter on the left to
choose which curves to overlay.

--separate  old layout: one run per run_id, tags prefixed by the stage (train/CICE/mse.avg).
--chain     puts the runs one after the other on the x-axis (S2 continues S1).
Re-running the script rewrites the event files (useful while a job is still training).
"""

import argparse
import json
import math
import pathlib
import shutil

WD = pathlib.Path("/e/scratch/weatherai/shared_work")
SKIP = {"weathergen.timestamp", "weathergen.time", "stage", "weathergen.step"}
STAGE_NAME = {"train": "train", "val": "val", "validation": "val", "valid": "val"}


def read_rows(path: pathlib.Path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # last line may be incomplete while the job is writing
    return rows


def tag_of(key: str, prefix: str = "") -> str:
    # LossPhysical.CICE.mse.avg -> [prefix/]CICE/mse.avg ; loss_avg_mean -> [prefix/]loss_avg_mean
    parts = key.split(".")
    t = f"{parts[1]}/{'.'.join(parts[2:])}" if parts[0] == "LossPhysical" and len(parts) > 2 else key
    return f"{prefix}/{t}" if prefix else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="+")
    ap.add_argument("--results", type=pathlib.Path, default=WD / "results")
    ap.add_argument("--out", type=pathlib.Path, default=WD / "tensorboard")
    ap.add_argument("--chain", action="store_true", help="continue the x-axis from run to run")
    ap.add_argument("--separate", action="store_true",
                    help="old layout: one run per run_id, stage as tag prefix (no train/val overlay)")
    a = ap.parse_args()

    from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

    offset = 0
    for rid in a.run_ids:
        f = a.results / rid / f"{rid}_train_metrics.json"
        if not f.exists():
            f = a.results / rid / "metrics.json"
        if not f.exists():
            print(f"{rid}: no metrics file in {a.results / rid}")
            continue
        rows = read_rows(f)
        out = a.out / rid
        shutil.rmtree(out, ignore_errors=True)
        writers = {}

        def writer(stage):
            key = "all" if a.separate else stage
            if key not in writers:
                writers[key] = SummaryWriter(log_dir=str(out if a.separate else out / stage))
            return writers[key]

        n_train = 0
        last_train_step = offset
        counts = {}
        for r in rows:
            raw = str(r.get("stage", "train"))
            stage = STAGE_NAME.get(raw.lower(), raw)
            if stage == "train":
                n_train += 1
                step = offset + int(r.get("weathergen.step", n_train))
                last_train_step = step
            else:  # validation rows: put them at the last training step
                step = last_train_step
            wall = r.get("weathergen.timestamp", 0) / 1000.0 or None
            w = writer(stage)
            for k, v in r.items():
                if k in SKIP or not isinstance(v, (int, float)) or not math.isfinite(v):
                    continue
                w.add_scalar(tag_of(k, stage if a.separate else ""), float(v), step, walltime=wall)
            counts[stage] = counts.get(stage, 0) + 1
        for w in writers.values():
            w.close()
        print(f"{rid}: {counts} -> {out}  (steps {offset}..{last_train_step})")
        if a.chain:
            offset = last_train_step
    print(f'\nview: uvx --with "setuptools<81" tensorboard --logdir {a.out} --host localhost --port 16006')


if __name__ == "__main__":
    main()
