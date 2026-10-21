#!/usr/bin/env python3
"""
Check that a streams directory is compatible with a trained run before fine-tuning:
every stream of the run json must exist in the directory with the same structural settings
(channels, token_size, embed/decoder sizes, stream_id, ...), otherwise its weights do not load
or load onto the wrong channels. New streams (e.g. CICE) are listed separately.

  uv run --offline python scripts/check_streams.py --run-id d5xirkmf \
      --streams-dir ../WeatherGenerator-configs/config/streams/earthgen3_cice/
"""

import argparse
import json
import pathlib
import sys

import yaml

# keys that do not change the network / data mapping (allowed to differ)
IGNORED = {"filenames", "loss_weight", "location_weight", "masking_rate", "masking_rate_none",
           "max_num_targets", "comment"}


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--streams-dir", required=True, type=pathlib.Path)
    a = ap.parse_args()

    from weathergen.common.config import get_path_model  # noqa: PLC0415

    cfg = json.load(open(get_path_model(run_id=a.run_id) / f"model_{a.run_id}_latest.json"))
    run_streams = cfg["streams"]
    if isinstance(run_streams, list):
        run_streams = {s["name"]: s for s in run_streams}

    new = {}
    for f in sorted(a.streams_dir.glob("*.y*ml")):
        for name, s in (yaml.safe_load(open(f)) or {}).items():
            new[name] = (f.name, s)

    ok = True
    ids = {}
    for name, rs in run_streams.items():
        if name not in new:
            print(f"MISSING  {name}: in run {a.run_id} but not in {a.streams_dir}")
            ok = False
            continue
        fname, ns = new[name]
        fr, fn = flatten(rs), flatten(ns)
        diffs = [k for k in fn if k.split(".")[0] not in IGNORED and k in fr and fr[k] != fn[k]]
        only_new = [k for k in fn if k.split(".")[0] not in IGNORED and k not in fr]
        if diffs or only_new:
            ok = False
            print(f"DIFFERENT {name} ({fname}):")
            for k in diffs:
                print(f"     {k}: run={fr[k]!r}  dir={fn[k]!r}")
            for k in only_new:
                print(f"     {k}: not in run, dir={fn[k]!r}")
        else:
            print(f"OK       {name} ({fname})")
    for name, (fname, ns) in new.items():
        sid = ns.get("stream_id")
        if sid in ids:
            print(f"ERROR    stream_id {sid} used by {ids[sid]} and {name}")
            ok = False
        ids[sid] = name
        if name not in run_streams:
            print(f"NEW      {name} ({fname}): stream_id {sid}, "
                  f"{len(ns.get('source') or [])} source / {len(ns.get('target') or [])} target channels "
                  "-> new embedder/decoder, randomly initialised")
    print("\n=> compatible" if ok else "\n=> NOT compatible: fix the files above before fine-tuning")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
