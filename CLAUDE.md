# CLAUDE.md — sea-ice work on WeatherGenerator (branch `beauchamp1/dev/seaice`)

Context for Claude Code sessions. The owner (Maxime Beauchamp) writes in French; code, comments,
docs and figure labels are in **English**.

## Project in one paragraph

Adapt the CMCC WeatherGenerator ocean model `d5xirkmf` (earthgen3) to sea ice (CICE coupled to
C-GLORS v8), on JUPITER (JSC). One doc per experiment; the code switches are all **off by
default** (no module created, original code path when off):

| Doc | Switch | Main files |
|---|---|---|
| `xp_01_ice_era5.md` CICE stream + ice + ERA5 base | stream file `cice.yml` | `packages/readers_extra/.../data_reader_grep.py` (`zero_as_nan`, `domain`, `stats_override`) |
| `xp_02_d5xirkmf_cice.md` CICE in `d5xirkmf` | `streams/earthgen3_cice/` | configs only |
| `xp_03_forcing_reinjection.md` | `forcing_reinjection` | `model/forcing_injection.py`, `model/model.py` (`forward`, `inject_forcing`), `datasets/multi_stream_data_sampler.py` (`_build_forcing_samples`), `datasets/batch.py` |
| `xp_04_flow_matching.md` | `flow_matching` | `model/flow_matching.py`, `model/model.py` (`predict_flow_matching`), sampler `_fm_values`, `datasets/stream_data.py`, `train/trainer.py` |
| `xp_05_decoder_partition_of_unity.md` | `partition_of_unity` | `model/partition_of_unity.py`, `model/model.py` (`predict_decoders`), `datasets/tokenizer_masking.py`, `datasets/stream_data.py` |

**Docs live in `../WeatherGenerator-configs/docs/`:**
- `seaice/README.md` is the index;
- each `seaice/xp_0*.md` follows the same plan: goal, method (maths, figures), implementation (files touched), configs, running, status/decisions/open questions, where the outputs are;
- `seaice/runbook.md` holds the shared commands and troubleshooting;
- `handoff.md` holds the current state, the preferences and the next actions.

Read the relevant `xp_0*.md` before changing a feature (see *Documentation workflow* below).

## Documentation workflow (mandatory)

Every experiment has its own file `../WeatherGenerator-configs/docs/seaice/xp_NN_<name>.md`.
**Whenever an experiment is created or changed, create or update its xp file in the same piece of
work, before committing.** Do not leave a feature, config or run undocumented.

- **New experiment** (new feature, new training chain, new config family):
  - create `xp_NN_<name>.md`, with NN the next free number;
  - follow the plan of the existing files: 1. Goal, 2. Method (text, maths, figures),
    3. Implementation (files touched, level, why), 4. Configs (path, settings),
    5. Running (train / inference / eval commands), 6. Status, decisions, open questions,
    7. Where the outputs are (one row per run: checkpoints, logs, results / evaluation);
  - add it to the tables and the dependency graph of `seaice/README.md`, and its tags to the file map.
- **Change to an existing experiment** (code, config, new run, result, decision): update the
  matching sections of its xp file:
  - the files table (§3);
  - the configs (§4);
  - the commands (§5);
  - the status, decisions and open questions (§6);
  - **the outputs table (§7): add a row for every new run (checkpoints, logs, results, eval)**;
  - the run table of `seaice/README.md`.
- **Figures**: `docs/seaice/figs/xp_NN_*.png`, produced by a script in `figs/src/`. Sketches and
  toy problems must be labelled as such, never presented as model output.
- **Shared commands or a new pitfall**: `seaice/runbook.md`.
- **End of every work step**: update `docs/handoff.md` ("Where we are" and "Next actions"; keep
  it under one page).

## Design rules (keep them)

- New modules that are added to a pretrained model are **zero-initialised so that the model is
  exactly unchanged at start**, and keep their parameters directly on the module (not an
  `nn.Linear`), so that `reset_parameters()` on checkpoint loading keeps the zeros.
  **Exception (owner's decision): the partition of unity (xp_05)** has no parameters but cannot
  be the identity at start (the identity is the discontinuous limit); it needs a decoder fine-tuning.
- Every feature has a non-regression check (`./run_cice_pipeline.sh reinj-check | fm-check |
  pu-check`): zero-init feature vs original on the same dates, compared with
  `scripts/compare_inference_zips.py`. For `pu-check`: off == original, on = report the change at start.
- Every feature has unit tests in `tests/test_<feature>.py`. Run them after any change:
  `uv run --offline pytest -q tests/test_forcing_reinjection.py tests/test_flow_matching.py tests/test_partition_of_unity.py tests/test_seam_diagnostic.py`
- Defaults in `config/default_config.yml` stay disabled.
- Keep diffs to upstream files small and localised (this branch must stay mergeable).

## Environment

- Real paths: `/e/project1/weatherai/beauchamp1/{WeatherGenerator,WeatherGenerator-configs}`.
  Runs: `/e/scratch/weatherai/shared_work/{models,results,logs}/<run_id>/`.
- **Never put backups or large files in `$HOME`** (quota). Backups go to `/e/project1/weatherai/beauchamp1/.backup_*`.
- JUPITER login nodes have internet (git, `uv sync`, `sbatch`); **compute nodes do not** (always `uv run --offline`).
- Booster: 4 GPUs per node, 72 CPU cores per GPU, 12 h max, account `e-ext-2025e01-128`.
  `srun` binds one GPU per task: `LOCAL_RANK=0`, `RANK=$SLURM_PROCID`, `WORLD_SIZE=$SLURM_NTASKS`
  (see `run()` in `slurm_submissions/*.sbatch`).
- sbatch jobs take **no code snapshot**: do not edit code/configs used by queued or running jobs.
- The team launcher `../WeatherGenerator-private/hpc/launch-slurm.py` is set aside (pitfalls: `docs/seaice/runbook.md` §8).

## Do not commit

- The temporary CICE path in `WeatherGenerator-configs/config/streams/{earthgen3_cice,ice_era5}/cice.yml`
  (`/e/scratch/weatherai/broccoli1/cice_cglorsv8_1993_2023_subset.zarr`). Revert with
  `git checkout -- earthgen3_cice/cice.yml ice_era5/cice.yml` before committing configs.
- `log_*.txt`, `config/evaluate/eval_inf_*.yml` (generated).

## Common commands

```bash
CF=../WeatherGenerator-configs/config; WD=/e/scratch/weatherai/shared_work
uv run --offline train_continue --from-run-id <run> --mini-epoch <N> --run-id <new> --config <cfg>
uv run --offline inference --from-run-id <run> --mini-epoch <N> --run-id inf_<x> --config <cfg>
./run_cice_pipeline.sh <step>            # check | ie-smoke | reinj-check | fm-check | pu-check | ...
bash slurm_submissions/submit_ice_era5.sh      # ice + ERA5 base (S1 -> S2)
bash slurm_submissions/submit_ice_era5_fm.sh   # FM chain on ice_era5_s2
bash slurm_submissions/submit_ice_era5_pu.sh   # partition of unity (+ FM) chain on ice_era5_s2
```

Always pass `--mini-epoch N` (`_latest` exists only every 250 batches).

## Workflow

Develop locally (VS Code + Claude Code; CPU tests in `.venv`, see `docs/seaice/runbook.md` §3) → `git push` → on JUPITER login node `git pull` → test on a
GPU node. Lint: `ruff check` / `ruff format` as configured in the repo.
