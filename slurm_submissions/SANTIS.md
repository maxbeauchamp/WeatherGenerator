# Sea-ice experiments on Santis (CSCS)

Ice + ERA5 training, then flow-matching (FM) fine-tuning of the CICE decoder, with inference and
evaluation, from the `beauchamp1/dev/seaice` branch. Same scripts as on JUPITER, with `SITE=santis`
(machine defaults in `slurm_submissions/site.sh`).

**Install and setup** (SSH, `cscs-key`, `uv`, `gh`, repos, `scripts/actions.sh sync` /
`create-links`, `agpu*` / `sq` aliases): follow the hackathon page
<https://gitlab.jsc.fz-juelich.de/hedgedoc/zG639KpwTGm4FWVeXvH4lg?view>.

## 0. Once

From the `WeatherGenerator` directory:
```bash
git -C ../WeatherGenerator-private pull   # course paths
scripts/actions.sh sync
```

Streams: create the Santis copy of `streams/ice_era5/` (do not commit it), with the Santis CICE path:
```bash
CFG=../WeatherGenerator-configs/config
cp -r $CFG/streams/ice_era5 $CFG/streams/ice_era5_santis
sed -i "s#/e/scratch/weatherai/broccoli1/#/capstor/store/cscs/userlab/ch17/data/#" $CFG/streams/ice_era5_santis/cice.yml
```
`site.sh` passes `streams_directory=.../streams/ice_era5_santis/` to every train / inference step.

Check that the ERA5 zarr used for training is there (relative name in `era5.yml`, resolved by
the private config):
```bash
ls -d /capstor/store/cscs/userlab/ch17/data/aifs-ea-an-oper-0001-mars-o96-1979-2024-1h-v3-with-era51.zarr
```
> `ERA5-1deg-6h-mean-forcing-1978-2024.zarr` is another dataset (1°, 6-hourly means): not the
> ERA5 stream the models were trained with.

## 1. Smoke test (interactive, ~20 min)

```bash
agpuN
cd ~/WeatherGenerator; CFG=../WeatherGenerator-configs/config
uv run --offline train --run-id ice_era5_smoke --config $CFG/train_ice_era5_scratch.yml \
  --options streams_directory=$CFG/streams/ice_era5_santis/ \
            training_config.num_mini_epochs=1 training_config.samples_per_mini_epoch=32 \
            validation_config.samples_per_mini_epoch=16 validation_config.validate_before_training=false
```
Check: the CICE and ERA5 streams load, the loss is finite and decreases.

## 2. Pipeline (login node, chained Slurm jobs, 1 node x 4 GPUs each)

```bash
# everything: S1 -> S2, then base inference | FM1 -> FM2 -> FM inference, then evaluation
SITE=santis FM=1 bash slurm_submissions/submit_ice_era5.sh

# only the ice + ERA5 base (S1 -> S2)
SITE=santis bash slurm_submissions/submit_ice_era5.sh

# only the FM experiment, on an existing base (default ice_era5_s2, chkpt 6)
SITE=santis bash slurm_submissions/submit_ice_era5_fm.sh
sq
```

| Job | What | Config | Run id |
|---|---|---|---|
| `ice_era5_s1` | from scratch, 2-day rollouts | `train_ice_era5_scratch.yml` | `ice_era5_s1` |
| `ice_era5_s2` | 7-day rollouts, from the final S1 | `train_ice_era5_long.yml` | `ice_era5_s2` |
| `ice_era5_binfer` | base inference, 32 dates of 2023 | `inference_ice_era5.yml` | `inf_ice_era5_base` |
| `ice_era5_fm1` | backbone frozen: CICE decoder + FM conditioner | `finetune_cice_fm_phase1.yml` | `ice_era5_fm1` |
| `ice_era5_fm2` | all trainable, lr 1e-5 | `finetune_cice_fm_phase2.yml` | `ice_era5_fm2` |
| `ice_era5_finfer` | FM inference (8 members x 16 Heun steps), same dates | `inference_cice_fm.yml` | `inf_ice_era5_fm` |
| `ice_era5_eval` | FastEval: FM ensemble vs base, with persistence | `evaluate/eval_ice_era5.yml` (template) | — |

**Options** (environment variables):
- **Epochs:** `S1_EPOCHS` (16), `S2_EPOCHS` (8), `FM1_EPOCHS` (7), `FM2_EPOCHS` (6).
- **Base of the FM:** `BASE_ID`, `BASE_EPOCH` (default `ice_era5_s2`, 6; set automatically with `FM=1`).
- **Run ids:** `S1_ID`, `S2_ID`, `FM1_ID`, `FM2_ID`, `BINF_ID`, `FINF_ID`. The results directory is
  shared by the whole course: if you copy the JUPITER `ice_era5_s2` there, retrain with other ids
  (e.g. `S1_ID=ice_era5_s1_santis S2_ID=ice_era5_s2_santis`) so that it is not overwritten.
- **Start the FM from the JUPITER base** without retraining: copy `models/ice_era5_s2/` into
  `$WD/models/`, then `SITE=santis bash slurm_submissions/submit_ice_era5_fm.sh`.
- `ACCOUNT`, `PARTITION`, `WD`, `LOGDIR`, `EXTRA_OPTS` override the defaults of `site.sh`.

Do not edit the code or the configs while jobs are queued or running (no code snapshot).

## 2b. xp_06: surface radiation (ssrd, strd) on the base

Once, the Santis copy of the streams with radiation:
```bash
CFG=../WeatherGenerator-configs/config
cp -r $CFG/streams/ice_era5_rad $CFG/streams/ice_era5_rad_santis
sed -i "s#/e/scratch/weatherai/broccoli1/#/capstor/store/cscs/userlab/ch17/data/#" $CFG/streams/ice_era5_rad_santis/cice.yml
```
Then (base `ice_era5_s2` in `$WD/models/`, or `BASE_ID` / `BASE_EPOCH` of a Santis base):
```bash
SITE=santis bash slurm_submissions/submit_ice_era5_rad.sh
```
Chain: `binfer` → `rcheck` (zero-init = base, comparison printed in its log) | `r1` (new embedder
only) → `r2` (all, lr 2e-5) → `rinfer` | `eval` (radiation vs base). Logs in
`$WD/logs/ice_era5_rad/`, run ids `ice_era5_rad1`, `ice_era5_rad2`, `inf_ice_era5_rad`, scores and
plots in `$WD/results/inf_ice_era5_rad/plots/`. Details: `WeatherGenerator-configs/docs/seaice/xp_06_radiation_forcing.md`.

## 3. Where everything goes

`WD=/iopsstor/scratch/cscs/course_29fe58c1/shared_work`

| What | Where |
|---|---|
| Slurm logs (one per job) | `$WD/logs/ice_era5/<job>_<jobid>.out/.err` (S1, S2), `$WD/logs/ice_era5_fm/<job>_<jobid>.out/.err` (FM chain) |
| Training logs | `logs/<run id>/` in the repo |
| Checkpoints | `$WD/models/<run id>/` (`<run id>_chkpt0000k` = end of mini-epoch k) |
| Training / validation metrics | `$WD/results/<run id>/<run id>_train_metrics.json` |
| Inference outputs | `$WD/results/inf_ice_era5_base/`, `$WD/results/inf_ice_era5_fm/` (`validation_chkpt*_rank*.zip`) |
| Evaluation config (generated, do not commit) | `config/evaluate/eval_inf_ice_era5_fm_vs_base.yml` |
| Evaluation scores and plots | `$WD/results/inf_ice_era5_fm/plots/` (and `evaluation/` of each inference run) |

Maps and animations of any inference run (GPU or CPU node):
```bash
uv run --offline python scripts/plot_seaice.py $WD/results/inf_ice_era5_fm/validation_chkpt*_rank*.zip \
  --samples 0 1 --vars aice hi --days 1 4 7 --animate --timeseries \
  --label inf_ice_era5_fm --out-dir $WD/results/inf_ice_era5_fm/figs_seaice
```
To look at them from your Mac:
```bash
scp -r santis:/iopsstor/scratch/cscs/course_29fe58c1/shared_work/results/inf_ice_era5_fm/figs_seaice .
```
