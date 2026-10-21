#!/usr/bin/env bash
# Sea-ice pipeline on top of d5xirkmf: checks -> fine-tuning (2 phases) -> inference -> FastEval
# -> NetCDF. Run from the WeatherGenerator repo root (branch beauchamp1/dev/seaice), one step
# at a time:
#
#   ./run_cice_pipeline.sh check      # patches, data, stream compatibility
#   ./run_cice_pipeline.sh p1         # phase 1 fine-tuning (frozen backbone)
#   ./run_cice_pipeline.sh p2         # phase 2 fine-tuning (all trainable)
#   ./run_cice_pipeline.sh infer      # inference on 2023
#   ./run_cice_pipeline.sh eval       # FastEval (CICE arctic + ocean)
#   ./run_cice_pipeline.sh export     # NetCDF on native grids
#   ./run_cice_pipeline.sh plots      # WG vs CICE maps + animations from the zip (SAMPLES="0 1")
#   --- forcing re-injection (ERA5 re-encoded at every forecast step) ---
#   ./run_cice_pipeline.sh reinj-check  # non-regression: zero-init injection == no injection
#   ./run_cice_pipeline.sh r1           # learn the injection only (from P2_ID)
#   ./run_cice_pipeline.sh r2           # everything trainable, low LR
#   ./run_cice_pipeline.sh longroll     # 7-day rollouts without / with re-injection
#   --- flow-matching decoder (generative, members = ODE samples; ERA5 re-injection on) ---
#   ./run_cice_pipeline.sh fm-check     # non-regression: zero-init FM decoder == deterministic
#   ./run_cice_pipeline.sh fm-smoke     # short FM training on the ocean stream (no CICE needed)
#   ./run_cice_pipeline.sh fm1          # train the CICE FM decoder only (from R2_ID)
#   ./run_cice_pipeline.sh fm2          # everything trainable, low LR
#   ./run_cice_pipeline.sh fm-infer     # 7-day ensemble inference on 2023 + plots
#
# Training steps are long: with SLURM=1 they are submitted with launch-slurm.py. The slurm
# launcher generates its own run id: then set P1_ID / P2_ID to the printed ids for the next steps.
set -euo pipefail

BASE_ID=${BASE_ID:-d5xirkmf}
P1_ID=${P1_ID:-ft_cice_p1}
P2_ID=${P2_ID:-ft_cice_p2}
INF_ID=${INF_ID:-inf_cice01}
R1_ID=${R1_ID:-ft_cice_r1}
R2_ID=${R2_ID:-ft_cice_r2}
FM1_ID=${FM1_ID:-ft_cice_fm1}
FM2_ID=${FM2_ID:-ft_cice_fm2}
FMINF_ID=${FMINF_ID:-inf_cice_fm}
NODES=${NODES:-2}
SLURM=${SLURM:-0}

CFG=${CFG:-../WeatherGenerator-configs/config}
WD=${WD:-/e/scratch/weatherai/shared_work}
DATA=${DATA:-/e/data1/slmet/ml_training}
READER=packages/readers_extra/src/weathergen/readers_extra/data_reader_grep.py
LAUNCH=../WeatherGenerator-private/hpc/launch-slurm.py

train() {  # train <from_run_id> <run_id> <config>
  if [[ "$SLURM" == "1" ]]; then
    "$LAUNCH" --nodes "$NODES" --from-run-id "$1" --config "$3"
    echo ">> submitted. Set the printed run id for the next step, e.g.: export P1_ID=<id>"
  else
    uv run --offline train_continue --from-run-id "$1" --run-id "$2" --config "$3" 2>&1 | tee "log_$2.txt"
  fi
}

case "${1:-}" in
  check)
    echo "== reader patches"
    for k in zero_as_nan "Optional spatial domain" stats_override; do
      grep -q "$k" "$READER" && echo "OK   $k" || { echo "MISSING patch: $k"; exit 1; }
    done
    echo "== data"
    for z in "$DATA/cice_cglorsv8_1993_2023.zarr" "$DATA/nemo_cglorsv8_subset_1993_2023.zarr"; do
      [[ -e "$z" ]] && echo "OK   $z" || { echo "MISSING $z"; exit 1; }
    done
    echo "== streams vs $BASE_ID"
    uv run --offline python scripts/check_streams.py --run-id "$BASE_ID" \
      --streams-dir "$CFG/streams/earthgen3_cice/"
    ;;
  p1)
    train "$BASE_ID" "$P1_ID" "$CFG/finetune_cice_arctic_phase1.yml"
    echo ">> check the log: trainable parameters only on CICE, CICE loss decreasing"
    ;;
  p2)
    train "$P1_ID" "$P2_ID" "$CFG/finetune_cice_arctic_phase2.yml"
    ;;
  infer)
    uv run --offline inference --from-run-id "$P2_ID" --run-id "$INF_ID" \
      --config "$CFG/inference_cice_arctic.yml" 2>&1 | tee "log_$INF_ID.txt"
    grep -E "Checkpoint|Missing keys|Unused keys|stats_override|domain|selected source channels" \
      "log_$INF_ID.txt" || true
    ;;
  eval)
    mkdir -p config/evaluate
    sed -e "s#__RUN__#$INF_ID#g" -e "s#__WD__#$WD#g" "$CFG/evaluate/eval_cice_arctic.yml" \
      > "config/evaluate/eval_$INF_ID.yml"
    uv run --offline evaluate --config "config/evaluate/eval_$INF_ID.yml" 2>&1 | tee "log_eval_$INF_ID.txt"
    ;;
  export)
    uv run --offline python scripts/wg_zip2netcdf.py --run-id "$INF_ID" --stream CICE \
      --grid-zarr "$DATA/cice_cglorsv8_1993_2023.zarr"
    uv run --offline python scripts/wg_zip2netcdf.py --run-id "$INF_ID" --stream CGLORS-OCEAN \
      --channels thetao_0.5 so_0.5 uo_0.5 vo_0.5 zos \
      --grid-zarr "$DATA/nemo_cglorsv8_subset_1993_2023.zarr"
    echo ">> NetCDF in $WD/results/$INF_ID/netcdf/"
    ;;
  plots)
    # reads the inference zip directly (FORMAT=netcdf to use the NetCDF export instead)
    if [[ "${FORMAT:-zarr}" == "netcdf" ]]; then
      files=(); for s in ${SAMPLES:-0}; do files+=("$WD/results/$INF_ID/netcdf/"*_CICE_s$(printf %04d "$s").nc); done
      sel=()
    else
      files=("$WD/results/$INF_ID/"validation_chkpt*_rank*.zip); sel=(--samples ${SAMPLES:-0})
    fi
    uv run --offline python scripts/plot_seaice.py "${files[@]}" "${sel[@]}" --vars aice hi speed \
      --days 1 5 10 --animate --timeseries --label "$INF_ID" --out-dir "$WD/results/$INF_ID/figs_seaice"
    echo ">> figures in $WD/results/$INF_ID/figs_seaice/"
    ;;
  reinj-check)
    # same model, same dates, without and with the (zero-initialised) re-injection
    common=(test_config.start_date=2023-01-01T00:00 test_config.end_date=2023-03-31T00:00
            test_config.shuffle=False test_config.samples_per_mini_epoch=2
            test_config.output.num_samples=2 test_config.forecast.num_steps=5)
    uv run --offline inference --from-run-id "$BASE_ID" --run-id inf_reinj_off \
      --options "${common[@]}" 2>&1 | tee log_inf_reinj_off.txt
    uv run --offline inference --from-run-id "$BASE_ID" --run-id inf_reinj_zero \
      --options "${common[@]}" forcing_reinjection.enabled=True "forcing_reinjection.streams=[ERA5]" \
      2>&1 | tee log_inf_reinj_zero.txt
    grep -E "Forcing re-injection|Initializing new module" log_inf_reinj_zero.txt || true
    uv run --offline python scripts/compare_inference_zips.py \
      "$WD"/results/inf_reinj_off/validation_chkpt*_rank0000.zip \
      "$WD"/results/inf_reinj_zero/validation_chkpt*_rank0000.zip
    ;;
  r1)
    train "$P2_ID" "$R1_ID" "$CFG/finetune_cice_reinj_phase1.yml"
    echo ">> check: 'Forcing re-injection: N' and N = total trainable parameters"
    ;;
  r2)
    train "$R1_ID" "$R2_ID" "$CFG/finetune_cice_reinj_phase2.yml"
    ;;
  longroll)
    for pair in "$P2_ID:inf_long_noreinj" "$R2_ID:inf_long_reinj"; do
      uv run --offline inference --from-run-id "${pair%%:*}" --run-id "${pair##*:}" \
        --config "$CFG/inference_cice_longroll.yml" 2>&1 | tee "log_${pair##*:}.txt"
    done
    for r in inf_long_noreinj inf_long_reinj; do
      uv run --offline python scripts/plot_seaice.py "$WD/results/$r/"validation_chkpt*_rank*.zip \
        --samples 0 --vars aice hi --days 1 4 7 --animate --timeseries --label "$r" \
        --out-dir "$WD/results/$r/figs_seaice"
    done
    ;;
  fm-check)
    # same model, same dates: deterministic vs zero-initialised FM decoder on the ocean stream.
    # Every FM member must equal the deterministic prediction (IDENTICAL).
    common=(test_config.start_date=2023-01-01T00:00 test_config.end_date=2023-03-31T00:00
            test_config.shuffle=False test_config.samples_per_mini_epoch=2
            test_config.output.num_samples=2 test_config.forecast.num_steps=3)
    uv run --offline inference --from-run-id "$BASE_ID" --run-id inf_fm_off \
      --options "${common[@]}" 2>&1 | tee log_inf_fm_off.txt
    uv run --offline inference --from-run-id "$BASE_ID" --run-id inf_fm_zero \
      --options "${common[@]}" flow_matching.enabled=True "flow_matching.streams=[CGLORS-OCEAN]" \
      flow_matching.inference.num_members=2 flow_matching.inference.num_steps=4 \
      2>&1 | tee log_inf_fm_zero.txt
    grep -E "Flow matching|Flow-matching|Initializing new module" log_inf_fm_zero.txt || true
    uv run --offline python scripts/compare_inference_zips.py --streams CGLORS-OCEAN \
      "$WD"/results/inf_fm_off/validation_chkpt*_rank0000.zip \
      "$WD"/results/inf_fm_zero/validation_chkpt*_rank0000.zip
    ;;
  fm-smoke)
    uv run --offline train_continue --from-run-id "$BASE_ID" --run-id ft_ocean_fm_smoke \
      --config "$CFG/finetune_ocean_fm_smoke.yml" 2>&1 | tee log_ft_ocean_fm_smoke.txt
    echo ">> check: finite loss, trainable = ocean surface decoder + Flow-matching conditioning"
    grep -E "Flow matching|Flow-matching|Forcing re-injection|nan" log_ft_ocean_fm_smoke.txt | head -20 || true
    ;;
  fm1)
    train "$R2_ID" "$FM1_ID" "$CFG/finetune_cice_fm_phase1.yml"
    echo ">> check: trainable = CICE decoder + 'Flow-matching conditioning'; CICE loss starts at the R2 value"
    ;;
  fm2)
    train "$FM1_ID" "$FM2_ID" "$CFG/finetune_cice_fm_phase2.yml"
    ;;
  fm-infer)
    uv run --offline inference --from-run-id "$FM2_ID" --run-id "$FMINF_ID" \
      --config "$CFG/inference_cice_fm.yml" 2>&1 | tee "log_$FMINF_ID.txt"
    # plot_seaice.py shows the ENSEMBLE MEAN; spread and members: FastEval (CRPS, spread/skill)
    uv run --offline python scripts/plot_seaice.py "$WD/results/$FMINF_ID/"validation_chkpt*_rank*.zip \
      --samples ${SAMPLES:-0} --vars aice hi --days 1 4 7 --animate --timeseries --label "$FMINF_ID" \
      --out-dir "$WD/results/$FMINF_ID/figs_seaice"
    ;;
  *)
    sed -n '2,27p' "$0"; exit 1 ;;
esac
