#!/bin/bash -x
# xp_06: surface radiation (ssrd, strd) added to the ERA5 forcing of the ice + ERA5 base model,
# WITHOUT the team launcher (no snapshot, no MLflow). One node, 4 GH200 GPUs (one task per GPU).
# Phases chained by submit_ice_era5_rad.sh:
#   binfer (base inference) -> rcheck (zero-init check) | r1 -> r2 -> rinfer | eval (after binfer + rinfer)
# Submit from a LOGIN node:  [SITE=santis] bash slurm_submissions/submit_ice_era5_rad.sh
# Do not edit the code or the configs used here while the jobs run (no snapshot).
#SBATCH --job-name=ice_era5_rad
#SBATCH --account=e-ext-2025e01-128
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

set -euo pipefail
PHASE=${PHASE:?PHASE = binfer | rcheck | r1 | r2 | rinfer | eval}
WG=${WG:-/e/project1/weatherai/beauchamp1/WeatherGenerator}
CFG=../WeatherGenerator-configs/config
WD=${WD:-/e/scratch/weatherai/shared_work}
BASE_ID=${BASE_ID:-ice_era5_s2};       BASE_EPOCH=${BASE_EPOCH:-6}
R1_ID=${R1_ID:-ice_era5_rad1};         R1_EPOCHS=${R1_EPOCHS:-3}
R2_ID=${R2_ID:-ice_era5_rad2};         R2_EPOCHS=${R2_EPOCHS:-3}
BINF_ID=${BINF_ID:-inf_ice_era5_base}; RINF_ID=${RINF_ID:-inf_ice_era5_rad}
CHECK_ID=${CHECK_ID:-inf_ice_era5_radcheck}
# both inferences on the SAME 32 start dates of 2023 (no shuffling), as in ice_era5_fm.sbatch
SAME_DATES=(test_config.shuffle=False test_config.samples_per_mini_epoch=32 test_config.output.num_samples=32)
# extra --options: streams with ssrd / strd (EXTRA_OPTS) and, for the base inference, the base
# streams (BASE_EXTRA_OPTS); both set by submit_ice_era5_rad.sh from site.sh
read -r -a XO <<< "${EXTRA_OPTS:-}"
read -r -a BXO <<< "${BASE_EXTRA_OPTS:-}"

cd "$WG"
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
if [[ "$SLURM_JOB_NUM_NODES" == "1" ]]; then export MASTER_ADDR=127.0.0.1
else export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1); fi
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
run() {  # one task per GPU (Slurm binds one GPU per task -> it is cuda:0 for the task)
  srun ${SRUN_OPTS:-} bash -c '
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]]; then export LOCAL_RANK=0
    else export LOCAL_RANK=$SLURM_LOCALID; fi
    export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS
    echo "task $SLURM_PROCID: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} LOCAL_RANK=$LOCAL_RANK"
    exec "$@"' _ "$@"
}
echo "phase=$PHASE node=$SLURM_JOB_NODELIST tasks=$SLURM_NTASKS extra_opts=${XO[*]:-none}"

case "$PHASE" in
  binfer)  # base model, base streams
    rm -rf "$WD/results/$BINF_ID"
    run uv run --offline inference --from-run-id "$BASE_ID" --mini-epoch "$BASE_EPOCH" --run-id "$BINF_ID" \
      --config "$CFG/inference_ice_era5.yml" --options "${SAME_DATES[@]}" ${BXO[@]+"${BXO[@]}"}
    ;;
  rcheck)  # base checkpoint + streams with ssrd / strd: the new embedder is 0 -> same forecasts
    rm -rf "$WD/results/$CHECK_ID"
    run uv run --offline inference --from-run-id "$BASE_ID" --mini-epoch "$BASE_EPOCH" --run-id "$CHECK_ID" \
      --config "$CFG/inference_ice_era5.yml" --options "${SAME_DATES[@]}" ${XO[@]+"${XO[@]}"}
    uv run --offline python scripts/compare_inference_zips.py \
      "$WD"/results/"$BINF_ID"/validation_chkpt*_rank0000.zip \
      "$WD"/results/"$CHECK_ID"/validation_chkpt*_rank0000.zip
    ;;
  r1)  # only the zero-initialised ssrd / strd embedder
    run uv run --offline train_continue --from-run-id "$BASE_ID" --mini-epoch "$BASE_EPOCH" --run-id "$R1_ID" \
      --config "$CFG/finetune_ice_era5_rad_phase1.yml" \
      --options training_config.num_mini_epochs="$R1_EPOCHS" ${XO[@]+"${XO[@]}"}
    ;;
  r2)  # everything trainable, low learning rate, from the final R1 model
    run uv run --offline train_continue --from-run-id "$R1_ID" --mini-epoch "$R1_EPOCHS" --run-id "$R2_ID" \
      --config "$CFG/finetune_ice_era5_rad_phase2.yml" \
      --options training_config.num_mini_epochs="$R2_EPOCHS" ${XO[@]+"${XO[@]}"}
    ;;
  rinfer)  # same dates as binfer
    rm -rf "$WD/results/$RINF_ID"
    run uv run --offline inference --from-run-id "$R2_ID" --mini-epoch "$R2_EPOCHS" --run-id "$RINF_ID" \
      --config "$CFG/inference_ice_era5.yml" --options "${SAME_DATES[@]}" ${XO[@]+"${XO[@]}"}
    ;;
  eval)  # FastEval (single process): with radiation vs base, with persistence
    mkdir -p config/evaluate
    EV="config/evaluate/eval_${RINF_ID}_vs_base.yml"
    uv run --offline python - "$CFG/evaluate/eval_ice_era5.yml" "$EV" "$WD" "$RINF_ID" "$BINF_ID" <<'PY'
import sys, yaml
tpl, out, wd, rad, base = sys.argv[1:]
txt = open(tpl).read()
def cfg(rid): return yaml.safe_load(txt.replace("__RUN__", rid).replace("__WD__", wd))
c = cfg(rad)
c["run_ids"][rad]["label"] = "ice + ERA5 + ssrd/strd"
c["run_ids"][base] = cfg(base)["run_ids"][base]
c["run_ids"][base]["label"] = "ice + ERA5, base"
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
print("eval config:", out, list(c["run_ids"]))
PY
    uv run --offline evaluate --config "$EV"
    echo ">> scores and figures in $WD/results/$RINF_ID/plots/"
    ;;
  *) echo "unknown PHASE=$PHASE"; exit 1 ;;
esac
