#!/bin/bash -x
# Flow matching on the ice + ERA5 base model, WITHOUT the team launcher (no snapshot, no MLflow).
# One node, 4 GH200 GPUs (one task per GPU). Phases chained by submit_ice_era5_fm.sh:
#   binfer (base inference) | fm1 -> fm2 -> finfer (FM inference) | eval (after binfer + finfer)
# Submit from a LOGIN node:  bash slurm_submissions/submit_ice_era5_fm.sh
# Logs: /e/scratch/weatherai/shared_work/logs/ice_era5_fm/<jobname>_<jobid>.out|.err
# Do not edit the code or the configs used here while the jobs run (no snapshot).
#SBATCH --job-name=ice_era5_fm
#SBATCH --account=e-ext-2025e01-128
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

set -euo pipefail
PHASE=${PHASE:?PHASE = binfer | fm1 | fm2 | finfer | eval}
WG=${WG:-/e/project1/weatherai/beauchamp1/WeatherGenerator}
CFG=../WeatherGenerator-configs/config
WD=/e/scratch/weatherai/shared_work
BASE_ID=${BASE_ID:-ice_era5_s2};  BASE_EPOCH=${BASE_EPOCH:-6}
FM1_ID=${FM1_ID:-ice_era5_fm1};  FM1_EPOCHS=${FM1_EPOCHS:-7}
FM2_ID=${FM2_ID:-ice_era5_fm2};  FM2_EPOCHS=${FM2_EPOCHS:-6}
BINF_ID=${BINF_ID:-inf_ice_era5_base}; FINF_ID=${FINF_ID:-inf_ice_era5_fm}
# both inferences on the SAME 32 start dates of 2023 (no shuffling)
SAME_DATES=(test_config.shuffle=False test_config.samples_per_mini_epoch=32 test_config.output.num_samples=32)

cd "$WG"
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
if [[ "$SLURM_JOB_NUM_NODES" == "1" ]]; then export MASTER_ADDR=127.0.0.1
else export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1); fi
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
run() {  # one task per GPU (Slurm binds one GPU per task -> it is cuda:0 for the task)
  srun bash -c '
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]]; then export LOCAL_RANK=0
    else export LOCAL_RANK=$SLURM_LOCALID; fi
    export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS
    echo "task $SLURM_PROCID: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} LOCAL_RANK=$LOCAL_RANK"
    exec "$@"' _ "$@"
}
echo "phase=$PHASE node=$SLURM_JOB_NODELIST tasks=$SLURM_NTASKS"

case "$PHASE" in
  binfer)
    rm -rf "$WD/results/$BINF_ID"
    run uv run --offline inference --from-run-id "$BASE_ID" --mini-epoch "$BASE_EPOCH" --run-id "$BINF_ID" \
      --config "$CFG/inference_ice_era5.yml" --options "${SAME_DATES[@]}"
    ;;
  fm1)   # CICE decoder + flow-matching conditioning only (ERA5 re-injection kept, as in the base)
    run uv run --offline train_continue --from-run-id "$BASE_ID" --mini-epoch "$BASE_EPOCH" --run-id "$FM1_ID" \
      --config "$CFG/finetune_cice_fm_phase1.yml" --options training_config.num_mini_epochs="$FM1_EPOCHS"
    ;;
  fm2)   # everything trainable, lr 1e-5, from the final FM1 model
    run uv run --offline train_continue --from-run-id "$FM1_ID" --mini-epoch "$FM1_EPOCHS" --run-id "$FM2_ID" \
      --config "$CFG/finetune_cice_fm_phase2.yml" --options training_config.num_mini_epochs="$FM2_EPOCHS"
    ;;
  finfer)  # 8 members x 16 Heun steps, same dates as binfer
    rm -rf "$WD/results/$FINF_ID"
    run uv run --offline inference --from-run-id "$FM2_ID" --mini-epoch "$FM2_EPOCHS" --run-id "$FINF_ID" \
      --config "$CFG/inference_cice_fm.yml" --options "${SAME_DATES[@]}"
    ;;
  eval)  # FastEval (single process): FM ensemble vs deterministic base, with persistence
    mkdir -p config/evaluate
    EV="config/evaluate/eval_${FINF_ID}_vs_base.yml"
    uv run --offline python - "$CFG/evaluate/eval_ice_era5.yml" "$EV" "$WD" "$FINF_ID" "$BINF_ID" <<'PY'
import sys, yaml
tpl, out, wd, fm, base = sys.argv[1:]
txt = open(tpl).read()
def cfg(rid): return yaml.safe_load(txt.replace("__RUN__", rid).replace("__WD__", wd))
c = cfg(fm)
c["run_ids"][fm]["label"] = "ice + ERA5, flow matching (8 members)"
c["run_ids"][base] = cfg(base)["run_ids"][base]
c["run_ids"][base]["label"] = "ice + ERA5, deterministic base"
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
print("eval config:", out, list(c["run_ids"]))
PY
    uv run --offline evaluate --config "$EV"
    echo ">> scores and figures in $WD/results/$FINF_ID/plots/"
    ;;
  *) echo "unknown PHASE=$PHASE"; exit 1 ;;
esac

