#!/bin/bash -x
# Ice + ERA5 from scratch on JUPITER: one node, 4 GH200 GPUs (DDP, one task per GPU).
# JUPITER limits: partition booster, max 12 h per job -> the ~24 h experiment is 2 chained jobs:
#   S1 (from scratch, 2-day rollouts)  then  S2 (7-day rollouts, from the final S1 model).
# Submit both from a LOGIN node (WeatherGenerator repo root):
#   bash submit_ice_era5.sh
# or one phase by hand:
#   sbatch --export=ALL,PHASE=s1 slurm_ice_era5.sbatch
#SBATCH --job-name=ice_era5
#SBATCH --account=e-ext-2025e01-128
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_ice_era5_%x_%j.out
#SBATCH --error=slurm_ice_era5_%x_%j.err

set -euo pipefail
PHASE=${PHASE:-s1}
WG=${WG:-/e/project1/weatherai/beauchamp1/WeatherGenerator}
CFG=../WeatherGenerator-configs/config
S1_ID=${S1_ID:-ice_era5_s1}
S2_ID=${S2_ID:-ice_era5_s2}
S1_EPOCHS=${S1_EPOCHS:-16}     # 16 x 4096 samples: fits in 12 h on 4 GPUs with margin
S2_EPOCHS=${S2_EPOCHS:-8}

cd "$WG"
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}   # srun does not inherit --cpus-per-task
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "phase=$PHASE nodes=$SLURM_JOB_NODELIST tasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT"
nvidia-smi --query-gpu=index,name,memory.used --format=csv

case "$PHASE" in
  s1)
    srun uv run --offline train --run-id "$S1_ID" \
      --config "$CFG/train_ice_era5_scratch.yml" \
      --options training_config.num_mini_epochs="$S1_EPOCHS"
    ;;
  s2)
    # final S1 model = chkpt000<S1_EPOCHS>
    srun uv run --offline train_continue --from-run-id "$S1_ID" --mini-epoch "$S1_EPOCHS" \
      --run-id "$S2_ID" --config "$CFG/train_ice_era5_long.yml" \
      --options training_config.num_mini_epochs="$S2_EPOCHS"
    ;;
  *) echo "unknown PHASE=$PHASE (s1 | s2)"; exit 1 ;;
esac

