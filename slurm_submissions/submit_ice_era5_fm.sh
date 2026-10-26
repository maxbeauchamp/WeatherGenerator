#!/usr/bin/env bash
# Submit the whole flow-matching experiment on ice + ERA5 (no launcher, no MLflow), from a LOGIN node:
#   bash slurm_submissions/submit_ice_era5_fm.sh [account]
# Chain: binfer (now) | fm1 (now) -> fm2 -> finfer ; eval after binfer AND finfer.
# Options as environment variables, e.g. FM1_EPOCHS=5 FM2_EPOCHS=5 bash ... (see ice_era5_fm.sbatch).
set -euo pipefail
ACCOUNT=${1:-e-ext-2025e01-128}
HERE=$(cd "$(dirname "$0")" && pwd)
LOGDIR=${LOGDIR:-/e/scratch/weatherai/shared_work/logs/ice_era5_fm}
mkdir -p "$LOGDIR"
sb() {  # sb <phase> <time> [dependency]
  local dep=(); [[ -n "${3:-}" ]] && dep=(--dependency="$3")
  sbatch --parsable -A "$ACCOUNT" -J "ice_era5_$1" -t "$2" -o "$LOGDIR/%x_%j.out" -e "$LOGDIR/%x_%j.err" \
    "${dep[@]}" --export=ALL,PHASE="$1" "$HERE/ice_era5_fm.sbatch"
}
jb=$(sb binfer 02:00:00)
j1=$(sb fm1    12:00:00)
j2=$(sb fm2    12:00:00 afterok:$j1)
jf=$(sb finfer 03:00:00 afterok:$j2)
je=$(sb eval   01:00:00 afterok:$jb:$jf)
echo "binfer $jb | fm1 $j1 -> fm2 $j2 -> finfer $jf | eval $je (after $jb and $jf)"
echo "follow: squeue -u $USER ; tail -f $LOGDIR/ice_era5_fm1_$j1.out"

