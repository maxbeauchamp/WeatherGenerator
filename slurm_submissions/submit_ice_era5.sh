#!/usr/bin/env bash
# Submit the ice + ERA5 experiment as 2 chained 12 h jobs (S1 then S2), from a LOGIN node.
#   bash slurm_submissions/submit_ice_era5.sh [account]     (default e-ext-2025e01-128)
# Options: S1_EPOCHS=10 bash ... (fewer mini-epochs if S1 is too slow for 12 h)
set -euo pipefail
ACCOUNT=${1:-e-ext-2025e01-128}
HERE=$(cd "$(dirname "$0")" && pwd)
LOGDIR=${LOGDIR:-/e/scratch/weatherai/shared_work/logs/ice_era5}
mkdir -p "$LOGDIR"
common=(--parsable -A "$ACCOUNT" -o "$LOGDIR/%x_%j.out" -e "$LOGDIR/%x_%j.err")
j1=$(sbatch "${common[@]}" -J ice_era5_s1 --export=ALL,PHASE=s1 "$HERE/ice_era5.sbatch")
j2=$(sbatch "${common[@]}" -J ice_era5_s2 --dependency=afterok:"$j1" --export=ALL,PHASE=s2 \
  "$HERE/ice_era5.sbatch")
echo "S1: job $j1    S2: job $j2 (starts when S1 ends successfully)"
echo "follow: squeue -u $USER ; tail -f $LOGDIR/ice_era5_s1_$j1.out"
