#!/usr/bin/env bash
# Submit the ice + ERA5 experiment as 2 chained 12 h jobs (S1 then S2), from a LOGIN node.
#   bash slurm_submissions/submit_ice_era5.sh [account]     (default: see site.sh)
# Options: S1_EPOCHS=10 bash ... (fewer mini-epochs if S1 is too slow for 12 h)
#          SITE=santis bash ...  (CSCS Santis instead of JUPITER, see site.sh)
#          FM=1 bash ...         (then chain the flow-matching experiment on the new S2,
#                                 see submit_ice_era5_fm.sh)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ACCOUNT=${1:-${ACCOUNT:-}}; [[ -z "$ACCOUNT" ]] && unset ACCOUNT
source "$HERE/site.sh"
S1_ID=${S1_ID:-ice_era5_s1}; S2_ID=${S2_ID:-ice_era5_s2}
S1_EPOCHS=${S1_EPOCHS:-16};  S2_EPOCHS=${S2_EPOCHS:-8}
export S1_ID S2_ID S1_EPOCHS S2_EPOCHS
LOGDIR=${LOGDIR:-$WD/logs/ice_era5}
mkdir -p "$LOGDIR"
common=(--parsable -A "$ACCOUNT" -p "$PARTITION" ${GPU_OPTS[@]+"${GPU_OPTS[@]}"}
  -o "$LOGDIR/%x_%j.out" -e "$LOGDIR/%x_%j.err")
j1=$(sbatch "${common[@]}" -J ice_era5_s1 --export=ALL,PHASE=s1 "$HERE/ice_era5.sbatch")
j2=$(sbatch "${common[@]}" -J ice_era5_s2 --dependency=afterok:"$j1" --export=ALL,PHASE=s2 \
  "$HERE/ice_era5.sbatch")
echo "[$SITE] S1: job $j1    S2: job $j2 (starts when S1 ends successfully)"
echo "follow: squeue -u $USER ; tail -f $LOGDIR/ice_era5_s1_$j1.out"
if [[ "${FM:-0}" == "1" ]]; then
  AFTER=$j2 BASE_ID=$S2_ID BASE_EPOCH=$S2_EPOCHS bash "$HERE/submit_ice_era5_fm.sh"
fi
