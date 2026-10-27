#!/usr/bin/env bash
# Submit the partition-of-unity experiment on ice + ERA5 (no launcher, no MLflow), from a LOGIN node:
#   bash slurm_submissions/submit_ice_era5_pu.sh [account]
# Chain: pucheck (now) | pu1 (now) -> puinfer | pufm1 (now) -> pufm2 -> pufinfer
#        | eval after puinfer AND pufinfer | seams after everything.
# Needs inf_ice_era5_base (ice_era5_fm chain, phase binfer). Options as environment variables,
# e.g. PU1_EPOCHS=5 bash ... (see ice_era5_pu.sbatch). NO_FM=1: skip the PU + FM runs.
set -euo pipefail
ACCOUNT=${1:-e-ext-2025e01-128}
HERE=$(cd "$(dirname "$0")" && pwd)
LOGDIR=${LOGDIR:-/e/scratch/weatherai/shared_work/logs/ice_era5_pu}
BINF=/e/scratch/weatherai/shared_work/results/${BINF_ID:-inf_ice_era5_base}
[[ -d "$BINF" ]] || { echo "missing $BINF: run the binfer phase of ice_era5_fm.sbatch first"; exit 1; }
mkdir -p "$LOGDIR"
sb() {  # sb <phase> <time> [dependency]
  local dep=(); [[ -n "${3:-}" ]] && dep=(--dependency="$3")
  sbatch --parsable -A "$ACCOUNT" -J "ice_era5_$1" -t "$2" -o "$LOGDIR/%x_%j.out" -e "$LOGDIR/%x_%j.err" \
    "${dep[@]}" --export=ALL,PHASE="$1" "$HERE/ice_era5_pu.sbatch"
}
jc=$(sb pucheck 03:00:00)
j1=$(sb pu1     12:00:00)
ji=$(sb puinfer 03:00:00 afterok:$j1)
if [[ -z "${NO_FM:-}" ]]; then
  jf1=$(sb pufm1    12:00:00)
  jf2=$(sb pufm2    12:00:00 afterok:$jf1)
  jfi=$(sb pufinfer 03:00:00 afterok:$jf2)
  je=$(sb eval  01:00:00 afterok:$ji:$jfi)
  js=$(sb seams 01:00:00 afterany:$jc:$ji:$jfi)
  echo "pucheck $jc | pu1 $j1 -> puinfer $ji | pufm1 $jf1 -> pufm2 $jf2 -> pufinfer $jfi | eval $je | seams $js"
else
  je=$(sb eval  01:00:00 afterok:$ji)
  js=$(sb seams 01:00:00 afterany:$jc:$ji)
  echo "pucheck $jc | pu1 $j1 -> puinfer $ji | eval $je | seams $js"
fi
echo "follow: squeue -u $USER ; tail -f $LOGDIR/ice_era5_pu1_$j1.out"
