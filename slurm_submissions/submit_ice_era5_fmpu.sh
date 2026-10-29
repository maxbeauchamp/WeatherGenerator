#!/usr/bin/env bash
# Submit the PU fine-tuning of the FM sea-ice model (xp_05 on xp_04), from ice_era5_fm2 chkpt 4,
# from a LOGIN node (no launcher, no MLflow):
#   SMOKE=1 bash slurm_submissions/submit_ice_era5_fmpu.sh [account]   # first: PU + FM, solver in validation
#   bash slurm_submissions/submit_ice_era5_fmpu.sh [account]           # then the chain
# Chain: fpu1 -> fpuinfer (7 days, 32 dates) -> fpueval | fpu1 -> fpulong x 2 (90 days, one job per date)
#        | fpuseams after fpuinfer and fpulong.
# Needs inf_ice_era5_fm (finfer) and inf_ice_era5_fm_long_* for the comparisons.
# Options as environment variables, e.g. FPU1_EPOCHS=2 bash ... (see ice_era5_pu.sbatch).
set -euo pipefail
ACCOUNT=${1:-e-ext-2025e01-128}
HERE=$(cd "$(dirname "$0")" && pwd)
# the jobs run the code of THIS checkout, unless WG is set
export WG=${WG:-$(dirname "$HERE")}
LOGDIR=${LOGDIR:-/e/scratch/weatherai/shared_work/logs/ice_era5_fmpu}
mkdir -p "$LOGDIR"
sb() {  # sb <phase> <time> [dependency] [extra exports, e.g. LONG_STARTS=2023-03-01]
  local dep=(); [[ -n "${3:-}" ]] && dep=(--dependency="$3")
  sbatch --parsable -A "$ACCOUNT" -J "ice_era5_$1" -t "$2" -o "$LOGDIR/%x_%j.out" -e "$LOGDIR/%x_%j.err" \
    "${dep[@]}" --export=ALL,PHASE="$1"${4:+,$4} "$HERE/ice_era5_pu.sbatch"
}
if [[ -n "${SMOKE:-}" ]]; then
  js=$(sb fpusmoke 01:00:00)
  echo "fpusmoke $js"
  echo "check: grep -E 'Partition of unity|Flow matching|trainable|loss =|validation \\(' $LOGDIR/ice_era5_fpusmoke_$js.out"
  exit 0
fi
j1=$(sb fpu1     08:00:00)
ji=$(sb fpuinfer 06:00:00 afterok:$j1)
# one job per 90-day start date, in parallel (the PU makes the FM sampling ~3x longer)
jl1=$(sb fpulong 12:00:00 afterok:$j1 LONG_STARTS=2023-03-01)
jl2=$(sb fpulong 12:00:00 afterok:$j1 LONG_STARTS=2023-09-01)
je=$(sb fpueval  01:00:00 afterok:$ji)
js=$(sb fpuseams 01:00:00 afterany:$ji:$jl1:$jl2)
echo "fpu1 $j1 -> fpuinfer $ji -> fpueval $je | fpulong $jl1 $jl2 | fpuseams $js"
echo "follow: squeue -u $USER ; grep 'loss =' $LOGDIR/ice_era5_fpu1_$j1.out | tail"
