#!/usr/bin/env bash
# Submit the partition-of-unity test on d5xirkmf, surface decoder only (no launcher, no MLflow),
# from a LOGIN node:
#   SMOKE=1 bash slurm_submissions/submit_ocean_pu.sh [account]   # first: memory and samples/s
#   bash slurm_submissions/submit_ocean_pu.sh [account]           # then the chain
# Chain: opucheck (now) | opu1 (now) -> opuinfer | eval after opucheck AND opuinfer
#        | seams after everything.
# Options as environment variables, e.g. OPU1_EPOCHS=2 N_DATES=4 bash ... (see ocean_pu.sbatch).
set -euo pipefail
ACCOUNT=${1:-e-ext-2025e01-128}
HERE=$(cd "$(dirname "$0")" && pwd)
# the jobs run the code of THIS checkout (e.g. a separate git worktree), unless WG is set
export WG=${WG:-$(dirname "$HERE")}
LOGDIR=${LOGDIR:-/e/scratch/weatherai/shared_work/logs/ocean_pu}
mkdir -p "$LOGDIR"
sb() {  # sb <phase> <time> [dependency]
  local dep=(); [[ -n "${3:-}" ]] && dep=(--dependency="$3")
  sbatch --parsable -A "$ACCOUNT" -J "ocean_$1" -t "$2" -o "$LOGDIR/%x_%j.out" -e "$LOGDIR/%x_%j.err" \
    "${dep[@]}" --export=ALL,PHASE="$1" "$HERE/ocean_pu.sbatch"
}
if [[ -n "${SMOKE:-}" ]]; then
  js=$(sb opusmoke 01:00:00)
  echo "opusmoke $js"
  echo "check: grep -E 'Partition of unity|trainable|samples/s|out of memory' $LOGDIR/ocean_opusmoke_$js.out"
  exit 0
fi
jc=$(sb opucheck 02:00:00)
j1=$(sb opu1     06:00:00)
ji=$(sb opuinfer 01:00:00 afterok:$j1)
je=$(sb eval     01:00:00 afterok:$jc:$ji)
js=$(sb seams    01:00:00 afterany:$jc:$ji)
echo "opucheck $jc | opu1 $j1 -> opuinfer $ji | eval $je | seams $js"
echo "follow: squeue -u $USER ; tail -f $LOGDIR/ocean_opu1_$j1.out"
