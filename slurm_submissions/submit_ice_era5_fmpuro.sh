#!/usr/bin/env bash
# Rollout of the FM condition (xp_06) on FM + PU: from ice_era5_fmpu1 chkpt 3, x_(k-1) is the
# model's own previous prediction with probability 0.5 (finetune_ice_era5_fmpuro.yml), so that the
# CICE decoder learns to regenerate the small scales in the rollouts. Same phases as
# submit_ice_era5_fmpu.sh (ice_era5_pu.sbatch), from a LOGIN node:
#   SMOKE=1 bash slurm_submissions/submit_ice_era5_fmpuro.sh [account]   # first: cost per batch
#   bash slurm_submissions/submit_ice_era5_fmpuro.sh [account]           # then the chain
# Compared with FM + PU (inf_ice_era5_fmpu*), FM and the base: FastEval 7 days, seams and texture
# vs lead time (seams_ice_era5_fmpuro*/texture_<ch>.png).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export FM2_ID=ice_era5_fmpu1 FM2_EPOCH=3 FPU1_ID=ice_era5_fmpuro1 FPU1_EPOCHS=${FPU1_EPOCHS:-3}
export FPU1_CFG=finetune_ice_era5_fmpuro.yml FPINF_ID=inf_ice_era5_fmpuro
export FPU_LABEL=fm_pu_ro FPU_DESC="FM + PU, rollout of the condition"
export LOGDIR=${LOGDIR:-/e/scratch/weatherai/shared_work/logs/ice_era5_fmpuro}
exec bash "$HERE/submit_ice_era5_fmpu.sh" "$@"
