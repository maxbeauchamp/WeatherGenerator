#!/usr/bin/env bash
# xp_07: ice + ERA5 base trained FROM SCRATCH with surface radiation (ssrd, strd: 8 ERA5 channels,
# AIFS o96 zarr), then inference on the same 32 dates as the base and evaluation against it.
# From a LOGIN node:
#   SMOKE=1 [SITE=santis] bash slurm_submissions/submit_ice_era5_rad8.sh   # 1 short job: multi-GPU start + data
#   [SITE=santis] bash slurm_submissions/submit_ice_era5_rad8.sh           # S1 -> S2 -> rinfer -> eval
# Streams: streams/ice_era5_rad8/ (Santis: its copy ice_era5_rad8_santis/ with the Santis CICE path).
# The base inference inf_ice_era5_base (same 32 dates) must exist (binfer of submit_ice_era5_rad.sh).
# Options: S1_EPOCHS (16), S2_EPOCHS (6: the base ice_era5_s2 is used at chkpt 6), NGPUS, AFTER=<jobid>.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STREAMS=${STREAMS:-ice_era5_rad8}
source "$HERE/site.sh"
export S1_ID=${S1_ID:-ice_era5_rad8_s1} S2_ID=${S2_ID:-ice_era5_rad8_s2}
export S1_EPOCHS=${S1_EPOCHS:-16} S2_EPOCHS=${S2_EPOCHS:-6}
# rinfer / eval phases of ice_era5_rad.sbatch on the final S2 model
export R2_ID=$S2_ID R2_EPOCHS=$S2_EPOCHS RINF_ID=${RINF_ID:-inf_ice_era5_rad8}
export BINF_ID=${BINF_ID:-inf_ice_era5_base} RAD_LABEL="ice + ERA5 + ssrd/strd, from scratch"
LOGDIR=${LOGDIR:-$WD/logs/ice_era5_rad8}
mkdir -p "$LOGDIR"
sb() {  # sb <sbatch file> <phase> <time> [dependency]
  local dep=(); [[ -n "${4:-}" ]] && dep=(--dependency="$4")
  sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J "ice_rad8_$2" -t "$3" \
    ${GPU_OPTS[@]+"${GPU_OPTS[@]}"} -o "$LOGDIR/%x_%j.out" -e "$LOGDIR/%x_%j.err" \
    "${dep[@]}" --export=ALL,PHASE="$2" "$HERE/$1"
}
if [[ "${SMOKE:-0}" == "1" ]]; then
  export S1_ID=ice_era5_rad8_smoke S1_EPOCHS=1
  export EXTRA_OPTS="$EXTRA_OPTS training_config.samples_per_mini_epoch=64 validation_config.samples_per_mini_epoch=16"
  j=$(sb ice_era5.sbatch s1 00:40:00 "${AFTER:+afterok:$AFTER}")
  echo "[$SITE] smoke: job $j (run id $S1_ID). Check: 'task i: CUDA_VISIBLE_DEVICES=i' for every task, a finite loss."
  echo "follow: tail -f $LOGDIR/ice_rad8_s1_$j.out"
  exit 0
fi
j1=$(sb ice_era5.sbatch     s1     12:00:00 "${AFTER:+afterok:$AFTER}")
j2=$(sb ice_era5.sbatch     s2     12:00:00 afterok:$j1)
jf=$(sb ice_era5_rad.sbatch rinfer 01:00:00 afterok:$j2)
je=$(sb ice_era5_rad.sbatch eval   01:00:00 afterok:$jf)
echo "[$SITE] streams: $EXTRA_OPTS"
echo "s1 $j1 ($S1_ID, $S1_EPOCHS mini-epochs) -> s2 $j2 ($S2_ID, $S2_EPOCHS) -> rinfer $jf ($RINF_ID) -> eval $je (vs $BINF_ID)"
echo "follow: squeue -u $USER ; tail -f $LOGDIR/ice_rad8_s1_$j1.out"
