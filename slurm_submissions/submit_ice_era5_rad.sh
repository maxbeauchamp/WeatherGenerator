#!/usr/bin/env bash
# xp_06: surface radiation (ssrd, strd) in the ERA5 forcing, from a LOGIN node:
#   [SITE=santis] bash slurm_submissions/submit_ice_era5_rad.sh [account]
# Chain: binfer (now) -> rcheck | r1 (now) -> r2 -> rinfer ; eval after binfer AND rinfer.
# Options as environment variables (see ice_era5_rad.sbatch), e.g. R1_EPOCHS=2 R2_EPOCHS=4 bash ...
#   SKIP_BINFER=1     the base inference BINF_ID already exists (JUPITER: inf_ice_era5_base of the
#                     FM chain, same 32 dates); then rcheck and eval do not wait for it
#   SKIP_RCHECK=1     no zero-init check
#   AFTER=<jobid>     r1 and binfer wait for this job
# Streams: STREAMS=ice_era5_rad (default here), its Santis copy ice_era5_rad_santis on Santis.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ACCOUNT=${1:-${ACCOUNT:-}}; [[ -z "$ACCOUNT" ]] && unset ACCOUNT
STREAMS=${STREAMS:-ice_era5_rad}
source "$HERE/site.sh"
# base streams for the base inference: the run json on JUPITER, the Santis copy on Santis
if [[ "$SITE" == "santis" ]]; then
  BASE_EXTRA_OPTS=${BASE_EXTRA_OPTS:-streams_directory=$CFGS/streams/ice_era5_santis/}
else
  BASE_EXTRA_OPTS=${BASE_EXTRA_OPTS:-}
fi
export BASE_EXTRA_OPTS
LOGDIR=${LOGDIR:-$WD/logs/ice_era5_rad}
mkdir -p "$LOGDIR"
sb() {  # sb <phase> <time> [dependency]
  local dep=(); [[ -n "${3:-}" ]] && dep=(--dependency="$3")
  sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -J "ice_era5_$1" -t "$2" \
    ${GPU_OPTS[@]+"${GPU_OPTS[@]}"} -o "$LOGDIR/%x_%j.out" -e "$LOGDIR/%x_%j.err" \
    "${dep[@]}" --export=ALL,PHASE="$1" "$HERE/ice_era5_rad.sbatch"
}
after=${AFTER:+afterok:$AFTER}
jb=""
[[ "${SKIP_BINFER:-0}" == "1" ]] || jb=$(sb binfer 02:00:00 "$after")
jc=""
[[ "${SKIP_RCHECK:-0}" == "1" ]] || jc=$(sb rcheck 02:00:00 "${jb:+afterok:$jb}")
j1=$(sb r1     12:00:00 "$after")
j2=$(sb r2     12:00:00 afterok:$j1)
jf=$(sb rinfer 03:00:00 afterok:$j2)
je=$(sb eval   01:00:00 "afterok:${jb:+$jb:}$jf")
echo "[$SITE] streams: $EXTRA_OPTS"
echo "binfer ${jb:-skipped} -> rcheck ${jc:-skipped} | r1 $j1 -> r2 $j2 -> rinfer $jf | eval $je"
echo "follow: squeue -u $USER ; tail -f $LOGDIR/ice_era5_r1_$j1.out"
