# Machine-specific defaults for the submit_*.sh scripts. Sourced, not run.
#   SITE=jupiter (default) | santis
#   STREAMS=<name>: stream directory of the configs (e.g. ice_era5_rad, xp_06); on Santis its copy
#   <name>_santis is used (default ice_era5_santis). Empty on JUPITER = the streams of the run json.
#   NGPUS=<n>: GPUs (= tasks) per job, default the 4 of the sbatch headers.
# Every value can still be overridden from the environment (e.g. WD=... SITE=santis bash ...).
HERE=${HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
SITE=${SITE:-jupiter}
case "$SITE" in
  jupiter)
    ACCOUNT=${ACCOUNT:-e-ext-2025e01-128}
    PARTITION=${PARTITION:-booster}
    WG=${WG:-/e/project1/weatherai/beauchamp1/WeatherGenerator}
    WD=${WD:-/e/scratch/weatherai/shared_work}
    CFGS=${CFGS:-$WG/../WeatherGenerator-configs/config}
    EXTRA_OPTS=${EXTRA_OPTS:-${STREAMS:+streams_directory=$CFGS/streams/$STREAMS/}}
    ;;
  santis)   # CSCS, Zürich hackathon (see slurm_submissions/SANTIS.md)
    ACCOUNT=${ACCOUNT:-wghack-course2026-cscs}
    PARTITION=${PARTITION:-normal}
    WG=${WG:-$(cd "$HERE/.." && pwd)}
    WD=${WD:-/iopsstor/scratch/cscs/course_29fe58c1/shared_work}
    # CICE path of Santis: streams/<name>_santis/ is the Santis copy of streams/<name>/
    CFGS=${CFGS:-$WG/../WeatherGenerator-configs/config}
    EXTRA_OPTS=${EXTRA_OPTS:-streams_directory=$CFGS/streams/${STREAMS:-ice_era5}_santis/}
    ;;
  *) echo "unknown SITE=$SITE (jupiter | santis)"; exit 1 ;;
esac
export SITE ACCOUNT PARTITION WG WD EXTRA_OPTS
# NGPUS=<n>: n tasks / n GPUs per job instead of the 4 of the sbatch headers (e.g. NGPUS=1 when
# the multi-GPU start fails on a machine). Used by the submit_*.sh scripts.
GPU_OPTS=()
if [[ -n "${NGPUS:-}" ]]; then GPU_OPTS=(--ntasks-per-node="$NGPUS" --gres=gpu:"$NGPUS"); fi
