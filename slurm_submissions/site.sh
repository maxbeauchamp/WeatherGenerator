# Machine-specific defaults for the submit_*.sh scripts. Sourced, not run.
#   SITE=jupiter (default) | santis
# Every value can still be overridden from the environment (e.g. WD=... SITE=santis bash ...).
HERE=${HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
SITE=${SITE:-jupiter}
case "$SITE" in
  jupiter)
    ACCOUNT=${ACCOUNT:-e-ext-2025e01-128}
    PARTITION=${PARTITION:-booster}
    WG=${WG:-/e/project1/weatherai/beauchamp1/WeatherGenerator}
    WD=${WD:-/e/scratch/weatherai/shared_work}
    EXTRA_OPTS=${EXTRA_OPTS:-}
    ;;
  santis)   # CSCS, Zürich hackathon (see slurm_submissions/SANTIS.md)
    ACCOUNT=${ACCOUNT:-wghack-course2026-cscs}
    PARTITION=${PARTITION:-normal}
    WG=${WG:-$(cd "$HERE/.." && pwd)}
    WD=${WD:-/iopsstor/scratch/cscs/course_29fe58c1/shared_work}
    # CICE path of Santis: streams/ice_era5_santis/ is the Santis copy of streams/ice_era5/
    EXTRA_OPTS=${EXTRA_OPTS:-streams_directory=$WG/../WeatherGenerator-configs/config/streams/ice_era5_santis/}
    ;;
  *) echo "unknown SITE=$SITE (jupiter | santis)"; exit 1 ;;
esac
export SITE ACCOUNT PARTITION WG WD EXTRA_OPTS
