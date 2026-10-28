#!/bin/bash -x
# Partition of unity (xp_05) on the ocean model d5xirkmf, SURFACE decoder only (CGLORS-OCEAN),
# WITHOUT the team launcher (no snapshot, no MLflow). One node, 4 GH200 GPUs (one task per GPU).
# Phases chained by submit_ocean_pu.sh:
#   opusmoke (memory, samples/s) | opucheck (base and PU at start, no training) | opu1 -> opuinfer
#   | eval (after opucheck + opuinfer) | seams (after everything)
# Submit from a LOGIN node:  bash slurm_submissions/submit_ocean_pu.sh
# Logs: /e/scratch/weatherai/shared_work/logs/ocean_pu/<jobname>_<jobid>.out|.err
# Do not edit the code or the configs used here while the jobs run (no snapshot).
#SBATCH --job-name=ocean_pu
#SBATCH --account=e-ext-2025e01-128
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00

set -euo pipefail
PHASE=${PHASE:?PHASE = opusmoke | opucheck | opu1 | opuinfer | eval | seams}
WG=${WG:-/e/project1/weatherai/beauchamp1/WeatherGenerator}
CFG=../WeatherGenerator-configs/config
WD=/e/scratch/weatherai/shared_work
BASE_ID=${BASE_ID:-d5xirkmf}
BASE_EPOCH=${BASE_EPOCH:-}                 # empty: <run>_latest
OPU1_ID=${OPU1_ID:-ocean_pu1};  OPU1_EPOCHS=${OPU1_EPOCHS:-4}
BINF_ID=${BINF_ID:-inf_ocean_base}         # d5xirkmf, PU off
P0INF_ID=${P0INF_ID:-inf_ocean_pu0}        # d5xirkmf, PU on, before any training
PINF_ID=${PINF_ID:-inf_ocean_pu}           # after opu1
STREAM=CGLORS-OCEAN
# all inferences on the SAME start dates (no shuffling): N_DATES dates from 2023-01-01, 10 days
N_DATES=${N_DATES:-8}
SAME_DATES=(test_config.start_date=2023-01-01T00:00 test_config.end_date=2023-12-31T00:00
            test_config.shuffle=False test_config.samples_per_mini_epoch="$N_DATES"
            test_config.output.num_samples="$N_DATES" test_config.forecast.num_steps=10)
PU_ON=(partition_of_unity.enabled=True "partition_of_unity.streams=[$STREAM]")
base_epoch=(); [[ -n "$BASE_EPOCH" ]] && base_epoch=(--mini-epoch "$BASE_EPOCH")

cd "$WG"
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
if [[ "$SLURM_JOB_NUM_NODES" == "1" ]]; then export MASTER_ADDR=127.0.0.1
else export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1); fi
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
run() {  # one task per GPU (Slurm binds one GPU per task -> it is cuda:0 for the task)
  srun bash -c '
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]]; then export LOCAL_RANK=0
    else export LOCAL_RANK=$SLURM_LOCALID; fi
    export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS
    echo "task $SLURM_PROCID: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} LOCAL_RANK=$LOCAL_RANK"
    exec "$@"' _ "$@"
}
zip0() { ls "$WD/results/$1/"validation_chkpt*_rank0000.zip 2>/dev/null | head -n 1; }
seams() {  # seams <out_dir> <run_id:label>...
  local out=$1 zips=() labels=(); shift
  for pair in "$@"; do
    z=$(zip0 "${pair%%:*}"); [[ -n "$z" ]] && { zips+=("$z"); labels+=("${pair##*:}"); }
  done
  uv run --offline python scripts/seam_diagnostic.py "${zips[@]}" --labels "${labels[@]}" \
    --stream "$STREAM" --channels thetao_0.5 so_0.5 zos --healpix-level 5 --out-dir "$out"
}
echo "phase=$PHASE node=$SLURM_JOB_NODELIST tasks=$SLURM_NTASKS"

case "$PHASE" in
  opusmoke)  # 1 short mini-epoch: memory of the surface decoder x3.2 and samples/s
    run uv run --offline train_continue --from-run-id "$BASE_ID" "${base_epoch[@]}" --run-id ocean_pu_smoke \
      --config "$CFG/finetune_ocean_pu.yml" \
      --options training_config.num_mini_epochs=1 training_config.samples_per_mini_epoch=64 \
                validation_config.samples_per_mini_epoch=8
    ;;
  opucheck)  # d5xirkmf with the PU off and on, same dates: change at start (not the identity) + seams
    rm -rf "$WD/results/$BINF_ID" "$WD/results/$P0INF_ID"
    run uv run --offline inference --from-run-id "$BASE_ID" "${base_epoch[@]}" --run-id "$BINF_ID" \
      --options "${SAME_DATES[@]}"
    run uv run --offline inference --from-run-id "$BASE_ID" "${base_epoch[@]}" --run-id "$P0INF_ID" \
      --options "${SAME_DATES[@]}" "${PU_ON[@]}"
    uv run --offline python scripts/compare_inference_zips.py --streams "$STREAM" --tol 1e9 \
      "$(zip0 "$BINF_ID")" "$(zip0 "$P0INF_ID")" || true
    seams "$WD/results/$P0INF_ID/seams" "$BINF_ID:base" "$P0INF_ID:pu_start"
    ;;
  opu1)   # surface decoder only, PU on (finetune_ocean_pu.yml)
    run uv run --offline train_continue --from-run-id "$BASE_ID" "${base_epoch[@]}" --run-id "$OPU1_ID" \
      --config "$CFG/finetune_ocean_pu.yml" --options training_config.num_mini_epochs="$OPU1_EPOCHS"
    ;;
  opuinfer)  # the PU block comes from the json of the run
    rm -rf "$WD/results/$PINF_ID"
    run uv run --offline inference --from-run-id "$OPU1_ID" --mini-epoch "$OPU1_EPOCHS" --run-id "$PINF_ID" \
      --options "${SAME_DATES[@]}"
    ;;
  eval)  # FastEval (single process): base, PU at start, PU fine-tuned (runs that exist), with persistence
    mkdir -p config/evaluate
    EV="config/evaluate/eval_${PINF_ID}_vs_base.yml"
    uv run --offline python - "$CFG/evaluate/eval_ocean_pu.yml" "$EV" "$WD" \
      "$PINF_ID:d5xirkmf, PU on the surface (fine-tuned)" "$BINF_ID:d5xirkmf" \
      "$P0INF_ID:d5xirkmf, PU at start (no training)" <<'PY'
import os, sys, yaml
tpl, out, wd, *runs = sys.argv[1:]
txt = open(tpl).read()
def cfg(rid): return yaml.safe_load(txt.replace("__RUN__", rid).replace("__WD__", wd))
c = None
for r in runs:
    rid, label = r.split(":", 1)
    if not os.path.isdir(f"{wd}/results/{rid}"):
        print("skip (no results):", rid); continue
    rc = cfg(rid)
    if c is None: c = rc
    else: c["run_ids"][rid] = rc["run_ids"][rid]
    c["run_ids"][rid]["label"] = label
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
print("eval config:", out, list(c["run_ids"]))
PY
    uv run --offline evaluate --config "$EV"
    echo ">> scores and figures in $WD/results/<first run of the eval config>/plots/"
    ;;
  seams)  # seam diagnostic of every run that exists
    seams "$WD/results/seams_ocean" "$BINF_ID:base" "$P0INF_ID:pu_start" "$PINF_ID:pu"
    ;;
  *) echo "unknown PHASE=$PHASE"; exit 1 ;;
esac
