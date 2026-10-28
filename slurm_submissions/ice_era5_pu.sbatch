#!/bin/bash -x
# Partition of unity (xp_05) on the ice + ERA5 base, WITHOUT the team launcher (no snapshot, no
# MLflow). One node, 4 GH200 GPUs (one task per GPU). Phases chained by submit_ice_era5_pu.sh:
#   pucheck (PU on the base, no training) | pu1 -> puinfer | pufm1 -> pufm2 -> pufinfer
#   | eval (after puinfer + pufinfer) | seams (after everything)
# Needs the base inference inf_ice_era5_base (phase binfer of ice_era5_fm.sbatch, same dates).
# Submit from a LOGIN node:  bash slurm_submissions/submit_ice_era5_pu.sh
# Logs: /e/scratch/weatherai/shared_work/logs/ice_era5_pu/<jobname>_<jobid>.out|.err
# Do not edit the code or the configs used here while the jobs run (no snapshot).
#SBATCH --job-name=ice_era5_pu
#SBATCH --account=e-ext-2025e01-128
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

set -euo pipefail
PHASE=${PHASE:?PHASE = pucheck | pu1 | puinfer | pufm1 | pufm2 | pufinfer | eval | seams}
WG=${WG:-/e/project1/weatherai/beauchamp1/WeatherGenerator}
CFG=../WeatherGenerator-configs/config
WD=/e/scratch/weatherai/shared_work
BASE_ID=${BASE_ID:-ice_era5_s2};      BASE_EPOCH=${BASE_EPOCH:-6}
PU1_ID=${PU1_ID:-ice_era5_pu1};       PU1_EPOCHS=${PU1_EPOCHS:-7}
PUFM1_ID=${PUFM1_ID:-ice_era5_pufm1}; PUFM1_EPOCHS=${PUFM1_EPOCHS:-7}
PUFM2_ID=${PUFM2_ID:-ice_era5_pufm2}; PUFM2_EPOCHS=${PUFM2_EPOCHS:-6}
BINF_ID=${BINF_ID:-inf_ice_era5_base}   # deterministic base (ice_era5_fm.sbatch, phase binfer)
FINF_ID=${FINF_ID:-inf_ice_era5_fm}     # FM without PU (ice_era5_fm.sbatch, phase finfer)
P0INF_ID=${P0INF_ID:-inf_ice_era5_pu0}  # PU on the base, before any training
PINF_ID=${PINF_ID:-inf_ice_era5_pu}; PFINF_ID=${PFINF_ID:-inf_ice_era5_pufm}
# all inferences on the SAME 32 start dates of 2023 as binfer (no shuffling)
SAME_DATES=(test_config.shuffle=False test_config.samples_per_mini_epoch=32 test_config.output.num_samples=32)
PU_ON=(partition_of_unity.enabled=True "partition_of_unity.streams=[CICE]")

cd "$WG"
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
if [[ "$SLURM_JOB_NUM_NODES" == "1" ]]; then export MASTER_ADDR=127.0.0.1
else export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1); fi
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
run() {  # one task per GPU (Slurm binds one GPU per task -> it is cuda:0 for the task);
       # a task that fails stops the others (no NCCL hang until the time limit)
  srun --kill-on-bad-exit=1 bash -c '
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]]; then export LOCAL_RANK=0
    else export LOCAL_RANK=$SLURM_LOCALID; fi
    export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS
    echo "task $SLURM_PROCID: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} LOCAL_RANK=$LOCAL_RANK"
    exec "$@"' _ "$@"
}
zip0() { ls "$WD/results/$1/"validation_chkpt*_rank0000.zip 2>/dev/null | head -n 1; }
echo "phase=$PHASE node=$SLURM_JOB_NODELIST tasks=$SLURM_NTASKS"

case "$PHASE" in
  pucheck)  # PU switched on in the trained base: change at start (not the identity) + seams
    rm -rf "$WD/results/$P0INF_ID"
    run uv run --offline inference --from-run-id "$BASE_ID" --mini-epoch "$BASE_EPOCH" --run-id "$P0INF_ID" \
      --config "$CFG/inference_ice_era5.yml" --options "${SAME_DATES[@]}" "${PU_ON[@]}"
    uv run --offline python scripts/compare_inference_zips.py --streams CICE --tol 1e9 \
      "$(zip0 "$BINF_ID")" "$(zip0 "$P0INF_ID")" || true
    uv run --offline python scripts/seam_diagnostic.py "$(zip0 "$BINF_ID")" "$(zip0 "$P0INF_ID")" \
      --labels base pu_start --stream CICE --channels aice hi --out-dir "$WD/results/$P0INF_ID/seams" \
      --maps --zoom -20 60 70 85
    ;;
  pu1)   # CICE decoder only, PU on (finetune_ice_era5_pu.yml)
    run uv run --offline train_continue --from-run-id "$BASE_ID" --mini-epoch "$BASE_EPOCH" --run-id "$PU1_ID" \
      --config "$CFG/finetune_ice_era5_pu.yml" --options training_config.num_mini_epochs="$PU1_EPOCHS"
    ;;
  puinfer)
    rm -rf "$WD/results/$PINF_ID"
    run uv run --offline inference --from-run-id "$PU1_ID" --mini-epoch "$PU1_EPOCHS" --run-id "$PINF_ID" \
      --config "$CFG/inference_ice_era5.yml" --options "${SAME_DATES[@]}"
    ;;
  pufm1)  # flow matching + PU: CICE decoder + FM conditioning (configs of the FM chain + PU)
    run uv run --offline train_continue --from-run-id "$BASE_ID" --mini-epoch "$BASE_EPOCH" --run-id "$PUFM1_ID" \
      --config "$CFG/finetune_cice_fm_phase1.yml" \
      --options training_config.num_mini_epochs="$PUFM1_EPOCHS" "${PU_ON[@]}"
    ;;
  pufm2)
    run uv run --offline train_continue --from-run-id "$PUFM1_ID" --mini-epoch "$PUFM1_EPOCHS" --run-id "$PUFM2_ID" \
      --config "$CFG/finetune_cice_fm_phase2.yml" \
      --options training_config.num_mini_epochs="$PUFM2_EPOCHS" "${PU_ON[@]}"
    ;;
  pufinfer)  # 8 members x 16 Heun steps, same dates as binfer
    rm -rf "$WD/results/$PFINF_ID"
    run uv run --offline inference --from-run-id "$PUFM2_ID" --mini-epoch "$PUFM2_EPOCHS" --run-id "$PFINF_ID" \
      --config "$CFG/inference_cice_fm.yml" --options "${SAME_DATES[@]}"
    ;;
  eval)  # FastEval (single process): base, PU, FM, PU + FM (runs that exist), with persistence
    mkdir -p config/evaluate
    EV="config/evaluate/eval_${PINF_ID}_vs_base.yml"
    uv run --offline python - "$CFG/evaluate/eval_ice_era5.yml" "$EV" "$WD" \
      "$PINF_ID:ice + ERA5, partition of unity" "$BINF_ID:ice + ERA5, deterministic base" \
      "$FINF_ID:ice + ERA5, flow matching (8 members)" "$PFINF_ID:ice + ERA5, PU + flow matching (8 members)" <<'PY'
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
    echo ">> scores and figures in $WD/results/$PINF_ID/plots/"
    ;;
  seams)  # seam diagnostic of every run that exists (ensemble mean + 2 members for FM)
    zips=(); labels=()
    for pair in "$BINF_ID:base" "$P0INF_ID:pu_start" "$PINF_ID:pu" "$FINF_ID:fm" "$PFINF_ID:pu_fm"; do
      z=$(zip0 "${pair%%:*}"); [[ -n "$z" ]] && { zips+=("$z"); labels+=("${pair##*:}"); }
    done
    uv run --offline python scripts/seam_diagnostic.py "${zips[@]}" --labels "${labels[@]}" \
      --stream CICE --channels aice hi --out-dir "$WD/results/seams_ice_era5" \
      --maps --zoom -20 60 70 85   # side-by-side maps; zoom: Fram Strait - Barents Sea
    ;;
  *) echo "unknown PHASE=$PHASE"; exit 1 ;;
esac
