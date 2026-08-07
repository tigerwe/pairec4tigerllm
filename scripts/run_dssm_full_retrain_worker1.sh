#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
IMAGE="${IMAGE:-docker.io/library/zcx-pairec-image:v1.1}"
GPU_DEVICE="${GPU_DEVICE:-all}"
CSV_PATH="${CSV_PATH:-$REPO_DIR/data/ctr_data_1M.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/dssm_all_candidates_out}"

case "$CSV_PATH" in
  "$REPO_DIR"/*) CONTAINER_CSV="/workspace/${CSV_PATH#"$REPO_DIR"/}" ;;
  *) echo "ERROR: CSV_PATH must be under REPO_DIR" >&2; exit 1 ;;
esac
case "$OUTPUT_DIR" in
  "$REPO_DIR"/*) CONTAINER_OUTPUT="/workspace/${OUTPUT_DIR#"$REPO_DIR"/}" ;;
  *) echo "ERROR: OUTPUT_DIR must be under REPO_DIR" >&2; exit 1 ;;
esac

gpu_args=(--gpus all)
if [[ "$GPU_DEVICE" != "all" ]]; then
  gpu_args=(--gpus "device=$GPU_DEVICE")
fi

echo "== Worker1 DSSM full retraining =="
echo "repo_dir=$REPO_DIR"
echo "image=$IMAGE"
echo "gpu_device=$GPU_DEVICE"
echo "csv_path=$CSV_PATH"
echo "output_dir=$OUTPUT_DIR"

docker image inspect "$IMAGE" >/dev/null
docker run --rm -i \
  "${gpu_args[@]}" \
  --ipc host \
  -e LD_PRELOAD= \
  -e CSV_PATH="$CONTAINER_CSV" \
  -e OUTPUT_DIR="$CONTAINER_OUTPUT" \
  -e DEVICE="${DEVICE:-cuda}" \
  -e TRAIN_ROWS="${TRAIN_ROWS:-0}" \
  -e VOCAB_ROWS="${VOCAB_ROWS:-0}" \
  -e PROFILE_ROWS="${PROFILE_ROWS:-0}" \
  -e BATCH_SIZE="${BATCH_SIZE:-4096}" \
  -e EPOCHS="${EPOCHS:-3}" \
  -e PATIENCE="${PATIENCE:-2}" \
  -e VAL_FRACTION="${VAL_FRACTION:-0.1}" \
  -e SEED="${SEED:-20260807}" \
  -e EVAL_QUERY_LIMIT="${EVAL_QUERY_LIMIT:-1000}" \
  -e EVAL_QUERY_BATCH_SIZE="${EVAL_QUERY_BATCH_SIZE:-32}" \
  -e MIN_RECALL_AT_50="${MIN_RECALL_AT_50:-0}" \
  -e ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}" \
  -e START_STAGE="${START_STAGE:-train}" \
  -e CANDIDATE_MODE="${CANDIDATE_MODE:-all_rows}" \
  -v "$REPO_DIR:/workspace" \
  --workdir /workspace \
  --entrypoint bash \
  "$IMAGE" \
  scripts/run_dssm_full_retrain.sh
