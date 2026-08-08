#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
IMAGE="${IMAGE:-docker.io/library/zcx-pairec-image:v1.1}"
GPU_DEVICE="${GPU_DEVICE:-all}"
CSV_PATH="${CSV_PATH:-$REPO_DIR/data/ctr_data_1M.csv}"
DSSM_VOCAB_PATH="${DSSM_VOCAB_PATH:-$REPO_DIR/dssm_all_candidates_out/vocab.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/deepfm_full_vocab_out}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"

to_container_path() {
  local path="$1" name="$2"
  case "$path" in
    "$REPO_DIR"/*) printf '/workspace/%s\n' "${path#"$REPO_DIR"/}" ;;
    *) echo "ERROR: $name must be under REPO_DIR" >&2; exit 1 ;;
  esac
}

[[ "$ALLOW_EXISTING_OUTPUT" = 0 || "$ALLOW_EXISTING_OUTPUT" = 1 ]] || {
  echo "ERROR: ALLOW_EXISTING_OUTPUT must be 0 or 1" >&2
  exit 1
}
test -f "$CSV_PATH" || { echo "ERROR: missing CSV: $CSV_PATH" >&2; exit 1; }
test -f "$DSSM_VOCAB_PATH" || {
  echo "ERROR: missing full DSSM vocab: $DSSM_VOCAB_PATH" >&2
  exit 1
}
if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  [[ "$ALLOW_EXISTING_OUTPUT" = 1 ]] || {
    echo "ERROR: output directory is not empty: $OUTPUT_DIR" >&2
    echo "Use a new directory or set ALLOW_EXISTING_OUTPUT=1 explicitly." >&2
    exit 1
  }
fi

CONTAINER_CSV="$(to_container_path "$CSV_PATH" CSV_PATH)"
CONTAINER_VOCAB="$(to_container_path "$DSSM_VOCAB_PATH" DSSM_VOCAB_PATH)"
CONTAINER_OUTPUT="$(to_container_path "$OUTPUT_DIR" OUTPUT_DIR)"

gpu_args=(--gpus all)
if [[ "$GPU_DEVICE" != all ]]; then
  gpu_args=(--gpus "device=$GPU_DEVICE")
fi

echo "== Worker1 DeepFM full-vocabulary training =="
echo "repo_dir=$REPO_DIR"
echo "image=$IMAGE"
echo "gpu_device=$GPU_DEVICE"
echo "csv_path=$CSV_PATH"
echo "dssm_vocab_path=$DSSM_VOCAB_PATH"
echo "output_dir=$OUTPUT_DIR"

docker image inspect "$IMAGE" >/dev/null
docker run --rm -i \
  "${gpu_args[@]}" \
  --ipc host \
  -e LD_PRELOAD= \
  -e CSV_PATH="$CONTAINER_CSV" \
  -e DSSM_VOCAB_PATH="$CONTAINER_VOCAB" \
  -e OUTPUT_DIR="$CONTAINER_OUTPUT" \
  -e DEVICE="${DEVICE:-cuda}" \
  -e BATCH_SIZE="${BATCH_SIZE:-4096}" \
  -e EPOCHS="${EPOCHS:-10}" \
  -e PATIENCE="${PATIENCE:-2}" \
  -e MAX_ROWS="${MAX_ROWS:-0}" \
  -v "$REPO_DIR:/workspace" \
  --workdir /workspace \
  --entrypoint bash \
  "$IMAGE" \
  scripts/run_deepfm_train.sh
