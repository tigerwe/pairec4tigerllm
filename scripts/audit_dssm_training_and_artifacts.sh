#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
IMAGE="${IMAGE:-docker.io/library/zcx-pairec-image:v1.1}"
DSSM_DIR="${DSSM_DIR:-$REPO_DIR/dssm_out}"
DEEPFM_DIR="${DEEPFM_DIR:-$REPO_DIR/deepfm_out}"
CSV_PATH="${CSV_PATH:-$REPO_DIR/data/ctr_data_1M.csv}"
CSV_SCAN_ROWS="${CSV_SCAN_ROWS:-5000000}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/dssm-training-audit-$(date +%Y%m%d-%H%M%S)}"

for path in \
  "$DSSM_DIR/vocab.json" \
  "$DSSM_DIR/dssm_model.pt" \
  "$DSSM_DIR/export/item_vectors.npy" \
  "$DSSM_DIR/export/item_ids.json" \
  "$DSSM_DIR/export/item_categories.json" \
  "$DSSM_DIR/export/user_profiles.json"; do
  test -s "$path" || { echo "ERROR: missing DSSM artifact: $path" >&2; exit 1; }
done
test -s "$CSV_PATH" || { echo "ERROR: missing CSV: $CSV_PATH" >&2; exit 1; }
test -s "$DEEPFM_DIR/training_summary.json" || {
  echo "ERROR: missing DeepFM full-data evidence: $DEEPFM_DIR/training_summary.json" >&2
  exit 1
}
test -s "$DEEPFM_DIR/item_categories.json" || {
  echo "ERROR: missing DeepFM item universe: $DEEPFM_DIR/item_categories.json" >&2
  exit 1
}

mkdir -p "$OUTPUT_DIR"

relative_path() {
  local path="$1"
  case "$path" in
    "$REPO_DIR"/*) printf '/workspace/%s' "${path#"$REPO_DIR"/}" ;;
    *) return 1 ;;
  esac
}

CONTAINER_DSSM_DIR="$(relative_path "$DSSM_DIR")" || {
  echo "ERROR: DSSM_DIR must be under REPO_DIR" >&2
  exit 1
}
CONTAINER_DEEPFM_DIR="$(relative_path "$DEEPFM_DIR")" || {
  echo "ERROR: DEEPFM_DIR must be under REPO_DIR" >&2
  exit 1
}
CONTAINER_CSV_PATH="$(relative_path "$CSV_PATH")" || {
  echo "ERROR: CSV_PATH must be under REPO_DIR" >&2
  exit 1
}

echo "== DSSM audit configuration =="
echo "repo_dir=$REPO_DIR"
echo "image=$IMAGE"
echo "dssm_dir=$DSSM_DIR"
echo "deepfm_dir=$DEEPFM_DIR"
echo "csv_path=$CSV_PATH"
echo "csv_scan_rows=$CSV_SCAN_ROWS"
echo "output_dir=$OUTPUT_DIR"

docker image inspect "$IMAGE" >/dev/null
docker run --rm -i \
  -v "$REPO_DIR:/workspace:ro" \
  -v "$OUTPUT_DIR:/audit-output" \
  --entrypoint env \
  "$IMAGE" \
  -u LD_PRELOAD \
  python /workspace/scripts/audit_dssm_training_and_artifacts.py \
    --dssm-dir "$CONTAINER_DSSM_DIR" \
    --deepfm-dir "$CONTAINER_DEEPFM_DIR" \
    --csv-path "$CONTAINER_CSV_PATH" \
    --csv-scan-rows "$CSV_SCAN_ROWS" \
    --pipeline-script /workspace/scripts/run_dssm_train_and_export.sh \
    --output /audit-output/report.json

echo "HOST_REPORT=$OUTPUT_DIR/report.json"
