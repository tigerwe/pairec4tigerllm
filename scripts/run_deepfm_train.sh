#!/usr/bin/env bash
set -euo pipefail

CSV_PATH="${CSV_PATH:-/home/workspace/zcx/pairec4tigerllm/data/ctr_data_1M.csv}"
DSSM_VOCAB_PATH="${DSSM_VOCAB_PATH:-dssm_out/vocab.json}"
OUTPUT_DIR="${OUTPUT_DIR:-deepfm_out}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-2}"
MAX_ROWS="${MAX_ROWS:-0}"

test -f "$CSV_PATH" || { echo "ERROR: missing CSV: $CSV_PATH" >&2; exit 1; }
test -f "$DSSM_VOCAB_PATH" || {
  echo "ERROR: missing DSSM vocab: $DSSM_VOCAB_PATH" >&2
  exit 1
}

python -m training.deepfm.train \
  --csv_path "$CSV_PATH" \
  --vocab_path "$DSSM_VOCAB_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --max_rows "$MAX_ROWS"

python - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
required = [
    "deepfm_best.pt", "feature_vocab.json", "model_config.json",
    "training_summary.json", "item_categories.json", "user_profiles.json",
]
for name in required:
    path = root / name
    assert path.is_file() and path.stat().st_size > 0, path
summary = json.loads((root / "training_summary.json").read_text())
assert summary["status"] == "PASS", summary
print("DEEPFM_TRAINING_ARTIFACTS_OK", "best_epoch=" + str(summary["best_epoch"]))
PY
