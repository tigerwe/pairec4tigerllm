#!/usr/bin/env bash
set -euo pipefail

CSV_PATH="${CSV_PATH:-/workspace/data/ctr_data_1M.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-dssm_full_out}"
DEVICE="${DEVICE:-cuda}"
TRAIN_ROWS="${TRAIN_ROWS:-0}"
VOCAB_ROWS="${VOCAB_ROWS:-0}"
PROFILE_ROWS="${PROFILE_ROWS:-0}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-3}"
PATIENCE="${PATIENCE:-2}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SEED="${SEED:-20260807}"
EVAL_QUERY_LIMIT="${EVAL_QUERY_LIMIT:-1000}"
EVAL_QUERY_BATCH_SIZE="${EVAL_QUERY_BATCH_SIZE:-32}"
MIN_RECALL_AT_50="${MIN_RECALL_AT_50:-0}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
START_STAGE="${START_STAGE:-train}"

test -s "$CSV_PATH" || { echo "ERROR: missing CSV: $CSV_PATH" >&2; exit 1; }
case "$START_STAGE" in
  train|export|evaluate) ;;
  *) echo "ERROR: START_STAGE must be train, export, or evaluate" >&2; exit 1 ;;
esac
if [[ "$START_STAGE" == "train" && -d "$OUTPUT_DIR" &&
      -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" &&
      "$ALLOW_EXISTING_OUTPUT" != "1" ]]; then
  echo "ERROR: output exists: $OUTPUT_DIR" >&2
  echo "Set ALLOW_EXISTING_OUTPUT=1 only when intentionally resuming." >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR/export"

echo "== DSSM full retraining configuration =="
echo "csv_path=$CSV_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "device=$DEVICE"
echo "train_rows=$TRAIN_ROWS vocab_rows=$VOCAB_ROWS profile_rows=$PROFILE_ROWS"
echo "batch_size=$BATCH_SIZE epochs=$EPOCHS patience=$PATIENCE"
echo "val_fraction=$VAL_FRACTION seed=$SEED"
echo "start_stage=$START_STAGE"

if [[ "$START_STAGE" == "train" ]]; then
  python -m training.dssm.train \
    --csv_path "$CSV_PATH" \
    --vocab_path "$OUTPUT_DIR/vocab.json" \
    --checkpoint_dir "$OUTPUT_DIR" \
    --train_rows "$TRAIN_ROWS" \
    --vocab_rows "$VOCAB_ROWS" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --val_fraction "$VAL_FRACTION" \
    --seed "$SEED" \
    --embed_dim 64 \
    --out_dim 64 \
    --learning_rate 1e-3 \
    --temperature 0.05 \
    --device "$DEVICE" \
    --log_every 100
fi

if [[ "$START_STAGE" != "evaluate" ]]; then
  test -s "$OUTPUT_DIR/dssm_model.pt" || {
    echo "ERROR: missing checkpoint for export: $OUTPUT_DIR/dssm_model.pt" >&2
    exit 1
  }
  python -m training.dssm.export_embeddings \
    --checkpoint "$OUTPUT_DIR/dssm_model.pt" \
    --vocab_path "$OUTPUT_DIR/vocab.json" \
    --csv_path "$CSV_PATH" \
    --out_dir "$OUTPUT_DIR/export" \
    --profile_rows "$PROFILE_ROWS" \
    --batch_size 8192 \
    --device "$DEVICE"
fi

for path in "$OUTPUT_DIR/dssm_model.pt" "$OUTPUT_DIR/vocab.json" \
  "$OUTPUT_DIR/export/item_vectors.npy" "$OUTPUT_DIR/export/item_ids.json"; do
  test -s "$path" || { echo "ERROR: missing evaluation input: $path" >&2; exit 1; }
done

python -m training.dssm.evaluate_retrieval \
  --csv_path "$CSV_PATH" \
  --checkpoint "$OUTPUT_DIR/dssm_model.pt" \
  --vocab_path "$OUTPUT_DIR/vocab.json" \
  --export_dir "$OUTPUT_DIR/export" \
  --output "$OUTPUT_DIR/retrieval_evaluation.json" \
  --query_limit "$EVAL_QUERY_LIMIT" \
  --query_batch_size "$EVAL_QUERY_BATCH_SIZE" \
  --val_fraction "$VAL_FRACTION" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --min_recall_at_50 "$MIN_RECALL_AT_50"

python - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
required = [
    root / "vocab.json",
    root / "dssm_model.pt",
    root / "training_summary.json",
    root / "retrieval_evaluation.json",
    root / "export/item_vectors.npy",
    root / "export/item_ids.json",
    root / "export/item_categories.json",
    root / "export/user_profiles.json",
]
for path in required:
    assert path.is_file() and path.stat().st_size > 0, path
training = json.loads((root / "training_summary.json").read_text())
evaluation = json.loads((root / "retrieval_evaluation.json").read_text())
assert training["status"] == "PASS", training
assert evaluation["status"] == "PASS", evaluation
print("DSSM_FULL_RETRAIN_ARTIFACTS_OK", "best_epoch=" + str(training["best_epoch"]))
PY
