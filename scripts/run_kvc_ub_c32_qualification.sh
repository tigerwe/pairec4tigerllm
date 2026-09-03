#!/usr/bin/env bash
set -euo pipefail

ITERATIONS="${ITERATIONS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/kvc-ub-c32-qualification/$(date +%Y%m%d-%H%M%S)}"
WORKER_HOST="${WORKER_HOST:-141.62.33.105}"
WORKER_PORT="${WORKER_PORT:-32501}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]] || die "ITERATIONS must be positive"
mkdir -p "$OUTPUT_DIR"

run_size() {
  local label="$1" bytes="$2" case_dir="$OUTPUT_DIR/$label"
  mkdir -p "$case_dir"
  env \
    WORKER_HOST="$WORKER_HOST" WORKER_PORT="$WORKER_PORT" \
    CONCURRENCY=32 ITERATIONS="$ITERATIONS" OBJECT_SIZE="$bytes" \
    PREFIX="PairecKvcUbC32_${label}_$(date +%Y%m%d_%H%M%S)_$$" \
    LOG_DIR="$case_dir/client-log" OUTPUT_DIR="$case_dir/evidence" \
    bash scripts/run_kvc_ub_integrity_probe.sh | tee "$case_dir/console.log"
}

run_size generation_3_5mib 3670016
run_size rank_8mib 8388608

grep -Fq 'concurrency=32 object_size=3670016' "$OUTPUT_DIR/generation_3_5mib/console.log"
grep -Fq 'concurrency=32 object_size=8388608' "$OUTPUT_DIR/rank_8mib/console.log"
grep -Fq 'tcp=0' "$OUTPUT_DIR/generation_3_5mib/console.log"
grep -Fq 'tcp=0' "$OUTPUT_DIR/rank_8mib/console.log"
echo "KVC_UB_C32_DUAL_OBJECT_QUALIFICATION_PASS iterations=$ITERATIONS operations_per_size=$((ITERATIONS * 32))"
echo "output_dir=$OUTPUT_DIR"
