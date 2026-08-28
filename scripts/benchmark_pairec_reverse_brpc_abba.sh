#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
PRIME_REQUESTS="${PRIME_REQUESTS:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pairec-reverse-brpc-abba/$(date +%Y%m%d-%H%M%S)}"
PATTERN="A,B,B,A,A,B,B,A,A,B,B,A,A,B,B,A,A,B,B,A"

mkdir -p "$OUTPUT_DIR"

echo "== Deploy master return pressure Sinks =="
NAMESPACE="$NAMESPACE" bash scripts/k8s_apply_brpc_return_pressure_sinks_master.sh \
  | tee "$OUTPUT_DIR/sinks-deploy.log"

echo "== Deploy Rank KVC c32 infrastructure once =="
NAMESPACE="$NAMESPACE" RANK_KVC_CONCURRENCY=32 RANK_KVC_PRESSURE_KEY_COUNT=4 \
  OUTPUT_DIR="$OUTPUT_DIR/rank-infrastructure" \
  bash scripts/deploy_deepfm_rank_burst_worker1.sh \
  | tee "$OUTPUT_DIR/rank-infrastructure.log"

common_env=(
  NAMESPACE="$NAMESPACE" WARMUP_REQUESTS=1 PRIME_REQUESTS="$PRIME_REQUESTS"
  BUILD_PAIREC_IMAGE=0 IMPORT_PAIREC_IMAGE=0
  WRAPPER_CONCURRENCY=1000 BURST_POOL_SIZE=10000 BURST_ACTIVE_CONNECTIONS=1000
  BUSINESS_PAYLOAD_BYTES=102400 BRPC_PRESSURE_PAYLOAD_BYTES=102400
  KVC_CONCURRENCY=32 KVC_PRESSURE_KEY_COUNT=4 KVC_OBJECT_SIZE=3670016
  KVC_PRESSURE_LEAD_US=1000 KVC_INPROCESS_PRESSURE=1
  EXPECTED_ONBOARDS_MIN=2 EXPECTED_ONBOARDS_MAX=2
  RANK_DEPLOYMENT=deepfm-rank-burst-wrapper
  RANK_SERVICE=deepfm-rank-burst-wrapper RANK_PORT=18213
  RANK_ENDPOINT_OVERRIDE=192.168.100.11:18213 RANK_TIMEOUT_MS=1500
  RANK_BURST_ENABLED=1 RANK_BURST_CONCURRENCY=1000 RANK_BURST_POOL_SIZE=1000
  RANK_BUSINESS_PAYLOAD_BYTES=102400 RANK_BURST_PAYLOAD_BYTES=102400
  RANK_BURST_PRESSURE_TIMEOUT_MS=5000
  RANK_KVC_ENABLED=1 RANK_KVC_CONCURRENCY=32 RANK_KVC_OBJECT_SIZE=8388608
  REVERSE_BURST_ENABLED=1 REVERSE_BURST_COMPLETION_TIMEOUT_SECONDS=15
)

echo "== Functional treatment smoke n1 =="
env "${common_env[@]}" REQUESTS=1 OUTPUT_DIR="$OUTPUT_DIR/smoke" \
  bash scripts/validate_pairec_brpc_wrapper_kvc_combined.sh \
  | tee "$OUTPUT_DIR/smoke.console.log"

echo "== Interleaved reverse BRPC A/B: five ABBA blocks =="
env "${common_env[@]}" REQUESTS=20 REVERSE_BURST_PATTERN="$PATTERN" \
  OUTPUT_DIR="$OUTPUT_DIR/abba" \
  bash scripts/validate_pairec_brpc_wrapper_kvc_combined.sh \
  | tee "$OUTPUT_DIR/abba.console.log"

python3 scripts/summarize_pairec_reverse_brpc_ab.py \
  --input "$OUTPUT_DIR/abba/summary.json" \
  --output "$OUTPUT_DIR/summary.json" \
  | tee "$OUTPUT_DIR/summary.txt"

echo "PAIREC_REVERSE_BRPC_FUNCTIONAL_AND_ABBA_OK"
echo "output_dir=$OUTPUT_DIR"
