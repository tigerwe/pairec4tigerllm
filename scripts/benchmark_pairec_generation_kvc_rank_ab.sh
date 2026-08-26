#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-pairec}
REQUESTS=${REQUESTS:-1}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-1}
PRIME_REQUESTS=${PRIME_REQUESTS:-20}
BUILD_PAIREC_IMAGE=${BUILD_PAIREC_IMAGE:-0}
IMPORT_PAIREC_IMAGE=${IMPORT_PAIREC_IMAGE:-0}
DEPLOY_RANK_INFRA=${DEPLOY_RANK_INFRA:-1}
RANK_DIRECT_ENDPOINT=${RANK_DIRECT_ENDPOINT:-192.168.100.11:18213}
RANK_PRESSURE_TIMEOUT_MS=${RANK_PRESSURE_TIMEOUT_MS:-5000}
GENERATION_BURST_POOL_SIZE=${GENERATION_BURST_POOL_SIZE:-10000}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/pairec-generation-kvc-rank-ab/$(date +%Y%m%d-%H%M%S)-n${REQUESTS}}

die() { echo "ERROR: $*" >&2; exit 1; }
[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "REQUESTS must be positive"
[[ "$WARMUP_REQUESTS" = 1 ]] || die "exactly one excluded warmup is required"
[[ "$PRIME_REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "PRIME_REQUESTS must be positive"
[[ "$RANK_DIRECT_ENDPOINT" == *:* ]] || die "RANK_DIRECT_ENDPOINT must be host:port"
[[ "$GENERATION_BURST_POOL_SIZE" = 10000 ]] \
  || die "GENERATION_BURST_POOL_SIZE must remain 10000 for the strict A/B"
for flag in "$BUILD_PAIREC_IMAGE" "$IMPORT_PAIREC_IMAGE" "$DEPLOY_RANK_INFRA"; do
  [[ "$flag" = 0 || "$flag" = 1 ]] || die "boolean flags must be 0 or 1"
done
mkdir -p "$OUTPUT_DIR"

if [[ "$DEPLOY_RANK_INFRA" = 1 ]]; then
  OUTPUT_DIR="$OUTPUT_DIR/infrastructure" \
    bash scripts/deploy_deepfm_rank_burst_worker1.sh \
    | tee "$OUTPUT_DIR/infrastructure.console.log"
else
  kubectl -n "$NAMESPACE" rollout status deployment/deepfm-rank-brpc-worker1 --timeout=60s
  kubectl -n "$NAMESPACE" rollout status deployment/deepfm-rank-burst-wrapper --timeout=60s
fi

run_case() {
  local name=$1 rank_concurrency=$2 build=$3 import_image=$4
  echo "== Combined A/B case=$name Generative=c1000 KVC=c32 Rank=c${rank_concurrency} =="
  env \
    NAMESPACE="$NAMESPACE" REQUESTS="$REQUESTS" WARMUP_REQUESTS="$WARMUP_REQUESTS" \
    PRIME_REQUESTS="$PRIME_REQUESTS" OUTPUT_DIR="$OUTPUT_DIR/$name" \
    BUILD_PAIREC_IMAGE="$build" IMPORT_PAIREC_IMAGE="$import_image" \
    WRAPPER_CONCURRENCY=1000 BURST_POOL_SIZE="$GENERATION_BURST_POOL_SIZE" \
    BURST_ACTIVE_CONNECTIONS=1000 \
    BUSINESS_PAYLOAD_BYTES=102400 BRPC_PRESSURE_PAYLOAD_BYTES=102400 \
    KVC_CONCURRENCY=32 KVC_PRESSURE_KEY_COUNT=4 KVC_OBJECT_SIZE=3670016 \
    KVC_PRESSURE_LEAD_US=1000 KVC_INPROCESS_PRESSURE=1 \
    EXPECTED_ONBOARDS_MIN=2 EXPECTED_ONBOARDS_MAX=2 \
    RANK_DEPLOYMENT=deepfm-rank-burst-wrapper \
    RANK_SERVICE=deepfm-rank-burst-wrapper RANK_PORT=18213 \
    RANK_ENDPOINT_OVERRIDE="$RANK_DIRECT_ENDPOINT" \
    RANK_BURST_ENABLED=1 RANK_BURST_CONCURRENCY="$rank_concurrency" \
    RANK_BURST_POOL_SIZE="$rank_concurrency" \
    RANK_BUSINESS_PAYLOAD_BYTES=102400 RANK_BURST_PAYLOAD_BYTES=102400 \
    RANK_BURST_PRESSURE_TIMEOUT_MS="$RANK_PRESSURE_TIMEOUT_MS" \
    bash scripts/validate_pairec_brpc_wrapper_kvc_combined.sh \
    | tee "$OUTPUT_DIR/$name.console.log"
}

run_case rank_c1 1 "$BUILD_PAIREC_IMAGE" "$IMPORT_PAIREC_IMAGE"
run_case rank_c1000 1000 0 0

python3 scripts/summarize_pairec_generation_kvc_rank_ab.py \
  --rank-c1 "$OUTPUT_DIR/rank_c1/summary.json" \
  --rank-c1000 "$OUTPUT_DIR/rank_c1000/summary.json" \
  --output "$OUTPUT_DIR/summary.json" \
  | tee "$OUTPUT_DIR/summary.txt"

echo "output_dir=$OUTPUT_DIR"
