#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-pairec}
KVC_CONCURRENCY=${KVC_CONCURRENCY:-10}
KVC_PRESSURE_KEY_COUNT=${KVC_PRESSURE_KEY_COUNT:-9}
REQUESTS=${REQUESTS:-3}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-1}
BURST_POOL_SIZE=${BURST_POOL_SIZE:-10000}
BURST_ACTIVE_CONNECTIONS=${BURST_ACTIVE_CONNECTIONS:-1}
BURST_PAYLOAD_BYTES=${BURST_PAYLOAD_BYTES:-0}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/pairec-brpc-wrapper-kvc-combined/$(date +%Y%m%d-%H%M%S)-c${KVC_CONCURRENCY}-n${REQUESTS}}
KVC_OVERLAY_BACKUP=${KVC_OVERLAY_BACKUP:-$OUTPUT_DIR/kvc-deployment-before.json}
WRAPPER_OUTPUT_DIR=${WRAPPER_OUTPUT_DIR:-$OUTPUT_DIR/wrapper}
CONTENTION_OUTPUT_DIR=${CONTENTION_OUTPUT_DIR:-$OUTPUT_DIR/contention}

die() { echo "ERROR: $*" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
[[ "$KVC_CONCURRENCY" =~ ^(1|10|100)$ ]] || die "KVC_CONCURRENCY must be 1, 10, or 100"
[[ "$KVC_PRESSURE_KEY_COUNT" =~ ^[0-9]+$ ]] || die "KVC_PRESSURE_KEY_COUNT must be non-negative"
(( KVC_PRESSURE_KEY_COUNT <= KVC_CONCURRENCY - 1 )) \
  || die "KVC_PRESSURE_KEY_COUNT must not exceed KVC pressure lanes"
[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "REQUESTS must be positive"
[[ "$WARMUP_REQUESTS" =~ ^[0-9]+$ ]] || die "WARMUP_REQUESTS must be non-negative"

restore_kvc() {
  if [[ -f "$KVC_OVERLAY_BACKUP" ]]; then
    NAMESPACE="$NAMESPACE" DEPLOYMENT=inference-brpc-trtllm \
      BACKUP_FILE="$KVC_OVERLAY_BACKUP" \
      bash scripts/deploy_f14_kvc_burst_overlay.sh restore || true
  fi
}
trap restore_kvc EXIT INT TERM

echo "== Apply KVC c${KVC_CONCURRENCY} overlay =="
NAMESPACE="$NAMESPACE" DEPLOYMENT=inference-brpc-trtllm \
  BACKUP_FILE="$KVC_OVERLAY_BACKUP" CONCURRENCY="$KVC_CONCURRENCY" \
  PRESSURE_KEY_COUNT="$KVC_PRESSURE_KEY_COUNT" KVC_BURST_ENABLED=1 \
  KVC_BURST_VERBOSE=1 KVC_BURST_INITIAL_ARMED=0 \
  bash scripts/deploy_f14_kvc_burst_overlay.sh apply \
  | tee "$OUTPUT_DIR/kvc-overlay.log"

echo "== Deploy preconnected BRPC Wrapper full chain =="
set +e
NAMESPACE="$NAMESPACE" REQUESTS=1 WARMUP_REQUESTS="$WARMUP_REQUESTS" \
  BURST_CONCURRENCY=1 BURST_POOL_SIZE="$BURST_POOL_SIZE" \
  BURST_ACTIVE_CONNECTIONS="$BURST_ACTIVE_CONNECTIONS" \
  BURST_PAYLOAD_BYTES="$BURST_PAYLOAD_BYTES" \
  BUILD_PAIREC_IMAGE=0 IMPORT_PAIREC_IMAGE=0 \
  OUTPUT_DIR="$WRAPPER_OUTPUT_DIR" \
  bash scripts/deploy_and_validate_pairec_brpc_wrapper_full.sh \
  | tee "$OUTPUT_DIR/wrapper-console.log"
wrapper_code=${PIPESTATUS[0]}
set -e
[[ "$wrapper_code" -eq 0 ]] || die "BRPC Wrapper full-chain validation failed: exit=$wrapper_code"

echo "== Run KVC contention through the deployed BRPC Wrapper =="
STARTED_AT="$(date --iso-8601=seconds)"
set +e
NAMESPACE="$NAMESPACE" REPEATS="$REQUESTS" MODE=baseline \
  KVC_BURST_CONTAINER=kvc-burst-wrapper KVC_BURST_REQUIRE_COMPLETE=1 \
  KVC_BURST_DYNAMIC_ARM=1 KVC_BURST_PRESSURE_KEY_COUNT="$KVC_PRESSURE_KEY_COUNT" \
  EXPECTED_OFFLOADS=3 EXPECTED_ONBOARDS=2 STRICT_COUNTS=1 \
  REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION=1 PRIME_REQUESTS=195 \
  RESET_INFERENCE_BEFORE_ROUND=1 RESET_INFERENCE_MODE=container-runtime \
  PAIREC_TARGET=deploy/pairec-brpc-observed-wrapper \
  BRPC_TARGET=deployment/inference-brpc-trtllm \
  OUT_DIR="$CONTENTION_OUTPUT_DIR" \
  bash scripts/benchmark_brpc_kvc_contention.sh \
  | tee "$OUTPUT_DIR/contention-console.log"
contention_code=${PIPESTATUS[0]}
set -e
[[ "$contention_code" -eq 0 ]] || die "KVC contention through Wrapper failed: exit=$contention_code"

WRAPPER_POD="$(kubectl -n "$NAMESPACE" get pod -l app=brpc-burst-wrapper \
  --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')"
[[ -n "$WRAPPER_POD" ]] || die "BRPC Wrapper Pod not found"
kubectl -n "$NAMESPACE" logs "$WRAPPER_POD" -c brpc-burst-wrapper \
  --since-time="$STARTED_AT" >"$OUTPUT_DIR/wrapper-measured.log" 2>&1 || true

python3 - "$CONTENTION_OUTPUT_DIR/result.json" "$OUTPUT_DIR/wrapper-measured.log" "$OUTPUT_DIR/summary.json" <<'PY'
import json, pathlib, re, sys
contention = json.load(open(sys.argv[1]))
wrapper_log = pathlib.Path(sys.argv[2]).read_text(errors="replace")
valid = contention.get("valid_repeats") == contention.get("expected_repeats")
rows = []
for row in contention.get("rows", []):
    request_id = row.get("request_id", "")
    if not request_id:
        replay = pathlib.Path(row["summary_path"]).parent
        request_id = json.load(open(replay / "client.json"))["request_id"]
    matches = [line for line in wrapper_log.splitlines()
               if "method=Recommend" in line and f"request_id={request_id}" in line]
    if len(matches) != 1 or " code=200 " not in matches[0]:
        valid = False
    rows.append({"request_id": request_id, "wrapper_log_matches": len(matches)})
result = {
    "classification": "PAIREC_BRPC_WRAPPER_KVC_COMBINED_C10_OK" if valid else "PAIREC_BRPC_WRAPPER_KVC_COMBINED_FAIL",
    "contention": contention,
    "wrapper_request_evidence": rows,
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + "\n")
print(f"classification={result['classification']}")
print(f"requests={len(rows)}")
print(f"summary_json={sys.argv[3]}")
if not valid:
    raise SystemExit(1)
PY

echo "output_dir=$OUTPUT_DIR"
