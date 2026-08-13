#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
REQUESTS="${REQUESTS:-100}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
QUALIFICATION_REQUESTS="${QUALIFICATION_REQUESTS:-10}"
USER_ID="${USER_ID:-6312}"
POOL_SIZE="${POOL_SIZE:-10000}"
ACTIVE_CONNECTIONS="${ACTIVE_CONNECTIONS:-1000}"
PRESSURE_PAYLOAD_BYTES="${PRESSURE_PAYLOAD_BYTES:-102400}"
BURST_CPU_SHARDS="${BURST_CPU_SHARDS:-[]}"
MAX_RUNNER_P99_DELTA_MS="${MAX_RUNNER_P99_DELTA_MS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pairec-brpc-wrapper-runner-diagnosis/$(date +%Y%m%d-%H%M%S)-n${REQUESTS}}"
BUILD_PAIREC_IMAGE="${BUILD_PAIREC_IMAGE:-1}"
IMPORT_PAIREC_IMAGE="${IMPORT_PAIREC_IMAGE:-1}"

[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: REQUESTS must be positive" >&2; exit 1; }
[[ "$POOL_SIZE" = 10000 ]] || { echo "ERROR: diagnostic POOL_SIZE must be 10000" >&2; exit 1; }
[[ "$ACTIVE_CONNECTIONS" = 1000 ]] || { echo "ERROR: diagnostic ACTIVE_CONNECTIONS must be 1000" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"

run_arm() {
  local name="$1" concurrency="$2" active="$3" payload="$4" build="$5" import="$6"
  echo "== Diagnostic arm: $name =="
  echo "pool_size=$POOL_SIZE active_connections=$active payload_bytes=$payload"
  BURST_CONCURRENCY="$concurrency" \
  BURST_POOL_SIZE="$POOL_SIZE" \
  BURST_ACTIVE_CONNECTIONS="$active" \
  BURST_CPU_SHARDS="$BURST_CPU_SHARDS" \
  BURST_PAYLOAD_BYTES="$payload" \
  NAMESPACE="$NAMESPACE" \
  REQUESTS="$REQUESTS" \
  WARMUP_REQUESTS="$WARMUP_REQUESTS" \
  QUALIFICATION_REQUESTS="$QUALIFICATION_REQUESTS" \
  USER_ID="$USER_ID" \
  OUTPUT_DIR="$OUTPUT_DIR/$name" \
  BUILD_PAIREC_IMAGE="$build" \
  IMPORT_PAIREC_IMAGE="$import" \
    bash scripts/deploy_and_validate_pairec_brpc_wrapper_full.sh
}

echo "== BRPC Wrapper runner interference diagnosis =="
echo "requests=$REQUESTS qualification_requests=$QUALIFICATION_REQUESTS user_id=$USER_ID pool_size=$POOL_SIZE cpu_shards=$BURST_CPU_SHARDS"
echo "arms=idle_pool,rpc_only,payload_100k output_dir=$OUTPUT_DIR"

# Every arm preconnects the same 10,000 sessions before warmup or measurement.
run_arm idle_pool 1 1 0 "$BUILD_PAIREC_IMAGE" "$IMPORT_PAIREC_IMAGE"
run_arm rpc_only 1000 "$ACTIVE_CONNECTIONS" 0 0 0
run_arm payload_100k 1000 "$ACTIVE_CONNECTIONS" "$PRESSURE_PAYLOAD_BYTES" 0 0

python3 - "$OUTPUT_DIR/idle_pool/summary.json" "$OUTPUT_DIR/rpc_only/summary.json" \
  "$OUTPUT_DIR/payload_100k/summary.json" "$OUTPUT_DIR/summary.json" \
  "$MAX_RUNNER_P99_DELTA_MS" <<'PY'
import json,pathlib,sys
idle_path,rpc_path,payload_path,output_path,max_delta=sys.argv[1:]
arms={
    "idle_pool":json.load(open(idle_path)),
    "rpc_only":json.load(open(rpc_path)),
    "payload_100k":json.load(open(payload_path)),
}
for name,item in arms.items():
    assert item["pool_size"]==10000,(name,item.get("pool_size"))
idle=arms["idle_pool"]["metrics"]
rpc=arms["rpc_only"]["metrics"]
payload=arms["payload_100k"]["metrics"]
result={
    "classification":"PAIREC_BRPC_WRAPPER_RUNNER_INTERFERENCE_DIAGNOSIS",
    "samples_per_arm":len(arms["idle_pool"]["samples"]),
    "arms":{name:item["metrics"] for name,item in arms.items()},
    "runner_p99_rpc_only_delta_ms":rpc["runner_ms"]["p99"]-idle["runner_ms"]["p99"],
    "runner_p99_payload_increment_ms":payload["runner_ms"]["p99"]-rpc["runner_ms"]["p99"],
    "runner_p99_total_delta_ms":payload["runner_ms"]["p99"]-idle["runner_ms"]["p99"],
    "runner_p99_limit_ms":float(max_delta),
}
if result["runner_p99_rpc_only_delta_ms"] > result["runner_p99_payload_increment_ms"]:
    result["dominant_observed_factor"]="rpc_scheduling"
else:
    result["dominant_observed_factor"]="payload_processing"
result["runner_gate_passed"]=result["runner_p99_total_delta_ms"]<=float(max_delta)
pathlib.Path(output_path).write_text(json.dumps(result,indent=2)+"\n")
print("arm runner_p50_ms runner_p95_ms runner_p99_ms front_brpc_p99_ms start_skew_p99_us")
for name in ("idle_pool","rpc_only","payload_100k"):
    metrics=arms[name]["metrics"]
    print(name,
          round(metrics["runner_ms"]["p50"],3),
          round(metrics["runner_ms"]["p95"],3),
          round(metrics["runner_ms"]["p99"],3),
          round(metrics["front_brpc_ms"]["p99"],3),
          round(metrics["start_skew_us"]["p99"],3))
print(f"runner_p99_rpc_only_delta_ms={result['runner_p99_rpc_only_delta_ms']:.3f}")
print(f"runner_p99_payload_increment_ms={result['runner_p99_payload_increment_ms']:.3f}")
print(f"runner_p99_total_delta_ms={result['runner_p99_total_delta_ms']:.3f}")
print(f"dominant_observed_factor={result['dominant_observed_factor']}")
print(f"runner_gate_passed={str(result['runner_gate_passed']).lower()}")
PY

echo "summary_json=$OUTPUT_DIR/summary.json"
echo "PAIREC_BRPC_WRAPPER_RUNNER_DIAGNOSIS_COMPLETE"
