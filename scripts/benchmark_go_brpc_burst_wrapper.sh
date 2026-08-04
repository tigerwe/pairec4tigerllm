#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
INFERENCE_SERVICE="${INFERENCE_SERVICE:-inference-brpc-trtllm}"
INFERENCE_APP="${INFERENCE_APP:-inference-brpc-trtllm}"
WRAPPER_APP="${WRAPPER_APP:-brpc-burst-wrapper}"
BRPC_PORT="${BRPC_PORT:-18100}"
ENDPOINT="${ENDPOINT:-}"
BURST_CONCURRENCY_LEVELS="${BURST_CONCURRENCY_LEVELS:-10 100 1000}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"
REPEATS="${REPEATS:-1000}"
PRESSURE_PAYLOAD_BYTES="${PRESSURE_PAYLOAD_BYTES:-102400}"
BUSINESS_PAYLOAD_BYTES="${BUSINESS_PAYLOAD_BYTES:-0}"
MIN_ACTIVE_RATIO="${MIN_ACTIVE_RATIO:-0.95}"
REQUIRE_SERVER_WRAPPER="${REQUIRE_SERVER_WRAPPER:-0}"
TIMEOUT_MS="${TIMEOUT_MS:-5000}"
MAX_RETRIES="${MAX_RETRIES:-0}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-0}"
CHECK_K8S_STATE="${CHECK_K8S_STATE:-1}"
HISTORY_SOURCE="${HISTORY_SOURCE:-user_features}"
USER_ID="${USER_ID:-5}"
UIDS="${UIDS:-$USER_ID}"
TOPK="${TOPK:-1}"
HISTORY_MAX_LENGTH="${HISTORY_MAX_LENGTH:-20}"
USER_FEATURES_PATH="${USER_FEATURES_PATH:-data/user_features.json}"
SEMANTIC_MAP_PATH="${SEMANTIC_MAP_PATH:-data/tenrec/processed/semantic_id_map.json}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/go_brpc_burst_wrapper/${RUN_ID}}"
PROBE_BIN="${OUT_DIR}/probe_go_brpc_client"
SUMMARY_JSON="${OUT_DIR}/summary.json"
RUN_INDEX="${OUT_DIR}/runs.tsv"
STARTED_AT="$(date --iso-8601=seconds)"

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing command: $1" >&2; exit 1; }
}

validate_positive_integer() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: ${name} must be a positive integer: ${value}" >&2
    exit 2
  }
}

resolve_endpoint() {
  [ -n "$ENDPOINT" ] && return
  local service_ip
  service_ip="$(kubectl -n "$NAMESPACE" get svc "$INFERENCE_SERVICE" -o jsonpath='{.spec.clusterIP}')"
  if [ -z "$service_ip" ] || [ "$service_ip" = "None" ]; then
    echo "ERROR: service ${NAMESPACE}/${INFERENCE_SERVICE} has no ClusterIP" >&2
    exit 1
  fi
  ENDPOINT="${service_ip}:${BRPC_PORT}"
}

capture_pod_state() {
  local output="$1" app="$2"
  kubectl -n "$NAMESPACE" get pods -l "app=${app}" -o json |
    python3 -c '
import json, sys
pods = json.load(sys.stdin).get("items", [])
ready = []
for pod in pods:
    statuses = pod.get("status", {}).get("containerStatuses", [])
    if pod.get("status", {}).get("phase") == "Running" and statuses and all(s.get("ready") for s in statuses):
        ready.append((pod, statuses))
if len(ready) != 1:
    raise SystemExit(f"expected one running inference pod, found {len(ready)}")
pod, statuses = ready[0]
print("pod=" + pod["metadata"]["name"])
print("uid=" + pod["metadata"]["uid"])
print("restarts=" + str(sum(s.get("restartCount", 0) for s in statuses)))
' >"$output"
}

mkdir -p "$OUT_DIR/runs"
require_command go
require_command python3
if [ -z "$ENDPOINT" ] || [ "$CHECK_K8S_STATE" = "1" ]; then
  require_command kubectl
fi
resolve_endpoint
validate_positive_integer REPEATS "$REPEATS"
case "$REQUIRE_SERVER_WRAPPER" in
  0|1) ;;
  *) echo "ERROR: REQUIRE_SERVER_WRAPPER must be 0 or 1" >&2; exit 2 ;;
esac

BURST_CONCURRENCY_LEVELS="${BURST_CONCURRENCY_LEVELS//,/ }"
read -r -a LEVELS <<<"$BURST_CONCURRENCY_LEVELS"
if [ "$INCLUDE_BASELINE" = "1" ]; then
  LEVELS+=(1)
fi
for concurrency in "${LEVELS[@]}"; do
  validate_positive_integer BURST_CONCURRENCY_LEVEL "$concurrency"
done
mapfile -t LEVELS < <(printf '%s\n' "${LEVELS[@]}" | sort -n -u)

if [ "$CHECK_K8S_STATE" = "1" ]; then
  capture_pod_state "${OUT_DIR}/inference-pod.before" "$INFERENCE_APP"
  if [ "$REQUIRE_SERVER_WRAPPER" = "1" ]; then
    capture_pod_state "${OUT_DIR}/wrapper-pod.before" "$WRAPPER_APP"
  fi
fi

echo "== Build Go BRPC burst probe =="
GOPROXY="${GOPROXY:-off}" GOSUMDB="${GOSUMDB:-off}" \
  go build -mod=vendor -o "$PROBE_BIN" ./scripts/probe_go_brpc_client.go

echo "== BRPC burst p99 benchmark =="
echo "endpoint=${ENDPOINT}"
echo "levels=${LEVELS[*]} repeats=${REPEATS} pressure_payload_bytes=${PRESSURE_PAYLOAD_BYTES}"
if [ "$REPEATS" -lt 1000 ]; then
  echo "WARNING: REPEATS=${REPEATS} is a smoke sample; use REPEATS>=1000 for a formal p99 result" >&2
fi
printf 'concurrency\trepeat\texit_code\tresult_json\tlog\n' >"$RUN_INDEX"

for concurrency in "${LEVELS[@]}"; do
  mkdir -p "${OUT_DIR}/runs/c${concurrency}"
done

# Interleave the baseline and pressure levels to reduce time/temperature drift.
for repeat in $(seq 1 "$REPEATS"); do
  for concurrency in "${LEVELS[@]}"; do
    level_dir="${OUT_DIR}/runs/c${concurrency}"
    result_json="${level_dir}/run-$(printf '%04d' "$repeat").json"
    run_log="${level_dir}/run-$(printf '%04d' "$repeat").log"
    echo "-- concurrency=${concurrency} repeat=${repeat}/${REPEATS} --"
    set +e
    "$PROBE_BIN" \
      --endpoint="$ENDPOINT" \
      --method=burst \
      --burst_concurrency="$concurrency" \
      --pressure_payload_bytes="$PRESSURE_PAYLOAD_BYTES" \
      --business_payload_bytes="$BUSINESS_PAYLOAD_BYTES" \
      --history_source="$HISTORY_SOURCE" \
      --user_id="$USER_ID" \
      --uids="$UIDS" \
      --user_features_path="$USER_FEATURES_PATH" \
      --semantic_map_path="$SEMANTIC_MAP_PATH" \
      --history_max_length="$HISTORY_MAX_LENGTH" \
      --topk="$TOPK" \
      --timeout_ms="$TIMEOUT_MS" \
      --max_retries="$MAX_RETRIES" \
      --quiet=true \
      --result_json="$result_json" \
      >"$run_log" 2>&1
    run_status="$?"
    set -e
    cat "$run_log"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$concurrency" "$repeat" "$run_status" "$result_json" "$run_log" >>"$RUN_INDEX"
    if [ "$COOLDOWN_SECONDS" != "0" ]; then
      sleep "$COOLDOWN_SECONDS"
    fi
  done
done

K8S_STATE_OK=1
CRASH_MARKER_COUNT=0
if [ "$CHECK_K8S_STATE" = "1" ]; then
  capture_pod_state "${OUT_DIR}/inference-pod.after" "$INFERENCE_APP"
  if ! cmp -s "${OUT_DIR}/inference-pod.before" "${OUT_DIR}/inference-pod.after"; then
    K8S_STATE_OK=0
    echo "ERROR: inference pod identity or restart count changed" >&2
    diff -u "${OUT_DIR}/inference-pod.before" "${OUT_DIR}/inference-pod.after" || true
  fi
  if [ "$REQUIRE_SERVER_WRAPPER" = "1" ]; then
    capture_pod_state "${OUT_DIR}/wrapper-pod.after" "$WRAPPER_APP"
    if ! cmp -s "${OUT_DIR}/wrapper-pod.before" "${OUT_DIR}/wrapper-pod.after"; then
      K8S_STATE_OK=0
      echo "ERROR: wrapper pod identity or restart count changed" >&2
      diff -u "${OUT_DIR}/wrapper-pod.before" "${OUT_DIR}/wrapper-pod.after" || true
    fi
    wrapper_pod="$(awk -F= '$1 == "pod" {print $2}' "${OUT_DIR}/wrapper-pod.after")"
    kubectl -n "$NAMESPACE" logs "$wrapper_pod" --since-time="$STARTED_AT" >"${OUT_DIR}/wrapper.log" 2>&1 || true
    wrapper_crashes="$(grep -Eic 'Segmentation|core dumped|Out of memory|terminate called' "${OUT_DIR}/wrapper.log" || true)"
    CRASH_MARKER_COUNT=$((CRASH_MARKER_COUNT + wrapper_crashes))
  fi
  pod_name="$(awk -F= '$1 == "pod" {print $2}' "${OUT_DIR}/inference-pod.after")"
  kubectl -n "$NAMESPACE" logs "$pod_name" --since-time="$STARTED_AT" >"${OUT_DIR}/inference.log" 2>&1 || true
  inference_crashes="$(grep -Eic 'EngineCore failed|Segmentation|core dumped|Out of memory|terminate called' "${OUT_DIR}/inference.log" || true)"
  CRASH_MARKER_COUNT=$((CRASH_MARKER_COUNT + inference_crashes))
  if [ "$CRASH_MARKER_COUNT" -ne 0 ]; then
    K8S_STATE_OK=0
    echo "ERROR: inference/wrapper crash markers detected: ${CRASH_MARKER_COUNT}" >&2
  fi
fi

python3 - "$RUN_INDEX" "$SUMMARY_JSON" "$REPEATS" "$MIN_ACTIVE_RATIO" \
  "$K8S_STATE_OK" "$CRASH_MARKER_COUNT" "$PRESSURE_PAYLOAD_BYTES" \
  "$BUSINESS_PAYLOAD_BYTES" "$ENDPOINT" "$REQUIRE_SERVER_WRAPPER" "${LEVELS[@]}" <<'PY'
import json
import math
import pathlib
import sys

index_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
repeats = int(sys.argv[3])
min_active_ratio = float(sys.argv[4])
k8s_state_ok = sys.argv[5] == "1"
crash_marker_count = int(sys.argv[6])
pressure_payload_bytes = int(sys.argv[7])
business_payload_bytes = int(sys.argv[8])
endpoint = sys.argv[9]
require_server_wrapper = sys.argv[10] == "1"
levels = [int(value) for value in sys.argv[11:]]

def percentile(values, quantile):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)

def metric(values):
    return {
        "count": len(values),
        "avg_ms": sum(values) / len(values) if values else None,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "p999_ms": percentile(values, 0.999),
        "max_ms": max(values) if values else None,
    }

indexed = {level: [] for level in levels}
for line in index_path.read_text(encoding="utf-8").splitlines()[1:]:
    concurrency, repeat, exit_code, result_path, log_path = line.split("\t")
    result = None
    path = pathlib.Path(result_path)
    if path.exists():
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    indexed[int(concurrency)].append({
        "repeat": int(repeat), "exit_code": int(exit_code), "result": result,
        "result_path": result_path, "log_path": log_path,
    })

cases = []
for concurrency in levels:
    records = indexed[concurrency]
    results = [record["result"] for record in records if record["result"] is not None]
    successful = [result for result in results if result.get("business_success")]
    pressure_ok = sum(
        int(result.get("pressure_success", -1)) == int(result.get("pressure_requests", -2))
        for result in results
    )
    threshold = math.ceil(concurrency * min_active_ratio)
    active_ok = sum(int(result.get("max_active_workers", 0)) >= threshold for result in results)
    armed_ok = sum(int(result.get("armed_workers", 0)) == concurrency for result in results)
    wrapper_trace_ok = sum(float(result.get("wrapper_total_ms", 0.0)) > 0.0 for result in successful)
    wrapper_overlap_ok = sum(
        int(result.get("wrapper_max_active_total", 0)) >= threshold for result in successful
    )
    process_ok = sum(record["exit_code"] == 0 and record["result"] is not None for record in records)
    valid = (
        len(records) == repeats and len(results) == repeats and process_ok == repeats
        and len(successful) == repeats and pressure_ok == repeats
        and active_ok == repeats and armed_ok == repeats and k8s_state_ok
        and (not require_server_wrapper or (
            wrapper_trace_ok == repeats and wrapper_overlap_ok == repeats
        ))
    )
    cases.append({
        "concurrency": concurrency,
        "planned_repeats": repeats,
        "result_count": len(results),
        "process_success": process_ok,
        "business_success": len(successful),
        "business_failure": repeats - len(successful),
        "full_pressure_success": pressure_ok,
        "active_pass": active_ok,
        "armed_pass": armed_ok,
        "active_threshold": threshold,
        "wrapper_required": require_server_wrapper,
        "wrapper_trace_pass": wrapper_trace_ok,
        "wrapper_overlap_pass": wrapper_overlap_ok,
        "wrapper_max_active_total_min": min(
            (int(r.get("wrapper_max_active_total", 0)) for r in successful), default=0
        ),
        "sample_sufficient_for_p99": len(successful) >= 1000,
        "successful_requests_only": True,
        "client_wall": metric([float(r["business_client_wall_ms"]) for r in successful]),
        "inference": metric([float(r["business_inference_ms"]) for r in successful]),
        "runner_generate": metric([float(r.get("business_runner_generate_ms", 0.0)) for r in successful]),
        "brpc_delta": metric([float(r["business_brpc_delta_ms"]) for r in successful]),
        "front_brpc": metric([float(r.get("business_front_brpc_ms", 0.0)) for r in successful]),
        "wrapper_total": metric([float(r.get("wrapper_total_ms", 0.0)) for r in successful]),
        "wrapper_backend_rpc": metric([float(r.get("wrapper_backend_rpc_ms", 0.0)) for r in successful]),
        "wrapper_overhead": metric([float(r.get("wrapper_overhead_ms", 0.0)) for r in successful]),
        "wrapper_backend_brpc": metric([float(r.get("wrapper_backend_brpc_ms", 0.0)) for r in successful]),
        "valid": valid,
    })

baseline = next((case for case in cases if case["concurrency"] == 1 and case["valid"]), None)
for case in cases:
    for name in (
        "client_wall", "inference", "runner_generate", "brpc_delta", "front_brpc",
        "wrapper_total", "wrapper_backend_rpc", "wrapper_overhead", "wrapper_backend_brpc",
    ):
        current = case[name]["p99_ms"]
        base = baseline[name]["p99_ms"] if baseline else None
        delta = current - base if current is not None and base is not None else None
        pct = delta / base * 100 if delta is not None and base not in (None, 0) else None
        case[name]["p99_delta_vs_baseline_ms"] = delta
        case[name]["p99_delta_vs_baseline_pct"] = pct

all_valid = bool(cases) and all(case["valid"] for case in cases)
formal = all(case["sample_sufficient_for_p99"] for case in cases)
status = "PASS" if all_valid and formal else "PASS_SMOKE" if all_valid else "FAIL"
summary = {
    "event": "brpc_burst_p99_benchmark",
    "endpoint": endpoint,
    "pressure_payload_bytes": pressure_payload_bytes,
    "business_payload_bytes": business_payload_bytes,
    "minimum_active_ratio": min_active_ratio,
    "require_server_wrapper": require_server_wrapper,
    "latency_samples": "successful business Recommend requests only; failures are reported separately",
    "baseline_concurrency": 1 if baseline else None,
    "k8s_state_ok": k8s_state_ok,
    "crash_marker_count": crash_marker_count,
    "cases": cases,
    "status": status,
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def show(value):
    return "n/a" if value is None else f"{value:.3f}"

print("\n== BRPC burst p99 summary ==")
print("concurrency valid samples formal client_p99 inference_p99 runner_p99 inference_delta runner_delta front_brpc_p99 wrapper_p99 backend_rpc_p99 backend_brpc_p99 overlap_min failures status")
for case in cases:
    print(
        f"{case['concurrency']:>11} {str(case['valid']):>5} "
        f"{case['business_success']:>7}/{repeats:<4} "
        f"{str(case['sample_sufficient_for_p99']):>6} "
        f"{show(case['client_wall']['p99_ms']):>10} "
        f"{show(case['inference']['p99_ms']):>13} "
        f"{show(case['runner_generate']['p99_ms']):>10} "
        f"{show(case['inference']['p99_delta_vs_baseline_ms']):>15} "
        f"{show(case['runner_generate']['p99_delta_vs_baseline_ms']):>12} "
        f"{show(case['front_brpc']['p99_ms']):>14} "
        f"{show(case['wrapper_total']['p99_ms']):>11} "
        f"{show(case['wrapper_backend_rpc']['p99_ms']):>15} "
        f"{show(case['wrapper_backend_brpc']['p99_ms']):>16} "
        f"{case['wrapper_max_active_total_min']:>11} "
        f"{case['business_failure']:>8} "
        f"{'PASS' if case['valid'] else 'FAIL'}"
    )
print("note=p99 is measured from one business Recommend per synchronized burst; no latency target is enforced")
print("note=front_brpc is client wall minus Wrapper total; backend_brpc is Wrapper backend RPC minus backend inference")
print("note=inference/runner deltas versus concurrency=1 reveal backend inference interference")
print("note=overlap_min is the minimum server-observed active lane peak across successful bursts")
print(f"summary_json={summary_path}")
print(f"RESULT={status}")
PY

if [ "$K8S_STATE_OK" -ne 1 ]; then
  exit 1
fi
if ! python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["status"] != "FAIL" else 1)' "$SUMMARY_JSON"; then
  exit 1
fi

echo "BRPC_BURST_P99_BENCHMARK_COMPLETE"
echo "summary_json=${SUMMARY_JSON}"
