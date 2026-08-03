#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
INFERENCE_SERVICE="${INFERENCE_SERVICE:-inference-brpc-trtllm}"
INFERENCE_APP="${INFERENCE_APP:-inference-brpc-trtllm}"
BRPC_PORT="${BRPC_PORT:-18100}"
ENDPOINT="${ENDPOINT:-}"
BURST_CONCURRENCY_LEVELS="${BURST_CONCURRENCY_LEVELS:-10 25 50 100 200 400 600 800 1000}"
REPEATS="${REPEATS:-20}"
PRESSURE_PAYLOAD_BYTES="${PRESSURE_PAYLOAD_BYTES:-102400}"
BUSINESS_PAYLOAD_BYTES="${BUSINESS_PAYLOAD_BYTES:-0}"
TARGET_BRPC_DELTA_MIN_MS="${TARGET_BRPC_DELTA_MIN_MS:-30}"
TARGET_BRPC_DELTA_MAX_MS="${TARGET_BRPC_DELTA_MAX_MS:-40}"
MIN_ACTIVE_RATIO="${MIN_ACTIVE_RATIO:-0.95}"
AUTO_REFINE="${AUTO_REFINE:-1}"
TIMEOUT_MS="${TIMEOUT_MS:-5000}"
MAX_RETRIES="${MAX_RETRIES:-0}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-1}"
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
SELECTED_ENV="${OUT_DIR}/selected.env"
RUN_INDEX="${OUT_DIR}/runs.tsv"
STARTED_AT="$(date --iso-8601=seconds)"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: missing command: $1" >&2
    exit 1
  fi
}

resolve_endpoint() {
  if [ -n "$ENDPOINT" ]; then
    return
  fi
  local service_ip
  service_ip="$(kubectl -n "$NAMESPACE" get svc "$INFERENCE_SERVICE" -o jsonpath='{.spec.clusterIP}')"
  if [ -z "$service_ip" ] || [ "$service_ip" = "None" ]; then
    echo "ERROR: service ${NAMESPACE}/${INFERENCE_SERVICE} has no ClusterIP" >&2
    exit 1
  fi
  ENDPOINT="${service_ip}:${BRPC_PORT}"
}

capture_pod_state() {
  local output="$1"
  kubectl -n "$NAMESPACE" get pods -l "app=${INFERENCE_APP}" -o json |
    python3 -c '
import json, sys
pods = json.load(sys.stdin).get("items", [])
ready = []
for pod in pods:
    statuses = pod.get("status", {}).get("containerStatuses", [])
    if (pod.get("status", {}).get("phase") == "Running" and statuses
            and all(status.get("ready") for status in statuses)):
        ready.append((pod, statuses))
if len(ready) != 1:
    raise SystemExit(f"expected one running inference pod, found {len(ready)}")
pod, statuses = ready[0]
print("pod=" + pod["metadata"]["name"])
print("uid=" + pod["metadata"]["uid"])
print("restarts=" + str(sum(s.get("restartCount", 0) for s in statuses)))
' >"$output"
}

validate_positive_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${name} must be a positive integer: ${value}" >&2
    exit 2
  fi
}

mkdir -p "$OUT_DIR/runs"
require_command go
require_command python3
if [ -z "$ENDPOINT" ] || [ "$CHECK_K8S_STATE" = "1" ]; then
  require_command kubectl
fi
resolve_endpoint
validate_positive_integer REPEATS "$REPEATS"

BURST_CONCURRENCY_LEVELS="${BURST_CONCURRENCY_LEVELS//,/ }"
read -r -a LEVELS <<<"$BURST_CONCURRENCY_LEVELS"
if [ "${#LEVELS[@]}" -eq 0 ]; then
  echo "ERROR: BURST_CONCURRENCY_LEVELS is empty" >&2
  exit 2
fi
for concurrency in "${LEVELS[@]}"; do
  validate_positive_integer BURST_CONCURRENCY_LEVEL "$concurrency"
done
mapfile -t LEVELS < <(printf '%s\n' "${LEVELS[@]}" | sort -n -u)

if [ "$CHECK_K8S_STATE" = "1" ]; then
  capture_pod_state "${OUT_DIR}/pod.before"
fi

echo "== Build Go BRPC burst probe =="
GOPROXY="${GOPROXY:-off}" GOSUMDB="${GOSUMDB:-off}" \
  go build -mod=vendor -o "$PROBE_BIN" ./scripts/probe_go_brpc_client.go

echo "== BRPC burst calibration =="
echo "endpoint=${ENDPOINT}"
echo "levels=${LEVELS[*]} repeats=${REPEATS} pressure_payload_bytes=${PRESSURE_PAYLOAD_BYTES}"
echo "target_brpc_delta_ms=${TARGET_BRPC_DELTA_MIN_MS}-${TARGET_BRPC_DELTA_MAX_MS}"
printf 'concurrency\trepeat\texit_code\tresult_json\tlog\n' >"$RUN_INDEX"

run_level() {
  local concurrency="$1"
  local level_dir
  local repeat
  local result_json
  local run_log
  local run_status
  level_dir="${OUT_DIR}/runs/c${concurrency}"
  mkdir -p "$level_dir"
  for repeat in $(seq 1 "$REPEATS"); do
    result_json="${level_dir}/run-$(printf '%03d' "$repeat").json"
    run_log="${level_dir}/run-$(printf '%03d' "$repeat").log"
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
}

for concurrency in "${LEVELS[@]}"; do
  run_level "$concurrency"
done

if [ "$AUTO_REFINE" = "1" ]; then
  REFINE_LEVELS="$(python3 - "$RUN_INDEX" "$TARGET_BRPC_DELTA_MIN_MS" "$TARGET_BRPC_DELTA_MAX_MS" \
    "$MIN_ACTIVE_RATIO" "$REPEATS" <<'PY'
import json
import math
import pathlib
import statistics
import sys

index_path = pathlib.Path(sys.argv[1])
target_min = float(sys.argv[2])
target_max = float(sys.argv[3])
minimum_active_ratio = float(sys.argv[4])
expected_repeats = int(sys.argv[5])
values = {}
for line in index_path.read_text(encoding="utf-8").splitlines()[1:]:
    concurrency, _, exit_code, result_path, _ = line.split("\t")
    path = pathlib.Path(result_path)
    if exit_code != "0" or not path.exists():
        continue
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    concurrency = int(concurrency)
    valid = (
        result.get("business_success")
        and int(result.get("pressure_success", -1)) == int(result.get("pressure_requests", -2))
        and int(result.get("armed_workers", 0)) == concurrency
        and int(result.get("max_active_workers", 0)) >= math.ceil(concurrency * minimum_active_ratio)
    )
    if valid:
        values.setdefault(concurrency, []).append(float(result["business_brpc_delta_ms"]))

medians = sorted(
    (concurrency, statistics.median(samples))
    for concurrency, samples in values.items()
    if len(samples) == expected_repeats
)
if any(target_min <= value <= target_max for _, value in medians):
    raise SystemExit(0)
for (left_c, left_v), (right_c, right_v) in zip(medians, medians[1:]):
    crosses = ((left_v < target_min and right_v > target_max)
               or (right_v < target_min and left_v > target_max))
    if not crosses or right_c - left_c <= 1:
        continue
    candidates = {
        round(left_c + (right_c - left_c) * fraction)
        for fraction in (0.25, 0.5, 0.75)
    }
    print(" ".join(str(value) for value in sorted(candidates) if left_c < value < right_c))
    break
PY
)"
  if [ -n "$REFINE_LEVELS" ]; then
    echo "== Refine bracket with concurrency: ${REFINE_LEVELS} =="
    read -r -a EXTRA_LEVELS <<<"$REFINE_LEVELS"
    for concurrency in "${EXTRA_LEVELS[@]}"; do
      run_level "$concurrency"
      LEVELS+=("$concurrency")
    done
    mapfile -t LEVELS < <(printf '%s\n' "${LEVELS[@]}" | sort -n -u)
  fi
fi

K8S_STATE_OK=1
CRASH_MARKER_COUNT=0
if [ "$CHECK_K8S_STATE" = "1" ]; then
  capture_pod_state "${OUT_DIR}/pod.after"
  if ! cmp -s "${OUT_DIR}/pod.before" "${OUT_DIR}/pod.after"; then
    K8S_STATE_OK=0
    echo "ERROR: inference pod identity or restart count changed" >&2
    diff -u "${OUT_DIR}/pod.before" "${OUT_DIR}/pod.after" || true
  fi
  pod_name="$(awk -F= '$1 == "pod" {print $2}' "${OUT_DIR}/pod.after")"
  kubectl -n "$NAMESPACE" logs "$pod_name" --since-time="$STARTED_AT" >"${OUT_DIR}/inference.log" 2>&1 || true
  CRASH_MARKER_COUNT="$(grep -Eic 'EngineCore failed|Segmentation|core dumped|Out of memory|terminate called' "${OUT_DIR}/inference.log" || true)"
  if [ "$CRASH_MARKER_COUNT" -ne 0 ]; then
    K8S_STATE_OK=0
    echo "ERROR: inference crash markers detected: ${CRASH_MARKER_COUNT}" >&2
  fi
fi

python3 - "$RUN_INDEX" "$SUMMARY_JSON" "$SELECTED_ENV" \
  "$REPEATS" "$TARGET_BRPC_DELTA_MIN_MS" "$TARGET_BRPC_DELTA_MAX_MS" \
  "$MIN_ACTIVE_RATIO" "$K8S_STATE_OK" "$CRASH_MARKER_COUNT" \
  "$PRESSURE_PAYLOAD_BYTES" "$BUSINESS_PAYLOAD_BYTES" "$ENDPOINT" "${LEVELS[@]}" <<'PY'
import json
import math
import pathlib
import statistics
import sys

index_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
selected_env_path = pathlib.Path(sys.argv[3])
repeats = int(sys.argv[4])
target_min = float(sys.argv[5])
target_max = float(sys.argv[6])
min_active_ratio = float(sys.argv[7])
k8s_state_ok = sys.argv[8] == "1"
crash_marker_count = int(sys.argv[9])
pressure_payload_bytes = int(sys.argv[10])
business_payload_bytes = int(sys.argv[11])
endpoint = sys.argv[12]
levels = [int(value) for value in sys.argv[13:]]

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

indexed = {level: [] for level in levels}
lines = index_path.read_text(encoding="utf-8").splitlines()[1:]
for line in lines:
    concurrency, repeat, exit_code, result_path, log_path = line.split("\t")
    record = {
        "repeat": int(repeat),
        "exit_code": int(exit_code),
        "result_path": result_path,
        "log_path": log_path,
        "result": None,
    }
    path = pathlib.Path(result_path)
    if path.exists():
        try:
            record["result"] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    indexed[int(concurrency)].append(record)

cases = []
selected = None
for concurrency in levels:
    records = indexed[concurrency]
    results = [record["result"] for record in records if record["result"] is not None]
    process_success = sum(record["exit_code"] == 0 and record["result"] is not None for record in records)
    deltas = [float(result["business_brpc_delta_ms"]) for result in results if result.get("business_success")]
    walls = [float(result["business_client_wall_ms"]) for result in results if result.get("business_success")]
    full_pressure = sum(
        int(result.get("pressure_success", -1)) == int(result.get("pressure_requests", -2))
        for result in results
    )
    business_success = sum(bool(result.get("business_success")) for result in results)
    active_threshold = math.ceil(concurrency * min_active_ratio)
    active_pass = sum(int(result.get("max_active_workers", 0)) >= active_threshold for result in results)
    armed_pass = sum(int(result.get("armed_workers", 0)) == concurrency for result in results)
    delta_p50 = percentile(deltas, 0.50)
    delta_p95 = percentile(deltas, 0.95)
    qualifies = (
        len(records) == repeats
        and len(results) == repeats
        and process_success == repeats
        and business_success == repeats
        and full_pressure == repeats
        and active_pass == repeats
        and armed_pass == repeats
        and delta_p50 is not None
        and target_min <= delta_p50 <= target_max
        and k8s_state_ok
    )
    case = {
        "concurrency": concurrency,
        "planned_repeats": repeats,
        "result_count": len(results),
        "process_success": process_success,
        "business_success": business_success,
        "full_pressure_success": full_pressure,
        "active_pass": active_pass,
        "armed_pass": armed_pass,
        "active_threshold": active_threshold,
        "business_brpc_delta_p50_ms": delta_p50,
        "business_brpc_delta_p95_ms": delta_p95,
        "business_client_wall_p50_ms": percentile(walls, 0.50),
        "business_client_wall_p95_ms": percentile(walls, 0.95),
        "qualifies": qualifies,
    }
    cases.append(case)
    if selected is None and qualifies:
        selected = case

summary = {
    "event": "brpc_burst_calibration",
    "target_brpc_delta_min_ms": target_min,
    "target_brpc_delta_max_ms": target_max,
    "minimum_active_ratio": min_active_ratio,
    "endpoint": endpoint,
    "pressure_payload_bytes": pressure_payload_bytes,
    "business_payload_bytes": business_payload_bytes,
    "k8s_state_ok": k8s_state_ok,
    "crash_marker_count": crash_marker_count,
    "cases": cases,
    "selected": selected,
    "status": "PASS" if selected is not None else "NO_TARGET",
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print("\n== BRPC burst calibration summary ==")
print("concurrency results process_ok business_ok pressure_ok active_ok delta_p50_ms delta_p95_ms status")
for case in cases:
    def show(value):
        return "n/a" if value is None else f"{value:.3f}"
    print(
        f"{case['concurrency']:>11} {case['result_count']:>7}/{repeats:<2} "
        f"{case['process_success']:>10}/{repeats:<2} "
        f"{case['business_success']:>11}/{repeats:<2} "
        f"{case['full_pressure_success']:>10}/{repeats:<2} "
        f"{case['active_pass']:>8}/{repeats:<2} "
        f"{show(case['business_brpc_delta_p50_ms']):>12} "
        f"{show(case['business_brpc_delta_p95_ms']):>12} "
        f"{'PASS' if case['qualifies'] else 'MISS'}"
    )

if selected is not None:
    selected_env_path.write_text(
        f"BURST_CONCURRENCY={selected['concurrency']}\n"
        f"PRESSURE_PAYLOAD_BYTES={pressure_payload_bytes}\n"
        f"BUSINESS_PAYLOAD_BYTES={business_payload_bytes}\n"
        f"ENDPOINT={endpoint}\n",
        encoding="utf-8",
    )
    print(f"selected_concurrency={selected['concurrency']}")
else:
    selected_env_path.unlink(missing_ok=True)
    print("selected_concurrency=none")
print(f"summary_json={summary_path}")
print(f"RESULT={'PASS' if selected is not None else 'FAIL'}")
PY

if [ "$K8S_STATE_OK" -ne 1 ]; then
  exit 1
fi
if [ ! -f "$SELECTED_ENV" ]; then
  exit 3
fi

echo "BRPC_BURST_WRAPPER_CALIBRATION_OK"
echo "selected_env=${SELECTED_ENV}"
