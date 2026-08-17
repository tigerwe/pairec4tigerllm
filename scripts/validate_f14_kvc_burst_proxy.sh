#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-pairec}
DEPLOYMENT=${DEPLOYMENT:-inference-brpc-trtllm}
PAIREC_TARGET=${PAIREC_TARGET:-deploy/pairec-brpc-observed}
CONCURRENCY_LEVELS=${CONCURRENCY_LEVELS:-1 10 100}
REPEATS=${REPEATS:-3}
PRIME_REQUESTS=${PRIME_REQUESTS:-195}
EXPECTED_SETS=${EXPECTED_SETS:-3}
EXPECTED_GETS=${EXPECTED_GETS:-2}
MAX_PRESSURE_KEYS=${MAX_PRESSURE_KEYS:-9}
OUT_DIR=${OUT_DIR:-/tmp/f14-kvc-burst-proxy/$(date +%Y%m%d-%H%M%S)-$$}
BACKUP_FILE=${BACKUP_FILE:-$OUT_DIR/deployment-before.json}

die() { echo "ERROR: $*" >&2; exit 1; }

restore_overlay() {
  if [[ -f "$BACKUP_FILE" ]]; then
    echo "== Restore original inference Deployment =="
    BACKUP_FILE="$BACKUP_FILE" NAMESPACE="$NAMESPACE" DEPLOYMENT="$DEPLOYMENT" \
      bash scripts/deploy_f14_kvc_burst_overlay.sh restore || true
  fi
}
trap restore_overlay EXIT INT TERM

run_case() {
  local name=$1 concurrency=$2 enabled=$3 measure_disabled=$4 require_complete=$5
  local pressure_key_count=$((concurrency - 1))
  if [ "$pressure_key_count" -gt "$MAX_PRESSURE_KEYS" ]; then
    pressure_key_count=$MAX_PRESSURE_KEYS
  fi
  local case_dir="$OUT_DIR/$name"
  mkdir -p "$case_dir"
  echo "== KVC Proxy case=$name concurrency=$concurrency enabled=$enabled =="
  BACKUP_FILE="$BACKUP_FILE" NAMESPACE="$NAMESPACE" DEPLOYMENT="$DEPLOYMENT" \
  CONCURRENCY="$concurrency" PRESSURE_KEY_COUNT="$pressure_key_count" \
  KVC_BURST_ENABLED="$enabled" MEASURE_DISABLED="$measure_disabled" \
  KVC_BURST_VERBOSE="$enabled" \
  KVC_BURST_INITIAL_ARMED=0 \
    bash scripts/deploy_f14_kvc_burst_overlay.sh apply \
    | tee "$case_dir/overlay.log"

  export KVC_BURST_CONTAINER=kvc-burst-wrapper
  export KVC_BURST_REQUIRE_COMPLETE="$require_complete"
  export KVC_BURST_DYNAMIC_ARM="$enabled"
  export KVC_BURST_PRESSURE_KEY_COUNT="$pressure_key_count"
  MODE=baseline \
  REPEATS="$REPEATS" \
  STRICT_COUNTS=1 \
  EXPECTED_OFFLOADS="$EXPECTED_SETS" \
  EXPECTED_ONBOARDS="$EXPECTED_GETS" \
  REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION=1 \
  PRIME_REQUESTS="$PRIME_REQUESTS" \
  RESET_INFERENCE_BEFORE_ROUND=1 \
  RESET_INFERENCE_MODE=container-runtime \
  NAMESPACE="$NAMESPACE" \
  PAIREC_TARGET="$PAIREC_TARGET" \
  BRPC_TARGET="deployment/$DEPLOYMENT" \
  OUT_DIR="$case_dir/contention" \
    bash scripts/benchmark_brpc_kvc_contention.sh \
    | tee "$case_dir/contention.log"
}

[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be positive"
[[ "$MAX_PRESSURE_KEYS" =~ ^[1-9][0-9]*$ ]] || die "MAX_PRESSURE_KEYS must be positive"
mkdir -p "$OUT_DIR"

run_case disabled 1 0 1 0
for concurrency in $CONCURRENCY_LEVELS; do
  [[ "$concurrency" =~ ^(1|10|100)$ ]] || die "concurrency must be 1, 10, or 100"
  run_case "c$concurrency" "$concurrency" 1 0 1
done

python3 - "$OUT_DIR" "$REPEATS" "$EXPECTED_SETS" "$EXPECTED_GETS" "$MAX_PRESSURE_KEYS" $CONCURRENCY_LEVELS <<'PY'
import glob
import json
import math
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
repeats = int(sys.argv[2])
expected_sets = int(sys.argv[3])
expected_gets = int(sys.argv[4])
max_pressure_keys = int(sys.argv[5])
levels = [int(value) for value in sys.argv[6:]]

def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(values[low])
    return values[low] * (high - position) + values[high] * (position - low)

def summaries(case):
    result = []
    for path in sorted(glob.glob(str(root / case / "contention/round-*/replay/summary.json"))):
        result.append(json.loads(pathlib.Path(path).read_text()))
    assert len(result) == repeats, (case, len(result), repeats)
    return result

disabled = summaries("disabled")
disabled_overheads = []
for sample in disabled:
    request_id = sample["request_id"]
    assert sample["response_code"] == 200, sample
    completion = sample["datasystem_request_complete"]
    assert completion["request_id"] == request_id and completion["attribution_complete"] is True
    events = [event for event in sample.get("kvc_proxy_events", [])
              if event.get("event") == "kvc_proxy_disabled_overhead"
              and event.get("request_id") == request_id]
    assert events, (request_id, sample.get("kvc_proxy_events"))
    disabled_overheads.extend(int(event["overhead_us"]) for event in events)
disabled_p99 = percentile(disabled_overheads, 0.99)
assert disabled_p99 <= 100, f"disabled Proxy overhead p99 {disabled_p99}us exceeds 100us"

rows = []
for concurrency in levels:
    actual = []
    adjusted = []
    barriers = []
    business = []
    pressure = []
    for sample in summaries(f"c{concurrency}"):
        request_id = sample["request_id"]
        assert sample["response_code"] == 200, sample
        completion = sample["datasystem_request_complete"]
        assert completion["request_id"] == request_id
        assert completion["attribution_complete"] is True
        assert completion["get_count"] == expected_gets, completion
        assert completion["set_count"] == expected_sets, completion
        events = sample.get("kvc_proxy_events", [])
        burst_events = [event for event in events
                        if event.get("event") == "kvc_burst_complete"
                        and event.get("request_id") == request_id]
        assert len(burst_events) == 1, (request_id, burst_events)
        burst = burst_events[0]
        assert burst["valid"] is True, burst
        assert burst["failure"] == 0, burst
        assert burst["pressure_success"] == concurrency - 1, burst
        assert burst["pressure_errors"] == 0, burst
        assert burst["pressure_key_count"] == min(concurrency - 1, max_pressure_keys), burst
        required_overlap = math.ceil((concurrency - 1) * 0.95)
        assert burst["business_overlap_gets"] >= required_overlap, burst
        barrier_ms = float(burst["barrier_wait_ms"])
        e2e_ms = float(sample["client"]["client_e2e_ms"])
        actual.append(e2e_ms)
        adjusted.append(max(0.0, e2e_ms - barrier_ms))
        barriers.append(barrier_ms)
        business.append(float(burst["business_get_ms"]))
        pressure.append(float(burst["pressure_get_p99_ms"]))
    rows.append({
        "concurrency": concurrency,
        "samples": len(actual),
        "actual_e2e_avg_ms": statistics.mean(actual),
        "actual_e2e_p99_ms": percentile(actual, 0.99),
        "adjusted_e2e_avg_ms": statistics.mean(adjusted),
        "adjusted_e2e_p99_ms": percentile(adjusted, 0.99),
        "barrier_wait_avg_ms": statistics.mean(barriers),
        "barrier_wait_p99_ms": percentile(barriers, 0.99),
        "business_get_avg_ms": statistics.mean(business),
        "business_get_p99_ms": percentile(business, 0.99),
        "pressure_get_p99_ms": percentile(pressure, 0.99),
    })

summary = {"classification": "F14_KVC_BURST_PROXY_PASS",
           "disabled_proxy_overhead_p99_us": disabled_p99, "cases": rows}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("concurrency samples actual_avg_ms actual_p99_ms adjusted_avg_ms adjusted_p99_ms barrier_avg_ms barrier_p99_ms business_get_p99_ms pressure_get_p99_ms")
for row in rows:
    print(f"{row['concurrency']:>11} {row['samples']:>7} "
          f"{row['actual_e2e_avg_ms']:>13.3f} {row['actual_e2e_p99_ms']:>13.3f} "
          f"{row['adjusted_e2e_avg_ms']:>15.3f} {row['adjusted_e2e_p99_ms']:>15.3f} "
          f"{row['barrier_wait_avg_ms']:>14.3f} {row['barrier_wait_p99_ms']:>14.3f} "
          f"{row['business_get_p99_ms']:>19.3f} {row['pressure_get_p99_ms']:>19.3f}")
print(f"disabled_proxy_overhead_p99_us={disabled_p99:.3f}")
print(f"summary_json={root / 'summary.json'}")
print("F14_KVC_BURST_PROXY_VALIDATION_OK")
PY

restore_overlay
trap - EXIT INT TERM
echo "output_dir=$OUT_DIR"
