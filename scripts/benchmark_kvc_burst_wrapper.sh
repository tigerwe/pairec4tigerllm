#!/usr/bin/env bash
set -euo pipefail

KVC_BURST_WRAPPER_BIN=${KVC_BURST_WRAPPER_BIN:-/opt/pairec-kvc-burst/bin/kvc_burst_wrapper}
KVC_BURST_BUSINESS_PROBE_BIN=${KVC_BURST_BUSINESS_PROBE_BIN:-/opt/pairec-kvc-burst/bin/kvc_burst_business_probe}
DS_ENDPOINT=${DS_ENDPOINT:-192.168.100.12:18482}
CONCURRENCY_LEVELS=${CONCURRENCY_LEVELS:-1 10 100}
REPEATS=${REPEATS:-3}
OBJECT_SIZE=${OBJECT_SIZE:-3670016}
BARRIER_TIMEOUT_MS=${BARRIER_TIMEOUT_MS:-10}
PRESSURE_LEAD_US=${PRESSURE_LEAD_US:-1000}
SHUFFLE_SEED=${SHUFFLE_SEED:-20260804}
PREFIX_BASE=${PREFIX_BASE:-PairecKvcBurstV2}
OUT_DIR=${OUT_DIR:-/tmp/kvc-burst-wrapper/$(date +%Y%m%d-%H%M%S)-$$}
CONTROL_DIR=${CONTROL_DIR:-$OUT_DIR/control}
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-120}
CLEANUP_KEYS=${CLEANUP_KEYS:-1}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "$DS_ENDPOINT" == *:* ]] || die "DS_ENDPOINT must be host:port"
DS_HOST=${DS_ENDPOINT%:*}
DS_PORT=${DS_ENDPOINT##*:}
[[ -x "$KVC_BURST_WRAPPER_BIN" ]] || die "wrapper binary is not executable: $KVC_BURST_WRAPPER_BIN"
[[ -x "$KVC_BURST_BUSINESS_PROBE_BIN" ]] \
  || die "business probe binary is not executable: $KVC_BURST_BUSINESS_PROBE_BIN"
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be positive"
[[ "$OBJECT_SIZE" =~ ^[1-9][0-9]*$ ]] || die "OBJECT_SIZE must be bytes"
[[ "$PRESSURE_LEAD_US" =~ ^[0-9]+$ ]] && (( PRESSURE_LEAD_US <= 1000000 )) \
  || die "PRESSURE_LEAD_US must be between 0 and 1000000"

mkdir -p "$OUT_DIR" "$CONTROL_DIR"
WRAPPER_PID=

stop_wrapper() {
  if [[ -n "${WRAPPER_PID:-}" ]] && kill -0 "$WRAPPER_PID" 2>/dev/null; then
    kill -TERM "$WRAPPER_PID" 2>/dev/null || true
    wait "$WRAPPER_PID" 2>/dev/null || true
  fi
  WRAPPER_PID=
}
trap stop_wrapper EXIT INT TERM

wait_ready() {
  local ready_file=$1
  local deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  while [[ ! -s "$ready_file" ]]; do
    if ! kill -0 "$WRAPPER_PID" 2>/dev/null; then
      die "KVC burst wrapper exited before readiness"
    fi
    (( SECONDS < deadline )) || die "timed out waiting for wrapper readiness"
    sleep 0.1
  done
}

for concurrency in $CONCURRENCY_LEVELS; do
  [[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || die "invalid concurrency: $concurrency"
  (( concurrency <= 256 )) || die "concurrency exceeds maximum: $concurrency"
  case_dir="$OUT_DIR/c$concurrency"
  mkdir -p "$case_dir"
  control_path="$CONTROL_DIR/control-c$concurrency"
  ready_file="$case_dir/ready"
  wrapper_log="$case_dir/wrapper.log"
  prefix="${PREFIX_BASE}_c${concurrency}_$$"

  echo "== KVC burst concurrency=$concurrency =="
  "$KVC_BURST_WRAPPER_BIN" \
    --host="$DS_HOST" \
    --port="$DS_PORT" \
    --concurrency="$concurrency" \
    --object_size="$OBJECT_SIZE" \
    --barrier_timeout_ms="$BARRIER_TIMEOUT_MS" \
    --pressure_lead_us="$PRESSURE_LEAD_US" \
    --seed="$SHUFFLE_SEED" \
    --prefix="$prefix" \
    --control_path="$control_path" \
    --ready_file="$ready_file" \
    --stats_file="$case_dir/results.jsonl" \
    --cleanup_keys="$CLEANUP_KEYS" \
    >"$wrapper_log" 2>&1 &
  WRAPPER_PID=$!
  wait_ready "$ready_file"
  cat "$ready_file"

  for ((repeat = 1; repeat <= REPEATS; ++repeat)); do
    request_id="kvc-c${concurrency}-r${repeat}-$$"
    echo "-- repeat=$repeat/$REPEATS request_id=$request_id --"
    "$KVC_BURST_BUSINESS_PROBE_BIN" \
      --host="$DS_HOST" \
      --port="$DS_PORT" \
      --object_size="$OBJECT_SIZE" \
      --prefix="$prefix" \
      --request_id="$request_id" \
      --control_path="$control_path" \
      --ready_timeout_ms="$((READY_TIMEOUT_SECONDS * 1000))" \
      --result_timeout_ms=120000 \
      --wait_result=true \
      --cleanup_key="$CLEANUP_KEYS" \
      | tee "$case_dir/probe-$repeat.log"
  done

  stop_wrapper
done

python3 - "$OUT_DIR" "$REPEATS" $CONCURRENCY_LEVELS <<'PY'
import json
import math
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
repeats = int(sys.argv[2])
levels = [int(value) for value in sys.argv[3:]]
failed = False

def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)

print("\n== KVC burst wrapper summary ==")
print("concurrency valid samples business_avg business_p95 business_p99 pressure_p99 overlap_min active_min status")
summary = {"cases": []}
for concurrency in levels:
    path = root / f"c{concurrency}" / "results.jsonl"
    events = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "kvc_burst_complete":
                events.append(event)
    valid = len(events) == repeats and all(event.get("valid") is True for event in events)
    valid = valid and all(
        int(event.get("pressure_inflight_at_business_start", 0))
        >= math.ceil(int(event.get("pressure_lanes", 0)) * .95)
        for event in events)
    business = [float(event["business_get_ms"]) for event in events]
    pressure = [float(event["pressure_get_p99_ms"]) for event in events]
    overlap_min = min((int(event["business_overlap_gets"]) for event in events), default=0)
    active_min = min((int(event["max_active_all_gets"]) for event in events), default=0)
    status = "PASS" if valid else "FAIL"
    failed |= not valid
    print(
        f"{concurrency:>11} {str(valid):>5} {len(events):>3}/{repeats:<3} "
        f"{(statistics.mean(business) if business else 0):>12.3f} "
        f"{percentile(business, 0.95):>12.3f} {percentile(business, 0.99):>12.3f} "
        f"{percentile(pressure, 0.99):>12.3f} {overlap_min:>11} {active_min:>10} {status}"
    )
    summary["cases"].append({
        "concurrency": concurrency,
        "valid": valid,
        "samples": len(events),
        "business_get_avg_ms": statistics.mean(business) if business else None,
        "business_get_p95_ms": percentile(business, 0.95),
        "business_get_p99_ms": percentile(business, 0.99),
        "pressure_get_p99_ms": percentile(pressure, 0.99),
        "business_overlap_min": overlap_min,
        "max_active_all_min": active_min,
    })

(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"summary_json={root / 'summary.json'}")
if failed:
    print("RESULT=FAIL")
    raise SystemExit(1)
print("KVC_BURST_WRAPPER_BENCHMARK_OK")
print("RESULT=PASS")
PY
