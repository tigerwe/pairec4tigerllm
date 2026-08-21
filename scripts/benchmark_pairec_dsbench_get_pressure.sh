#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/pairec-dsbench-get/${RUN_ID}}"
REPEATS="${REPEATS:-1}"
DSBENCH_CLIENTS="${DSBENCH_CLIENTS:-64}"
DSBENCH_THREADS="${DSBENCH_THREADS:-1}"
DSBENCH_KEY_COUNT="${DSBENCH_KEY_COUNT:-$((DSBENCH_CLIENTS * DSBENCH_THREADS))}"
DSBENCH_OBJECT_SIZE="${DSBENCH_OBJECT_SIZE:-3584KB}"
DSBENCH_DURATION_SECONDS="${DSBENCH_DURATION_SECONDS:-90}"
DSBENCH_SETTLE_SECONDS="${DSBENCH_SETTLE_SECONDS:-3}"
DSBENCH_CPP="${DSBENCH_CPP:-/home/zcx/bin/dsbench-v081-sustained}"
DSBENCH_TASKSET_CPUS="${DSBENCH_TASKSET_CPUS:-40-71}"
KVC_LOAD_HOST="${KVC_LOAD_HOST:-worker1}"
KVC_REMOTE_REPO="${KVC_REMOTE_REPO:-/home/zcx/workspace/pairec4tigerllm}"
KVC_DS_ENDPOINT="${KVC_DS_ENDPOINT:-192.168.100.12:18482}"
PRIME_REQUESTS="${PRIME_REQUESTS:-195}"
EXPECTED_ONBOARDS_MIN="${EXPECTED_ONBOARDS_MIN:-1}"
EXPECTED_ONBOARDS_MAX="${EXPECTED_ONBOARDS_MAX:-2}"
USER_FEATURES_PATH="${USER_FEATURES_PATH:-$REPO_ROOT/data/user_features.json}"
SEMANTIC_MAP_PATH="${SEMANTIC_MAP_PATH:-$REPO_ROOT/data/tenrec/processed/semantic_id_map.json}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

for value in "$REPEATS" "$DSBENCH_CLIENTS" "$DSBENCH_THREADS" \
  "$DSBENCH_KEY_COUNT" "$DSBENCH_DURATION_SECONDS" "$DSBENCH_SETTLE_SECONDS" \
  "$PRIME_REQUESTS" "$EXPECTED_ONBOARDS_MIN" "$EXPECTED_ONBOARDS_MAX"; do
  [[ "$value" =~ ^[0-9]+$ ]] || die "integer configuration contains invalid value: $value"
done
[ "$REPEATS" -gt 0 ] || die "REPEATS must be positive"
[ "$DSBENCH_CLIENTS" -gt 0 ] || die "DSBENCH_CLIENTS must be positive"
[ "$DSBENCH_THREADS" -gt 0 ] || die "DSBENCH_THREADS must be positive"
[ "$DSBENCH_KEY_COUNT" -eq $((DSBENCH_CLIENTS * DSBENCH_THREADS)) ] \
  || die "DSBENCH_KEY_COUNT must equal DSBENCH_CLIENTS * DSBENCH_THREADS in sustained mode"
[ "$DSBENCH_DURATION_SECONDS" -gt "$DSBENCH_SETTLE_SECONDS" ] \
  || die "DSBENCH_DURATION_SECONDS must exceed DSBENCH_SETTLE_SECONDS"
[ "$EXPECTED_ONBOARDS_MIN" -gt 0 ] || die "at least one business Get is required"
[ "$EXPECTED_ONBOARDS_MIN" -le "$EXPECTED_ONBOARDS_MAX" ] \
  || die "EXPECTED_ONBOARDS_MIN must not exceed EXPECTED_ONBOARDS_MAX"

if [ ! -s "$SEMANTIC_MAP_PATH" ]; then
  fallback="/home/zcx/workspace/pairec4tigerllm/data/tenrec/processed/semantic_id_map.json"
  [ -s "$fallback" ] && SEMANTIC_MAP_PATH="$fallback"
fi
[ -s "$USER_FEATURES_PATH" ] || die "user features not found: $USER_FEATURES_PATH"
[ -s "$SEMANTIC_MAP_PATH" ] || die "semantic map not found: $SEMANTIC_MAP_PATH"

mkdir -p "$OUT_DIR"
cat >"$OUT_DIR/config.txt" <<EOF
pressure_engine=dsbench
pressure_operation=get
dsbench_clients=${DSBENCH_CLIENTS}
dsbench_threads=${DSBENCH_THREADS}
dsbench_key_count=${DSBENCH_KEY_COUNT}
dsbench_object_size=${DSBENCH_OBJECT_SIZE}
dsbench_duration_seconds=${DSBENCH_DURATION_SECONDS}
dsbench_settle_seconds=${DSBENCH_SETTLE_SECONDS}
dsbench_cpp=${DSBENCH_CPP}
kvc_load_host=${KVC_LOAD_HOST}
kvc_ds_endpoint=${KVC_DS_ENDPOINT}
prime_requests=${PRIME_REQUESTS}
repeats=${REPEATS}
expected_onboards_min=${EXPECTED_ONBOARDS_MIN}
expected_onboards_max=${EXPECTED_ONBOARDS_MAX}
EOF

cat "$OUT_DIR/config.txt"

contention_code=0
MODE=kvc-get \
REPEATS="$REPEATS" \
OUT_DIR="$OUT_DIR/contention" \
KVC_PRESSURE_ENGINE=dsbench \
KVC_DSBENCH_SUSTAINED=1 \
KVC_DSBENCH_CPP="$DSBENCH_CPP" \
KVC_LOAD_HOST="$KVC_LOAD_HOST" \
KVC_REMOTE_REPO="$KVC_REMOTE_REPO" \
KVC_DS_ENDPOINT="$KVC_DS_ENDPOINT" \
KVC_OBJECT_SIZE="$DSBENCH_OBJECT_SIZE" \
KVC_THREAD_NUM="$DSBENCH_THREADS" \
KVC_GET_CLIENTS="$DSBENCH_CLIENTS" \
KVC_GET_KEY_COUNT="$DSBENCH_KEY_COUNT" \
KVC_SET_CLIENTS=1 \
KVC_SET_KEY_COUNT="$DSBENCH_THREADS" \
KVC_BATCH_NUM=1 \
KVC_TASKSET_CPUS="$DSBENCH_TASKSET_CPUS" \
KVC_LOAD_DURATION_SECONDS="$DSBENCH_DURATION_SECONDS" \
LOAD_SETTLE_SECONDS="$DSBENCH_SETTLE_SECONDS" \
PRIME_REQUESTS="$PRIME_REQUESTS" \
USER_FEATURES_PATH="$USER_FEATURES_PATH" \
SEMANTIC_MAP_PATH="$SEMANTIC_MAP_PATH" \
STRICT_COUNTS=1 \
EXPECTED_OFFLOADS=3 \
EXPECTED_ONBOARDS=2 \
EXPECTED_ONBOARDS_MIN="$EXPECTED_ONBOARDS_MIN" \
EXPECTED_ONBOARDS_MAX="$EXPECTED_ONBOARDS_MAX" \
REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION=1 \
REQUIRE_BUSINESS_ONBOARD_GET=1 \
RESET_INFERENCE_AFTER_PRIME=1 \
RESET_INFERENCE_MODE=container-runtime \
KVC_BURST_REQUIRE_COMPLETE=0 \
KVC_BURST_DYNAMIC_ARM=0 \
KVC_NIC_BURST_SAMPLE=1 \
  bash scripts/benchmark_brpc_kvc_contention.sh \
  | tee "$OUT_DIR/contention.console.log" || contention_code=$?

python3 - "$OUT_DIR" "$contention_code" <<'PY' | tee "$OUT_DIR/summary.txt"
import glob
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
contention_code = int(sys.argv[2])

try:
    result = json.loads((root / "contention" / "result.json").read_text())
except (FileNotFoundError, json.JSONDecodeError):
    result = {}

rows = result.get("rows", [])
all_samples = []
for path_text in sorted(glob.glob(str(root / "contention" / "round-*" / "replay" / "summary.json"))):
    path = pathlib.Path(path_text)
    replay = json.loads(path.read_text())
    native = replay.get("datasystem_request_complete") or {}
    access = replay.get("kvc_access") or {}
    executor = replay.get("trt_executor_request_completions") or []
    onboards = access.get("onboard_events") or []
    row = next(
        (item for item in rows if pathlib.Path(item.get("summary_path", "")) == path),
        None,
    )
    if row is None:
        continue
    nic_path = path.parent / "nic-burst.json"
    try:
        nic = json.loads(nic_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        nic = {}
    all_samples.append({
        "valid": bool(row.get("valid")),
        "round": row.get("round"),
        "request_id": replay.get("request_id"),
        "client_e2e_ms": row.get("e2e_ms", 0.0),
        "runner_ms": sum(float(item.get("runner_us", 0)) for item in executor) / 1000.0,
        "business_get_count": len(onboards),
        "business_get_total_ms": sum(float(item.get("get_ms", 0)) for item in onboards),
        "business_get_each_ms": [float(item.get("get_ms", 0)) for item in onboards],
        "datasystem_get_ms": float(native.get("get_us", 0)) / 1000.0,
        "datasystem_set_ms": float(native.get("set_us", 0)) / 1000.0,
        "dsbench_get_qps": row.get("kvc_get_qps", 0.0),
        "dsbench_get_gbps": row.get("kvc_get_gbps", 0.0),
        "dsbench_get_max_inflight": row.get("kvc_get_max_inflight", 0),
        "nic_peak_rx_gbps": nic.get("peak_rx_gbps", 0.0),
        "nic_peak_rx_link_pct": nic.get("peak_rx_link_pct", 0.0),
    })

samples = [sample for sample in all_samples if sample["valid"]]

def avg(key):
    return statistics.mean(float(sample[key]) for sample in samples) if samples else 0.0

def diagnostic_avg(key):
    return statistics.mean(float(sample[key]) for sample in all_samples) if all_samples else 0.0

valid = (
    contention_code == 0
    and result.get("status") == "PASS"
    and len(samples) == int(result.get("expected_repeats", -1))
)
summary = {
    "classification": "PAIREC_DSBENCH_GET_PRESSURE_VALID" if valid else "PAIREC_DSBENCH_GET_PRESSURE_INVALID",
    "valid": valid,
    "contention_exit_code": contention_code,
    "valid_repeats": len(samples),
    "expected_repeats": result.get("expected_repeats", 0),
    "averages": {
        "client_e2e_ms": avg("client_e2e_ms"),
        "runner_ms": avg("runner_ms"),
        "business_get_total_ms": avg("business_get_total_ms"),
        "datasystem_get_ms": avg("datasystem_get_ms"),
        "datasystem_set_ms": avg("datasystem_set_ms"),
        "dsbench_get_qps": avg("dsbench_get_qps"),
        "dsbench_get_gbps": avg("dsbench_get_gbps"),
        "dsbench_get_max_inflight": avg("dsbench_get_max_inflight"),
        "nic_peak_rx_gbps": avg("nic_peak_rx_gbps"),
        "nic_peak_rx_link_pct": avg("nic_peak_rx_link_pct"),
    },
    "samples": samples,
    "diagnostic_averages": {
        "client_e2e_ms": diagnostic_avg("client_e2e_ms"),
        "runner_ms": diagnostic_avg("runner_ms"),
        "business_get_total_ms": diagnostic_avg("business_get_total_ms"),
        "datasystem_get_ms": diagnostic_avg("datasystem_get_ms"),
        "datasystem_set_ms": diagnostic_avg("datasystem_set_ms"),
        "dsbench_get_qps": diagnostic_avg("dsbench_get_qps"),
        "dsbench_get_gbps": diagnostic_avg("dsbench_get_gbps"),
        "dsbench_get_max_inflight": diagnostic_avg("dsbench_get_max_inflight"),
        "nic_peak_rx_gbps": diagnostic_avg("nic_peak_rx_gbps"),
        "nic_peak_rx_link_pct": diagnostic_avg("nic_peak_rx_link_pct"),
    },
    "diagnostic_samples": all_samples,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

print(summary["classification"])
print(f"valid_repeats={len(samples)}/{summary['expected_repeats']}")
for key, value in summary["averages"].items():
    print(f"{key}={value:.3f}")
if not valid:
    print("diagnostic averages (includes invalid rounds):")
    for key, value in summary["diagnostic_averages"].items():
        print(f"diagnostic_{key}={value:.3f}")
print(f"summary_json={root / 'summary.json'}")
raise SystemExit(0 if valid else 1)
PY
