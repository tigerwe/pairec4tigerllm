#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-baseline}"
REPEATS="${REPEATS:-3}"
STRICT_COUNTS="${STRICT_COUNTS:-1}"
EXPECTED_OFFLOADS="${EXPECTED_OFFLOADS:-3}"
EXPECTED_ONBOARDS="${EXPECTED_ONBOARDS:-2}"

BRPC_ENDPOINT="${BRPC_ENDPOINT:-192.168.100.11:18100}"
BRPC_LOAD_CONCURRENCY="${BRPC_LOAD_CONCURRENCY:-16}"
BRPC_LOAD_PAYLOAD_BYTES="${BRPC_LOAD_PAYLOAD_BYTES:-102400}"
BRPC_LOAD_REQUESTS="${BRPC_LOAD_REQUESTS:-2000000}"
BRPC_LOAD_TIMEOUT_MS="${BRPC_LOAD_TIMEOUT_MS:-5000}"
BRPC_PROBE_BIN="${BRPC_PROBE_BIN:-/tmp/probe-go-brpc-client}"

KVC_LOAD_HOST="${KVC_LOAD_HOST:-worker1}"
KVC_REMOTE_REPO="${KVC_REMOTE_REPO:-/home/zcx/workspace/pairec4tigerllm}"
KVC_DS_ENDPOINT="${KVC_DS_ENDPOINT:-192.168.100.12:18482}"
KVC_DSBENCH_CPP="${KVC_DSBENCH_CPP:-}"
KVC_OBJECT_SIZE="${KVC_OBJECT_SIZE:-3584KB}"
KVC_KEY_COUNT="${KVC_KEY_COUNT:-256}"
KVC_BATCH_NUM="${KVC_BATCH_NUM:-1}"
KVC_THREAD_NUM="${KVC_THREAD_NUM:-1}"
KVC_GET_CLIENTS="${KVC_GET_CLIENTS:-4}"
KVC_SET_CLIENTS="${KVC_SET_CLIENTS:-6}"
KVC_TASKSET_CPUS="${KVC_TASKSET_CPUS:-}"
KVC_LOAD_DURATION_SECONDS="${KVC_LOAD_DURATION_SECONDS:-300}"

LOAD_SETTLE_SECONDS="${LOAD_SETTLE_SECONDS:-3}"
PRIME_REQUESTS="${PRIME_REQUESTS:-200}"
PRIME_UIDS="${PRIME_UIDS:-5,6312,130,2184,303,1190,1191,1192,1193,1194}"
USER_FEATURES_PATH="${USER_FEATURES_PATH:-/home/zcx/workspace/pairec4tigerllm/data/user_features.json}"
SEMANTIC_MAP_PATH="${SEMANTIC_MAP_PATH:-/home/zcx/workspace/pairec4tigerllm/data/tenrec/processed/semantic_id_map.json}"
HISTORY_MAX_LENGTH="${HISTORY_MAX_LENGTH:-20}"
REPLAY_USER_ID="${REPLAY_USER_ID:-5}"
REPLAY_SIZE="${REPLAY_SIZE:-1}"
REPLAY_TIMEOUT="${REPLAY_TIMEOUT:-30}"

NAMESPACE="${NAMESPACE:-pairec}"
PAIREC_TARGET="${PAIREC_TARGET:-deploy/pairec}"
BRPC_TARGET="${BRPC_TARGET:-deployment/inference-brpc-trtllm}"
BRPC_CONTAINER="${BRPC_CONTAINER:-brpc-inference}"
NETWORK_INTERFACE="${NETWORK_INTERFACE:-enp41s0f1}"
REMOTE_NETWORK_INTERFACE="${REMOTE_NETWORK_INTERFACE:-enp41s0f1}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/brpc-kvc-contention/${RUN_ID}-${MODE}}"
RESULT_JSON="${OUT_DIR}/result.json"
SUMMARY_TXT="${OUT_DIR}/summary.txt"

BRPC_LOAD_PID=""
KVC_SSH_PID=""
KVC_REMOTE_PID_FILE=""
KVC_REMOTE_READY_FILE=""

log() {
  printf '\n== %s ==\n' "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

mode_has_brpc() {
  [ "$MODE" = "brpc" ] || [ "$MODE" = "combined" ]
}

mode_has_kvc() {
  case "$MODE" in
    kvc-get|kvc-set|kvc-mixed|combined) return 0 ;;
    *) return 1 ;;
  esac
}

kvc_mode() {
  case "$MODE" in
    kvc-get) echo get ;;
    kvc-set) echo set ;;
    kvc-mixed|combined) echo mixed ;;
    *) echo "" ;;
  esac
}

validate() {
  case "$MODE" in
    baseline|brpc|kvc-get|kvc-set|kvc-mixed|combined) ;;
    *) die "MODE must be baseline, brpc, kvc-get, kvc-set, kvc-mixed, or combined" ;;
  esac
  local value
  for value in "$REPEATS" "$EXPECTED_OFFLOADS" "$EXPECTED_ONBOARDS" "$PRIME_REQUESTS"; do
    [[ "$value" =~ ^[0-9]+$ ]] || die "repeat/count parameters must be non-negative integers"
  done
  [ "$REPEATS" -gt 0 ] || die "REPEATS must be positive"
  [ "$PRIME_REQUESTS" -gt 0 ] || die "PRIME_REQUESTS must be positive"
  if mode_has_kvc && [ "$KVC_BATCH_NUM" -ne 1 ]; then
    die "KVC_BATCH_NUM must be 1 to match current single-key TRT KVC calls"
  fi
}

read_counter() {
  local interface="$1"
  local counter="$2"
  local path="/sys/class/net/${interface}/statistics/${counter}"
  if [ -r "$path" ]; then
    cat "$path"
  else
    echo 0
  fi
}

read_remote_counter() {
  local counter="$1"
  ssh "$KVC_LOAD_HOST" "cat /sys/class/net/${REMOTE_NETWORK_INTERFACE}/statistics/${counter} 2>/dev/null || echo 0"
}

build_brpc_probe() {
  if ! mode_has_brpc; then
    return
  fi
  log "Build BRPC pressure client"
  GOPROXY="${GOPROXY:-off}" GOSUMDB="${GOSUMDB:-off}" \
    go build -mod=vendor -o "$BRPC_PROBE_BIN" ./scripts/probe_go_brpc_client.go
}

start_brpc_load() {
  local round_dir="$1"
  if ! mode_has_brpc; then
    return
  fi
  "$BRPC_PROBE_BIN" \
    --endpoint="$BRPC_ENDPOINT" \
    --method=health \
    --requests="$BRPC_LOAD_REQUESTS" \
    --concurrency="$BRPC_LOAD_CONCURRENCY" \
    --payload_bytes="$BRPC_LOAD_PAYLOAD_BYTES" \
    --timeout_ms="$BRPC_LOAD_TIMEOUT_MS" \
    --max_retries=0 \
    --quiet=true \
    >"${round_dir}/brpc-load.log" 2>&1 &
  BRPC_LOAD_PID="$!"
  echo "$BRPC_LOAD_PID" >"${round_dir}/brpc-load.pid"
}

start_kvc_load() {
  local round="$1"
  local round_dir="$2"
  if ! mode_has_kvc; then
    return
  fi
  local remote_run_id="${RUN_ID//[^a-zA-Z0-9]/}_${round}"
  KVC_REMOTE_PID_FILE="/tmp/dsbench-pressure-${remote_run_id}.pid"
  KVC_REMOTE_READY_FILE="/tmp/dsbench-pressure-${remote_run_id}.ready"
  local remote_mode
  remote_mode="$(kvc_mode)"

  ssh "$KVC_LOAD_HOST" \
    "cd '$KVC_REMOTE_REPO' && env MODE='$remote_mode' DS_ENDPOINT='$KVC_DS_ENDPOINT' DSBENCH_CPP='$KVC_DSBENCH_CPP' OBJECT_SIZE='$KVC_OBJECT_SIZE' KEY_COUNT='$KVC_KEY_COUNT' BATCH_NUM='$KVC_BATCH_NUM' THREAD_NUM='$KVC_THREAD_NUM' GET_CLIENTS='$KVC_GET_CLIENTS' SET_CLIENTS='$KVC_SET_CLIENTS' TASKSET_CPUS='$KVC_TASKSET_CPUS' DURATION_SECONDS='$KVC_LOAD_DURATION_SECONDS' RUN_ID='$remote_run_id' PID_FILE='$KVC_REMOTE_PID_FILE' READY_FILE='$KVC_REMOTE_READY_FILE' bash scripts/run_datasystem_dsbench_pressure.sh" \
    >"${round_dir}/kvc-load.log" 2>&1 &
  KVC_SSH_PID="$!"
  echo "$KVC_SSH_PID" >"${round_dir}/kvc-load-ssh.pid"

  local attempt
  for attempt in $(seq 1 60); do
    if ssh "$KVC_LOAD_HOST" "test -f '$KVC_REMOTE_READY_FILE'" >/dev/null 2>&1; then
      return
    fi
    if ! kill -0 "$KVC_SSH_PID" >/dev/null 2>&1; then
      cat "${round_dir}/kvc-load.log" >&2 || true
      die "remote dsbench pressure exited before becoming ready"
    fi
    sleep 1
  done
  die "remote dsbench pressure did not become ready within 60 seconds"
}

stop_loads() {
  if [ -n "$BRPC_LOAD_PID" ]; then
    kill "$BRPC_LOAD_PID" >/dev/null 2>&1 || true
    wait "$BRPC_LOAD_PID" >/dev/null 2>&1 || true
    BRPC_LOAD_PID=""
  fi
  if [ -n "$KVC_REMOTE_PID_FILE" ]; then
    ssh "$KVC_LOAD_HOST" \
      "if test -f '$KVC_REMOTE_PID_FILE'; then kill -TERM \$(cat '$KVC_REMOTE_PID_FILE') 2>/dev/null || true; fi" \
      >/dev/null 2>&1 || true
  fi
  if [ -n "$KVC_SSH_PID" ]; then
    wait "$KVC_SSH_PID" >/dev/null 2>&1 || true
    KVC_SSH_PID=""
  fi
  KVC_REMOTE_PID_FILE=""
  KVC_REMOTE_READY_FILE=""
}

cleanup() {
  stop_loads
}

terminate() {
  cleanup
  trap - EXIT
  exit 130
}

trap cleanup EXIT
trap terminate INT TERM

run_prime() {
  local round_dir="$1"
  log "Prime cache state"
  ENDPOINT="$BRPC_ENDPOINT" \
  REQUESTS="$PRIME_REQUESTS" \
  CONCURRENCY=1 \
  TOPK=1 \
  TIMEOUT_MS=120000 \
  MAX_RETRIES=0 \
  HISTORY_SOURCE=user_features \
  UIDS="$PRIME_UIDS" \
  USER_FEATURES_PATH="$USER_FEATURES_PATH" \
  SEMANTIC_MAP_PATH="$SEMANTIC_MAP_PATH" \
  HISTORY_MAX_LENGTH="$HISTORY_MAX_LENGTH" \
  VARY_USER_ID=true \
  OUT_DIR="${round_dir}/prime" \
    bash scripts/benchmark_go_brpc_probe_kvc_latency.sh \
    >"${round_dir}/prime.console.log" 2>&1
}

run_replay() {
  local round_dir="$1"
  log "Replay one PaiRec request"
  NAMESPACE="$NAMESPACE" \
  PAIREC_TARGET="$PAIREC_TARGET" \
  BRPC_TARGET="$BRPC_TARGET" \
  BRPC_CONTAINER="$BRPC_CONTAINER" \
  USER_ID="$REPLAY_USER_ID" \
  SIZE="$REPLAY_SIZE" \
  TIMEOUT="$REPLAY_TIMEOUT" \
  OUT_DIR="${round_dir}/replay" \
    bash scripts/trace_single_brpc_datasystem_request.sh \
    >"${round_dir}/replay.console.log" 2>&1
}

summarize() {
  python3 - "$OUT_DIR" "$MODE" "$REPEATS" "$EXPECTED_OFFLOADS" "$EXPECTED_ONBOARDS" \
    "$STRICT_COUNTS" "$RESULT_JSON" <<'PY' | tee "$SUMMARY_TXT"
import glob
import json
import math
import os
import statistics
import sys

out_dir, mode, repeats, expected_offloads, expected_onboards, strict_counts, result_path = sys.argv[1:8]
repeats = int(repeats)
expected_offloads = int(expected_offloads)
expected_onboards = int(expected_onboards)
strict_counts = strict_counts == "1"

def number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def read_int(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (FileNotFoundError, ValueError):
        return 0

def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    return values[low] * (high - rank) + values[high] * (rank - low)

rows = []
for path in sorted(glob.glob(os.path.join(out_dir, "round-*", "replay", "summary.json"))):
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    client = raw.get("client") or {}
    trace = raw.get("pairec_generative_trace") or {}
    access = raw.get("kvc_access") or {}
    brpc_events = raw.get("brpc_events") or []
    offloads = access.get("offload_events") or []
    onboards = access.get("onboard_events") or []
    e2e_ms = number(client.get("client_e2e_ms"))
    server_ms = sum(number(event.get("latency_ms")) for event in brpc_events)
    rpc_ms = number(trace.get("rpc_ms", trace.get("brpc_ms")), server_ms)
    offload_ms = sum(number(event.get("total_ms")) for event in offloads)
    onboard_ms = sum(number(event.get("total_ms")) for event in onboards)
    kvc_ms = offload_ms + onboard_ms
    brpc_ms = max(0.0, rpc_ms - server_ms)
    outer_ms = max(0.0, e2e_ms - rpc_ms)
    server_other_ms = max(0.0, server_ms - kvc_ms)
    round_dir = os.path.dirname(os.path.dirname(path))
    local_rx_delta = max(0, read_int(os.path.join(round_dir, "local-rx.after")) - read_int(os.path.join(round_dir, "local-rx.before")))
    local_tx_delta = max(0, read_int(os.path.join(round_dir, "local-tx.after")) - read_int(os.path.join(round_dir, "local-tx.before")))
    remote_rx_delta = max(0, read_int(os.path.join(round_dir, "remote-rx.after")) - read_int(os.path.join(round_dir, "remote-rx.before")))
    remote_tx_delta = max(0, read_int(os.path.join(round_dir, "remote-tx.after")) - read_int(os.path.join(round_dir, "remote-tx.before")))
    count_ok = len(offloads) == expected_offloads and len(onboards) == expected_onboards
    response_ok = client.get("ok") is True and raw.get("response_code") == 200 and len(brpc_events) == 1
    rows.append({
        "round": os.path.basename(os.path.dirname(os.path.dirname(path))).split("-")[-1],
        "valid": response_ok and (count_ok or not strict_counts),
        "response_ok": response_ok,
        "count_ok": count_ok,
        "e2e_ms": e2e_ms,
        "rpc_ms": rpc_ms,
        "server_ms": server_ms,
        "brpc_ms": brpc_ms,
        "offload_count": len(offloads),
        "onboard_count": len(onboards),
        "offload_ms": offload_ms,
        "onboard_ms": onboard_ms,
        "kvc_ms": kvc_ms,
        "server_other_ms": server_other_ms,
        "outer_ms": outer_ms,
        "local_rx_bytes": local_rx_delta,
        "local_tx_bytes": local_tx_delta,
        "remote_rx_bytes": remote_rx_delta,
        "remote_tx_bytes": remote_tx_delta,
        "summary_path": path,
    })

valid = [row for row in rows if row["valid"]]
metrics = {}
for key in ("e2e_ms", "server_ms", "brpc_ms", "kvc_ms", "offload_ms", "onboard_ms", "server_other_ms", "outer_ms"):
    values = [row[key] for row in valid]
    metrics[key] = None if not values else {
        "avg": statistics.mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }

status = "PASS" if len(valid) == repeats else "FAIL"
result = {
    "mode": mode,
    "status": status,
    "expected_repeats": repeats,
    "valid_repeats": len(valid),
    "expected_counts": {"offload": expected_offloads, "onboard": expected_onboards},
    "strict_counts": strict_counts,
    "rows": rows,
    "metrics": metrics,
}
with open(result_path, "w", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)

print("BRPC/KVC contention summary")
print(f"  mode={mode} status={status} valid_repeats={len(valid)}/{repeats}")
print("  round valid e2e_ms server_ms brpc_ms kvc_ms offloads onboards server_other_ms outer_ms local_rx local_tx")
for row in rows:
    print(
        f"  {row['round']:>5} {str(row['valid']):>5} {row['e2e_ms']:>7.3f} {row['server_ms']:>9.3f} "
        f"{row['brpc_ms']:>7.3f} {row['kvc_ms']:>6.3f} {row['offload_count']:>8} "
        f"{row['onboard_count']:>8} {row['server_other_ms']:>15.3f} {row['outer_ms']:>8.3f} "
        f"{row['local_rx_bytes']:>8} {row['local_tx_bytes']:>8}"
    )
if valid:
    print("  aggregate averages:")
    for key in ("e2e_ms", "server_ms", "brpc_ms", "kvc_ms", "offload_ms", "onboard_ms", "server_other_ms", "outer_ms"):
        print(f"    {key}={metrics[key]['avg']:.3f}")
    e2e_avg = metrics["e2e_ms"]["avg"]
    print(f"    brpc_e2e_pct={metrics['brpc_ms']['avg'] / e2e_avg * 100:.2f}%")
    print(f"    kvc_e2e_pct={metrics['kvc_ms']['avg'] / e2e_avg * 100:.2f}%")
print(f"  result_json={result_path}")
raise SystemExit(0 if status == "PASS" else 1)
PY
}

mkdir -p "$OUT_DIR"
require_command bash
require_command go
require_command kubectl
require_command python3
if mode_has_kvc; then
  require_command ssh
fi
validate
build_brpc_probe

cat >"${OUT_DIR}/config.txt" <<EOF
mode=${MODE}
repeats=${REPEATS}
brpc_endpoint=${BRPC_ENDPOINT}
brpc_load_concurrency=${BRPC_LOAD_CONCURRENCY}
brpc_load_payload_bytes=${BRPC_LOAD_PAYLOAD_BYTES}
kvc_load_host=${KVC_LOAD_HOST}
kvc_ds_endpoint=${KVC_DS_ENDPOINT}
kvc_object_size=${KVC_OBJECT_SIZE}
kvc_key_count=${KVC_KEY_COUNT}
kvc_batch_num=${KVC_BATCH_NUM}
kvc_thread_num=${KVC_THREAD_NUM}
kvc_get_clients=${KVC_GET_CLIENTS}
kvc_set_clients=${KVC_SET_CLIENTS}
expected_offloads=${EXPECTED_OFFLOADS}
expected_onboards=${EXPECTED_ONBOARDS}
EOF

log "Experiment configuration"
cat "${OUT_DIR}/config.txt"

overall_code=0
for round in $(seq 1 "$REPEATS"); do
  round_dir="${OUT_DIR}/round-${round}"
  mkdir -p "$round_dir"
  log "Round ${round}/${REPEATS}: mode=${MODE}"

  run_prime "$round_dir"

  read_counter "$NETWORK_INTERFACE" rx_bytes >"${round_dir}/local-rx.before"
  read_counter "$NETWORK_INTERFACE" tx_bytes >"${round_dir}/local-tx.before"
  if mode_has_kvc; then
    read_remote_counter rx_bytes >"${round_dir}/remote-rx.before"
    read_remote_counter tx_bytes >"${round_dir}/remote-tx.before"
  fi

  start_brpc_load "$round_dir"
  start_kvc_load "$round" "$round_dir"
  sleep "$LOAD_SETTLE_SECONDS"
  if [ -n "$BRPC_LOAD_PID" ] && ! kill -0 "$BRPC_LOAD_PID" >/dev/null 2>&1; then
    cat "${round_dir}/brpc-load.log" >&2 || true
    die "BRPC pressure exited before replay"
  fi
  if [ -n "$KVC_SSH_PID" ] && ! kill -0 "$KVC_SSH_PID" >/dev/null 2>&1; then
    cat "${round_dir}/kvc-load.log" >&2 || true
    die "KVC pressure exited before replay"
  fi

  set +e
  run_replay "$round_dir"
  replay_code="$?"
  set -e
  echo "$replay_code" >"${round_dir}/replay.exit_code"
  [ "$replay_code" -eq 0 ] || overall_code=1

  read_counter "$NETWORK_INTERFACE" rx_bytes >"${round_dir}/local-rx.after"
  read_counter "$NETWORK_INTERFACE" tx_bytes >"${round_dir}/local-tx.after"
  if mode_has_kvc; then
    read_remote_counter rx_bytes >"${round_dir}/remote-rx.after"
    read_remote_counter tx_bytes >"${round_dir}/remote-tx.after"
  fi
  stop_loads
done

log "Summarize"
set +e
summarize
summary_code="$?"
set -e
if [ "$summary_code" -ne 0 ]; then
  overall_code=1
fi

echo "output_dir=${OUT_DIR}"
exit "$overall_code"
