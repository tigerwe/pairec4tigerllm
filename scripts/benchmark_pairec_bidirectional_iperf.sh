#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/pairec-bidirectional-iperf/${RUN_ID}}"
WORKER_NODE="${WORKER_NODE:-worker1}"
WORKER_SSH_HOST="${WORKER_SSH_HOST:-$WORKER_NODE}"
MASTER_DATA_IP="${MASTER_DATA_IP:-192.168.100.12}"
WORKER_DATA_IP="${WORKER_DATA_IP:-192.168.100.11}"
MASTER_SERVER_PORT="${MASTER_SERVER_PORT:-5212}"
WORKER_SERVER_PORT="${WORKER_SERVER_PORT:-5211}"
IPERF_STREAMS="${IPERF_STREAMS:-8}"
IPERF_DURATION_SECONDS="${IPERF_DURATION_SECONDS:-90}"
IPERF_OMIT_SECONDS="${IPERF_OMIT_SECONDS:-3}"
IPERF_SETTLE_SECONDS="${IPERF_SETTLE_SECONDS:-5}"
MIN_DIRECTION_GBPS="${MIN_DIRECTION_GBPS:-20}"
MASTER_CLIENT_CPUS="${MASTER_CLIENT_CPUS:-40-55}"
MASTER_SERVER_CPUS="${MASTER_SERVER_CPUS:-56-71}"
WORKER_SERVER_CPUS="${WORKER_SERVER_CPUS:-40-55}"
WORKER_CLIENT_CPUS="${WORKER_CLIENT_CPUS:-56-71}"
PRIME_REQUESTS="${PRIME_REQUESTS:-195}"
BRPC_ENDPOINT="${BRPC_ENDPOINT:-192.168.100.11:18100}"
REPLAY_USER_ID="${REPLAY_USER_ID:-5}"
REPLAY_SIZE="${REPLAY_SIZE:-1}"
REPLAY_TIMEOUT="${REPLAY_TIMEOUT:-30}"
USER_FEATURES_PATH="${USER_FEATURES_PATH:-$REPO_ROOT/data/user_features.json}"
SEMANTIC_MAP_PATH="${SEMANTIC_MAP_PATH:-$REPO_ROOT/data/tenrec/processed/semantic_id_map.json}"
PRIME_UIDS="${PRIME_UIDS:-5,6312,130,2184,303,1190,1191,1192,1193,1194}"
HISTORY_MAX_LENGTH="${HISTORY_MAX_LENGTH:-20}"

MASTER_SERVER_PID=""
MASTER_CLIENT_PID=""
WORKER_CLIENT_SSH_PID=""
REMOTE_SERVER_PID_FILE="/tmp/pairec-iperf-server-${RUN_ID}.pid"
REMOTE_CLIENT_PID_FILE="/tmp/pairec-iperf-client-${RUN_ID}.pid"

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

resolve_worker_host() {
  local configured="$WORKER_SSH_HOST"
  local user_prefix=""
  local node="$configured"
  local internal_ip
  if [[ "$configured" == *@* ]]; then
    user_prefix="${configured%@*}@"
    node="${configured##*@}"
  fi
  internal_ip="$(kubectl get node "$node" \
    -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}' \
    2>/dev/null || true)"
  if [ -n "$internal_ip" ]; then
    WORKER_SSH_HOST="${user_prefix}${internal_ip}"
  fi
}

stop_pid() {
  local pid="$1"
  if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  stop_pid "$MASTER_CLIENT_PID"
  stop_pid "$WORKER_CLIENT_SSH_PID"
  stop_pid "$MASTER_SERVER_PID"
  ssh "$WORKER_SSH_HOST" sh -s -- "$REMOTE_SERVER_PID_FILE" "$REMOTE_CLIENT_PID_FILE" \
    >/dev/null 2>&1 <<'SH' || true
for pid_file in "$@"; do
  if test -s "$pid_file"; then
    pid="$(cat "$pid_file")"
    kill "$pid" 2>/dev/null || true
    rm -f "$pid_file"
  fi
done
SH
}

terminate() {
  cleanup
  trap - EXIT
  exit 130
}

trap cleanup EXIT
trap terminate INT TERM

for command in bash ip iperf3 kubectl python3 ssh taskset; do
  require_command "$command"
done

for value in "$MASTER_SERVER_PORT" "$WORKER_SERVER_PORT" "$IPERF_STREAMS" \
  "$IPERF_DURATION_SECONDS" "$IPERF_OMIT_SECONDS" "$IPERF_SETTLE_SECONDS" \
  "$PRIME_REQUESTS" "$REPLAY_SIZE" "$REPLAY_TIMEOUT"; do
  [[ "$value" =~ ^[0-9]+$ ]] || die "integer configuration contains invalid value: $value"
done
[ "$IPERF_STREAMS" -gt 0 ] || die "IPERF_STREAMS must be positive"
[ "$IPERF_DURATION_SECONDS" -gt "$IPERF_SETTLE_SECONDS" ] \
  || die "IPERF_DURATION_SECONDS must exceed IPERF_SETTLE_SECONDS"

resolve_worker_host
mkdir -p "$OUT_DIR"

if [ ! -s "$SEMANTIC_MAP_PATH" ]; then
  fallback="/home/zcx/workspace/pairec4tigerllm/data/tenrec/processed/semantic_id_map.json"
  [ -s "$fallback" ] && SEMANTIC_MAP_PATH="$fallback"
fi
[ -s "$USER_FEATURES_PATH" ] || die "user features not found: $USER_FEATURES_PATH"
[ -s "$SEMANTIC_MAP_PATH" ] || die "semantic map not found: $SEMANTIC_MAP_PATH"

ip -o addr show | grep -F "${MASTER_DATA_IP}/" >/dev/null \
  || die "master data IP not present: $MASTER_DATA_IP"
ssh "$WORKER_SSH_HOST" \
  "command -v iperf3 >/dev/null && command -v taskset >/dev/null && ip -o addr show | grep -F '${WORKER_DATA_IP}/' >/dev/null" \
  || die "worker iperf/taskset/data-IP preflight failed: host=$WORKER_SSH_HOST ip=$WORKER_DATA_IP"

cat >"$OUT_DIR/config.txt" <<EOF
worker_node=$WORKER_NODE
worker_ssh_host=$WORKER_SSH_HOST
master_data_ip=$MASTER_DATA_IP
worker_data_ip=$WORKER_DATA_IP
master_server_port=$MASTER_SERVER_PORT
worker_server_port=$WORKER_SERVER_PORT
iperf_streams=$IPERF_STREAMS
iperf_duration_seconds=$IPERF_DURATION_SECONDS
iperf_omit_seconds=$IPERF_OMIT_SECONDS
iperf_settle_seconds=$IPERF_SETTLE_SECONDS
min_direction_gbps=$MIN_DIRECTION_GBPS
prime_requests=$PRIME_REQUESTS
brpc_endpoint=$BRPC_ENDPOINT
replay_user_id=$REPLAY_USER_ID
user_features_path=$USER_FEATURES_PATH
semantic_map_path=$SEMANTIC_MAP_PATH
EOF

log "Configuration"
cat "$OUT_DIR/config.txt"

if [ "$PRIME_REQUESTS" -gt 0 ]; then
  log "Prime deterministic DataSystem cache shape: ${PRIME_REQUESTS} requests"
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
  OUT_DIR="$OUT_DIR/prime" \
    bash scripts/benchmark_go_brpc_probe_kvc_latency.sh \
    >"$OUT_DIR/prime.console.log" 2>&1
  grep -F "recommend ok index=${PRIME_REQUESTS} " "$OUT_DIR/prime/go_brpc_probe.log" >/dev/null \
    || die "prime did not complete request ${PRIME_REQUESTS}"
fi

log "Start one-shot iperf servers"
taskset -c "$MASTER_SERVER_CPUS" \
  iperf3 -s -1 -B "$MASTER_DATA_IP" -p "$MASTER_SERVER_PORT" \
  >"$OUT_DIR/master-server.log" 2>&1 &
MASTER_SERVER_PID=$!

ssh "$WORKER_SSH_HOST" sh -s -- \
  "$WORKER_SERVER_CPUS" "$WORKER_DATA_IP" "$WORKER_SERVER_PORT" \
  "$REMOTE_SERVER_PID_FILE" "/tmp/pairec-iperf-worker-server-${RUN_ID}.log" <<'SH'
cpus="$1"
bind_ip="$2"
port="$3"
pid_file="$4"
log_file="$5"
nohup taskset -c "$cpus" iperf3 -s -1 -B "$bind_ip" -p "$port" \
  >"$log_file" 2>&1 &
echo $! >"$pid_file"
SH
sleep 1

log "Start full-duplex iperf traffic"
taskset -c "$MASTER_CLIENT_CPUS" \
  iperf3 -c "$WORKER_DATA_IP" -B "$MASTER_DATA_IP" -p "$WORKER_SERVER_PORT" \
  -P "$IPERF_STREAMS" -t "$IPERF_DURATION_SECONDS" -O "$IPERF_OMIT_SECONDS" \
  --get-server-output >"$OUT_DIR/master-to-worker.log" 2>&1 &
MASTER_CLIENT_PID=$!

ssh "$WORKER_SSH_HOST" sh -s -- \
  "$WORKER_CLIENT_CPUS" "$MASTER_DATA_IP" "$WORKER_DATA_IP" "$MASTER_SERVER_PORT" \
  "$IPERF_STREAMS" "$IPERF_DURATION_SECONDS" "$IPERF_OMIT_SECONDS" \
  "$REMOTE_CLIENT_PID_FILE" <<'SH' >"$OUT_DIR/worker-to-master.log" 2>&1 &
cpus="$1"
server_ip="$2"
bind_ip="$3"
port="$4"
streams="$5"
duration="$6"
omit="$7"
pid_file="$8"
echo $$ >"$pid_file"
exec taskset -c "$cpus" iperf3 -c "$server_ip" -B "$bind_ip" -p "$port" \
  -P "$streams" -t "$duration" -O "$omit" --get-server-output
SH
WORKER_CLIENT_SSH_PID=$!

sleep "$IPERF_SETTLE_SECONDS"
kill -0 "$MASTER_CLIENT_PID" >/dev/null 2>&1 \
  || die "master-to-worker iperf exited before replay"
kill -0 "$WORKER_CLIENT_SSH_PID" >/dev/null 2>&1 \
  || die "worker-to-master iperf exited before replay"

log "Replay one full PaiRec request under full-duplex NIC pressure"
set +e
OUT_DIR="$OUT_DIR/replay" \
USER_ID="$REPLAY_USER_ID" \
SIZE="$REPLAY_SIZE" \
TIMEOUT="$REPLAY_TIMEOUT" \
REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION=1 \
  bash scripts/trace_single_brpc_datasystem_request.sh \
  >"$OUT_DIR/replay.console.log" 2>&1
replay_code=$?
set -e

log "Wait for iperf completion"
set +e
wait "$MASTER_CLIENT_PID"
master_client_code=$?
MASTER_CLIENT_PID=""
wait "$WORKER_CLIENT_SSH_PID"
worker_client_code=$?
WORKER_CLIENT_SSH_PID=""
wait "$MASTER_SERVER_PID"
master_server_code=$?
MASTER_SERVER_PID=""
set -e

python3 - "$OUT_DIR" "$MIN_DIRECTION_GBPS" "$replay_code" \
  "$master_client_code" "$worker_client_code" "$master_server_code" <<'PY'
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
minimum = float(sys.argv[2])
codes = {
    "replay": int(sys.argv[3]),
    "master_to_worker": int(sys.argv[4]),
    "worker_to_master": int(sys.argv[5]),
    "master_server": int(sys.argv[6]),
}

def receiver_gbps(path):
    values = []
    for line in path.read_text(errors="replace").splitlines():
        if "[SUM]" not in line or "receiver" not in line:
            continue
        match = re.search(r"([0-9.]+)\s+Gbits/sec", line)
        if match:
            values.append(float(match.group(1)))
    return values[-1] if values else 0.0

def last_event(path, event_name):
    found = None
    for line in path.read_text(errors="replace").splitlines():
        if f'"event":"{event_name}"' not in line:
            continue
        try:
            found = json.loads(line[line.index("{"):])
        except (ValueError, json.JSONDecodeError):
            continue
    return found

m2w = receiver_gbps(root / "master-to-worker.log")
w2m = receiver_gbps(root / "worker-to-master.log")
trt_log = root / "replay" / "brpc_trtllm.log"
native = last_event(trt_log, "datasystem_request_complete") if trt_log.exists() else None
runner = last_event(trt_log, "trt_executor_request_complete") if trt_log.exists() else None
get_count = int((native or {}).get("get_count", 0))
valid = (
    all(code == 0 for code in codes.values())
    and m2w >= minimum
    and w2m >= minimum
    and get_count >= 1
)
result = {
    "classification": "PAIREC_BIDIRECTIONAL_IPERF_VALID" if valid else "PAIREC_BIDIRECTIONAL_IPERF_INVALID",
    "valid": valid,
    "exit_codes": codes,
    "master_to_worker_gbps": m2w,
    "worker_to_master_gbps": w2m,
    "minimum_direction_gbps": minimum,
    "datasystem_request_complete": native,
    "trt_executor_request_complete": runner,
}
(root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(f"classification={result['classification']}")
print(f"master_to_worker_gbps={m2w:.3f}")
print(f"worker_to_master_gbps={w2m:.3f}")
if native:
    print(f"datasystem_get_count={get_count}")
    print(f"datasystem_get_ms={native.get('get_us', 0) / 1000:.3f}")
    print(f"datasystem_set_count={native.get('set_count', 0)}")
    print(f"datasystem_set_ms={native.get('set_us', 0) / 1000:.3f}")
if runner:
    print(f"runner_ms={runner.get('runner_us', 0) / 1000:.3f}")
print(f"summary_json={root / 'summary.json'}")
raise SystemExit(0 if valid else 1)
PY

echo "output_dir=$OUT_DIR"
echo "PAIREC_BIDIRECTIONAL_IPERF_COMPLETE"
