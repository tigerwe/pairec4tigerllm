#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NAMESPACE="${NAMESPACE:-pairec}"
BRPC_TARGET="${BRPC_TARGET:-deployment/inference-brpc-trtllm}"
BRPC_ENDPOINT="${BRPC_ENDPOINT:-192.168.100.11:18100}"
INFERENCE_ROLLOUT_TIMEOUT_SECONDS="${INFERENCE_ROLLOUT_TIMEOUT_SECONDS:-300}"
RESET_INFERENCE="${RESET_INFERENCE:-1}"

KVC_LOAD_HOST="${KVC_LOAD_HOST:-root@141.61.91.188}"
KVC_REMOTE_REPO="${KVC_REMOTE_REPO:-/home/zcx/workspace/pairec4tigerllm}"
KVC_DSBENCH_CPP="${KVC_DSBENCH_CPP:-/home/zcx/bin/dsbench-v081-sustained}"
KVC_DS_ENDPOINT="${KVC_DS_ENDPOINT:-192.168.100.12:18482}"
KVC_OBJECT_SIZE="${KVC_OBJECT_SIZE:-17920KB}"
KVC_GET_CLIENTS="${KVC_GET_CLIENTS:-4}"
KVC_SET_CLIENTS="${KVC_SET_CLIENTS:-6}"
KVC_GET_KEY_COUNT="${KVC_GET_KEY_COUNT:-4}"
KVC_SET_KEY_COUNT="${KVC_SET_KEY_COUNT:-6}"
KVC_DURATION_SECONDS="${KVC_DURATION_SECONDS:-300}"
KVC_READY_TIMEOUT_SECONDS="${KVC_READY_TIMEOUT_SECONDS:-300}"
LOAD_SETTLE_SECONDS="${LOAD_SETTLE_SECONDS:-3}"

PRIME_REQUESTS="${PRIME_REQUESTS:-195}"
PRIME_UIDS="${PRIME_UIDS:-5,6312,130,2184,303,1190,1191,1192,1193,1194}"
DEFAULT_DATA_ROOT="${REPO_ROOT}/data"
if [ ! -f "${DEFAULT_DATA_ROOT}/tenrec/processed/semantic_id_map.json" ] \
  && [ -f /home/zcx/workspace/pairec4tigerllm/data/tenrec/processed/semantic_id_map.json ]; then
  DEFAULT_DATA_ROOT=/home/zcx/workspace/pairec4tigerllm/data
fi
USER_FEATURES_PATH="${USER_FEATURES_PATH:-${DEFAULT_DATA_ROOT}/user_features.json}"
SEMANTIC_MAP_PATH="${SEMANTIC_MAP_PATH:-${DEFAULT_DATA_ROOT}/tenrec/processed/semantic_id_map.json}"
HISTORY_MAX_LENGTH="${HISTORY_MAX_LENGTH:-20}"
PRIME_BENCHMARK_SCRIPT="${PRIME_BENCHMARK_SCRIPT:-${REPO_ROOT}/scripts/benchmark_go_brpc_probe_kvc_latency.sh}"

RUN_ID="${RUN_ID:-kvcdebug-$(date +%Y%m%d%H%M%S)}"
SAFE_RUN_ID="${RUN_ID//[^a-zA-Z0-9_-]/_}"
OUT_DIR="${OUT_DIR:-/tmp/datasystem-sustained-get-lifecycle/${SAFE_RUN_ID}}"

REMOTE_PID_FILE="/tmp/dsbench-pressure-${SAFE_RUN_ID}.pid"
REMOTE_READY_FILE="/tmp/dsbench-pressure-${SAFE_RUN_ID}.ready"
REMOTE_PREPARED_FILE="/tmp/dsbench-pressure-${SAFE_RUN_ID}.prepared"
REMOTE_START_FILE="/tmp/dsbench-pressure-${SAFE_RUN_ID}.start"
REMOTE_STATS_FILE="/tmp/dsbench-pressure-${SAFE_RUN_ID}.stats.jsonl"
REMOTE_PREFIX="KvcLoad_${SAFE_RUN_ID}"
REMOTE_LOG="${OUT_DIR}/kvc-load.log"

mkdir -p "$OUT_DIR"

KVC_SSH_PID=""
STOPPED=0

log() {
  printf '\n== %s ==\n' "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

validate_positive() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "${name} must be positive"
}

remote_process_running() {
  ssh "$KVC_LOAD_HOST" \
    "test -s '$REMOTE_PID_FILE' && pid=\$(cat '$REMOTE_PID_FILE') && test -r /proc/\$pid/stat && test \"\$(awk '{ print \$3 }' /proc/\$pid/stat)\" != Z" \
    >/dev/null 2>&1
}

collect_remote_state() {
  ssh "$KVC_LOAD_HOST" \
    "for file in '$REMOTE_PREPARED_FILE' '$REMOTE_READY_FILE' '$REMOTE_STATS_FILE.get' '$REMOTE_STATS_FILE.set'; do if test -f \"\$file\"; then echo \"--- \$file ---\"; cat \"\$file\"; fi; done" \
    >"${OUT_DIR}/remote-state.txt" 2>/dev/null || true
}

stop_remote_pressure() {
  [ "$STOPPED" -eq 0 ] || return
  STOPPED=1
  if [ -n "$KVC_LOAD_HOST" ]; then
    ssh "$KVC_LOAD_HOST" \
      "if test -s '$REMOTE_PID_FILE'; then kill -TERM \$(cat '$REMOTE_PID_FILE') 2>/dev/null || true; fi" \
      >/dev/null 2>&1 || true
  fi
  if [ -n "$KVC_SSH_PID" ]; then
    wait "$KVC_SSH_PID" >/dev/null 2>&1 || true
    KVC_SSH_PID=""
  fi
}

cleanup() {
  collect_remote_state
  stop_remote_pressure
}

trap cleanup EXIT
trap 'exit 130' INT TERM

wait_remote_file() {
  local file="$1"
  local description="$2"
  local attempt
  for attempt in $(seq 1 "$KVC_READY_TIMEOUT_SECONDS"); do
    if ssh "$KVC_LOAD_HOST" "test -s '$file'" >/dev/null 2>&1; then
      ssh "$KVC_LOAD_HOST" "cat '$file'" | tee "${OUT_DIR}/${description}.txt"
      return
    fi
    if ! remote_process_running; then
      tail -100 "$REMOTE_LOG" >&2 || true
      die "remote dsbench exited before ${description}"
    fi
    sleep 1
  done
  tail -100 "$REMOTE_LOG" >&2 || true
  die "remote dsbench did not reach ${description} within ${KVC_READY_TIMEOUT_SECONDS}s"
}

probe_get_keys() {
  local stage="$1"
  local output="${OUT_DIR}/probe-${stage}.log"
  log "Probe four Get keys: ${stage}"
  set +e
  ssh "$KVC_LOAD_HOST" \
    "'$KVC_DSBENCH_CPP' kv --action=get --worker_address='$KVC_DS_ENDPOINT' --prefix='$REMOTE_PREFIX'_Get --client_num=1 --thread_num=1 --num='$KVC_GET_KEY_COUNT' --size='$KVC_OBJECT_SIZE' --batch_num=1 --worker_num=1 --worker_index=0" \
    2>&1 | tee "$output"
  local status="${PIPESTATUS[0]}"
  set -e
  echo "$status" >"${OUT_DIR}/probe-${stage}.exit_code"
  if [ "$status" -ne 0 ]; then
    return "$status"
  fi
  grep -q 'BENCHMARK-RESULT:get-' "$output" \
    || die "Get probe ${stage} returned success without benchmark evidence"
}

start_remote_pressure() {
  log "Start gated dsbench pressure on ${KVC_LOAD_HOST}"
  ssh "$KVC_LOAD_HOST" \
    "rm -f '$REMOTE_PID_FILE' '$REMOTE_READY_FILE' '$REMOTE_PREPARED_FILE' '$REMOTE_START_FILE' '$REMOTE_STATS_FILE' '$REMOTE_STATS_FILE.get' '$REMOTE_STATS_FILE.set'; cd '$KVC_REMOTE_REPO' && env PRESSURE_ENGINE=dsbench MODE=mixed DS_ENDPOINT='$KVC_DS_ENDPOINT' DSBENCH_CPP='$KVC_DSBENCH_CPP' DSBENCH_SUSTAINED=1 DSBENCH_READY_TIMEOUT_SECONDS='$KVC_READY_TIMEOUT_SECONDS' OBJECT_SIZE='$KVC_OBJECT_SIZE' KEY_COUNT=256 GET_KEY_COUNT='$KVC_GET_KEY_COUNT' SET_KEY_COUNT='$KVC_SET_KEY_COUNT' BATCH_NUM=1 THREAD_NUM=1 GET_CLIENTS='$KVC_GET_CLIENTS' SET_CLIENTS='$KVC_SET_CLIENTS' DURATION_SECONDS='$KVC_DURATION_SECONDS' CLEANUP_KEYS=1 REPORT_INTERVAL_SECONDS=1 RUN_ID='$SAFE_RUN_ID' PID_FILE='$REMOTE_PID_FILE' READY_FILE='$REMOTE_READY_FILE' PREPARED_FILE='$REMOTE_PREPARED_FILE' START_FILE='$REMOTE_START_FILE' STATS_FILE='$REMOTE_STATS_FILE' bash scripts/run_datasystem_dsbench_pressure.sh" \
    >"$REMOTE_LOG" 2>&1 &
  KVC_SSH_PID="$!"
  wait_remote_file "$REMOTE_PREPARED_FILE" prepared
}

reset_and_prime() {
  if [ "$RESET_INFERENCE" = "1" ]; then
    log "Reset inference cache state"
    kubectl -n "$NAMESPACE" rollout restart "$BRPC_TARGET" \
      | tee "${OUT_DIR}/inference-rollout-restart.log"
    kubectl -n "$NAMESPACE" rollout status "$BRPC_TARGET" \
      --timeout="${INFERENCE_ROLLOUT_TIMEOUT_SECONDS}s" \
      | tee "${OUT_DIR}/inference-rollout-status.log"
  fi

  log "Prime foreground cache state"
  (
    cd "$REPO_ROOT"
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
    OUT_DIR="${OUT_DIR}/prime" \
      bash "$PRIME_BENCHMARK_SCRIPT"
  ) | tee "${OUT_DIR}/prime.console.log"
}

release_pressure() {
  log "Release ten dsbench workers"
  ssh "$KVC_LOAD_HOST" "touch '$REMOTE_START_FILE'"
  wait_remote_file "$REMOTE_READY_FILE" ready
  sleep "$LOAD_SETTLE_SECONDS"
  if ! remote_process_running; then
    tail -100 "$REMOTE_LOG" >&2 || true
    die "remote dsbench exited during the settle window"
  fi
  collect_remote_state
}

for pair in \
  "KVC_GET_CLIENTS:$KVC_GET_CLIENTS" \
  "KVC_SET_CLIENTS:$KVC_SET_CLIENTS" \
  "KVC_GET_KEY_COUNT:$KVC_GET_KEY_COUNT" \
  "KVC_SET_KEY_COUNT:$KVC_SET_KEY_COUNT" \
  "KVC_DURATION_SECONDS:$KVC_DURATION_SECONDS" \
  "KVC_READY_TIMEOUT_SECONDS:$KVC_READY_TIMEOUT_SECONDS" \
  "PRIME_REQUESTS:$PRIME_REQUESTS"; do
  validate_positive "${pair%%:*}" "${pair#*:}"
done
case "$RESET_INFERENCE" in
  0|1) ;;
  *) die "RESET_INFERENCE must be 0 or 1" ;;
esac
[ "$KVC_GET_KEY_COUNT" -eq "$KVC_GET_CLIENTS" ] \
  || die "KVC_GET_KEY_COUNT must equal KVC_GET_CLIENTS"
[ "$KVC_SET_KEY_COUNT" -eq "$KVC_SET_CLIENTS" ] \
  || die "KVC_SET_KEY_COUNT must equal KVC_SET_CLIENTS"
[ -f "$PRIME_BENCHMARK_SCRIPT" ] \
  || die "prime benchmark script is missing"
[ -f "$USER_FEATURES_PATH" ] || die "user features file is missing: ${USER_FEATURES_PATH}"
[ -f "$SEMANTIC_MAP_PATH" ] || die "semantic map file is missing: ${SEMANTIC_MAP_PATH}"

cat >"${OUT_DIR}/config.txt" <<EOF
run_id=${SAFE_RUN_ID}
kvc_load_host=${KVC_LOAD_HOST}
kvc_remote_repo=${KVC_REMOTE_REPO}
kvc_dsbench_cpp=${KVC_DSBENCH_CPP}
kvc_ds_endpoint=${KVC_DS_ENDPOINT}
kvc_object_size=${KVC_OBJECT_SIZE}
kvc_get_clients=${KVC_GET_CLIENTS}
kvc_set_clients=${KVC_SET_CLIENTS}
prime_requests=${PRIME_REQUESTS}
prime_benchmark_script=${PRIME_BENCHMARK_SCRIPT}
reset_inference=${RESET_INFERENCE}
brpc_endpoint=${BRPC_ENDPOINT}
EOF

start_remote_pressure

if ! probe_get_keys before_reset; then
  cat >"${OUT_DIR}/diagnosis.txt" <<EOF
result=FAIL
failure_stage=before_reset
reason=background Get keys are unavailable immediately after dsbench preparation
EOF
  cat "${OUT_DIR}/diagnosis.txt"
  exit 1
fi

reset_and_prime

if ! probe_get_keys after_prime; then
  cat >"${OUT_DIR}/diagnosis.txt" <<EOF
result=FAIL
failure_stage=after_prime
reason=background Get keys existed before reset but disappeared during inference reset or prime
EOF
  cat "${OUT_DIR}/diagnosis.txt"
  exit 1
fi

release_pressure

cat >"${OUT_DIR}/diagnosis.txt" <<EOF
result=PASS
before_reset_get=PASS
after_prime_get=PASS
pressure_ready=PASS
next=the key lifecycle and pressure release path are healthy; rerun the KVC matrix
EOF

log "Diagnosis"
cat "${OUT_DIR}/diagnosis.txt"
echo "output_dir=${OUT_DIR}"
