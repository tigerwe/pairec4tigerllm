#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-baseline}"
REPEATS="${REPEATS:-3}"
STRICT_COUNTS="${STRICT_COUNTS:-1}"
EXPECTED_OFFLOADS="${EXPECTED_OFFLOADS:-3}"
EXPECTED_ONBOARDS="${EXPECTED_ONBOARDS:-2}"
EXPECTED_ONBOARDS_MIN="${EXPECTED_ONBOARDS_MIN:-$EXPECTED_ONBOARDS}"
EXPECTED_ONBOARDS_MAX="${EXPECTED_ONBOARDS_MAX:-$EXPECTED_ONBOARDS}"
REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION="${REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION:-0}"

BRPC_ENDPOINT="${BRPC_ENDPOINT:-192.168.100.11:18100}"
BRPC_LOAD_ENDPOINT="${BRPC_LOAD_ENDPOINT:-$BRPC_ENDPOINT}"
BRPC_LOAD_CONCURRENCY="${BRPC_LOAD_CONCURRENCY:-10}"
BRPC_LOAD_QPS="${BRPC_LOAD_QPS:-0}"
BRPC_LOAD_PAYLOAD_BYTES="${BRPC_LOAD_PAYLOAD_BYTES:-102400}"
BRPC_LOAD_REQUESTS="${BRPC_LOAD_REQUESTS:-1000000}"
BRPC_LOAD_TIMEOUT_MS="${BRPC_LOAD_TIMEOUT_MS:-5000}"
BRPC_LOAD_REUSE_CONNECTIONS="${BRPC_LOAD_REUSE_CONNECTIONS:-1}"
BRPC_LOAD_READY_TIMEOUT_SECONDS="${BRPC_LOAD_READY_TIMEOUT_SECONDS:-30}"
BRPC_PROBE_BIN="${BRPC_PROBE_BIN:-/tmp/probe-go-brpc-client}"

KVC_LOAD_HOST="${KVC_LOAD_HOST:-worker1}"
KVC_REMOTE_REPO="${KVC_REMOTE_REPO:-/home/zcx/workspace/pairec4tigerllm}"
KVC_DS_ENDPOINT="${KVC_DS_ENDPOINT:-192.168.100.12:18482}"
KVC_PRESSURE_ENGINE="${KVC_PRESSURE_ENGINE:-dsbench}"
KVC_DSBENCH_CPP="${KVC_DSBENCH_CPP:-}"
KVC_DATASYSTEM_SDK_DIR="${KVC_DATASYSTEM_SDK_DIR:-}"
KVC_PERSISTENT_BIN="${KVC_PERSISTENT_BIN:-/tmp/datasystem_kv_pressure}"
KVC_OBJECT_SIZE="${KVC_OBJECT_SIZE:-3584KB}"
KVC_KEY_COUNT="${KVC_KEY_COUNT:-256}"
KVC_BATCH_NUM="${KVC_BATCH_NUM:-1}"
KVC_THREAD_NUM="${KVC_THREAD_NUM:-1}"
KVC_GET_CLIENTS="${KVC_GET_CLIENTS:-4}"
KVC_SET_CLIENTS="${KVC_SET_CLIENTS:-6}"
KVC_GET_KEY_COUNT="${KVC_GET_KEY_COUNT:-$KVC_GET_CLIENTS}"
KVC_SET_KEY_COUNT="${KVC_SET_KEY_COUNT:-$KVC_SET_CLIENTS}"
KVC_TASKSET_CPUS="${KVC_TASKSET_CPUS:-}"
KVC_LOAD_DURATION_SECONDS="${KVC_LOAD_DURATION_SECONDS:-300}"
KVC_REPORT_INTERVAL_SECONDS="${KVC_REPORT_INTERVAL_SECONDS:-1}"
KVC_LOAD_READY_TIMEOUT_SECONDS="${KVC_LOAD_READY_TIMEOUT_SECONDS:-300}"
KVC_DSBENCH_SUSTAINED="${KVC_DSBENCH_SUSTAINED:-0}"

LOAD_SETTLE_SECONDS="${LOAD_SETTLE_SECONDS:-3}"
ROUND_COOLDOWN_SECONDS="${ROUND_COOLDOWN_SECONDS:-5}"
PRIME_REQUESTS="${PRIME_REQUESTS:-200}"
PRIME_MAX_ATTEMPTS="${PRIME_MAX_ATTEMPTS:-3}"
PRIME_RETRY_DELAY_SECONDS="${PRIME_RETRY_DELAY_SECONDS:-10}"
PRIME_UIDS="${PRIME_UIDS:-5,6312,130,2184,303,1190,1191,1192,1193,1194}"
USER_FEATURES_PATH="${USER_FEATURES_PATH:-/home/zcx/workspace/pairec4tigerllm/data/user_features.json}"
SEMANTIC_MAP_PATH="${SEMANTIC_MAP_PATH:-/home/zcx/workspace/pairec4tigerllm/data/tenrec/processed/semantic_id_map.json}"
HISTORY_MAX_LENGTH="${HISTORY_MAX_LENGTH:-20}"
REPLAY_USER_ID="${REPLAY_USER_ID:-5}"
REPLAY_SIZE="${REPLAY_SIZE:-1}"
REPLAY_TIMEOUT="${REPLAY_TIMEOUT:-30}"
RESET_INFERENCE_BEFORE_ROUND="${RESET_INFERENCE_BEFORE_ROUND:-0}"
RESET_INFERENCE_MODE="${RESET_INFERENCE_MODE:-rollout}"
INFERENCE_ROLLOUT_TIMEOUT_SECONDS="${INFERENCE_ROLLOUT_TIMEOUT_SECONDS:-600}"
INFERENCE_CONTAINER_RESTART_TIMEOUT_SECONDS="${INFERENCE_CONTAINER_RESTART_TIMEOUT_SECONDS:-120}"
INFERENCE_RUNTIME_SSH_HOST="${INFERENCE_RUNTIME_SSH_HOST:-}"
MIN_ROOT_AVAILABLE_KB="${MIN_ROOT_AVAILABLE_KB:-0}"

NAMESPACE="${NAMESPACE:-pairec}"
PAIREC_TARGET="${PAIREC_TARGET:-deploy/pairec}"
BRPC_TARGET="${BRPC_TARGET:-deployment/inference-brpc-trtllm}"
BRPC_CONTAINER="${BRPC_CONTAINER:-brpc-inference}"
KVC_BURST_CONTAINER="${KVC_BURST_CONTAINER:-}"
KVC_BURST_DYNAMIC_ARM="${KVC_BURST_DYNAMIC_ARM:-0}"
KVC_BURST_PRESSURE_KEY_COUNT="${KVC_BURST_PRESSURE_KEY_COUNT:-0}"
KVC_BURST_CONTROL_BIN="${KVC_BURST_CONTROL_BIN:-/opt/pairec-f19/bin/kvc_burst_wrapper}"
BRPC_LOAD_POD_SELECTOR="${BRPC_LOAD_POD_SELECTOR:-app=brpc-pressure-target}"
BRPC_LOAD_CONTAINER="${BRPC_LOAD_CONTAINER:-brpc-pressure-target}"
NETWORK_INTERFACE="${NETWORK_INTERFACE:-enp41s0f1}"
REMOTE_NETWORK_INTERFACE="${REMOTE_NETWORK_INTERFACE:-enp41s0f1}"
DS_WORKER_METRICS="${DS_WORKER_METRICS:-1}"
DS_WORKER_POD_SELECTOR="${DS_WORKER_POD_SELECTOR:-app=datasystem-25g-master}"
DS_WORKER_CONTAINER="${DS_WORKER_CONTAINER:-datasystem-worker}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/brpc-kvc-contention/${RUN_ID}-${MODE}}"
RESULT_JSON="${OUT_DIR}/result.json"
SUMMARY_TXT="${OUT_DIR}/summary.txt"

BRPC_LOAD_PID=""
KVC_SSH_PID=""
KVC_REMOTE_PID_FILE=""
KVC_REMOTE_READY_FILE=""
KVC_REMOTE_PREPARED_FILE=""
KVC_REMOTE_START_FILE=""
KVC_REMOTE_STATS_FILE=""

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
  for value in "$REPEATS" "$EXPECTED_OFFLOADS" "$EXPECTED_ONBOARDS" \
    "$EXPECTED_ONBOARDS_MIN" "$EXPECTED_ONBOARDS_MAX" "$PRIME_REQUESTS" \
    "$BRPC_LOAD_QPS" "$BRPC_LOAD_CONCURRENCY" "$BRPC_LOAD_PAYLOAD_BYTES" "$BRPC_LOAD_REQUESTS" \
    "$BRPC_LOAD_TIMEOUT_MS" "$BRPC_LOAD_READY_TIMEOUT_SECONDS" "$PRIME_MAX_ATTEMPTS" \
    "$PRIME_RETRY_DELAY_SECONDS" "$ROUND_COOLDOWN_SECONDS" "$KVC_LOAD_READY_TIMEOUT_SECONDS" \
    "$INFERENCE_ROLLOUT_TIMEOUT_SECONDS" "$INFERENCE_CONTAINER_RESTART_TIMEOUT_SECONDS" \
    "$MIN_ROOT_AVAILABLE_KB" \
    "$KVC_KEY_COUNT" "$KVC_GET_KEY_COUNT" \
    "$KVC_SET_KEY_COUNT" "$KVC_BATCH_NUM" "$KVC_THREAD_NUM" "$KVC_GET_CLIENTS" \
    "$KVC_SET_CLIENTS" "$KVC_LOAD_DURATION_SECONDS" "$KVC_REPORT_INTERVAL_SECONDS"; do
    [[ "$value" =~ ^[0-9]+$ ]] || die "repeat/count parameters must be non-negative integers"
  done
  [ "$REPEATS" -gt 0 ] || die "REPEATS must be positive"
  [ "$EXPECTED_ONBOARDS_MIN" -le "$EXPECTED_ONBOARDS_MAX" ] \
    || die "EXPECTED_ONBOARDS_MIN must not exceed EXPECTED_ONBOARDS_MAX"
  [ "$PRIME_REQUESTS" -gt 0 ] || die "PRIME_REQUESTS must be positive"
  [ "$PRIME_MAX_ATTEMPTS" -gt 0 ] || die "PRIME_MAX_ATTEMPTS must be positive"
  [ "$INFERENCE_CONTAINER_RESTART_TIMEOUT_SECONDS" -gt 0 ] \
    || die "INFERENCE_CONTAINER_RESTART_TIMEOUT_SECONDS must be positive"
  [ "$BRPC_LOAD_CONCURRENCY" -gt 0 ] || die "BRPC_LOAD_CONCURRENCY must be positive"
  [ "$BRPC_LOAD_PAYLOAD_BYTES" -gt 0 ] || die "BRPC_LOAD_PAYLOAD_BYTES must be positive"
  [ "$BRPC_LOAD_REQUESTS" -ge "$BRPC_LOAD_CONCURRENCY" ] \
    || die "BRPC_LOAD_REQUESTS must be at least BRPC_LOAD_CONCURRENCY"
  [ "$BRPC_LOAD_READY_TIMEOUT_SECONDS" -gt 0 ] || die "BRPC_LOAD_READY_TIMEOUT_SECONDS must be positive"
  [ "$KVC_LOAD_READY_TIMEOUT_SECONDS" -gt 0 ] || die "KVC_LOAD_READY_TIMEOUT_SECONDS must be positive"
  if mode_has_kvc; then
    [ "$KVC_GET_CLIENTS" -gt 0 ] || die "KVC_GET_CLIENTS must be positive"
    [ "$KVC_SET_CLIENTS" -gt 0 ] || die "KVC_SET_CLIENTS must be positive"
    [ "$KVC_GET_KEY_COUNT" -gt 0 ] || die "KVC_GET_KEY_COUNT must be positive"
    [ "$KVC_SET_KEY_COUNT" -gt 0 ] || die "KVC_SET_KEY_COUNT must be positive"
    [ "$KVC_THREAD_NUM" -gt 0 ] || die "KVC_THREAD_NUM must be positive"
    [ "$KVC_LOAD_DURATION_SECONDS" -gt 0 ] || die "KVC_LOAD_DURATION_SECONDS must be positive"
  fi
  case "$BRPC_LOAD_REUSE_CONNECTIONS" in
    0|1) ;;
    *) die "BRPC_LOAD_REUSE_CONNECTIONS must be 0 or 1" ;;
  esac
  case "$RESET_INFERENCE_BEFORE_ROUND" in
    0|1) ;;
    *) die "RESET_INFERENCE_BEFORE_ROUND must be 0 or 1" ;;
  esac
  case "$RESET_INFERENCE_MODE" in
    rollout|container-runtime|pod-recreate) ;;
    *) die "RESET_INFERENCE_MODE must be rollout, container-runtime, or pod-recreate" ;;
  esac
  case "$REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION" in
    0|1) ;;
    *) die "REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION must be 0 or 1" ;;
  esac
  case "$KVC_BURST_DYNAMIC_ARM" in
    0|1) ;;
    *) die "KVC_BURST_DYNAMIC_ARM must be 0 or 1" ;;
  esac
  case "$KVC_DSBENCH_SUSTAINED" in
    0|1) ;;
    *) die "KVC_DSBENCH_SUSTAINED must be 0 or 1" ;;
  esac
  if mode_has_brpc && [ "$BRPC_LOAD_QPS" -eq 0 ] && [ "$BRPC_LOAD_REUSE_CONNECTIONS" != "1" ]; then
		die "unlimited BRPC load requires BRPC_LOAD_REUSE_CONNECTIONS=1 to avoid ephemeral-port exhaustion"
  fi
  if mode_has_kvc && [ "$KVC_BATCH_NUM" -ne 1 ]; then
    die "KVC_BATCH_NUM must be 1 to match current single-key TRT KVC calls"
  fi
  if mode_has_kvc && [ "$KVC_DSBENCH_SUSTAINED" = "1" ]; then
    [ "$KVC_GET_KEY_COUNT" -eq $((KVC_GET_CLIENTS * KVC_THREAD_NUM)) ] \
      || die "KVC_GET_KEY_COUNT must equal KVC_GET_CLIENTS * KVC_THREAD_NUM"
    [ "$KVC_SET_KEY_COUNT" -eq $((KVC_SET_CLIENTS * KVC_THREAD_NUM)) ] \
      || die "KVC_SET_KEY_COUNT must equal KVC_SET_CLIENTS * KVC_THREAD_NUM"
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

capture_inference_state() {
  local round_dir="$1"
  local phase="$2"
  local pod
  pod="$(kubectl -n "$NAMESPACE" get pod -l app=inference-brpc-trtllm \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1].metadata.name}')"
  [ -n "$pod" ] || die "inference pod not found"
  printf '%s\n' "$pod" >"${round_dir}/inference-pod.${phase}"
  kubectl -n "$NAMESPACE" get pod "$pod" \
    -o jsonpath='{.status.containerStatuses[?(@.name=="brpc-inference")].restartCount}' \
    >"${round_dir}/inference-restarts.${phase}"
  printf '\n' >>"${round_dir}/inference-restarts.${phase}"
}

capture_datasystem_worker_metrics() {
  local round_dir="$1"
  local phase="$2"
  [ "$DS_WORKER_METRICS" = "1" ] || return 0
  # Metrics are observational: a collection failure is recorded as an artifact
  # but never fails the round itself.
  if ! DS_WORKER_POD_SELECTOR="$DS_WORKER_POD_SELECTOR" \
       DS_WORKER_CONTAINER="$DS_WORKER_CONTAINER" NAMESPACE="$NAMESPACE" \
       bash scripts/collect_datasystem_worker_metrics.sh snapshot \
         "${round_dir}/datasystem-worker-metrics.${phase}.json" \
         >"${round_dir}/datasystem-worker-metrics.${phase}.log" 2>&1; then
    echo "snapshot phase=${phase} failed; see ${round_dir}/datasystem-worker-metrics.${phase}.log" \
      >"${round_dir}/datasystem-worker-metrics.error"
    return 0
  fi
  if [ "$phase" = "after" ] && [ -f "${round_dir}/datasystem-worker-metrics.before.json" ]; then
    if ! bash scripts/collect_datasystem_worker_metrics.sh delta \
        "${round_dir}/datasystem-worker-metrics.before.json" \
        "${round_dir}/datasystem-worker-metrics.after.json" \
        "${round_dir}/datasystem-worker-metrics.json" \
        >>"${round_dir}/datasystem-worker-metrics.${phase}.log" 2>&1; then
      echo "delta computation failed" >"${round_dir}/datasystem-worker-metrics.error"
    fi
  fi
}

capture_brpc_pressure_cpu_stat() {
  local round_dir="$1"
  local phase="$2"
  if ! mode_has_brpc; then
    return
  fi
  local pod
  pod="$(kubectl -n "$NAMESPACE" get pod -l "$BRPC_LOAD_POD_SELECTOR" \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1].metadata.name}')"
  [ -n "$pod" ] || die "BRPC pressure target pod not found: ${BRPC_LOAD_POD_SELECTOR}"
  printf '%s\n' "$pod" >"${round_dir}/brpc-pressure-pod.${phase}"
  kubectl -n "$NAMESPACE" exec "$pod" -c "$BRPC_LOAD_CONTAINER" -- \
    env -u LD_PRELOAD sh -c '
      if test -r /sys/fs/cgroup/cpu.stat; then
        cat /sys/fs/cgroup/cpu.stat
      elif test -r /sys/fs/cgroup/cpu/cpu.stat; then
        cat /sys/fs/cgroup/cpu/cpu.stat
      else
        exit 1
      fi
    ' >"${round_dir}/brpc-pressure-cpu-stat.${phase}"
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
	local ready_file="${round_dir}/brpc-pressure.ready"
	rm -f "$ready_file"
	"$BRPC_PROBE_BIN" \
		--endpoint="$BRPC_LOAD_ENDPOINT" \
    --method=health \
    --requests="$BRPC_LOAD_REQUESTS" \
    --concurrency="$BRPC_LOAD_CONCURRENCY" \
    --qps="$BRPC_LOAD_QPS" \
    --payload_bytes="$BRPC_LOAD_PAYLOAD_BYTES" \
    --timeout_ms="$BRPC_LOAD_TIMEOUT_MS" \
		--max_retries=0 \
		--reuse_connections="$BRPC_LOAD_REUSE_CONNECTIONS" \
		--ready_file="$ready_file" \
		--stats_file="${round_dir}/brpc-pressure-stats.jsonl" \
		--stats_interval_ms=1000 \
		--quiet=true \
    >"${round_dir}/brpc-load.log" 2>&1 &
	BRPC_LOAD_PID="$!"
	echo "$BRPC_LOAD_PID" >"${round_dir}/brpc-load.pid"
	local attempt
	for attempt in $(seq 1 "$BRPC_LOAD_READY_TIMEOUT_SECONDS"); do
		if [ -s "$ready_file" ]; then
			return
		fi
		if ! kill -0 "$BRPC_LOAD_PID" >/dev/null 2>&1; then
			cat "${round_dir}/brpc-load.log" >&2 || true
			die "BRPC pressure exited before reaching configured concurrency"
		fi
		sleep 1
	done
	die "BRPC pressure did not reach concurrency=${BRPC_LOAD_CONCURRENCY} within ${BRPC_LOAD_READY_TIMEOUT_SECONDS}s"
}

check_brpc_load_endpoint() {
  if ! mode_has_brpc; then
    return
  fi
  log "Check BRPC pressure endpoint"
  local endpoint_count
  endpoint_count="$(awk -F, '{print NF}' <<<"$BRPC_LOAD_ENDPOINT")"
  "$BRPC_PROBE_BIN" \
    --endpoint="$BRPC_LOAD_ENDPOINT" \
    --method=health \
    --requests="$endpoint_count" \
    --concurrency="$endpoint_count" \
    --timeout_ms="$BRPC_LOAD_TIMEOUT_MS" \
    --max_retries=0 \
    >"${OUT_DIR}/brpc-pressure-endpoint-check.log" 2>&1 \
    || die "BRPC pressure endpoint is unavailable: ${BRPC_LOAD_ENDPOINT}"
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
  KVC_REMOTE_PREPARED_FILE="/tmp/dsbench-pressure-${remote_run_id}.prepared"
  KVC_REMOTE_START_FILE="/tmp/dsbench-pressure-${remote_run_id}.start"
  KVC_REMOTE_STATS_FILE="/tmp/dsbench-pressure-${remote_run_id}.stats.jsonl"
  local remote_mode
  remote_mode="$(kvc_mode)"

  ssh "$KVC_LOAD_HOST" \
    "cd '$KVC_REMOTE_REPO' && env PRESSURE_ENGINE='$KVC_PRESSURE_ENGINE' MODE='$remote_mode' DS_ENDPOINT='$KVC_DS_ENDPOINT' DSBENCH_CPP='$KVC_DSBENCH_CPP' DSBENCH_SUSTAINED='$KVC_DSBENCH_SUSTAINED' DSBENCH_READY_TIMEOUT_SECONDS='$KVC_LOAD_READY_TIMEOUT_SECONDS' DATASYSTEM_SDK_DIR='$KVC_DATASYSTEM_SDK_DIR' PERSISTENT_BIN='$KVC_PERSISTENT_BIN' OBJECT_SIZE='$KVC_OBJECT_SIZE' KEY_COUNT='$KVC_KEY_COUNT' GET_KEY_COUNT='$KVC_GET_KEY_COUNT' SET_KEY_COUNT='$KVC_SET_KEY_COUNT' BATCH_NUM='$KVC_BATCH_NUM' THREAD_NUM='$KVC_THREAD_NUM' GET_CLIENTS='$KVC_GET_CLIENTS' SET_CLIENTS='$KVC_SET_CLIENTS' TASKSET_CPUS='$KVC_TASKSET_CPUS' DURATION_SECONDS='$KVC_LOAD_DURATION_SECONDS' REPORT_INTERVAL_SECONDS='$KVC_REPORT_INTERVAL_SECONDS' RUN_ID='$remote_run_id' PID_FILE='$KVC_REMOTE_PID_FILE' READY_FILE='$KVC_REMOTE_READY_FILE' PREPARED_FILE='$KVC_REMOTE_PREPARED_FILE' START_FILE='$KVC_REMOTE_START_FILE' STATS_FILE='$KVC_REMOTE_STATS_FILE' bash scripts/run_datasystem_dsbench_pressure.sh" \
    >"${round_dir}/kvc-load.log" 2>&1 &
  KVC_SSH_PID="$!"
  echo "$KVC_SSH_PID" >"${round_dir}/kvc-load-ssh.pid"

  local wait_file="$KVC_REMOTE_READY_FILE"
  local state_name="ready"
  if [ "$KVC_DSBENCH_SUSTAINED" = "1" ]; then
    wait_file="$KVC_REMOTE_PREPARED_FILE"
    state_name="prepared"
  fi
  local attempt
  for attempt in $(seq 1 "$KVC_LOAD_READY_TIMEOUT_SECONDS"); do
    if ssh "$KVC_LOAD_HOST" "test -s '$wait_file'" >/dev/null 2>&1; then
      ssh "$KVC_LOAD_HOST" "cat '$wait_file'" \
        >"${round_dir}/kvc-pressure-${state_name}.txt" 2>/dev/null || true
      return
    fi
    if ! kill -0 "$KVC_SSH_PID" >/dev/null 2>&1; then
      cat "${round_dir}/kvc-load.log" >&2 || true
      die "remote KVC pressure exited before becoming ${state_name}"
    fi
    sleep 1
  done
  die "remote KVC pressure did not become ${state_name} within ${KVC_LOAD_READY_TIMEOUT_SECONDS} seconds"
}

release_kvc_load() {
  local round_dir="$1"
  if ! mode_has_kvc || [ "$KVC_DSBENCH_SUSTAINED" != "1" ]; then
    return
  fi
  ssh "$KVC_LOAD_HOST" "touch '$KVC_REMOTE_START_FILE'"
  local attempt
  for attempt in $(seq 1 "$KVC_LOAD_READY_TIMEOUT_SECONDS"); do
    if ssh "$KVC_LOAD_HOST" "test -s '$KVC_REMOTE_READY_FILE'" >/dev/null 2>&1; then
      ssh "$KVC_LOAD_HOST" "cat '$KVC_REMOTE_READY_FILE'" \
        >"${round_dir}/kvc-pressure-ready.txt" 2>/dev/null || true
      return
    fi
    if ! kill -0 "$KVC_SSH_PID" >/dev/null 2>&1; then
      cat "${round_dir}/kvc-load.log" >&2 || true
      die "remote KVC pressure exited before becoming ready"
    fi
    sleep 1
  done
  die "remote KVC pressure did not become ready within ${KVC_LOAD_READY_TIMEOUT_SECONDS} seconds"
}

capture_kvc_stats() {
  local round_dir="$1"
  if [ -z "$KVC_REMOTE_STATS_FILE" ]; then
    return
  fi
  if [ "$KVC_PRESSURE_ENGINE" = "dsbench" ] && [ "$KVC_DSBENCH_SUSTAINED" = "1" ]; then
    ssh "$KVC_LOAD_HOST" \
      "cat '$KVC_REMOTE_STATS_FILE.get' '$KVC_REMOTE_STATS_FILE.set' 2>/dev/null || true" \
      >"${round_dir}/kvc-pressure-stats.jsonl" 2>/dev/null || true
  else
    ssh "$KVC_LOAD_HOST" "cat '$KVC_REMOTE_STATS_FILE' 2>/dev/null || true" \
      >"${round_dir}/kvc-pressure-stats.jsonl" 2>/dev/null || true
  fi
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
  KVC_REMOTE_PREPARED_FILE=""
  KVC_REMOTE_START_FILE=""
  KVC_REMOTE_STATS_FILE=""
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

run_prime_once() {
  local attempt_dir="$1"
  local script_code=0
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
  OUT_DIR="$attempt_dir" \
    bash scripts/benchmark_go_brpc_probe_kvc_latency.sh \
    >"${attempt_dir}.console.log" 2>&1 || script_code="$?"

  if [ "$script_code" -eq 0 ]; then
    return 0
  fi

  # The prime workload is complete once both the client and server observed every
  # request. Do not repeat 195 requests for a post-processing/cleanup exit code.
  if python3 - "$attempt_dir" "$PRIME_REQUESTS" <<'PY'
import json
import pathlib
import sys

attempt_dir = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])

try:
    probe_code = int((attempt_dir / "probe.exit_code").read_text().strip())
    summary = json.loads((attempt_dir / "summary.json").read_text())
except (FileNotFoundError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)

counts = summary.get("counts") or {}
valid = (
    probe_code == 0
    and int(counts.get("probe_recommend_events", -1)) == expected
    and int(counts.get("brpc_server_events", -1)) == expected
)
raise SystemExit(0 if valid else 1)
PY
  then
    echo "$script_code" >"${attempt_dir}.postprocess_exit_code"
    echo "WARN: prime request workload completed despite benchmark script exit=${script_code}; accepting semantic evidence" \
      >>"${attempt_dir}.console.log"
    return 0
  fi

  return "$script_code"
}

reset_inference() {
  local round_dir="$1"
  if [ "$RESET_INFERENCE_BEFORE_ROUND" != "1" ]; then
    return
  fi

  log "Reset inference cache state"
  if [ "$RESET_INFERENCE_MODE" = "container-runtime" ]; then
    reset_inference_container_runtime "$round_dir"
    return
  fi
  if [ "$RESET_INFERENCE_MODE" = "pod-recreate" ]; then
    reset_inference_pod_recreate "$round_dir"
    return
  fi

  kubectl -n "$NAMESPACE" rollout restart "$BRPC_TARGET" \
    >"${round_dir}/inference-rollout-restart.log" 2>&1
  kubectl -n "$NAMESPACE" rollout status "$BRPC_TARGET" \
    --timeout="${INFERENCE_ROLLOUT_TIMEOUT_SECONDS}s" \
    >"${round_dir}/inference-rollout-status.log" 2>&1
  kubectl -n "$NAMESPACE" get pods -l app=inference-brpc-trtllm -o wide \
    >"${round_dir}/inference-pods-after-reset.txt"
}

reset_inference_pod_recreate() {
  local round_dir="$1"
  local pod old_uid candidate candidate_uid ready deadline
  pod="$(kubectl -n "$NAMESPACE" get pod -l app=inference-brpc-trtllm \
    --sort-by=.metadata.creationTimestamp \
    -o 'jsonpath={range .items[*]}{.metadata.name}{"\n"}{end}' | tail -1)"
  [ -n "$pod" ] || die "inference pod not found for pod recreation"
  old_uid="$(kubectl -n "$NAMESPACE" get pod "$pod" -o jsonpath='{.metadata.uid}')"
  printf 'old_pod=%s old_uid=%s\n' "$pod" "$old_uid" \
    >"${round_dir}/inference-pod-recreate.txt"

  kubectl -n "$NAMESPACE" delete pod "$pod" --wait=false \
    >"${round_dir}/inference-pod-delete.log" 2>&1

  deadline="$((SECONDS + INFERENCE_ROLLOUT_TIMEOUT_SECONDS))"
  while (( SECONDS < deadline )); do
    candidate="$(kubectl -n "$NAMESPACE" get pod -l app=inference-brpc-trtllm \
      --sort-by=.metadata.creationTimestamp \
      -o 'jsonpath={range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null | tail -1 || true)"
    if [ -n "$candidate" ]; then
      candidate_uid="$(kubectl -n "$NAMESPACE" get pod "$candidate" \
        -o jsonpath='{.metadata.uid}' 2>/dev/null || true)"
      ready="$(kubectl -n "$NAMESPACE" get pod "$candidate" \
        -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)"
      if [ -n "$candidate_uid" ] && [ "$candidate_uid" != "$old_uid" ] && [ "$ready" = "True" ]; then
        printf 'new_pod=%s new_uid=%s ready=%s\n' "$candidate" "$candidate_uid" "$ready" \
          >>"${round_dir}/inference-pod-recreate.txt"
        kubectl -n "$NAMESPACE" get pod "$candidate" -o wide \
          >"${round_dir}/inference-pods-after-reset.txt"
        return 0
      fi
    fi
    sleep 2
  done

  kubectl -n "$NAMESPACE" get pods -l app=inference-brpc-trtllm -o wide \
    >"${round_dir}/inference-pod-recreate-timeout.txt" 2>&1 || true
  kubectl -n "$NAMESPACE" get events --sort-by=.lastTimestamp \
    >"${round_dir}/inference-pod-recreate-events.txt" 2>&1 || true
  echo "ERROR: replacement inference pod did not become Ready" >&2
  return 1
}

check_root_free_space() {
  [ "$MIN_ROOT_AVAILABLE_KB" -gt 0 ] || return 0
  local available_kb
  available_kb="$(df -Pk / | awk 'NR == 2 { print $4 + 0 }')"
  if [ "$available_kb" -lt "$MIN_ROOT_AVAILABLE_KB" ]; then
    echo "ERROR: root filesystem has ${available_kb}KiB available; require at least ${MIN_ROOT_AVAILABLE_KB}KiB" >&2
    return 1
  fi
}

reset_inference_container_runtime() {
  local round_dir="$1"
  local pod node runtime_host container_id before_restart current_restart main_ready sidecar_ready deadline
  pod="$(kubectl -n "$NAMESPACE" get pod -l app=inference-brpc-trtllm \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1].metadata.name}')"
  [ -n "$pod" ] || die "inference pod not found"
  before_restart="$(kubectl -n "$NAMESPACE" get pod "$pod" \
    -o "jsonpath={.status.containerStatuses[?(@.name==\"${BRPC_CONTAINER}\")].restartCount}")"
  [[ "$before_restart" =~ ^[0-9]+$ ]] || die "cannot read ${BRPC_CONTAINER} restart count"
  node="$(kubectl -n "$NAMESPACE" get pod "$pod" -o jsonpath='{.spec.nodeName}')"
  container_id="$(kubectl -n "$NAMESPACE" get pod "$pod" \
    -o "jsonpath={.status.containerStatuses[?(@.name==\"${BRPC_CONTAINER}\")].containerID}")"
  container_id="${container_id#*://}"
  [ -n "$node" ] || die "cannot read inference node name"
  [[ "$container_id" =~ ^[a-f0-9]+$ ]] || die "cannot read ${BRPC_CONTAINER} runtime container ID"
  runtime_host="$INFERENCE_RUNTIME_SSH_HOST"
  if [ -z "$runtime_host" ]; then
    runtime_host="$(kubectl get node "$node" \
      -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')"
  fi
  [ -n "$runtime_host" ] || die "cannot resolve runtime SSH host for node ${node}"

  printf 'pod=%s node=%s runtime_host=%s container=%s container_id=%s restart_before=%s\n' \
    "$pod" "$node" "$runtime_host" "$BRPC_CONTAINER" "$container_id" "$before_restart" \
    >"${round_dir}/inference-container-reset.txt"
  if ! ssh "$runtime_host" sh -s -- "$container_id" \
      >"${round_dir}/inference-container-runtime-stop.log" 2>&1 <<'SH'
set -eu
container_id="$1"
run_privileged() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo -n "$@"
  fi
}
if command -v crictl >/dev/null 2>&1; then
  run_privileged crictl stop --timeout 0 "$container_id"
elif command -v ctr >/dev/null 2>&1; then
  run_privileged ctr -n k8s.io tasks kill --signal SIGKILL "$container_id"
else
  echo "ERROR: worker node has neither crictl nor ctr" >&2
  exit 1
fi
SH
  then
    cat "${round_dir}/inference-container-runtime-stop.log" >&2 || true
    echo "ERROR: failed to stop inference container through worker runtime" >&2
    return 1
  fi

  deadline="$((SECONDS + INFERENCE_CONTAINER_RESTART_TIMEOUT_SECONDS))"
  while (( SECONDS < deadline )); do
    current_restart="$(kubectl -n "$NAMESPACE" get pod "$pod" \
      -o "jsonpath={.status.containerStatuses[?(@.name==\"${BRPC_CONTAINER}\")].restartCount}" \
      2>/dev/null || true)"
    main_ready="$(kubectl -n "$NAMESPACE" get pod "$pod" \
      -o "jsonpath={.status.containerStatuses[?(@.name==\"${BRPC_CONTAINER}\")].ready}" \
      2>/dev/null || true)"
    sidecar_ready=true
    if [ -n "${KVC_BURST_CONTAINER:-}" ]; then
      sidecar_ready="$(kubectl -n "$NAMESPACE" get pod "$pod" \
        -o "jsonpath={.status.containerStatuses[?(@.name==\"${KVC_BURST_CONTAINER}\")].ready}" \
        2>/dev/null || true)"
    fi
    if [[ "$current_restart" =~ ^[0-9]+$ ]] \
        && [ "$current_restart" -gt "$before_restart" ] \
        && [ "$main_ready" = "true" ] \
        && [ "$sidecar_ready" = "true" ]; then
      printf 'restart_after=%s main_ready=%s sidecar_ready=%s\n' \
        "$current_restart" "$main_ready" "$sidecar_ready" \
        >>"${round_dir}/inference-container-reset.txt"
      kubectl -n "$NAMESPACE" get pod "$pod" -o wide \
        >"${round_dir}/inference-pods-after-reset.txt"
      return 0
    fi
    sleep 1
  done

  kubectl -n "$NAMESPACE" get pod "$pod" -o yaml \
    >"${round_dir}/inference-container-reset-timeout.yaml" 2>/dev/null || true
  kubectl -n "$NAMESPACE" logs "$pod" -c "$BRPC_CONTAINER" --tail=200 \
    >"${round_dir}/inference-container-reset-timeout.log" 2>&1 || true
  echo "ERROR: runtime-stopped inference container did not restart Ready while sidecar remained Ready" >&2
  return 1
}

set_kvc_burst_arm() {
  local round_dir="$1"
  local action="$2"
  if [ "$KVC_BURST_DYNAMIC_ARM" != "1" ]; then
    return
  fi
  [ -n "$KVC_BURST_CONTAINER" ] || die "KVC_BURST_CONTAINER is required for dynamic arm"
  local pod
  local control_action="$action"
  if [ "$action" = "arm" ]; then
    control_action=refresh-and-arm
  fi
  local ds_host="${KVC_DS_ENDPOINT%:*}"
  local ds_port="${KVC_DS_ENDPOINT##*:}"
  pod="$(kubectl -n "$NAMESPACE" get pod -l app=inference-brpc-trtllm \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1].metadata.name}')"
  [ -n "$pod" ] || die "inference pod not found for KVC burst ${action}"
  kubectl -n "$NAMESPACE" exec "$pod" -c "$KVC_BURST_CONTAINER" -- \
    "$KVC_BURST_CONTROL_BIN" \
    "--control_action=${control_action}" \
    "--host=${ds_host}" \
    "--port=${ds_port}" \
    "--prefix=PairecKvcBurstV2" \
    "--control_path=/run/pairec-kvc-burst/control" \
    >>"${round_dir}/kvc-burst-control.log" 2>&1
}

capture_kvc_burst_failure() {
  local round_dir="$1"
  [ -n "$KVC_BURST_CONTAINER" ] || return 0
  local pod
  pod="$(kubectl -n "$NAMESPACE" get pod -l app=inference-brpc-trtllm \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || true)"
  [ -n "$pod" ] || return 0
  kubectl -n "$NAMESPACE" get pod "$pod" -o json \
    >"${round_dir}/kvc-burst-failure-pod.json" 2>&1 || true
  kubectl -n "$NAMESPACE" get pod "$pod" -o wide \
    >"${round_dir}/kvc-burst-failure-pod.txt" 2>&1 || true
  kubectl -n "$NAMESPACE" logs "$pod" -c "$KVC_BURST_CONTAINER" \
    --timestamps --tail=2000 \
    >"${round_dir}/kvc-burst-failure-sidecar.log" 2>&1 || true
  kubectl -n "$NAMESPACE" logs "$pod" -c "$BRPC_CONTAINER" \
    --timestamps --tail=4000 \
    >"${round_dir}/kvc-burst-failure-inference.log" 2>&1 || true
  kubectl -n "$NAMESPACE" exec "$pod" -c "$KVC_BURST_CONTAINER" -- sh -c '
    echo "== ready =="
    cat /run/pairec-kvc-burst/ready 2>&1 || true
    echo "== control u32 =="
    od -An -t u4 -N 128 /run/pairec-kvc-burst/control 2>&1 || true
    echo "== process =="
    grep -E "^(Name|State|Pid|Threads):" /proc/1/status 2>&1 || true
  ' >"${round_dir}/kvc-burst-failure-control.txt" 2>&1 || true
}

run_prime() {
  local round_dir="$1"
  local attempt
  for attempt in $(seq 1 "$PRIME_MAX_ATTEMPTS"); do
    log "Prime cache state attempt ${attempt}/${PRIME_MAX_ATTEMPTS}"
    set +e
    run_prime_once "${round_dir}/prime-attempt-${attempt}"
    local code="$?"
    set -e
    if [ "$code" -eq 0 ]; then
      echo "$attempt" >"${round_dir}/prime.success_attempt"
      return 0
    fi
    echo "$code" >"${round_dir}/prime-attempt-${attempt}.exit_code"
    if [ "$attempt" -lt "$PRIME_MAX_ATTEMPTS" ]; then
      sleep "$PRIME_RETRY_DELAY_SECONDS"
    fi
  done
  return 1
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
  REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION="$REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION" \
  OUT_DIR="${round_dir}/replay" \
    bash scripts/trace_single_brpc_datasystem_request.sh \
    >"${round_dir}/replay.console.log" 2>&1
}

summarize() {
  python3 - "$OUT_DIR" "$MODE" "$REPEATS" "$EXPECTED_OFFLOADS" "$EXPECTED_ONBOARDS" \
    "$EXPECTED_ONBOARDS_MIN" "$EXPECTED_ONBOARDS_MAX" \
    "$STRICT_COUNTS" "$RESULT_JSON" "$KVC_PRESSURE_ENGINE" "$KVC_GET_CLIENTS" "$KVC_SET_CLIENTS" \
    "$KVC_DSBENCH_SUSTAINED" "$BRPC_LOAD_CONCURRENCY" "$REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION" \
    <<'PY' | tee "$SUMMARY_TXT"
import glob
import json
import math
import os
import statistics
import sys

(
    out_dir, mode, repeats, expected_offloads, expected_onboards,
    expected_onboards_min, expected_onboards_max, strict_counts, result_path,
    pressure_engine, get_clients, set_clients, dsbench_sustained, brpc_load_concurrency,
    require_exact_attribution,
) = sys.argv[1:16]
repeats = int(repeats)
expected_offloads = int(expected_offloads)
expected_onboards = int(expected_onboards)
expected_onboards_min = int(expected_onboards_min)
expected_onboards_max = int(expected_onboards_max)
strict_counts = strict_counts == "1"
get_clients = int(get_clients)
set_clients = int(set_clients)
brpc_load_concurrency = int(brpc_load_concurrency)
dsbench_sustained = dsbench_sustained == "1"
require_exact_attribution = require_exact_attribution == "1"
kvc_pressure_required = (
    mode in {"kvc-get", "kvc-set", "kvc-mixed", "combined"}
    and (pressure_engine == "persistent" or dsbench_sustained)
)
brpc_pressure_required = mode in {"brpc", "combined"}
pressure_required = kvc_pressure_required or brpc_pressure_required
expected_pressure_active = (
    get_clients if mode == "kvc-get" else
    set_clients if mode == "kvc-set" else
    get_clients + set_clients if mode in {"kvc-mixed", "combined"} else 0
)

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

def read_last_json(path):
    try:
        last = None
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    last = json.loads(line)
        return last or {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def read_latest_by_action(path):
    latest = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                action = item.get("action")
                if action in {"get", "set"}:
                    latest[action] = item
    except FileNotFoundError:
        return {}
    return latest

def read_key_values(path):
    try:
        values = {}
        with open(path, "r", encoding="utf-8") as handle:
            for token in handle.read().split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    values[key] = value
        return values
    except FileNotFoundError:
        return {}

def read_cpu_stat(path):
    try:
        values = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) == 2:
                    values[parts[0]] = parts[1]
        return values
    except FileNotFoundError:
        return {}

def cpu_stat_delta(before_path, after_path):
    before = read_cpu_stat(before_path)
    after = read_cpu_stat(after_path)
    periods = max(0, int(number(after.get("nr_periods"))) - int(number(before.get("nr_periods"))))
    throttled = max(0, int(number(after.get("nr_throttled"))) - int(number(before.get("nr_throttled"))))
    if "throttled_usec" in after:
        throttled_ms = max(
            0.0,
            number(after.get("throttled_usec")) - number(before.get("throttled_usec")),
        ) / 1000.0
    else:
        throttled_ms = max(
            0.0,
            number(after.get("throttled_time")) - number(before.get("throttled_time")),
        ) / 1_000_000.0
    return {
        "available": bool(before) and bool(after),
        "periods": periods,
        "throttled_periods": throttled,
        "throttled_period_pct": 0.0 if periods == 0 else throttled / periods * 100.0,
        "throttled_ms": throttled_ms,
    }

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
    exact = raw.get("datasystem_request_complete") or {}
    exact_completion_count = int(number(raw.get("datasystem_request_completion_count")))
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
    kvc_pressure_path = os.path.join(round_dir, "kvc-pressure-stats.jsonl")
    kvc_pressure = read_last_json(kvc_pressure_path)
    kvc_pressure_by_action = read_latest_by_action(kvc_pressure_path)
    kvc_get_pressure = kvc_pressure_by_action.get("get", {})
    kvc_set_pressure = kvc_pressure_by_action.get("set", {})
    kvc_pressure_ready = read_key_values(os.path.join(round_dir, "kvc-pressure-ready.txt"))
    brpc_pressure = read_last_json(os.path.join(round_dir, "brpc-pressure-stats.jsonl"))
    brpc_pressure_ready = read_key_values(os.path.join(round_dir, "brpc-pressure.ready"))
    brpc_pressure_cpu = cpu_stat_delta(
        os.path.join(round_dir, "brpc-pressure-cpu-stat.before"),
        os.path.join(round_dir, "brpc-pressure-cpu-stat.after"),
    )
    local_rx_delta = max(0, read_int(os.path.join(round_dir, "local-rx.after")) - read_int(os.path.join(round_dir, "local-rx.before")))
    local_tx_delta = max(0, read_int(os.path.join(round_dir, "local-tx.after")) - read_int(os.path.join(round_dir, "local-tx.before")))
    remote_rx_delta = max(0, read_int(os.path.join(round_dir, "remote-rx.after")) - read_int(os.path.join(round_dir, "remote-rx.before")))
    remote_tx_delta = max(0, read_int(os.path.join(round_dir, "remote-tx.after")) - read_int(os.path.join(round_dir, "remote-tx.before")))
    restart_before = read_int(os.path.join(round_dir, "inference-restarts.before"))
    restart_after = read_int(os.path.join(round_dir, "inference-restarts.after"))
    try:
        with open(os.path.join(round_dir, "inference-pod.before"), "r", encoding="utf-8") as handle:
            pod_before_name = handle.read().strip()
        with open(os.path.join(round_dir, "inference-pod.after"), "r", encoding="utf-8") as handle:
            pod_after_name = handle.read().strip()
    except FileNotFoundError:
        pod_before_name = ""
        pod_after_name = ""
    try:
        with open(os.path.join(round_dir, "replay", "brpc_trtllm.log"), "r", encoding="utf-8", errors="replace") as handle:
            replay_log = handle.read()
    except FileNotFoundError:
        replay_log = ""
    crash_markers = sum(
        replay_log.count(marker)
        for marker in ("Segmentation fault", "core dumped", "Assertion failed", "CUDA error", "std::terminate")
    )
    runtime_ok = (
        restart_after == restart_before
        and pod_before_name != ""
        and pod_before_name == pod_after_name
        and crash_markers == 0
    )
    onboard_count_ok = expected_onboards_min <= len(onboards) <= expected_onboards_max
    count_ok = len(offloads) == expected_offloads and onboard_count_ok
    exact_count_ok = (
        exact_completion_count == 1
        and exact.get("request_id") == raw.get("request_id")
        and exact.get("attribution_complete") is True
        and int(number(exact.get("set_count"))) == expected_offloads
        and expected_onboards_min <= int(number(exact.get("get_count"))) <= expected_onboards_max
        and int(number(exact.get("get_failed_count"))) == 0
        and int(number(exact.get("set_failed_count"))) == 0
        and int(number(exact.get("pending_count"))) == 0
        and int(number(exact.get("unknown_count"))) == 0
    )
    exact_attribution_ok = exact_count_ok or not require_exact_attribution
    response_ok = client.get("ok") is True and raw.get("response_code") == 200 and len(brpc_events) == 1
    kvc_get_calls = int(number(kvc_get_pressure.get("calls")))
    kvc_set_calls = int(number(kvc_set_pressure.get("calls")))
    kvc_get_errors = int(number(kvc_get_pressure.get("errors")))
    kvc_set_errors = int(number(kvc_set_pressure.get("errors")))
    kvc_get_qps = number(kvc_get_pressure.get("qps"))
    kvc_set_qps = number(kvc_set_pressure.get("qps"))
    kvc_get_gbps = number(kvc_get_pressure.get("gbps"))
    kvc_set_gbps = number(kvc_set_pressure.get("gbps"))
    kvc_get_max_inflight = int(number(kvc_get_pressure.get("max_inflight")))
    kvc_set_max_inflight = int(number(kvc_set_pressure.get("max_inflight")))
    kvc_ready_workers = int(number(kvc_pressure_ready.get("ready_workers")))
    if pressure_engine == "persistent":
        kvc_pressure_calls = int(number(kvc_pressure.get("calls")))
        kvc_pressure_errors = int(number(kvc_pressure.get("errors")))
        kvc_pressure_qps = number(kvc_pressure.get("qps"))
        kvc_pressure_gbps = number(kvc_pressure.get("gbps"))
    else:
        kvc_pressure_calls = kvc_get_calls + kvc_set_calls
        kvc_pressure_errors = kvc_get_errors + kvc_set_errors
        kvc_pressure_qps = kvc_get_qps + kvc_set_qps
        kvc_pressure_gbps = kvc_get_gbps + kvc_set_gbps
    kvc_pressure_max_active = max(kvc_get_max_inflight, kvc_set_max_inflight)
    brpc_pressure_calls = int(number(brpc_pressure.get("calls")))
    brpc_pressure_errors = int(number(brpc_pressure.get("errors")))
    brpc_pressure_max_active = max(
        int(number(brpc_pressure.get("max_active_calls"))),
        int(number(brpc_pressure.get("active_calls"))),
        int(number(brpc_pressure.get("max_active"))),
        int(number(brpc_pressure_ready.get("max_active"))),
        int(number(brpc_pressure_ready.get("active_calls"))),
    )
    if not kvc_pressure_required:
        kvc_pressure_ok = True
    elif pressure_engine == "persistent":
        kvc_pressure_max_active = max(
            int(number(kvc_pressure.get("max_active_calls"))),
            int(number(kvc_pressure_ready.get("max_active"))),
            int(number(kvc_pressure_ready.get("active_calls"))),
        )
        kvc_pressure_ok = (
            kvc_pressure_calls > 0
            and kvc_pressure_errors == 0
            and kvc_pressure_max_active >= expected_pressure_active
        )
    else:
        get_required = mode in {"kvc-get", "kvc-mixed", "combined"}
        set_required = mode in {"kvc-set", "kvc-mixed", "combined"}
        get_ok = (
            not get_required
            or (
                kvc_get_calls > 0
                and kvc_get_errors == 0
                and kvc_get_qps > 0
                and kvc_get_max_inflight >= get_clients
            )
        )
        set_ok = (
            not set_required
            or (
                kvc_set_calls > 0
                and kvc_set_errors == 0
                and kvc_set_qps > 0
                and kvc_set_max_inflight >= set_clients
            )
        )
        kvc_pressure_ok = (
            get_ok
            and set_ok
            and kvc_ready_workers >= expected_pressure_active
        )
        kvc_pressure_max_active = max(kvc_get_max_inflight, kvc_set_max_inflight)
    brpc_pressure_ok = (
        not brpc_pressure_required
        or (
            brpc_pressure_calls > 0
            and brpc_pressure_errors == 0
            and brpc_pressure_max_active >= brpc_load_concurrency
        )
    )
    pressure_ok = kvc_pressure_ok and brpc_pressure_ok
    rows.append({
        "round": os.path.basename(os.path.dirname(os.path.dirname(path))).split("-")[-1],
        "valid": response_ok and (count_ok or not strict_counts) and exact_attribution_ok and pressure_ok and runtime_ok,
        "response_ok": response_ok,
        "count_ok": count_ok,
        "exact_attribution_ok": exact_attribution_ok,
        "exact_count_ok": exact_count_ok,
        "exact_completion_count": exact_completion_count,
        "exact_request_id": exact.get("request_id", ""),
        "exact_get_count": int(number(exact.get("get_count"))),
        "exact_get_us": int(number(exact.get("get_us"))),
        "exact_set_count": int(number(exact.get("set_count"))),
        "exact_set_us": int(number(exact.get("set_us"))),
        "exact_get_failed_count": int(number(exact.get("get_failed_count"))),
        "exact_set_failed_count": int(number(exact.get("set_failed_count"))),
        "exact_pending_count": int(number(exact.get("pending_count"))),
        "exact_unknown_count": int(number(exact.get("unknown_count"))),
        "exact_attribution_complete": exact.get("attribution_complete") is True,
        "pressure_ok": pressure_ok,
        "runtime_ok": runtime_ok,
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
        "pod_before": pod_before_name,
        "pod_after": pod_after_name,
        "restart_before": restart_before,
        "restart_after": restart_after,
        "restart_delta": restart_after - restart_before,
        "crash_markers": crash_markers,
        "pressure_calls": brpc_pressure_calls if brpc_pressure_required else kvc_pressure_calls,
        "pressure_errors": brpc_pressure_errors if brpc_pressure_required else kvc_pressure_errors,
        "pressure_qps": number(brpc_pressure.get("qps")) if brpc_pressure_required else kvc_pressure_qps,
        "pressure_gbps": number(brpc_pressure.get("gbps")) if brpc_pressure_required else kvc_pressure_gbps,
        "pressure_max_active_calls": max(brpc_pressure_max_active, kvc_pressure_max_active),
        "brpc_pressure_ok": brpc_pressure_ok,
        "brpc_pressure_qps": number(brpc_pressure.get("qps")),
        "brpc_pressure_gbps": number(brpc_pressure.get("gbps")),
        "brpc_pressure_max_active": brpc_pressure_max_active,
        "brpc_pressure_cpu_stat_available": brpc_pressure_cpu["available"],
        "brpc_pressure_cpu_periods": brpc_pressure_cpu["periods"],
        "brpc_pressure_cpu_throttled_periods": brpc_pressure_cpu["throttled_periods"],
        "brpc_pressure_cpu_throttled_period_pct": brpc_pressure_cpu["throttled_period_pct"],
        "brpc_pressure_cpu_throttled_ms": brpc_pressure_cpu["throttled_ms"],
        "kvc_pressure_ok": kvc_pressure_ok,
        "kvc_pressure_calls": kvc_pressure_calls,
        "kvc_pressure_errors": kvc_pressure_errors,
        "kvc_pressure_ready_workers": kvc_ready_workers,
        "kvc_get_calls": kvc_get_calls,
        "kvc_get_errors": kvc_get_errors,
        "kvc_get_qps": kvc_get_qps,
        "kvc_get_gbps": kvc_get_gbps,
        "kvc_get_max_inflight": kvc_get_max_inflight,
        "kvc_set_calls": kvc_set_calls,
        "kvc_set_errors": kvc_set_errors,
        "kvc_set_qps": kvc_set_qps,
        "kvc_set_gbps": kvc_set_gbps,
        "kvc_set_max_inflight": kvc_set_max_inflight,
        "kvc_pressure_qps": kvc_pressure_qps,
        "kvc_pressure_gbps": kvc_pressure_gbps,
        "kvc_pressure_max_active": kvc_pressure_max_active,
        "summary_path": path,
    })

valid = [row for row in rows if row["valid"]]
brpc_cpu_periods = sum(row["brpc_pressure_cpu_periods"] for row in valid)
brpc_cpu_throttled_periods = sum(row["brpc_pressure_cpu_throttled_periods"] for row in valid)
brpc_cpu_throttled_period_pct = (
    0.0 if brpc_cpu_periods == 0
    else brpc_cpu_throttled_periods / brpc_cpu_periods * 100.0
)
brpc_cpu_throttled_ms = sum(row["brpc_pressure_cpu_throttled_ms"] for row in valid)
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
    "expected_counts": {
        "offload": expected_offloads,
        "onboard": expected_onboards,
        "onboard_min": expected_onboards_min,
        "onboard_max": expected_onboards_max,
    },
    "strict_counts": strict_counts,
    "require_exact_datasystem_attribution": require_exact_attribution,
    "pressure_engine": pressure_engine,
    "dsbench_sustained": dsbench_sustained,
    "pressure_required": pressure_required,
    "brpc_pressure_required": brpc_pressure_required,
    "kvc_pressure_required": kvc_pressure_required,
    "expected_pressure_active": expected_pressure_active,
    "brpc_pressure_cpu": {
        "periods": brpc_cpu_periods,
        "throttled_periods": brpc_cpu_throttled_periods,
        "throttled_period_pct": brpc_cpu_throttled_period_pct,
        "throttled_ms": brpc_cpu_throttled_ms,
    },
    "rows": rows,
    "metrics": metrics,
}
with open(result_path, "w", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)

print("BRPC/KVC contention summary")
print(f"  mode={mode} status={status} valid_repeats={len(valid)}/{repeats}")
if pressure_required:
    print(
        f"  pressure_engine={pressure_engine} expected_get_inflight={get_clients if mode in {'kvc-get', 'kvc-mixed', 'combined'} else 0} "
        f"expected_set_inflight={set_clients if mode in {'kvc-set', 'kvc-mixed', 'combined'} else 0} "
        f"expected_brpc_active={brpc_load_concurrency if brpc_pressure_required else 0}"
    )
print("  round valid e2e_ms server_ms brpc_ms kvc_ms offloads onboards exact_set exact_get exact_ok server_other_ms outer_ms brpc_qps brpc_gbps brpc_active cpu_thr_pct get_qps get_inflight set_qps set_inflight restarts crashes")
for row in rows:
    print(
        f"  {row['round']:>5} {str(row['valid']):>5} {row['e2e_ms']:>7.3f} {row['server_ms']:>9.3f} "
        f"{row['brpc_ms']:>7.3f} {row['kvc_ms']:>6.3f} {row['offload_count']:>8} "
        f"{row['onboard_count']:>8} {row['exact_set_count']:>9} {row['exact_get_count']:>9} "
        f"{str(row['exact_attribution_ok']):>8} {row['server_other_ms']:>15.3f} {row['outer_ms']:>8.3f} "
        f"{row['brpc_pressure_qps']:>9.3f} {row['brpc_pressure_gbps']:>10.3f} "
        f"{row['brpc_pressure_max_active']:>11} {row['brpc_pressure_cpu_throttled_period_pct']:>11.3f} "
        f"{row['kvc_get_qps']:>7.3f} {row['kvc_get_max_inflight']:>12} "
        f"{row['kvc_set_qps']:>7.3f} {row['kvc_set_max_inflight']:>12} "
        f"{row['restart_delta']:>8} {row['crash_markers']:>7}"
    )
if valid:
    print("  aggregate averages:")
    for key in ("e2e_ms", "server_ms", "brpc_ms", "kvc_ms", "offload_ms", "onboard_ms", "server_other_ms", "outer_ms"):
        print(f"    {key}={metrics[key]['avg']:.3f}")
    e2e_avg = metrics["e2e_ms"]["avg"]
    print(f"    brpc_e2e_pct={metrics['brpc_ms']['avg'] / e2e_avg * 100:.2f}%")
    print(f"    kvc_e2e_pct={metrics['kvc_ms']['avg'] / e2e_avg * 100:.2f}%")
    if brpc_pressure_required:
        print(f"    brpc_pressure_qps={statistics.mean(row['brpc_pressure_qps'] for row in valid):.3f}")
        print(f"    brpc_pressure_gbps={statistics.mean(row['brpc_pressure_gbps'] for row in valid):.3f}")
        print(f"    brpc_pressure_cpu_throttled_period_pct={brpc_cpu_throttled_period_pct:.3f}%")
        print(f"    brpc_pressure_cpu_throttled_ms={brpc_cpu_throttled_ms:.3f}")
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
brpc_load_endpoint=${BRPC_LOAD_ENDPOINT}
brpc_load_concurrency=${BRPC_LOAD_CONCURRENCY}
brpc_load_qps=${BRPC_LOAD_QPS}
brpc_load_payload_bytes=${BRPC_LOAD_PAYLOAD_BYTES}
brpc_load_reuse_connections=${BRPC_LOAD_REUSE_CONNECTIONS}
brpc_load_pod_selector=${BRPC_LOAD_POD_SELECTOR}
brpc_load_container=${BRPC_LOAD_CONTAINER}
kvc_load_host=${KVC_LOAD_HOST}
kvc_ds_endpoint=${KVC_DS_ENDPOINT}
kvc_pressure_engine=${KVC_PRESSURE_ENGINE}
kvc_object_size=${KVC_OBJECT_SIZE}
kvc_key_count=${KVC_KEY_COUNT}
kvc_get_key_count=${KVC_GET_KEY_COUNT}
kvc_set_key_count=${KVC_SET_KEY_COUNT}
kvc_batch_num=${KVC_BATCH_NUM}
kvc_thread_num=${KVC_THREAD_NUM}
kvc_get_clients=${KVC_GET_CLIENTS}
kvc_set_clients=${KVC_SET_CLIENTS}
kvc_load_ready_timeout_seconds=${KVC_LOAD_READY_TIMEOUT_SECONDS}
kvc_dsbench_sustained=${KVC_DSBENCH_SUSTAINED}
expected_offloads=${EXPECTED_OFFLOADS}
expected_onboards=${EXPECTED_ONBOARDS}
expected_onboards_min=${EXPECTED_ONBOARDS_MIN}
expected_onboards_max=${EXPECTED_ONBOARDS_MAX}
strict_counts=${STRICT_COUNTS}
require_exact_datasystem_attribution=${REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION}
reset_inference_before_round=${RESET_INFERENCE_BEFORE_ROUND}
reset_inference_mode=${RESET_INFERENCE_MODE}
inference_rollout_timeout_seconds=${INFERENCE_ROLLOUT_TIMEOUT_SECONDS}
inference_container_restart_timeout_seconds=${INFERENCE_CONTAINER_RESTART_TIMEOUT_SECONDS}
inference_runtime_ssh_host=${INFERENCE_RUNTIME_SSH_HOST:-auto-node-internal-ip}
min_root_available_kb=${MIN_ROOT_AVAILABLE_KB}
kvc_burst_dynamic_arm=${KVC_BURST_DYNAMIC_ARM}
kvc_burst_pressure_key_count=${KVC_BURST_PRESSURE_KEY_COUNT}
EOF

log "Experiment configuration"
cat "${OUT_DIR}/config.txt"
check_brpc_load_endpoint

overall_code=0
for round in $(seq 1 "$REPEATS"); do
  round_dir="${OUT_DIR}/round-${round}"
  mkdir -p "$round_dir"
  log "Round ${round}/${REPEATS}: mode=${MODE}"

  if ! check_root_free_space; then
    overall_code=1
    echo "ERROR: round ${round} root filesystem free-space gate failed" >&2
    break
  fi

  start_kvc_load "$round" "$round_dir"

  set +e
  reset_inference "$round_dir"
  reset_code="$?"
  set -e
  if [ "$reset_code" -ne 0 ]; then
    echo "$reset_code" >"${round_dir}/inference-reset.exit_code"
    overall_code=1
    echo "ERROR: round ${round} inference reset failed" >&2
    stop_loads
    break
  fi

  set +e
  set_kvc_burst_arm "$round_dir" disarm
  arm_code="$?"
  set -e
  if [ "$arm_code" -ne 0 ]; then
    echo "$arm_code" >"${round_dir}/kvc-burst-disarm.exit_code"
    overall_code=1
    echo "ERROR: round ${round} failed to disarm KVC burst before prime" >&2
    stop_loads
    break
  fi

  set +e
  run_prime "$round_dir"
  prime_code="$?"
  set -e
  if [ "$prime_code" -ne 0 ]; then
    echo "$prime_code" >"${round_dir}/prime.exit_code"
    overall_code=1
    echo "ERROR: round ${round} prime failed after ${PRIME_MAX_ATTEMPTS} attempts" >&2
    stop_loads
    break
  fi

  set +e
  set_kvc_burst_arm "$round_dir" arm
  arm_code="$?"
  set -e
  if [ "$arm_code" -ne 0 ]; then
    echo "$arm_code" >"${round_dir}/kvc-burst-arm.exit_code"
    capture_kvc_burst_failure "$round_dir"
    overall_code=1
    echo "ERROR: round ${round} failed to arm KVC burst before replay" >&2
    stop_loads
    break
  fi

  capture_inference_state "$round_dir" before
  capture_brpc_pressure_cpu_stat "$round_dir" before
  capture_datasystem_worker_metrics "$round_dir" before

  read_counter "$NETWORK_INTERFACE" rx_bytes >"${round_dir}/local-rx.before"
  read_counter "$NETWORK_INTERFACE" tx_bytes >"${round_dir}/local-tx.before"
  if mode_has_kvc; then
    read_remote_counter rx_bytes >"${round_dir}/remote-rx.before"
    read_remote_counter tx_bytes >"${round_dir}/remote-tx.before"
  fi

  start_brpc_load "$round_dir"
  release_kvc_load "$round_dir"
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
  if [ "$replay_code" -ne 0 ] && [ "$KVC_BURST_REQUIRE_COMPLETE" = "1" ]; then
    # Preserve the armed control state and Proxy skip reason before disarm.
    capture_kvc_burst_failure "$round_dir"
  fi
  set_kvc_burst_arm "$round_dir" disarm
  disarm_code="$?"
  set -e
  echo "$replay_code" >"${round_dir}/replay.exit_code"
  [ "$replay_code" -eq 0 ] || overall_code=1
  if [ "$disarm_code" -ne 0 ]; then
    echo "$disarm_code" >"${round_dir}/kvc-burst-disarm-after-replay.exit_code"
    overall_code=1
  fi
  capture_inference_state "$round_dir" after
  capture_brpc_pressure_cpu_stat "$round_dir" after
  capture_datasystem_worker_metrics "$round_dir" after

  read_counter "$NETWORK_INTERFACE" rx_bytes >"${round_dir}/local-rx.after"
  read_counter "$NETWORK_INTERFACE" tx_bytes >"${round_dir}/local-tx.after"
  if mode_has_kvc; then
    read_remote_counter rx_bytes >"${round_dir}/remote-rx.after"
    read_remote_counter tx_bytes >"${round_dir}/remote-tx.after"
    capture_kvc_stats "$round_dir"
  fi
  stop_loads
  if [ "$round" -lt "$REPEATS" ] && [ "$ROUND_COOLDOWN_SECONDS" -gt 0 ]; then
    log "Round cooldown ${ROUND_COOLDOWN_SECONDS}s"
    sleep "$ROUND_COOLDOWN_SECONDS"
  fi
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
