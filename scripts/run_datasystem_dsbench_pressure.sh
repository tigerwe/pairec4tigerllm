#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-mixed}"
DS_ENDPOINT="${DS_ENDPOINT:-192.168.100.12:18482}"
PRESSURE_ENGINE="${PRESSURE_ENGINE:-persistent}"
DSBENCH_CPP="${DSBENCH_CPP:-}"
DSBENCH_SUSTAINED="${DSBENCH_SUSTAINED:-0}"
DSBENCH_READY_TIMEOUT_SECONDS="${DSBENCH_READY_TIMEOUT_SECONDS:-60}"
PERSISTENT_SOURCE="${PERSISTENT_SOURCE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/datasystem_kv_pressure.cpp}"
PERSISTENT_BIN="${PERSISTENT_BIN:-/tmp/datasystem_kv_pressure}"
DATASYSTEM_SDK_DIR="${DATASYSTEM_SDK_DIR:-}"
CXX="${CXX:-g++}"
OBJECT_SIZE="${OBJECT_SIZE:-3584KB}"
KEY_COUNT="${KEY_COUNT:-256}"
BATCH_NUM="${BATCH_NUM:-1}"
THREAD_NUM="${THREAD_NUM:-1}"
TASKSET_CPUS="${TASKSET_CPUS:-}"
GET_CLIENTS="${GET_CLIENTS:-4}"
SET_CLIENTS="${SET_CLIENTS:-6}"
GET_KEY_COUNT="${GET_KEY_COUNT:-$GET_CLIENTS}"
SET_KEY_COUNT="${SET_KEY_COUNT:-$SET_CLIENTS}"
DURATION_SECONDS="${DURATION_SECONDS:-0}"
CLEANUP_KEYS="${CLEANUP_KEYS:-1}"
REPORT_INTERVAL_SECONDS="${REPORT_INTERVAL_SECONDS:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d%H%M%S)}"
PREFIX="${PREFIX:-KvcLoad_${RUN_ID}}"
PID_FILE="${PID_FILE:-/tmp/dsbench-pressure-${RUN_ID}.pid}"
READY_FILE="${READY_FILE:-/tmp/dsbench-pressure-${RUN_ID}.ready}"
PREPARED_FILE="${PREPARED_FILE:-/tmp/dsbench-pressure-${RUN_ID}.prepared}"
START_FILE="${START_FILE:-/tmp/dsbench-pressure-${RUN_ID}.start}"
STATS_FILE="${STATS_FILE:-/tmp/dsbench-pressure-${RUN_ID}.stats.jsonl}"

GET_PREFIX="${PREFIX}_Get"
SET_PREFIX="${PREFIX}_Set"
STOPPING=0
CHILD_PIDS=()
CHILD_READY_FILES=()
CHILD_PREPARED_FILES=()
CHILD_STATS_FILES=()

log() {
  printf '[dsbench-pressure] %s\n' "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

find_dsbench() {
  if [ -n "$DSBENCH_CPP" ]; then
    return
  fi
  if command -v dsbench_cpp >/dev/null 2>&1; then
    DSBENCH_CPP="$(command -v dsbench_cpp)"
    return
  fi
  local candidate
  for candidate in \
    /usr/local/lib/python*/site-packages/yr/datasystem/dsbench_cpp \
    /usr/lib/python*/site-packages/yr/datasystem/dsbench_cpp; do
    if [ -x "$candidate" ]; then
      DSBENCH_CPP="$candidate"
      return
    fi
  done
  die "dsbench_cpp not found; set DSBENCH_CPP explicitly"
}

parse_bytes() {
  local value="$1"
  local number unit multiplier
  if [[ "$value" =~ ^([1-9][0-9]*)([KkMmGg][Bb]|[Bb])?$ ]]; then
    number="${BASH_REMATCH[1]}"
    unit="${BASH_REMATCH[2]:-B}"
  else
    die "invalid byte size: $value"
  fi
  case "${unit^^}" in
    B) multiplier=1 ;;
    KB) multiplier=1024 ;;
    MB) multiplier=$((1024 * 1024)) ;;
    GB) multiplier=$((1024 * 1024 * 1024)) ;;
    *) die "unsupported byte size unit: $unit" ;;
  esac
  echo $((number * multiplier))
}

build_persistent_pressure() {
  [ "$PRESSURE_ENGINE" = "persistent" ] || return
  command -v "$CXX" >/dev/null 2>&1 || die "C++ compiler not found: $CXX"
  [ -f "$PERSISTENT_SOURCE" ] || die "persistent pressure source not found: $PERSISTENT_SOURCE"
  if [ -z "$DATASYSTEM_SDK_DIR" ]; then
    [ -n "$DSBENCH_CPP" ] || find_dsbench
    DATASYSTEM_SDK_DIR="$(dirname "$DSBENCH_CPP")"
  fi
  local include_dir="${DATASYSTEM_SDK_DIR}/include"
  local lib_dir="${DATASYSTEM_SDK_DIR}/lib"
  [ -f "${include_dir}/datasystem/kv_client.h" ] \
    || die "DataSystem headers not found under ${include_dir}"
  [ -f "${lib_dir}/libdatasystem.so" ] \
    || die "libdatasystem.so not found under ${lib_dir}"
  if [ ! -x "$PERSISTENT_BIN" ] || [ "$PERSISTENT_SOURCE" -nt "$PERSISTENT_BIN" ]; then
    log "build persistent pressure client: ${PERSISTENT_BIN}"
    "$CXX" -std=c++17 -O2 -pthread \
      -I"$include_dir" "$PERSISTENT_SOURCE" \
      -L"$lib_dir" -Wl,-rpath,"$lib_dir" -ldatasystem \
      -o "$PERSISTENT_BIN"
  fi
}

validate() {
  case "$PRESSURE_ENGINE" in
    persistent|dsbench) ;;
    *) die "PRESSURE_ENGINE must be persistent or dsbench" ;;
  esac
  case "$MODE" in
    get|set|mixed) ;;
    *) die "MODE must be get, set, or mixed" ;;
  esac
  [[ "$KEY_COUNT" =~ ^[1-9][0-9]*$ ]] || die "KEY_COUNT must be positive"
  [[ "$BATCH_NUM" =~ ^[1-9][0-9]*$ ]] || die "BATCH_NUM must be positive"
  [[ "$THREAD_NUM" =~ ^[1-9][0-9]*$ ]] || die "THREAD_NUM must be positive"
  [[ "$GET_CLIENTS" =~ ^[1-9][0-9]*$ ]] || die "GET_CLIENTS must be positive"
  [[ "$SET_CLIENTS" =~ ^[1-9][0-9]*$ ]] || die "SET_CLIENTS must be positive"
  [[ "$GET_KEY_COUNT" =~ ^[1-9][0-9]*$ ]] || die "GET_KEY_COUNT must be positive"
  [[ "$SET_KEY_COUNT" =~ ^[1-9][0-9]*$ ]] || die "SET_KEY_COUNT must be positive"
  [[ "$DURATION_SECONDS" =~ ^[0-9]+$ ]] || die "DURATION_SECONDS must be non-negative"
  [[ "$REPORT_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "REPORT_INTERVAL_SECONDS must be positive"
  [[ "$DSBENCH_READY_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] \
    || die "DSBENCH_READY_TIMEOUT_SECONDS must be positive"
  case "$DSBENCH_SUSTAINED" in
    0|1) ;;
    *) die "DSBENCH_SUSTAINED must be 0 or 1" ;;
  esac
  if [ "$DSBENCH_SUSTAINED" = "1" ] && [ "$PRESSURE_ENGINE" != "dsbench" ]; then
    die "DSBENCH_SUSTAINED=1 requires PRESSURE_ENGINE=dsbench"
  fi
  if [ "$DSBENCH_SUSTAINED" = "1" ] && [ "$DURATION_SECONDS" -eq 0 ]; then
    die "DSBENCH_SUSTAINED=1 requires a positive DURATION_SECONDS"
  fi
  if [ "$DSBENCH_SUSTAINED" = "1" ] && [ "$BATCH_NUM" -ne 1 ]; then
    die "sustained observability requires BATCH_NUM=1"
  fi
  if [ "$DSBENCH_SUSTAINED" = "1" ]; then
    case "$MODE" in
      get|mixed)
        [ "$GET_KEY_COUNT" -eq $((GET_CLIENTS * THREAD_NUM)) ] \
          || die "GET_KEY_COUNT must equal GET_CLIENTS * THREAD_NUM for exact RPC accounting"
        ;;
    esac
    case "$MODE" in
      set|mixed)
        [ "$SET_KEY_COUNT" -eq $((SET_CLIENTS * THREAD_NUM)) ] \
          || die "SET_KEY_COUNT must equal SET_CLIENTS * THREAD_NUM for exact RPC accounting"
        ;;
    esac
  fi
  if [ "$BATCH_NUM" -ne 1 ]; then
    log "warning: BATCH_NUM=${BATCH_NUM} uses vector Get/MSet; current TRT KVC comparison expects BATCH_NUM=1"
  fi
}

run_persistent_pressure() {
  local host="${DS_ENDPOINT%:*}"
  local port="${DS_ENDPOINT##*:}"
  local object_size_bytes
  object_size_bytes="$(parse_bytes "$OBJECT_SIZE")"
  rm -f "$READY_FILE" "$STATS_FILE"
  log "start persistent pressure client"
  local -a command=("$PERSISTENT_BIN"
    --mode="$MODE"
    --host="$host"
    --port="$port"
    --object_size="$object_size_bytes"
    --key_count="$KEY_COUNT"
    --get_clients="$GET_CLIENTS"
    --set_clients="$SET_CLIENTS"
    --duration_seconds="$DURATION_SECONDS"
    --report_interval_seconds="$REPORT_INTERVAL_SECONDS"
    --prefix="$PREFIX"
    --ready_file="$READY_FILE"
    --stats_file="$STATS_FILE"
    --cleanup_keys="$CLEANUP_KEYS")
  if [ -n "$TASKSET_CPUS" ]; then
    exec taskset -c "$TASKSET_CPUS" "${command[@]}"
  fi
  exec "${command[@]}"
}

run_dsbench() {
  local action="$1"
  local prefix="$2"
  local clients="$3"
  local duration_seconds="${4:-0}"
  local ready_file="${5:-}"
  local key_count="${6:-$KEY_COUNT}"
  local prepared_file="${7:-}"
  local start_file="${8:-}"
  local stats_file="${9:-}"
  local -a args=(
    kv
    --action="$action"
    --worker_address="$DS_ENDPOINT"
    --prefix="$prefix"
    --client_num="$clients"
    --thread_num="$THREAD_NUM"
    --num="$key_count"
    --size="$OBJECT_SIZE"
    --batch_num="$BATCH_NUM"
  )
  if [ "$action" = "get" ] || [ "$action" = "del" ]; then
    args+=(--worker_num=1 --worker_index=0)
  else
    args+=(--worker_index=0)
  fi
  if [ "$duration_seconds" -gt 0 ]; then
    args+=(--duration_seconds="$duration_seconds")
  fi
  if [ -n "$ready_file" ]; then
    args+=(--ready_file="$ready_file")
  fi
  if [ -n "$prepared_file" ]; then
    args+=(--prepared_file="$prepared_file")
  fi
  if [ -n "$start_file" ]; then
    args+=(--start_file="$start_file")
  fi
  if [ -n "$stats_file" ]; then
    args+=(--stats_file="$stats_file" --report_interval_ms=$((REPORT_INTERVAL_SECONDS * 1000)))
  fi
  if [ -n "$TASKSET_CPUS" ]; then
    taskset -c "$TASKSET_CPUS" "$DSBENCH_CPP" "${args[@]}"
  else
    "$DSBENCH_CPP" "${args[@]}"
  fi
}

start_sustained_action() {
  local action="$1"
  local prefix="$2"
  local clients="$3"
  local ready_file="${READY_FILE}.${action}"
  local prepared_file="${PREPARED_FILE}.${action}"
  local stats_file="${STATS_FILE}.${action}"
  local key_count
  if [ "$action" = "get" ]; then
    key_count="$GET_KEY_COUNT"
  else
    key_count="$SET_KEY_COUNT"
  fi
  rm -f "$ready_file" "$prepared_file" "$stats_file"
  run_dsbench "$action" "$prefix" "$clients" "$DURATION_SECONDS" "$ready_file" \
    "$key_count" "$prepared_file" "$START_FILE" "$stats_file" &
  CHILD_PIDS+=("$!")
  CHILD_READY_FILES+=("$ready_file")
  CHILD_PREPARED_FILES+=("$prepared_file")
  CHILD_STATS_FILES+=("$stats_file")
}

wait_sustained_files() {
  local description="$1"
  shift
  local -a files=("$@")
  local attempt state_file pid all_ready
  for attempt in $(seq 1 "$DSBENCH_READY_TIMEOUT_SECONDS"); do
    all_ready=1
    for state_file in "${files[@]}"; do
      [ -s "$state_file" ] || all_ready=0
    done
    if [ "$all_ready" -eq 1 ]; then
      return
    fi
    for pid in "${CHILD_PIDS[@]}"; do
      local child_state
      child_state="$(awk '{ print $3 }' "/proc/${pid}/stat" 2>/dev/null || true)"
      if [ -z "$child_state" ] || [ "$child_state" = "Z" ]; then
        local child_status=0
        wait "$pid" || child_status="$?"
        die "sustained dsbench child pid=${pid} exited with status=${child_status} before ${description}"
      fi
    done
    sleep 1
  done
  die "sustained dsbench did not reach ${description} within ${DSBENCH_READY_TIMEOUT_SECONDS}s"
}

run_sustained_pressure() {
  local help_output
  help_output="$("$DSBENCH_CPP" kv --help 2>&1 || true)"
  for marker in --duration_seconds --prepared_file --start_file --stats_file; do
    grep -q -- "$marker" <<<"$help_output" \
      || die "dsbench_cpp does not contain ${marker}; reapply and rebuild sustained pressure patches"
  done

  rm -f "$READY_FILE" "$PREPARED_FILE" "$START_FILE" "$STATS_FILE"

  case "$MODE" in
    get)
      start_sustained_action get "$GET_PREFIX" "$GET_CLIENTS"
      ;;
    set)
      start_sustained_action set "$SET_PREFIX" "$SET_CLIENTS"
      ;;
    mixed)
      start_sustained_action get "$GET_PREFIX" "$GET_CLIENTS"
      start_sustained_action set "$SET_PREFIX" "$SET_CLIENTS"
      ;;
  esac

  wait_sustained_files prepared "${CHILD_PREPARED_FILES[@]}"
  local prepared_workers=0 state_file
  for state_file in "${CHILD_PREPARED_FILES[@]}"; do
    cat "$state_file"
    prepared_workers=$((prepared_workers + $(sed -n 's/.*prepared_workers=\([0-9][0-9]*\).*/\1/p' "$state_file")))
  done
  printf 'engine=dsbench sustained=1 prepared_workers=%s duration_seconds=%s\n' \
    "$prepared_workers" "$DURATION_SECONDS" >"$PREPARED_FILE"
  log "prepared sustained=1 prepared_workers=${prepared_workers} prepared_file=${PREPARED_FILE}"

  wait_sustained_files ready "${CHILD_READY_FILES[@]}"
  local ready_workers=0
  for state_file in "${CHILD_READY_FILES[@]}"; do
    cat "$state_file"
    ready_workers=$((ready_workers + $(sed -n 's/.*ready_workers=\([0-9][0-9]*\).*/\1/p' "$state_file")))
  done
  printf 'engine=dsbench sustained=1 ready_workers=%s duration_seconds=%s\n' \
    "$ready_workers" "$DURATION_SECONDS" >"$READY_FILE"
  log "ready sustained=1 ready_workers=${ready_workers} ready_file=${READY_FILE}"
  wait "${CHILD_PIDS[@]}"
}

should_continue() {
  [ "$STOPPING" -eq 0 ] || return 1
  [ "$DURATION_SECONDS" -eq 0 ] || [ "$SECONDS" -lt "$DURATION_SECONDS" ]
}

run_loop() {
  local action="$1"
  local prefix="$2"
  local clients="$3"
  local iterations=0
  while should_continue; do
    run_dsbench "$action" "$prefix" "$clients"
    iterations=$((iterations + 1))
  done
  log "action=${action} iterations=${iterations} clients=${clients} prefix=${prefix}"
}

cleanup_keys() {
  [ "$CLEANUP_KEYS" = "1" ] || return 0
  set +e
  if [ "$MODE" = "get" ] || [ "$MODE" = "mixed" ]; then
    run_dsbench del "$GET_PREFIX" 1 0 "" "$GET_KEY_COUNT" >/dev/null 2>&1
  fi
  if [ "$MODE" = "set" ] || [ "$MODE" = "mixed" ]; then
    run_dsbench del "$SET_PREFIX" 1 0 "" "$SET_KEY_COUNT" >/dev/null 2>&1
  fi
  set -e
}

cleanup() {
  STOPPING=1
  rm -f "$READY_FILE"
  local pid
  for pid in "${CHILD_PIDS[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" >/dev/null 2>&1 || true
  done
  rm -f "$PREPARED_FILE" "$START_FILE"
  rm -f "${CHILD_READY_FILES[@]:-}" "${CHILD_PREPARED_FILES[@]:-}"
  for pid in "${CHILD_PIDS[@]:-}"; do
    [ -n "$pid" ] && wait "$pid" >/dev/null 2>&1 || true
  done
  cleanup_keys
  rm -f "$PID_FILE"
}

terminate() {
  cleanup
  trap - EXIT
  exit 130
}

trap cleanup EXIT
trap terminate INT TERM

validate
if [ "$PRESSURE_ENGINE" = "persistent" ]; then
  build_persistent_pressure
else
  find_dsbench
fi
printf '%s\n' "$$" >"$PID_FILE"

log "mode=${MODE} engine=${PRESSURE_ENGINE} endpoint=${DS_ENDPOINT}"
log "object_size=${OBJECT_SIZE} key_count=${KEY_COUNT} batch_num=${BATCH_NUM} thread_num=${THREAD_NUM}"
log "get_clients=${GET_CLIENTS} set_clients=${SET_CLIENTS} duration_seconds=${DURATION_SECONDS}"
log "get_key_count=${GET_KEY_COUNT} set_key_count=${SET_KEY_COUNT}"
log "dsbench_sustained=${DSBENCH_SUSTAINED}"
log "taskset_cpus=${TASKSET_CPUS:-none}"

if [ "$PRESSURE_ENGINE" = "persistent" ]; then
  run_persistent_pressure
fi

log "dsbench=${DSBENCH_CPP}"

if [ "$MODE" = "get" ] || [ "$MODE" = "mixed" ]; then
  log "prefill prefix=${GET_PREFIX}"
  run_dsbench set "$GET_PREFIX" "$GET_CLIENTS" 0 "" "$GET_KEY_COUNT"
fi

if [ "$MODE" = "set" ] || [ "$MODE" = "mixed" ]; then
  log "initialize set prefix=${SET_PREFIX}"
  run_dsbench set "$SET_PREFIX" "$SET_CLIENTS" 0 "" "$SET_KEY_COUNT"
fi

if [ "$DSBENCH_SUSTAINED" = "1" ]; then
  run_sustained_pressure
  exit 0
fi

: >"$READY_FILE"
log "ready pid=$$ ready_file=${READY_FILE}"

case "$MODE" in
  get)
    run_loop get "$GET_PREFIX" "$GET_CLIENTS"
    ;;
  set)
    run_loop set "$SET_PREFIX" "$SET_CLIENTS"
    ;;
  mixed)
    run_loop get "$GET_PREFIX" "$GET_CLIENTS" &
    CHILD_PIDS+=("$!")
    run_loop set "$SET_PREFIX" "$SET_CLIENTS" &
    CHILD_PIDS+=("$!")
    wait "${CHILD_PIDS[@]}"
    ;;
esac
