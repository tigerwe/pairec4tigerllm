#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-mixed}"
DS_ENDPOINT="${DS_ENDPOINT:-192.168.100.12:18482}"
DSBENCH_CPP="${DSBENCH_CPP:-}"
OBJECT_SIZE="${OBJECT_SIZE:-3584KB}"
KEY_COUNT="${KEY_COUNT:-256}"
BATCH_NUM="${BATCH_NUM:-1}"
THREAD_NUM="${THREAD_NUM:-1}"
TASKSET_CPUS="${TASKSET_CPUS:-}"
GET_CLIENTS="${GET_CLIENTS:-4}"
SET_CLIENTS="${SET_CLIENTS:-6}"
DURATION_SECONDS="${DURATION_SECONDS:-0}"
CLEANUP_KEYS="${CLEANUP_KEYS:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d%H%M%S)}"
PREFIX="${PREFIX:-KvcLoad_${RUN_ID}}"
PID_FILE="${PID_FILE:-/tmp/dsbench-pressure-${RUN_ID}.pid}"
READY_FILE="${READY_FILE:-/tmp/dsbench-pressure-${RUN_ID}.ready}"

GET_PREFIX="${PREFIX}_Get"
SET_PREFIX="${PREFIX}_Set"
STOPPING=0
CHILD_PIDS=()

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

validate() {
  case "$MODE" in
    get|set|mixed) ;;
    *) die "MODE must be get, set, or mixed" ;;
  esac
  [[ "$KEY_COUNT" =~ ^[1-9][0-9]*$ ]] || die "KEY_COUNT must be positive"
  [[ "$BATCH_NUM" =~ ^[1-9][0-9]*$ ]] || die "BATCH_NUM must be positive"
  [[ "$THREAD_NUM" =~ ^[1-9][0-9]*$ ]] || die "THREAD_NUM must be positive"
  [[ "$GET_CLIENTS" =~ ^[1-9][0-9]*$ ]] || die "GET_CLIENTS must be positive"
  [[ "$SET_CLIENTS" =~ ^[1-9][0-9]*$ ]] || die "SET_CLIENTS must be positive"
  [[ "$DURATION_SECONDS" =~ ^[0-9]+$ ]] || die "DURATION_SECONDS must be non-negative"
  if [ "$BATCH_NUM" -ne 1 ]; then
    log "warning: BATCH_NUM=${BATCH_NUM} uses vector Get/MSet; current TRT KVC comparison expects BATCH_NUM=1"
  fi
}

run_dsbench() {
  local action="$1"
  local prefix="$2"
  local clients="$3"
  local -a args=(
    kv
    --action="$action"
    --worker_address="$DS_ENDPOINT"
    --prefix="$prefix"
    --client_num="$clients"
    --thread_num="$THREAD_NUM"
    --num="$KEY_COUNT"
    --size="$OBJECT_SIZE"
    --batch_num="$BATCH_NUM"
  )
  if [ "$action" = "get" ] || [ "$action" = "del" ]; then
    args+=(--worker_num=1 --worker_index=0)
  else
    args+=(--worker_index=0)
  fi
  if [ -n "$TASKSET_CPUS" ]; then
    taskset -c "$TASKSET_CPUS" "$DSBENCH_CPP" "${args[@]}"
  else
    "$DSBENCH_CPP" "${args[@]}"
  fi
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
    run_dsbench del "$GET_PREFIX" 1 >/dev/null 2>&1
  fi
  if [ "$MODE" = "set" ] || [ "$MODE" = "mixed" ]; then
    run_dsbench del "$SET_PREFIX" 1 >/dev/null 2>&1
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

find_dsbench
validate
printf '%s\n' "$$" >"$PID_FILE"

log "mode=${MODE} endpoint=${DS_ENDPOINT} dsbench=${DSBENCH_CPP}"
log "object_size=${OBJECT_SIZE} key_count=${KEY_COUNT} batch_num=${BATCH_NUM} thread_num=${THREAD_NUM}"
log "get_clients=${GET_CLIENTS} set_clients=${SET_CLIENTS} duration_seconds=${DURATION_SECONDS}"
log "taskset_cpus=${TASKSET_CPUS:-none}"

if [ "$MODE" = "get" ] || [ "$MODE" = "mixed" ]; then
  log "prefill prefix=${GET_PREFIX}"
  run_dsbench set "$GET_PREFIX" "$GET_CLIENTS"
fi

if [ "$MODE" = "set" ] || [ "$MODE" = "mixed" ]; then
  log "initialize set prefix=${SET_PREFIX}"
  run_dsbench set "$SET_PREFIX" "$SET_CLIENTS"
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
