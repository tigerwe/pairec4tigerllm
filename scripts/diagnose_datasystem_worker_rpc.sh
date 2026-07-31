#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
KVC_LOAD_HOST="${KVC_LOAD_HOST:-root@141.61.91.188}"
KVC_DS_ENDPOINT="${KVC_DS_ENDPOINT:-192.168.100.12:18482}"
KVC_DSBENCH_CPP="${KVC_DSBENCH_CPP:-/home/zcx/bin/dsbench-v081-sustained}"
TCP_TIMEOUT_SECONDS="${TCP_TIMEOUT_SECONDS:-3}"
RPC_TIMEOUT_SECONDS="${RPC_TIMEOUT_SECONDS:-90}"
KUBECTL_LOG_SINCE="${KUBECTL_LOG_SINCE:-15m}"
RUN_ID="${RUN_ID:-dsrpc-$(date +%Y%m%d%H%M%S)}"
SAFE_RUN_ID="${RUN_ID//[^a-zA-Z0-9_-]/_}"
OUT_DIR="${OUT_DIR:-/tmp/datasystem-worker-rpc/${SAFE_RUN_ID}}"
SMOKE_PREFIX="pairec_ds_rpc_${SAFE_RUN_ID}"

log() {
  printf '\n== %s ==\n' "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

case "$KVC_DS_ENDPOINT" in
  *:*)
    DS_HOST="${KVC_DS_ENDPOINT%:*}"
    DS_PORT="${KVC_DS_ENDPOINT##*:}"
    ;;
  *)
    die "KVC_DS_ENDPOINT must use host:port format"
    ;;
esac

[[ "$DS_PORT" =~ ^[1-9][0-9]*$ ]] || die "invalid DataSystem port: ${DS_PORT}"
[[ "$TCP_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "TCP_TIMEOUT_SECONDS must be positive"
[[ "$RPC_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "RPC_TIMEOUT_SECONDS must be positive"

mkdir -p "$OUT_DIR"
cat >"${OUT_DIR}/config.txt" <<EOF
namespace=${NAMESPACE}
kvc_load_host=${KVC_LOAD_HOST}
kvc_ds_endpoint=${KVC_DS_ENDPOINT}
kvc_dsbench_cpp=${KVC_DSBENCH_CPP}
tcp_timeout_seconds=${TCP_TIMEOUT_SECONDS}
rpc_timeout_seconds=${RPC_TIMEOUT_SECONDS}
smoke_prefix=${SMOKE_PREFIX}
EOF

log "Remote load-host connectivity and residual processes"
set +e
ssh "$KVC_LOAD_HOST" \
  "env DS_HOST='$DS_HOST' DS_PORT='$DS_PORT' TCP_TIMEOUT_SECONDS='$TCP_TIMEOUT_SECONDS' DSBENCH_CPP='$KVC_DSBENCH_CPP' bash -s" \
  2>&1 <<'REMOTE' | tee "${OUT_DIR}/remote-state.log"
set -u
echo "remote_hostname=$(hostname)"
echo "remote_time=$(date --iso-8601=seconds 2>/dev/null || date)"

if timeout "${TCP_TIMEOUT_SECONDS}s" bash -c "</dev/tcp/${DS_HOST}/${DS_PORT}" 2>/dev/null; then
  echo "REMOTE_TCP_STATUS=PASS"
else
  echo "REMOTE_TCP_STATUS=FAIL"
fi

if [ -x "$DSBENCH_CPP" ]; then
  echo "REMOTE_DSBENCH_STATUS=EXECUTABLE"
else
  echo "REMOTE_DSBENCH_STATUS=MISSING_OR_NOT_EXECUTABLE"
fi

echo "-- residual-processes --"
pgrep -af "dsbench|datasystem_kv_pressure|run_datasystem_dsbench_pressure" \
  || echo "RESIDUAL_PROCESS_COUNT=0"

echo "-- pressure-pid-files --"
found_pid_file=0
for file in /tmp/dsbench-pressure-*.pid; do
  [ -e "$file" ] || continue
  found_pid_file=1
  pid="$(cat "$file" 2>/dev/null || true)"
  echo "PID_FILE=${file} PID=${pid}"
  if [ -n "$pid" ]; then
    ps -o pid,ppid,state,etime,cmd -p "$pid" || true
  fi
done
[ "$found_pid_file" -eq 1 ] || echo "PRESSURE_PID_FILE_COUNT=0"
REMOTE
REMOTE_STATE_STATUS="${PIPESTATUS[0]}"
set -e
echo "$REMOTE_STATE_STATUS" >"${OUT_DIR}/remote-state.exit_code"

REMOTE_TCP_STATUS="$(
  sed -n 's/^REMOTE_TCP_STATUS=//p' "${OUT_DIR}/remote-state.log" | tail -1
)"
REMOTE_DSBENCH_STATUS="$(
  sed -n 's/^REMOTE_DSBENCH_STATUS=//p' "${OUT_DIR}/remote-state.log" | tail -1
)"

log "DataSystem endpoint state from master"
{
  echo "master_hostname=$(hostname)"
  echo "master_time=$(date --iso-8601=seconds 2>/dev/null || date)"
  if timeout "${TCP_TIMEOUT_SECONDS}s" bash -c "</dev/tcp/${DS_HOST}/${DS_PORT}" 2>/dev/null; then
    echo "MASTER_TCP_STATUS=PASS"
  else
    echo "MASTER_TCP_STATUS=FAIL"
  fi
  if command -v ss >/dev/null 2>&1; then
    echo "-- listening-sockets --"
    ss -lntp 2>&1 | grep -E ":${DS_PORT}([[:space:]]|$)" || echo "LISTENER_NOT_FOUND"
  else
    echo "SS_COMMAND=UNAVAILABLE"
  fi
  echo "-- local-worker-processes --"
  pgrep -af '(^|[ /])datasystem_worker([ ]|$)' \
    || echo "LOCAL_DATASYSTEM_WORKER_COUNT=0"
} | tee "${OUT_DIR}/master-state.log"

MASTER_TCP_STATUS="$(
  sed -n 's/^MASTER_TCP_STATUS=//p' "${OUT_DIR}/master-state.log" | tail -1
)"

log "Kubernetes DataSystem state"
if command -v kubectl >/dev/null 2>&1; then
  set +e
  kubectl -n "$NAMESPACE" get pods -o wide 2>&1 \
    | tee "${OUT_DIR}/kubernetes-pods.log"
  KUBECTL_GET_STATUS="${PIPESTATUS[0]}"
  set -e
  echo "$KUBECTL_GET_STATUS" >"${OUT_DIR}/kubernetes-pods.exit_code"

  if [ "$KUBECTL_GET_STATUS" -eq 0 ]; then
    mapfile -t DATASYSTEM_PODS < <(
      kubectl -n "$NAMESPACE" get pods -o name 2>/dev/null | grep -i datasystem || true
    )
    : >"${OUT_DIR}/kubernetes-worker-logs.log"
    for pod in "${DATASYSTEM_PODS[@]}"; do
      {
        echo "===== ${pod} ====="
        kubectl -n "$NAMESPACE" logs "$pod" --all-containers \
          --since="$KUBECTL_LOG_SINCE" --tail=500 2>&1 || true
      } >>"${OUT_DIR}/kubernetes-worker-logs.log"
    done
    grep -E \
      'ERROR|Error|timeout|Timeout|unavailable|queue|RPC_RECV_TIMEOUT|Try again|18482' \
      "${OUT_DIR}/kubernetes-worker-logs.log" \
      | tail -200 \
      | tee "${OUT_DIR}/kubernetes-worker-errors.log" \
      || true
  fi
else
  echo "KUBECTL_STATUS=UNAVAILABLE" | tee "${OUT_DIR}/kubernetes-pods.log"
fi

RPC_SMOKE_STATUS="SKIPPED"
RPC_SMOKE_EXIT_CODE=125
if [ "$REMOTE_TCP_STATUS" = "PASS" ] && [ "$REMOTE_DSBENCH_STATUS" = "EXECUTABLE" ]; then
  log "Remote 1KB DataSystem Set smoke"
  set +e
  ssh "$KVC_LOAD_HOST" \
    "timeout '${RPC_TIMEOUT_SECONDS}s' '$KVC_DSBENCH_CPP' kv --action=set --worker_address='$KVC_DS_ENDPOINT' --prefix='$SMOKE_PREFIX' --client_num=1 --thread_num=1 --num=1 --size=1KB --batch_num=1 --worker_index=0" \
    2>&1 | tee "${OUT_DIR}/rpc-set-smoke.log"
  RPC_SMOKE_EXIT_CODE="${PIPESTATUS[0]}"
  set -e
  echo "$RPC_SMOKE_EXIT_CODE" >"${OUT_DIR}/rpc-set-smoke.exit_code"
  if [ "$RPC_SMOKE_EXIT_CODE" -eq 0 ]; then
    RPC_SMOKE_STATUS="PASS"
    log "Delete smoke key"
    set +e
    ssh "$KVC_LOAD_HOST" \
      "timeout '${RPC_TIMEOUT_SECONDS}s' '$KVC_DSBENCH_CPP' kv --action=del --worker_address='$KVC_DS_ENDPOINT' --prefix='$SMOKE_PREFIX' --client_num=1 --thread_num=1 --num=1 --size=1KB --batch_num=1 --worker_num=1 --worker_index=0" \
      >"${OUT_DIR}/rpc-delete-smoke.log" 2>&1
    DELETE_STATUS="$?"
    set -e
    echo "$DELETE_STATUS" >"${OUT_DIR}/rpc-delete-smoke.exit_code"
  else
    RPC_SMOKE_STATUS="FAIL"
  fi
else
  echo "RPC smoke skipped because TCP or dsbench preflight failed" \
    | tee "${OUT_DIR}/rpc-set-smoke.log"
  echo "$RPC_SMOKE_EXIT_CODE" >"${OUT_DIR}/rpc-set-smoke.exit_code"
fi

RESULT="FAIL"
if [ "$REMOTE_STATE_STATUS" -ne 0 ]; then
  CLASSIFICATION="REMOTE_DIAGNOSTIC_SSH_FAILED"
elif [ "$REMOTE_DSBENCH_STATUS" != "EXECUTABLE" ]; then
  CLASSIFICATION="REMOTE_DSBENCH_UNAVAILABLE"
elif [ "$REMOTE_TCP_STATUS" != "PASS" ]; then
  CLASSIFICATION="ENDPOINT_TCP_UNREACHABLE_FROM_LOAD_HOST"
elif [ "$RPC_SMOKE_STATUS" = "FAIL" ]; then
  CLASSIFICATION="DATASYSTEM_RPC_UNAVAILABLE"
elif [ "$RPC_SMOKE_STATUS" = "PASS" ]; then
  CLASSIFICATION="DATASYSTEM_RPC_SMOKE_OK"
  RESULT="PASS"
else
  CLASSIFICATION="DATASYSTEM_RPC_DIAGNOSTIC_INCOMPLETE"
fi

cat >"${OUT_DIR}/diagnosis.txt" <<EOF
result=${RESULT}
classification=${CLASSIFICATION}
remote_tcp_status=${REMOTE_TCP_STATUS:-UNKNOWN}
master_tcp_status=${MASTER_TCP_STATUS:-UNKNOWN}
remote_dsbench_status=${REMOTE_DSBENCH_STATUS:-UNKNOWN}
rpc_smoke_status=${RPC_SMOKE_STATUS}
rpc_smoke_exit_code=${RPC_SMOKE_EXIT_CODE}
output_dir=${OUT_DIR}
EOF

log "Diagnosis"
cat "${OUT_DIR}/diagnosis.txt"
echo "DATASYSTEM_WORKER_RPC_DIAGNOSTIC_COMPLETE"

[ "$RESULT" = "PASS" ]
