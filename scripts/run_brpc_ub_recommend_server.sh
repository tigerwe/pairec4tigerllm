#!/usr/bin/env bash
set -euo pipefail

SERVER_BIN=${SERVER_BIN:-/opt/pairec-brpc-ub-probe/bin/minimal_recommend_server}
PORT=${PORT:-18100}
MAX_PAYLOAD_BYTES=${MAX_PAYLOAD_BYTES:-4194304}
LOG_DIR=${LOG_DIR:-/root/brpc-ub-recommend/server-log}
NOFILE_LIMIT=${NOFILE_LIMIT:-1048576}
URMA_RUNTIME_LIB_DIR=${URMA_RUNTIME_LIB_DIR:-/usr/lib64}
URMA_PROVIDER_LIB_DIR=${URMA_PROVIDER_LIB_DIR:-/usr/lib64/urma}
URMA_RUNTIME_LD_LIBRARY_PATH=${URMA_RUNTIME_LD_LIBRARY_PATH:-$URMA_RUNTIME_LIB_DIR:$URMA_PROVIDER_LIB_DIR}

[[ -x "$SERVER_BIN" ]] || { echo "ERROR: server binary is not executable: $SERVER_BIN" >&2; exit 1; }
[[ -r "$URMA_RUNTIME_LIB_DIR/liburma.so" ]] || {
    echo "ERROR: host URMA runtime is missing: $URMA_RUNTIME_LIB_DIR/liburma.so" >&2
    exit 1
}
mkdir -p "$LOG_DIR"
ulimit -n "$NOFILE_LIMIT"

# Keep a DataSystem virtualenv's bundled liburma from taking precedence over the host driver stack.
export LD_LIBRARY_PATH="$URMA_RUNTIME_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

log_file="$LOG_DIR/minimal-recommend-server-$(date +%Y%m%d-%H%M%S).log"
echo "server_log=$log_file"
echo "urma_runtime_library_path=$URMA_RUNTIME_LD_LIBRARY_PATH"
echo "transport_mode=functional_ub backup_link=default degrade=default"
exec "$SERVER_BIN" \
    --probe_port="$PORT" \
    --probe_max_payload_bytes="$MAX_PAYLOAD_BYTES" \
    --probe_echo_payload=true \
    --ubsocket_enable=true \
    --ubsocket_use_ub=true \
    2>&1 | tee "$log_file"
