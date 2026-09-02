#!/usr/bin/env bash
set -euo pipefail

SERVER_BIN=${SERVER_BIN:-/opt/pairec-brpc-ub-probe/bin/minimal_recommend_server}
PORT=${PORT:-18100}
MAX_PAYLOAD_BYTES=${MAX_PAYLOAD_BYTES:-4194304}
LOG_DIR=${LOG_DIR:-/root/brpc-ub-recommend/server-log}
NOFILE_LIMIT=${NOFILE_LIMIT:-1048576}

[[ -x "$SERVER_BIN" ]] || { echo "ERROR: server binary is not executable: $SERVER_BIN" >&2; exit 1; }
mkdir -p "$LOG_DIR"
ulimit -n "$NOFILE_LIMIT"

log_file="$LOG_DIR/minimal-recommend-server-$(date +%Y%m%d-%H%M%S).log"
echo "server_log=$log_file"
exec "$SERVER_BIN" \
    --probe_port="$PORT" \
    --probe_max_payload_bytes="$MAX_PAYLOAD_BYTES" \
    --probe_echo_payload=true \
    --ubsocket_enable=true \
    --ubsocket_use_ub=true \
    --ubsocket_backup_link_enable=false \
    --ubsocket_degrade_enable=false \
    2>&1 | tee "$log_file"
