#!/usr/bin/env bash
set -euo pipefail

ROLE="${ROLE:-}"
TRANSPORT="${TRANSPORT:-ub}"
HOP_BIN="${HOP_BIN:-/opt/pairec-brpc-ub-recommend-known-good/bin/brpc_post_rank_hop}"
LISTEN_PORT="${LISTEN_PORT:-}"
PRESSURE_LISTEN_PORT="${PRESSURE_LISTEN_PORT:-18313}"
HOP2_HOST="${HOP2_HOST:-127.0.0.1}"
LOG_DIR="${LOG_DIR:-/root/brpc-ub-post-rank/log}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ "$ROLE" = hop1 || "$ROLE" = hop2 ]] || die "ROLE must be hop1 or hop2"
[[ "$TRANSPORT" = tcp || "$TRANSPORT" = ub ]] || die "TRANSPORT must be tcp or ub"
[[ -x "$HOP_BIN" ]] || die "post-rank binary is not executable: $HOP_BIN"
ulimit -n 1048576
export LD_LIBRARY_PATH="/usr/lib64:/usr/lib64/urma${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$LOG_DIR"

if [[ "$ROLE" = hop2 ]]; then
  LISTEN_PORT="${LISTEN_PORT:-18312}"
  args=(
    --role=hop2
    "--transport=$TRANSPORT"
    "--listen_port=$LISTEN_PORT"
    "--pressure_listen_port=$PRESSURE_LISTEN_PORT"
    --idle_timeout_sec=-1
  )
else
  LISTEN_PORT="${LISTEN_PORT:-18311}"
  args=(
    --role=hop1
    "--transport=$TRANSPORT"
    "--listen_port=$LISTEN_PORT"
    "--business_backend=$HOP2_HOST:18312"
    "--pressure_backend=$HOP2_HOST:18313"
    --concurrency=1000
    --payload_bytes=102400
    --business_timeout_ms=1000
    --pressure_timeout_ms=5000
    --pressure_start_quorum=950
    --pressure_start_timeout_ms=250
    --startup_timeout_ms=120000
    --startup_batch_size=64
    --startup_max_retries=3
    --startup_retry_backoff_ms=100
    --idle_timeout_sec=-1
  )
fi

log_file="$LOG_DIR/post-rank-${ROLE}-${TRANSPORT}-$(date +%Y%m%d-%H%M%S).log"
echo "post_rank_role=$ROLE transport=$TRANSPORT log=$log_file"
exec "$HOP_BIN" "${args[@]}" 2>&1 | tee "$log_file"
