#!/usr/bin/env bash
# Collect the DataSystem worker's own "resource" log and parse its thread-pool
# saturation counters (idle/current/max/waiting/usage per service thread pool).
#
# Unlike collect_datasystem_worker_metrics.sh (host /proc + cgroup), this reads
# the worker's internal resource log, which the worker writes every
# log_monitor_interval_ms (default 10s) when log_monitor is enabled (default
# true). The "maxWaiting" field is the peak queued-task depth over the interval
# -- the direct evidence of whether Get/Set RPCs queue behind pressure.
#
# Used by benchmark_brpc_kvc_contention.sh around each replay round.
set -euo pipefail

ACTION=${1:-}
NAMESPACE=${NAMESPACE:-pairec}
DS_WORKER_POD_SELECTOR=${DS_WORKER_POD_SELECTOR:-app=datasystem-25g-master}
DS_WORKER_CONTAINER=${DS_WORKER_CONTAINER:-datasystem-worker}
DS_WORKER_LOG_DIR=${DS_WORKER_LOG_DIR:-/tmp/datasystem-25g-master/log}
DS_WORKER_TAIL_LINES=${DS_WORKER_TAIL_LINES:-20}
DS_WORKER_WINDOW_START_NS=${DS_WORKER_WINDOW_START_NS:-}
DS_WORKER_WAIT_UNTIL_NS=${DS_WORKER_WAIT_UNTIL_NS:-}
DS_WORKER_WAIT_TIMEOUT_SECONDS=${DS_WORKER_WAIT_TIMEOUT_SECONDS:-20}
DS_WORKER_WAIT_POLL_SECONDS=${DS_WORKER_WAIT_POLL_SECONDS:-1}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

die() { echo "ERROR: $*" >&2; exit 1; }

snapshot() {
  local out=${1:-}
  [[ -n "$out" ]] || die "usage: $0 snapshot <out.json>"
  local pod node raw ts deadline latest_resource_ts
  pod=$(kubectl -n "$NAMESPACE" get pod -l "$DS_WORKER_POD_SELECTOR" \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1:].metadata.name}')
  [[ -n "$pod" ]] || die "DataSystem worker pod not found: $DS_WORKER_POD_SELECTOR"
  node=$(kubectl -n "$NAMESPACE" get pod "$pod" -o jsonpath='{.spec.nodeName}')
  [[ -n "$node" ]] || die "DataSystem worker pod $pod has no nodeName"
  raw=$(mktemp /tmp/ds-worker-threadpool.XXXXXX.raw)
  deadline=$(( $(date +%s) + DS_WORKER_WAIT_TIMEOUT_SECONDS ))
  while true; do
    ts=$(date +%s%N)
    # Read the most recent resource log file (resource.log or
    # resource.<digits>.log) and tail the last N lines.
    kubectl -n "$NAMESPACE" exec "$pod" -c "$DS_WORKER_CONTAINER" -- \
      env -u LD_PRELOAD sh -c '
        f=$(ls -t "$1"/resource*.log 2>/dev/null | head -1 || true)
        [ -n "$f" ] && tail -n "$2" "$f" || true
      ' -- "$DS_WORKER_LOG_DIR" "$DS_WORKER_TAIL_LINES" >"$raw" 2>/dev/null
    args=(parse-resource-log --raw "$raw" --pod "$pod" --node "$node" --ts-ns "$ts" --out "$out")
    if [[ -n "$DS_WORKER_WINDOW_START_NS" && -n "$DS_WORKER_WAIT_UNTIL_NS" ]]; then
      args+=(--window-start-ns "$DS_WORKER_WINDOW_START_NS" --window-end-ns "$DS_WORKER_WAIT_UNTIL_NS")
    fi
    python3 "$SCRIPT_DIR/datasystem_worker_metrics.py" "${args[@]}"
    if [[ -z "$DS_WORKER_WAIT_UNTIL_NS" ]]; then
      break
    fi
    latest_resource_ts=$(python3 - "$out" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
values = [line.get("resource_ts_ns", 0) for line in data.get("lines", [])]
print(max(values, default=0))
PY
)
    if (( latest_resource_ts >= DS_WORKER_WAIT_UNTIL_NS )); then
      break
    fi
    if (( $(date +%s) >= deadline )); then
      rm -f "$raw"
      die "resource log did not cover measurement end within ${DS_WORKER_WAIT_TIMEOUT_SECONDS}s"
    fi
    sleep "$DS_WORKER_WAIT_POLL_SECONDS"
  done
  rm -f "$raw"
}

case "$ACTION" in
  snapshot)
    snapshot "${2:-}"
    ;;
  *)
    die "usage: $0 snapshot <out.json>"
    ;;
esac
