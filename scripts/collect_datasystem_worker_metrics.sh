#!/usr/bin/env bash
# Collect host-level metrics from the DataSystem worker pod (hostNetwork, so
# /proc and /proc/net reflect the node it runs on). Used by
# benchmark_brpc_kvc_contention.sh to capture before/after snapshots around
# each replay round; deltas are computed by scripts/datasystem_worker_metrics.py.
set -euo pipefail

ACTION=${1:-}
NAMESPACE=${NAMESPACE:-pairec}
DS_WORKER_POD_SELECTOR=${DS_WORKER_POD_SELECTOR:-app=datasystem-25g-master}
DS_WORKER_CONTAINER=${DS_WORKER_CONTAINER:-datasystem-worker}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

die() { echo "ERROR: $*" >&2; exit 1; }

snapshot() {
  local out=${1:-}
  [[ -n "$out" ]] || die "usage: $0 snapshot <out.json>"
  local pod node ts raw
  pod=$(kubectl -n "$NAMESPACE" get pod -l "$DS_WORKER_POD_SELECTOR" \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1:].metadata.name}')
  [[ -n "$pod" ]] || die "DataSystem worker pod not found: $DS_WORKER_POD_SELECTOR"
  node=$(kubectl -n "$NAMESPACE" get pod "$pod" -o jsonpath='{.spec.nodeName}')
  [[ -n "$node" ]] || die "DataSystem worker pod $pod has no nodeName"
  raw=$(mktemp /tmp/ds-worker-metrics.XXXXXX.raw)
  ts=$(date +%s%N)
  kubectl -n "$NAMESPACE" exec "$pod" -c "$DS_WORKER_CONTAINER" -- \
    env -u LD_PRELOAD sh -c '
      echo @cpu_stat
      cat /sys/fs/cgroup/cpu.stat 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.stat 2>/dev/null || true
      echo @cpuacct
      echo "usage $(cat /sys/fs/cgroup/cpuacct/cpuacct.usage 2>/dev/null)"
      echo @loadavg
      cat /proc/loadavg
      echo @ctxt
      grep -E "ctxt_switches" /proc/1/status
      echo @procstat
      cat /proc/1/stat
      echo @netdev
      cat /proc/net/dev
      echo @softirq
      grep -E "NET_(RX|TX)" /proc/softirqs
      echo @psi_cpu
      cat /proc/pressure/cpu 2>/dev/null || true
      echo @psi_memory
      cat /proc/pressure/memory 2>/dev/null || true
      echo @memstat
      (cat /sys/fs/cgroup/memory.stat 2>/dev/null || cat /sys/fs/cgroup/memory/memory.stat 2>/dev/null) \
        | grep -E "^(pgfault|pgmajfault) " || true
      echo @memcurrent
      cat /sys/fs/cgroup/memory.current 2>/dev/null \
        || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || true
    ' >"$raw"
  python3 "$SCRIPT_DIR/datasystem_worker_metrics.py" parse-snapshot \
    --raw "$raw" --pod "$pod" --node "$node" --ts-ns "$ts" --out "$out"
  rm -f "$raw"
}

case "$ACTION" in
  snapshot)
    snapshot "${2:-}"
    ;;
  delta)
    [[ $# -eq 4 ]] || die "usage: $0 delta <before.json> <after.json> <out.json>"
    python3 "$SCRIPT_DIR/datasystem_worker_metrics.py" delta \
      --before "$2" --after "$3" --out "$4"
    ;;
  *)
    die "usage: $0 snapshot <out.json> | delta <before.json> <after.json> <out.json>"
    ;;
esac
