#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
MANIFEST="${MANIFEST:-k8s/deployment-brpc-pressure-target-188.yaml}"
DEPLOYMENT="${DEPLOYMENT:-brpc-pressure-target}"
ENDPOINT="${ENDPOINT:-192.168.100.11:18101,192.168.100.11:18102}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-300s}"
PROBE_BIN="${PROBE_BIN:-/tmp/probe-go-brpc-client}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[ -f "$MANIFEST" ] || die "manifest not found: $MANIFEST"
command -v kubectl >/dev/null 2>&1 || die "kubectl is required"
command -v go >/dev/null 2>&1 || die "go is required"

kubectl apply -f "$MANIFEST"
kubectl -n "$NAMESPACE" rollout status "deployment/${DEPLOYMENT}" --timeout="$ROLLOUT_TIMEOUT"
kubectl -n "$NAMESPACE" get pod -l "app=${DEPLOYMENT}" -o wide

POD="$(kubectl -n "$NAMESPACE" get pod -l "app=${DEPLOYMENT}" \
  -o jsonpath='{.items[0].metadata.name}')"
[ -n "$POD" ] || die "pressure target pod was not found"
kubectl -n "$NAMESPACE" get pod "$POD" \
  -o jsonpath='requests={.spec.containers[0].resources.requests}{"\n"}limits={.spec.containers[0].resources.limits}{"\n"}'
kubectl -n "$NAMESPACE" exec "$POD" -- bash -lc '
  if test -f /sys/fs/cgroup/cpu.max; then
    echo -n "cpu.max="
    cat /sys/fs/cgroup/cpu.max
  elif test -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us; then
    echo -n "cpu.cfs_quota_us="
    cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
    echo -n "cpu.cfs_period_us="
    cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
  fi
  if test -f /sys/fs/cgroup/cpu.stat; then
    echo "cpu.stat:"
    cat /sys/fs/cgroup/cpu.stat
  elif test -f /sys/fs/cgroup/cpu/cpu.stat; then
    echo "cpu.stat:"
    cat /sys/fs/cgroup/cpu/cpu.stat
  fi
'

GOPROXY="${GOPROXY:-off}" GOSUMDB="${GOSUMDB:-off}" \
  go build -mod=vendor -o "$PROBE_BIN" ./scripts/probe_go_brpc_client.go

ENDPOINT_COUNT="$(awk -F, '{print NF}' <<<"$ENDPOINT")"
"$PROBE_BIN" \
  --endpoint="$ENDPOINT" \
  --method=health \
  --requests="$ENDPOINT_COUNT" \
  --concurrency="$ENDPOINT_COUNT" \
  --timeout_ms=5000 \
  --max_retries=0

echo "BRPC pressure targets are ready: ${ENDPOINT}"
