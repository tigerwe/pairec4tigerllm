#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
MANIFEST="${MANIFEST:-k8s/deployment-brpc-burst-wrapper-188.yaml}"
DEPLOYMENT="${DEPLOYMENT:-brpc-burst-wrapper}"
WRAPPER_ENDPOINT="${WRAPPER_ENDPOINT:-192.168.100.11:18103}"
BACKEND_ENDPOINT="${BACKEND_ENDPOINT:-192.168.100.11:18100}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-300s}"
WRAPPER_HOST_BIN="${WRAPPER_HOST_BIN:-/home/zcx/bin/brpc_burst_wrapper}"
WRAPPER_WORKER="${WRAPPER_WORKER:-root@192.168.100.11}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[ -f "$MANIFEST" ] || die "manifest not found: $MANIFEST"
command -v kubectl >/dev/null 2>&1 || die "kubectl is required"
command -v ssh >/dev/null 2>&1 || die "ssh is required"

echo "== Wrapper binary preflight on worker1 =="
kubectl get node worker1 >/dev/null 2>&1 || die "Kubernetes node worker1 does not exist"
ssh "$WRAPPER_WORKER" \
  "test -x '${WRAPPER_HOST_BIN}' && ls -lh '${WRAPPER_HOST_BIN}'" \
  || die "wrapper binary is missing on worker1: ${WRAPPER_HOST_BIN}"

echo "== Backend preflight =="
backend_host="${BACKEND_ENDPOINT%:*}"
backend_port="${BACKEND_ENDPOINT##*:}"
timeout 3 bash -c "cat </dev/null >/dev/tcp/${backend_host}/${backend_port}" \
  || die "backend is unreachable: ${BACKEND_ENDPOINT}"

echo "== Deploy BRPC burst wrapper =="
kubectl apply -f "$MANIFEST"
kubectl -n "$NAMESPACE" rollout status "deployment/${DEPLOYMENT}" --timeout="$ROLLOUT_TIMEOUT"
kubectl -n "$NAMESPACE" get pod -l "app=${DEPLOYMENT}" -o wide

POD="$(kubectl -n "$NAMESPACE" get pod -l "app=${DEPLOYMENT}" \
  -o jsonpath='{.items[0].metadata.name}')"
[ -n "$POD" ] || die "wrapper pod was not found"

echo "== Wrapper runtime and CPU placement =="
kubectl -n "$NAMESPACE" exec "$POD" -- bash -lc '
  test -x /opt/pairec-brpc/bin/brpc_burst_wrapper
  grep -E "Cpus_allowed_list|Mems_allowed_list" /proc/1/status
  if test -f /sys/fs/cgroup/cpu.max; then
    echo -n "cpu.max="; cat /sys/fs/cgroup/cpu.max
  elif test -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us; then
    echo -n "cpu.cfs_quota_us="; cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
    echo -n "cpu.cfs_period_us="; cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
  fi
'

echo "== Local Health terminates in wrapper =="
kubectl -n "$NAMESPACE" exec "$POD" -- \
  /opt/pairec-brpc/bin/brpc_recommend_client \
    --server=127.0.0.1:18103 \
    --method=health \
    --requests=1 \
    --timeout_ms=3000 \
    --max_retry=0

echo "== Recommend forwards to inference backend =="
kubectl -n "$NAMESPACE" exec "$POD" -- \
  /opt/pairec-brpc/bin/brpc_recommend_client \
    --server=127.0.0.1:18103 \
    --method=recommend \
    --topk=1 \
    --requests=1 \
    --timeout_ms=5000 \
    --max_retry=0

kubectl -n "$NAMESPACE" logs "$POD" --tail=30

echo "BRPC_BURST_WRAPPER_DEPLOYMENT_OK"
echo "wrapper_endpoint=${WRAPPER_ENDPOINT}"
echo "backend_endpoint=${BACKEND_ENDPOINT}"
echo "next=ENDPOINT=${WRAPPER_ENDPOINT} REQUIRE_SERVER_WRAPPER=1 REPEATS=3 bash scripts/benchmark_go_brpc_burst_wrapper.sh"
