#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
CONFIG_MANIFEST="${CONFIG_MANIFEST:-k8s/configmap-pairec-brpc-wrapper-c1.yaml}"
DEPLOYMENT_MANIFEST="${DEPLOYMENT_MANIFEST:-k8s/deployment-pairec-brpc-wrapper-master.yaml}"
DEPLOYMENT="${DEPLOYMENT:-pairec-brpc-wrapper}"
WRAPPER_ENDPOINT="${WRAPPER_ENDPOINT:-192.168.100.11:18103}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-300s}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

for command in kubectl timeout; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: ${command}"
done
test -f "$CONFIG_MANIFEST" || die "config manifest not found: ${CONFIG_MANIFEST}"
test -f "$DEPLOYMENT_MANIFEST" || die "deployment manifest not found: ${DEPLOYMENT_MANIFEST}"

wrapper_host="${WRAPPER_ENDPOINT%:*}"
wrapper_port="${WRAPPER_ENDPOINT##*:}"

echo "== BRPC Wrapper endpoint preflight from master =="
timeout 3 bash -c "cat </dev/null >/dev/tcp/${wrapper_host}/${wrapper_port}" \
  || die "BRPC Wrapper is unreachable: ${WRAPPER_ENDPOINT}"

echo "== Deploy isolated PaiRec Wrapper client =="
kubectl apply -f k8s/namespace.yaml
kubectl apply -f "$CONFIG_MANIFEST"
kubectl apply -f "$DEPLOYMENT_MANIFEST"
kubectl -n "$NAMESPACE" rollout restart "deployment/${DEPLOYMENT}"
kubectl -n "$NAMESPACE" rollout status "deployment/${DEPLOYMENT}" --timeout="$ROLLOUT_TIMEOUT"

POD="$(kubectl -n "$NAMESPACE" get pod -l "app=${DEPLOYMENT}" \
  -o jsonpath='{.items[0].metadata.name}')"
test -n "$POD" || die "PaiRec Wrapper pod was not found"

echo "== Deployment state =="
kubectl -n "$NAMESPACE" get pod "$POD" -o wide
kubectl -n "$NAMESPACE" get service "$DEPLOYMENT" -o wide

echo "== Runtime configuration and CPU placement =="
kubectl -n "$NAMESPACE" exec "$POD" -- sh -c '
  grep -n "brpc_endpoint" /app/configs/pairec_config.json
  grep -E "Cpus_allowed_list|Mems_allowed_list" /proc/1/status
  if test -f /sys/fs/cgroup/cpu.max; then
    echo -n "cpu.max="; cat /sys/fs/cgroup/cpu.max
  elif test -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us; then
    echo -n "cpu.cfs_quota_us="; cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
    echo -n "cpu.cfs_period_us="; cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
  fi
'

echo "PAIREC_BRPC_WRAPPER_DEPLOYMENT_READY"
echo "deployment=${DEPLOYMENT}"
echo "wrapper_endpoint=${WRAPPER_ENDPOINT}"
echo "next=REQUESTS=3 bash scripts/test_pairec_brpc_wrapper_smoke.sh"
