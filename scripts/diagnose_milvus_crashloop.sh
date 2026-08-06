#!/usr/bin/env bash
# Collect evidence for a Milvus standalone CrashLoopBackOff without changing it.
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
DEPLOYMENT="${DEPLOYMENT:-milvus-standalone}"
CONTAINER="${CONTAINER:-milvus}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/milvus-crashloop-diagnostic}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-500}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

matches() {
  local pattern="$1"
  local input_file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -qi "$pattern" "$input_file"
  else
    grep -Eqi "$pattern" "$input_file"
  fi
}

classify_log() {
  local input_file="$1"
  if matches 'exec format error|no matching manifest.*(arm64|aarch64)|unsupported platform' "$input_file"; then
    echo "IMAGE_ARCHITECTURE_MISMATCH"
  elif matches 'OOMKilled|out of memory|cannot allocate memory' "$input_file"; then
    echo "MEMORY_LIMIT_OR_OOM"
  elif matches 'no space left on device|disk quota exceeded' "$input_file"; then
    echo "MILVUS_DATA_DISK_FULL"
  elif matches 'permission denied|operation not permitted|read-only file system' "$input_file"; then
    echo "MILVUS_DATA_PERMISSION_FAILURE"
  elif matches '(\[::1\]|localhost).*(2379|etcd)|connect.*\[::1\].*refused' "$input_file"; then
    echo "EMBEDDED_ETCD_IPV6_ENDPOINT_FAILURE"
  elif matches '(connection refused|deadline exceeded|unavailable).*(2379|etcd)|(2379|etcd).*(connection refused|deadline exceeded|unavailable)' "$input_file"; then
    echo "EMBEDDED_ETCD_NOT_READY"
  elif matches 'yaml:|failed to.*config|cannot.*config|parse.*config|unknown field' "$input_file"; then
    echo "MILVUS_CONFIGURATION_INVALID"
  elif matches 'ImagePullBackOff|ErrImagePull|failed to pull image' "$input_file"; then
    echo "MILVUS_IMAGE_PULL_FAILURE"
  elif matches 'Liveness probe failed|Readiness probe failed' "$input_file"; then
    echo "MILVUS_HEALTH_PROBE_FAILURE"
  else
    echo "UNKNOWN_CRASHLOOP_CAUSE"
  fi
}

if [[ "${1:-}" == "--classify-file" ]]; then
  test -n "${2:-}" || fail "usage: $0 --classify-file LOG_FILE"
  test -f "$2" || fail "classification input does not exist: $2"
  classify_log "$2"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"

command -v kubectl >/dev/null 2>&1 || fail "kubectl is not installed"
kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" >/dev/null 2>&1 ||
  fail "deployment not found: ${NAMESPACE}/${DEPLOYMENT}"

POD="$(kubectl -n "$NAMESPACE" get pods \
  -l "app=${DEPLOYMENT}" \
  --sort-by=.metadata.creationTimestamp \
  -o jsonpath='{.items[-1:].metadata.name}')"
test -n "$POD" || fail "no pod found for app=${DEPLOYMENT}"

NODE="$(kubectl -n "$NAMESPACE" get pod "$POD" \
  -o jsonpath='{.spec.nodeName}')"
RESTARTS="$(kubectl -n "$NAMESPACE" get pod "$POD" \
  -o jsonpath='{.status.containerStatuses[0].restartCount}')"
WAITING_REASON="$(kubectl -n "$NAMESPACE" get pod "$POD" \
  -o jsonpath='{.status.containerStatuses[0].state.waiting.reason}')"
LAST_REASON="$(kubectl -n "$NAMESPACE" get pod "$POD" \
  -o jsonpath='{.status.containerStatuses[0].lastState.terminated.reason}')"
LAST_EXIT_CODE="$(kubectl -n "$NAMESPACE" get pod "$POD" \
  -o jsonpath='{.status.containerStatuses[0].lastState.terminated.exitCode}')"

echo "== Milvus target =="
echo "namespace=$NAMESPACE deployment=$DEPLOYMENT pod=$POD container=$CONTAINER"
echo "node=$NODE restarts=${RESTARTS:-0} waiting_reason=${WAITING_REASON:-none} last_reason=${LAST_REASON:-none} last_exit_code=${LAST_EXIT_CODE:-n/a}"

kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o yaml \
  >"$OUTPUT_DIR/deployment.yaml"
kubectl -n "$NAMESPACE" get pod "$POD" -o yaml \
  >"$OUTPUT_DIR/pod.yaml"
kubectl -n "$NAMESPACE" describe pod "$POD" \
  >"$OUTPUT_DIR/pod.describe.txt"
kubectl -n "$NAMESPACE" get events \
  --field-selector "involvedObject.name=${POD}" \
  --sort-by=.lastTimestamp \
  >"$OUTPUT_DIR/pod.events.txt" 2>&1 || true
kubectl get node "$NODE" -o yaml >"$OUTPUT_DIR/node.yaml" 2>&1 || true
kubectl -n "$NAMESPACE" get configmap milvus-embed-config -o yaml \
  >"$OUTPUT_DIR/configmap.yaml" 2>&1 || true
kubectl -n "$NAMESPACE" get service "$DEPLOYMENT" -o yaml \
  >"$OUTPUT_DIR/service.yaml" 2>&1 || true

kubectl -n "$NAMESPACE" logs "$POD" -c "$CONTAINER" \
  --tail="$LOG_TAIL_LINES" >"$OUTPUT_DIR/current.log" 2>&1 || true
kubectl -n "$NAMESPACE" logs "$POD" -c "$CONTAINER" --previous \
  --tail="$LOG_TAIL_LINES" >"$OUTPUT_DIR/previous.log" 2>&1 || true

cat "$OUTPUT_DIR/current.log" "$OUTPUT_DIR/previous.log" \
  "$OUTPUT_DIR/pod.describe.txt" "$OUTPUT_DIR/pod.events.txt" \
  >"$OUTPUT_DIR/combined.txt"

CLASSIFICATION="$(classify_log "$OUTPUT_DIR/combined.txt")"
NEXT_ACTION="inspect ${OUTPUT_DIR}/previous.log and ${OUTPUT_DIR}/pod.describe.txt"

if [[ "$CLASSIFICATION" == "IMAGE_ARCHITECTURE_MISMATCH" ]]; then
  NEXT_ACTION="use an ARM64 Milvus image and verify it with: kubectl get node ${NODE} -o jsonpath='{.status.nodeInfo.architecture}'"
elif [[ "$CLASSIFICATION" == "MEMORY_LIMIT_OR_OOM" ]]; then
  NEXT_ACTION="compare the container memory limit with peak usage and inspect node memory pressure"
elif [[ "$CLASSIFICATION" == "MILVUS_DATA_DISK_FULL" ]]; then
  NEXT_ACTION="inspect free space and inode usage for /home/zcx/milvus-data on node ${NODE}"
elif [[ "$CLASSIFICATION" == "MILVUS_DATA_PERMISSION_FAILURE" ]]; then
  NEXT_ACTION="inspect ownership and permissions of /home/zcx/milvus-data on node ${NODE}"
elif [[ "$CLASSIFICATION" == "EMBEDDED_ETCD_IPV6_ENDPOINT_FAILURE" ]]; then
  NEXT_ACTION="verify the mounted user.yaml and embedEtcd.yaml both use 127.0.0.1:2379"
elif [[ "$CLASSIFICATION" == "EMBEDDED_ETCD_NOT_READY" ]]; then
  NEXT_ACTION="inspect the first embedded-etcd error in ${OUTPUT_DIR}/previous.log before changing topology"
elif [[ "$CLASSIFICATION" == "MILVUS_CONFIGURATION_INVALID" ]]; then
  NEXT_ACTION="compare ${OUTPUT_DIR}/configmap.yaml with the configuration schema supported by the deployed Milvus version"
elif [[ "$CLASSIFICATION" == "MILVUS_IMAGE_PULL_FAILURE" ]]; then
  NEXT_ACTION="import or pull the configured image on node ${NODE} and verify imagePullPolicy"
elif [[ "$CLASSIFICATION" == "MILVUS_HEALTH_PROBE_FAILURE" ]]; then
  NEXT_ACTION="inspect Milvus startup logs and verify that port 9091 exposes /healthz before changing probe thresholds"
fi

cat >"$OUTPUT_DIR/result.env" <<EOF
namespace=$NAMESPACE
deployment=$DEPLOYMENT
pod=$POD
container=$CONTAINER
node=$NODE
restarts=${RESTARTS:-0}
waiting_reason=${WAITING_REASON:-none}
last_reason=${LAST_REASON:-none}
last_exit_code=${LAST_EXIT_CODE:-n/a}
classification=$CLASSIFICATION
output_dir=$OUTPUT_DIR
EOF

echo
echo "== Previous container log tail =="
tail -80 "$OUTPUT_DIR/previous.log"
echo
echo "== Relevant pod events =="
tail -40 "$OUTPUT_DIR/pod.events.txt"
echo
echo "== Diagnosis =="
echo "classification=$CLASSIFICATION"
echo "next_action=$NEXT_ACTION"
echo "output_dir=$OUTPUT_DIR"
echo "MILVUS_CRASHLOOP_DIAGNOSTIC_COMPLETE"

if [[ "$CLASSIFICATION" == "UNKNOWN_CRASHLOOP_CAUSE" ]]; then
  exit 2
fi
