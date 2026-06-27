#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
APP_LABEL="${APP_LABEL:-app=inference-brpc-trtllm}"
CONTAINER="${CONTAINER:-brpc-inference}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_SUDO="${REMOTE_SUDO:-sudo}"
MIN_LOG_SIZE="${MIN_LOG_SIZE:-50M}"
FOLLOW="${FOLLOW:-0}"
OUT_LOG="${OUT_LOG:-/tmp/server_single_100k_c1.log}"

log() {
  printf '\n== %s ==\n' "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: missing command: $1" >&2
    exit 1
  fi
}

require_command kubectl
require_command ssh

POD="${POD:-$(kubectl -n "$NAMESPACE" get pod -l "$APP_LABEL" -o jsonpath='{.items[0].metadata.name}')}"
if [ -z "$POD" ]; then
  echo "ERROR: no pod found for ${NAMESPACE}/${APP_LABEL}" >&2
  exit 1
fi

NODE="$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.spec.nodeName}')"
NODE_IP="$(kubectl get node "$NODE" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')"
if [ -z "$NODE_IP" ]; then
  echo "ERROR: cannot resolve InternalIP for node ${NODE}" >&2
  exit 1
fi

log "Target"
echo "namespace=${NAMESPACE}"
echo "pod=${POD}"
echo "container=${CONTAINER}"
echo "node=${NODE}"
echo "node_ip=${NODE_IP}"
echo "min_log_size=${MIN_LOG_SIZE}"

log "Clean large container logs on node"
ssh "${REMOTE_USER}@${NODE_IP}" \
  "set -euo pipefail
   echo '-- disk before --'
   df -h /
   echo '-- inode before --'
   df -ih /
   echo '-- log usage before --'
   ${REMOTE_SUDO} du -sh /var/log/pods /var/log/containers 2>/dev/null || true
   echo '-- largest logs before --'
   ${REMOTE_SUDO} find /var/log/pods /var/log/containers -type f -name '*.log' -printf '%s %p\n' 2>/dev/null | sort -nr | head -20 || true
   echo '-- truncating logs larger than ${MIN_LOG_SIZE} --'
   ${REMOTE_SUDO} find /var/log/pods /var/log/containers -type f -name '*.log' -size +${MIN_LOG_SIZE} -exec sh -c ': > \"\$1\"' _ {} \;
   echo '-- log usage after --'
   ${REMOTE_SUDO} du -sh /var/log/pods /var/log/containers 2>/dev/null || true
   echo '-- disk after --'
   df -h /
   echo '-- inode after --'
   df -ih /
  "

log "Verify kubectl logs"
kubectl -n "$NAMESPACE" logs "$POD" -c "$CONTAINER" --tail=20 >/tmp/k8s-brpc-log-verify.out 2>/tmp/k8s-brpc-log-verify.err || {
  echo "ERROR: kubectl logs still failed" >&2
  cat /tmp/k8s-brpc-log-verify.err >&2 || true
  exit 1
}
cat /tmp/k8s-brpc-log-verify.out

if [ "$FOLLOW" = "1" ]; then
  log "Follow logs"
  echo "writing to ${OUT_LOG}"
  kubectl -n "$NAMESPACE" logs "$POD" -c "$CONTAINER" --tail=0 -f | tee "$OUT_LOG"
else
  log "Next command"
  echo "FOLLOW=1 OUT_LOG=${OUT_LOG} bash scripts/k8s_prepare_brpc_log_capture.sh"
fi
