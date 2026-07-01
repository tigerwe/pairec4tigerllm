#!/usr/bin/env bash
set -euo pipefail

SOURCE_NODE="${SOURCE_NODE:-141.61.91.188}"
MASTER_NODE_NAME="${MASTER_NODE_NAME:-master}"
PAUSE_IMAGE="${PAUSE_IMAGE:-docker.io/library/pause-aarch64:3.8}"
ARCHIVE_REMOTE="${ARCHIVE_REMOTE:-/home/zcx/pause-aarch64-3.8.tar}"
ARCHIVE_LOCAL="${ARCHIVE_LOCAL:-/home/zcx/pause-aarch64-3.8.tar}"
RESTART_KUBELET="${RESTART_KUBELET:-0}"
RESTART_CONTAINERD="${RESTART_CONTAINERD:-0}"
DELETE_NON_RUNNING_PODS="${DELETE_NON_RUNNING_PODS:-0}"
NAMESPACE="${NAMESPACE:-pairec}"

if ! command -v ssh >/dev/null 2>&1; then
  echo "ERROR: ssh is required" >&2
  exit 1
fi

if ! command -v scp >/dev/null 2>&1; then
  echo "ERROR: scp is required" >&2
  exit 1
fi

echo "Sync pause sandbox image to local master"
echo "  source node:       ${SOURCE_NODE}"
echo "  master node name:  ${MASTER_NODE_NAME}"
echo "  pause image:       ${PAUSE_IMAGE}"
echo "  remote archive:    ${ARCHIVE_REMOTE}"
echo "  local archive:     ${ARCHIVE_LOCAL}"
echo "  restart containerd:${RESTART_CONTAINERD}"
echo "  restart kubelet:   ${RESTART_KUBELET}"
echo

echo "== Source node image check =="
ssh "$SOURCE_NODE" "sudo ctr -n k8s.io images ls | grep -F '${PAUSE_IMAGE}'"

echo
echo "== Export pause image on source node =="
ssh "$SOURCE_NODE" \
  "sudo ctr -n k8s.io images export '${ARCHIVE_REMOTE}' '${PAUSE_IMAGE}' && ls -lh '${ARCHIVE_REMOTE}'"

echo
echo "== Copy archive to local master =="
if [ "$ARCHIVE_LOCAL" != "$ARCHIVE_REMOTE" ]; then
  scp "${SOURCE_NODE}:${ARCHIVE_REMOTE}" "$ARCHIVE_LOCAL"
else
  scp "${SOURCE_NODE}:${ARCHIVE_REMOTE}" "${ARCHIVE_LOCAL}.tmp"
  mv "${ARCHIVE_LOCAL}.tmp" "$ARCHIVE_LOCAL"
fi
ls -lh "$ARCHIVE_LOCAL"

echo
echo "== Import into local k8s.io containerd namespace =="
sudo ctr -n k8s.io images import "$ARCHIVE_LOCAL"

echo
echo "== Local pause image checks =="
sudo ctr -n k8s.io images ls | grep -F "$PAUSE_IMAGE" || true
if command -v crictl >/dev/null 2>&1; then
  sudo crictl images | grep -E 'pause|pause-aarch64' || true
  sudo crictl inspecti "$PAUSE_IMAGE" >/dev/null 2>&1 \
    && echo "crictl inspecti ok: ${PAUSE_IMAGE}" \
    || echo "WARN: crictl inspecti did not find ${PAUSE_IMAGE}"
fi

if [ "$RESTART_CONTAINERD" = "1" ]; then
  echo
  echo "== Restart containerd =="
  sudo systemctl restart containerd
fi

if [ "$RESTART_KUBELET" = "1" ]; then
  echo
  echo "== Restart kubelet =="
  sudo systemctl restart kubelet
fi

if [ "$DELETE_NON_RUNNING_PODS" = "1" ]; then
  if ! command -v kubectl >/dev/null 2>&1; then
    echo "WARN: kubectl not found; skip deleting non-running pods"
  else
    echo
    echo "== Delete non-running pods on ${MASTER_NODE_NAME} in namespace ${NAMESPACE} =="
    mapfile -t pods < <(
      kubectl -n "$NAMESPACE" get pods -o wide --no-headers 2>/dev/null \
        | awk -v node="$MASTER_NODE_NAME" '$7 == node && $3 != "Running" {print "pod/" $1}'
    )
    if [ "${#pods[@]}" -gt 0 ]; then
      kubectl -n "$NAMESPACE" delete "${pods[@]}" --force --grace-period=0
    else
      echo "no non-running pods on ${MASTER_NODE_NAME} in namespace ${NAMESPACE}"
    fi
  fi
fi

echo
echo "Next checks:"
echo "  sudo ctr -n k8s.io images ls | grep -F '${PAUSE_IMAGE}'"
echo "  kubectl -n ${NAMESPACE} get pods -o wide"
echo
echo "If pods still report failed to get sandbox image after this import, run:"
echo "  RESTART_KUBELET=1 bash scripts/sync_pause_image_to_master.sh"
