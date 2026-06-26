#!/usr/bin/env bash
set -euo pipefail

SOURCE_NODE="${SOURCE_NODE:-141.61.91.188}"
MASTER_NODE_NAME="${MASTER_NODE_NAME:-master}"
DAEMONSET="${DAEMONSET:-nvidia-device-plugin-daemonset}"
NAMESPACE="${NAMESPACE:-kube-system}"
ARCHIVE_REMOTE="${ARCHIVE_REMOTE:-/home/zcx/nvidia-device-plugin-image.tar}"
ARCHIVE_LOCAL="${ARCHIVE_LOCAL:-/home/zcx/nvidia-device-plugin-image.tar}"
DELETE_MASTER_PLUGIN_PODS="${DELETE_MASTER_PLUGIN_PODS:-1}"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "ERROR: kubectl is required" >&2
  exit 1
fi

if ! command -v ssh >/dev/null 2>&1; then
  echo "ERROR: ssh is required" >&2
  exit 1
fi

image="$(kubectl -n "$NAMESPACE" get ds "$DAEMONSET" \
  -o jsonpath='{.spec.template.spec.containers[0].image}')"

if [ -z "$image" ]; then
  echo "ERROR: failed to resolve image from ${NAMESPACE}/daemonset/${DAEMONSET}" >&2
  exit 1
fi

echo "Sync NVIDIA device plugin image to local master node"
echo "  source node: ${SOURCE_NODE}"
echo "  daemonset:   ${NAMESPACE}/${DAEMONSET}"
echo "  image:       ${image}"
echo "  remote tar:  ${ARCHIVE_REMOTE}"
echo "  local tar:   ${ARCHIVE_LOCAL}"
echo

echo "== Export image on ${SOURCE_NODE} =="
ssh "$SOURCE_NODE" "sudo ctr -n k8s.io images export '${ARCHIVE_REMOTE}' '${image}' && ls -lh '${ARCHIVE_REMOTE}'"

echo
echo "== Copy image archive to local master =="
if [ "$ARCHIVE_LOCAL" != "$ARCHIVE_REMOTE" ]; then
  scp "${SOURCE_NODE}:${ARCHIVE_REMOTE}" "$ARCHIVE_LOCAL"
else
  scp "${SOURCE_NODE}:${ARCHIVE_REMOTE}" "${ARCHIVE_LOCAL}.tmp"
  mv "${ARCHIVE_LOCAL}.tmp" "$ARCHIVE_LOCAL"
fi
ls -lh "$ARCHIVE_LOCAL"

echo
echo "== Import image into local k8s.io containerd namespace =="
sudo ctr -n k8s.io images import "$ARCHIVE_LOCAL"

echo
echo "== Local NVIDIA device plugin image =="
sudo ctr -n k8s.io images ls | grep -F "$image" || sudo ctr -n k8s.io images ls | grep -i 'k8s-device-plugin' || true

if [ "$DELETE_MASTER_PLUGIN_PODS" = "1" ]; then
  echo
  echo "== Recreate NVIDIA device plugin pods scheduled on ${MASTER_NODE_NAME} =="
  mapfile -t master_plugin_pods < <(
    kubectl -n "$NAMESPACE" get pods -o wide --no-headers \
      | awk -v node="$MASTER_NODE_NAME" -v ds="$DAEMONSET" '$1 ~ ds && $7 == node {print "pod/" $1}'
  )
  if [ "${#master_plugin_pods[@]}" -gt 0 ]; then
    kubectl -n "$NAMESPACE" delete "${master_plugin_pods[@]}" --force --grace-period=0
  else
    echo "no ${DAEMONSET} pod currently scheduled on ${MASTER_NODE_NAME}"
  fi
fi

echo
echo "== NVIDIA device plugin pods =="
kubectl -n "$NAMESPACE" get pods -o wide | grep -i nvidia || true

echo
echo "== Master GPU allocatable check =="
kubectl describe node "$MASTER_NODE_NAME" | grep -A8 -E "Capacity:|Allocatable:|nvidia.com/gpu" || true

echo
echo "Next expected success condition:"
echo "  kubectl -n ${NAMESPACE} get pods -o wide | grep -i nvidia"
echo "should show a 1/1 Running pod on ${MASTER_NODE_NAME}, and:"
echo "  kubectl describe node ${MASTER_NODE_NAME} | grep -A8 -E 'Capacity:|Allocatable:|nvidia.com/gpu'"
echo "should show:"
echo "  nvidia.com/gpu: 1"
