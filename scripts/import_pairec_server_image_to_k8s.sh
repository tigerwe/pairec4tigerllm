#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAR="${IMAGE_TAR:-/home/zcx/pairec-server-k8s-arm64-brpc-v1.tar}"
IMAGE="${IMAGE:-docker.io/library/pairec-server:k8s-arm64-brpc-v1}"
NAMESPACE="${NAMESPACE:-pairec}"
APP_LABEL="${APP_LABEL:-app=pairec}"
DELETE_NON_RUNNING_PODS="${DELETE_NON_RUNNING_PODS:-0}"

echo "Import PaiRec server image into local k8s.io containerd namespace"
echo "  image tar:  ${IMAGE_TAR}"
echo "  image:      ${IMAGE}"
echo "  namespace:  ${NAMESPACE}"
echo "  app label:  ${APP_LABEL}"
echo

if [ ! -f "$IMAGE_TAR" ]; then
  echo "ERROR: image tar not found: ${IMAGE_TAR}" >&2
  echo "If it exists on worker1, copy it first, for example:" >&2
  echo "  scp 141.61.91.188:${IMAGE_TAR} ${IMAGE_TAR}" >&2
  exit 1
fi

ls -lh "$IMAGE_TAR"

echo
echo "== Import image =="
sudo ctr -n k8s.io images import "$IMAGE_TAR"

echo
echo "== Verify image in k8s.io namespace =="
sudo ctr -n k8s.io images ls | grep -F "$IMAGE" || {
  echo "WARN: exact image reference was not found: ${IMAGE}" >&2
  echo "Available PaiRec server images:" >&2
  sudo ctr -n k8s.io images ls | grep 'pairec-server' || true
}

if command -v crictl >/dev/null 2>&1; then
  echo
  echo "== CRI image view =="
  sudo crictl images | grep 'pairec-server' || true
fi

if [ "$DELETE_NON_RUNNING_PODS" = "1" ]; then
  if ! command -v kubectl >/dev/null 2>&1; then
    echo "WARN: kubectl not found; skip deleting non-running pods"
  else
    echo
    echo "== Delete non-running pods for ${APP_LABEL} =="
    mapfile -t pods < <(
      kubectl -n "$NAMESPACE" get pods -l "$APP_LABEL" --no-headers 2>/dev/null \
        | awk '$3 != "Running" {print "pod/" $1}'
    )
    if [ "${#pods[@]}" -gt 0 ]; then
      kubectl -n "$NAMESPACE" delete "${pods[@]}" --force --grace-period=0
    else
      echo "no non-running pods found for ${APP_LABEL}"
    fi
  fi
fi

echo
echo "Next checks:"
echo "  sudo ctr -n k8s.io images ls | grep 'pairec-server.*k8s-arm64-brpc-v1'"
echo "  kubectl -n ${NAMESPACE} get pods -l ${APP_LABEL} -o wide"
