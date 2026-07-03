#!/usr/bin/env bash
set -euo pipefail

SOURCE_NODE="${SOURCE_NODE:-141.61.91.188}"
MASTER_NODE_NAME="${MASTER_NODE_NAME:-master}"
PAUSE_IMAGE="${PAUSE_IMAGE:-docker.io/library/pause-aarch64:3.8}"
SYNC_PAUSE="${SYNC_PAUSE:-1}"
SYNC_CALICO="${SYNC_CALICO:-1}"
EXTRA_IMAGES="${EXTRA_IMAGES:-}"
ARCHIVE_REMOTE="${ARCHIVE_REMOTE:-/home/zcx/master-runtime-images.tar}"
ARCHIVE_LOCAL="${ARCHIVE_LOCAL:-/home/zcx/master-runtime-images.tar}"
RESTART_KUBELET="${RESTART_KUBELET:-0}"
RESTART_CONTAINERD="${RESTART_CONTAINERD:-0}"
DELETE_MASTER_CALICO_PODS="${DELETE_MASTER_CALICO_PODS:-1}"
DELETE_NON_RUNNING_PODS="${DELETE_NON_RUNNING_PODS:-0}"
NAMESPACE="${NAMESPACE:-pairec}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: missing command: $1" >&2
    exit 1
  fi
}

add_image() {
  local image="$1"
  image="${image//$'\r'/}"
  if [ -z "$image" ]; then
    return
  fi
  local existing
  for existing in "${images[@]}"; do
    if [ "$existing" = "$image" ]; then
      return
    fi
  done
  images+=("$image")
}

collect_calico_images() {
  require_command kubectl
  while IFS= read -r image; do
    add_image "$image"
  done < <(
    {
      kubectl -n kube-system get ds calico-node \
        -o jsonpath='{range .spec.template.spec.initContainers[*]}{.image}{"\n"}{end}{range .spec.template.spec.containers[*]}{.image}{"\n"}{end}' 2>/dev/null || true
      kubectl -n kube-system get deploy calico-kube-controllers \
        -o jsonpath='{range .spec.template.spec.initContainers[*]}{.image}{"\n"}{end}{range .spec.template.spec.containers[*]}{.image}{"\n"}{end}' 2>/dev/null || true
    } | awk 'NF'
  )
}

collect_extra_images() {
  local image
  while IFS= read -r image; do
    add_image "$image"
  done < <(printf '%s\n' "$EXTRA_IMAGES" | tr ',;' '\n' | awk '{for (i=1; i<=NF; i++) print $i}')
}

delete_master_calico_pods() {
  require_command kubectl
  echo
  echo "== Recreate Calico pods scheduled on ${MASTER_NODE_NAME} =="
  mapfile -t master_calico_pods < <(
    kubectl -n kube-system get pods \
      --field-selector "spec.nodeName=${MASTER_NODE_NAME}" \
      -o name 2>/dev/null | grep -i calico || true
  )
  if [ "${#master_calico_pods[@]}" -gt 0 ]; then
    kubectl -n kube-system delete "${master_calico_pods[@]}" --force --grace-period=0
  else
    echo "no Calico pods currently scheduled on ${MASTER_NODE_NAME}"
  fi
}

delete_non_running_namespace_pods() {
  require_command kubectl
  echo
  echo "== Delete non-running pods on ${MASTER_NODE_NAME} in namespace ${NAMESPACE} =="
  mapfile -t pods < <(
    kubectl -n "$NAMESPACE" get pods \
      --field-selector "spec.nodeName=${MASTER_NODE_NAME}" \
      --no-headers 2>/dev/null \
      | awk '$3 != "Running" {print "pod/" $1}'
  )
  if [ "${#pods[@]}" -gt 0 ]; then
    kubectl -n "$NAMESPACE" delete "${pods[@]}" --force --grace-period=0
  else
    echo "no non-running pods on ${MASTER_NODE_NAME} in namespace ${NAMESPACE}"
  fi
}

require_command ssh
require_command scp

images=()
if [ "$SYNC_PAUSE" = "1" ]; then
  add_image "$PAUSE_IMAGE"
fi
if [ "$SYNC_CALICO" = "1" ]; then
  collect_calico_images
fi
collect_extra_images

if [ "${#images[@]}" -eq 0 ]; then
  echo "ERROR: no images selected for sync" >&2
  exit 1
fi

echo "Sync runtime images to local master"
echo "  source node:              ${SOURCE_NODE}"
echo "  master node name:         ${MASTER_NODE_NAME}"
echo "  sync pause:               ${SYNC_PAUSE}"
echo "  sync calico:              ${SYNC_CALICO}"
echo "  remote archive:           ${ARCHIVE_REMOTE}"
echo "  local archive:            ${ARCHIVE_LOCAL}"
echo "  restart containerd:       ${RESTART_CONTAINERD}"
echo "  restart kubelet:          ${RESTART_KUBELET}"
echo "  delete master calico pods:${DELETE_MASTER_CALICO_PODS}"
echo "  delete non-running pods:  ${DELETE_NON_RUNNING_PODS}"
echo
echo "Images:"
printf '  %s\n' "${images[@]}"
echo

quoted_images=""
for image in "${images[@]}"; do
  quoted_images="${quoted_images} '${image}'"
done

echo "== Source node image checks =="
for image in "${images[@]}"; do
  ssh "$SOURCE_NODE" "sudo ctr -n k8s.io images ls | grep -F '${image}'"
done

echo
echo "== Export images on ${SOURCE_NODE} =="
ssh "$SOURCE_NODE" "sudo ctr -n k8s.io images export '${ARCHIVE_REMOTE}' ${quoted_images} && ls -lh '${ARCHIVE_REMOTE}'"

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
echo "== Import images into local k8s.io containerd namespace =="
sudo ctr -n k8s.io images import "$ARCHIVE_LOCAL"

echo
echo "== Local image checks =="
for image in "${images[@]}"; do
  sudo ctr -n k8s.io images ls | grep -F "$image" || true
  if command -v crictl >/dev/null 2>&1; then
    sudo crictl inspecti "$image" >/dev/null 2>&1 \
      && echo "crictl inspecti ok: ${image}" \
      || echo "WARN: crictl inspecti did not find ${image}"
  fi
done

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

if [ "$SYNC_CALICO" = "1" ] && [ "$DELETE_MASTER_CALICO_PODS" = "1" ]; then
  delete_master_calico_pods
fi

if [ "$DELETE_NON_RUNNING_PODS" = "1" ]; then
  delete_non_running_namespace_pods
fi

echo
echo "Next checks:"
echo "  sudo ctr -n k8s.io images ls | grep -E 'pause|calico'"
echo "  kubectl -n kube-system get pods -o wide | grep -E 'calico|tigera'"
echo "  kubectl -n ${NAMESPACE} get pods -o wide"
echo
echo "Examples:"
echo "  RESTART_KUBELET=1 bash scripts/sync_master_runtime_images.sh"
echo "  SYNC_PAUSE=0 SYNC_CALICO=1 bash scripts/sync_master_runtime_images.sh"
echo "  EXTRA_IMAGES='nvcr.io/nvidia/k8s-device-plugin:v0.19.2' bash scripts/sync_master_runtime_images.sh"
