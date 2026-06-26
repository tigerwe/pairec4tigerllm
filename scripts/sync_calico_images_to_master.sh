#!/usr/bin/env bash
set -euo pipefail

SOURCE_NODE="${SOURCE_NODE:-141.61.91.188}"
MASTER_NODE_NAME="${MASTER_NODE_NAME:-master}"
ARCHIVE_REMOTE="${ARCHIVE_REMOTE:-/home/zcx/calico-images-for-master.tar}"
ARCHIVE_LOCAL="${ARCHIVE_LOCAL:-/home/zcx/calico-images-for-master.tar}"
DELETE_MASTER_CALICO_PODS="${DELETE_MASTER_CALICO_PODS:-1}"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "ERROR: kubectl is required" >&2
  exit 1
fi

if ! command -v ssh >/dev/null 2>&1; then
  echo "ERROR: ssh is required" >&2
  exit 1
fi

echo "Collecting Calico images from Kubernetes manifests ..."
mapfile -t images < <(
  {
    kubectl -n kube-system get ds calico-node \
      -o jsonpath='{range .spec.template.spec.initContainers[*]}{.image}{"\n"}{end}{range .spec.template.spec.containers[*]}{.image}{"\n"}{end}' 2>/dev/null || true
    kubectl -n kube-system get deploy calico-kube-controllers \
      -o jsonpath='{range .spec.template.spec.initContainers[*]}{.image}{"\n"}{end}{range .spec.template.spec.containers[*]}{.image}{"\n"}{end}' 2>/dev/null || true
  } | awk 'NF && !seen[$0]++'
)

if [ "${#images[@]}" -eq 0 ]; then
  echo "ERROR: no Calico images were found from kube-system calico manifests" >&2
  exit 1
fi

printf '  %s\n' "${images[@]}"
echo

quoted_images=""
for image in "${images[@]}"; do
  quoted_images="${quoted_images} '${image}'"
done

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
echo "== Local Calico images =="
sudo ctr -n k8s.io images ls | grep -i calico || true

if [ "$DELETE_MASTER_CALICO_PODS" = "1" ]; then
  echo
  echo "== Recreate Calico pods scheduled on ${MASTER_NODE_NAME} =="
  mapfile -t master_calico_pods < <(
    kubectl -n kube-system get pods -o wide --no-headers \
      | awk -v node="$MASTER_NODE_NAME" 'tolower($1) ~ /calico/ && $7 == node {print "pod/" $1}'
  )
  if [ "${#master_calico_pods[@]}" -gt 0 ]; then
    kubectl -n kube-system delete "${master_calico_pods[@]}" --force --grace-period=0
  else
    echo "no Calico pods currently scheduled on ${MASTER_NODE_NAME}"
  fi
fi

echo
echo "== Current Calico pods =="
kubectl -n kube-system get pods -o wide | grep -i calico || true

echo
echo "Next checks:"
echo "  kubectl -n kube-system get pods -o wide | grep -i calico"
echo "  kubectl -n kube-system get pods -o wide | grep -i nvidia"
echo "  kubectl describe node ${MASTER_NODE_NAME} | grep -A8 -E 'Capacity:|Allocatable:|nvidia.com/gpu'"
