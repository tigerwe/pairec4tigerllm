#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
MANIFEST="${MANIFEST:-k8s/deployment-brpc-burst-wrapper-master.yaml}"
DEPLOYMENT="${DEPLOYMENT:-brpc-burst-wrapper-master}"
IMAGE="${IMAGE:-docker.io/library/pairec-brpc-inference:k8s-arm64-v1}"
HOST_BIN="${HOST_BIN:-/home/zcx/bin/brpc_burst_wrapper}"
WRAPPER_ENDPOINT="${WRAPPER_ENDPOINT:-192.168.100.12:18104}"
BACKEND_ENDPOINT="${BACKEND_ENDPOINT:-192.168.100.11:18100}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-5m}"
PAUSE_IMAGE="${PAUSE_IMAGE:-docker.io/library/pause-aarch64:3.8}"
PAUSE_ARCHIVE="${PAUSE_ARCHIVE:-/home/zcx/pause-aarch64-3.8.tar}"

die() { echo "ERROR: $*" >&2; exit 1; }
for command in ctr docker kubectl timeout; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done
test -f "$MANIFEST" || die "missing manifest: $MANIFEST"
kubectl get node master >/dev/null 2>&1 || die "Kubernetes node master does not exist"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "missing master runtime image: $IMAGE"

ctr_k8s() {
  if (( EUID == 0 )); then ctr -n k8s.io "$@"; else sudo -n ctr -n k8s.io "$@"; fi
}
has_k8s_image() {
  ctr_k8s images list | awk 'NR > 1 {print $1}' | grep -Fxq "$1"
}
if ! has_k8s_image "$PAUSE_IMAGE"; then
  test -f "$PAUSE_ARCHIVE" || die "sandbox image and archive are missing: $PAUSE_IMAGE / $PAUSE_ARCHIVE"
  ctr_k8s images import "$PAUSE_ARCHIVE" >/dev/null \
    || die "failed to import sandbox archive: $PAUSE_ARCHIVE"
fi
has_k8s_image "$PAUSE_IMAGE" || die "sandbox image import did not create $PAUSE_IMAGE"
has_k8s_image "$IMAGE" \
  || die "master k8s.io containerd is missing runtime image: $IMAGE"
echo "K8S_MASTER_RUNTIME_READY pause=$PAUSE_IMAGE image=$IMAGE"

if [[ ! -x "$HOST_BIN" ]]; then
  echo "== Extract Wrapper binary for master hostPath =="
  container="pairec-wrapper-master-extract-$$"
  cleanup_extract() { docker rm -f "$container" >/dev/null 2>&1 || true; }
  trap cleanup_extract EXIT
  mkdir -p "$(dirname "$HOST_BIN")"
  docker create --name "$container" --entrypoint /bin/true "$IMAGE" >/dev/null
  docker cp "$container:/opt/pairec-brpc/bin/brpc_burst_wrapper" "${HOST_BIN}.part"
  install -m 0755 "${HOST_BIN}.part" "$HOST_BIN"
  rm -f "${HOST_BIN}.part"
  cleanup_extract
  trap - EXIT
fi
test -x "$HOST_BIN" || die "Wrapper binary is not executable: $HOST_BIN"
sha256sum "$HOST_BIN"

backend_host="${BACKEND_ENDPOINT%:*}"
backend_port="${BACKEND_ENDPOINT##*:}"
timeout 3 bash -c "cat </dev/null >/dev/tcp/${backend_host}/${backend_port}" \
  || die "backend is unreachable: $BACKEND_ENDPOINT"

echo "== Deploy master BRPC burst Wrapper =="
kubectl apply -f "$MANIFEST"
kubectl -n "$NAMESPACE" rollout status "deployment/$DEPLOYMENT" --timeout="$ROLLOUT_TIMEOUT"
pod="$(kubectl -n "$NAMESPACE" get pod -l "app=$DEPLOYMENT" -o jsonpath='{.items[0].metadata.name}')"
test -n "$pod" || die "master Wrapper pod is missing"
node="$(kubectl -n "$NAMESPACE" get pod "$pod" -o jsonpath='{.spec.nodeName}')"
[[ "$node" = master ]] || die "master Wrapper scheduled on unexpected node: $node"

kubectl -n "$NAMESPACE" exec "$pod" -c brpc-burst-wrapper -- \
  /opt/pairec-brpc/bin/brpc_recommend_client \
    --server=127.0.0.1:18104 --method=health --requests=1 \
    --timeout_ms=3000 --max_retry=0
kubectl -n "$NAMESPACE" exec "$pod" -c brpc-burst-wrapper -- \
  /opt/pairec-brpc/bin/brpc_recommend_client \
    --server=127.0.0.1:18104 --method=recommend --topk=1 --requests=1 \
    --timeout_ms=5000 --max_retry=0

echo "BRPC_BURST_WRAPPER_MASTER_OK pod=$pod node=$node"
echo "wrapper_endpoint=$WRAPPER_ENDPOINT backend_endpoint=$BACKEND_ENDPOINT"
