#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-docker.io/library/pairec-brpc-inference:k8s-arm64-v1}"
TAR_PATH="${2:-/tmp/pairec-brpc-inference-k8s-arm64-v1.tar}"
WORKER="${WORKER:-${3:-root@141.61.91.188}}"
REMOTE_TAR="${REMOTE_TAR:-$TAR_PATH}"
IMPORT_CMD="${IMPORT_CMD:-ctr}"

echo "Exporting $IMAGE to $TAR_PATH"
docker save "$IMAGE" -o "$TAR_PATH"

echo "Copying $TAR_PATH to $WORKER:$REMOTE_TAR"
scp "$TAR_PATH" "$WORKER:$REMOTE_TAR"

case "$IMPORT_CMD" in
  ctr)
    echo "Importing on worker with containerd"
    ssh "$WORKER" "ctr -n k8s.io images import '$REMOTE_TAR' && ctr -n k8s.io images ls | grep -F 'pairec-brpc-inference'"
    ;;
  docker)
    echo "Importing on worker with docker"
    ssh "$WORKER" "docker load -i '$REMOTE_TAR' && docker images | grep -F 'pairec-brpc-inference'"
    ;;
  none)
    echo "Skipping remote import because IMPORT_CMD=none"
    ;;
  *)
    echo "Unsupported IMPORT_CMD=$IMPORT_CMD, expected ctr|docker|none" >&2
    exit 2
    ;;
esac

echo ""
echo "Native brpc inference image is available on $WORKER"
echo "Next:"
echo "  bash scripts/k8s_apply_inference_brpc_native.sh"
echo "  bash scripts/test_brpc_native_inference_smoke.sh"
