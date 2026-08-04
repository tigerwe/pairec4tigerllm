#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-docker.io/library/pairec-brpc-inference:k8s-arm64-v1}"
LOCAL_BIN="${LOCAL_BIN:-/home/zcx/brpc_burst_wrapper}"
WORKER="${WORKER:-root@192.168.100.11}"
REMOTE_BIN="${REMOTE_BIN:-/home/zcx/bin/brpc_burst_wrapper}"
CONTAINER_NAME="pairec-brpc-wrapper-extract-$$"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

command -v docker >/dev/null 2>&1 || die "docker is required"
command -v scp >/dev/null 2>&1 || die "scp is required"
command -v ssh >/dev/null 2>&1 || die "ssh is required"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image does not exist: $IMAGE"

echo "== Extract Wrapper binary from image =="
docker create --name "$CONTAINER_NAME" --entrypoint /bin/true "$IMAGE" >/dev/null
docker cp \
  "${CONTAINER_NAME}:/opt/pairec-brpc/bin/brpc_burst_wrapper" \
  "$LOCAL_BIN"
chmod 0755 "$LOCAL_BIN"
test -x "$LOCAL_BIN" || die "extracted wrapper is not executable: $LOCAL_BIN"
ls -lh "$LOCAL_BIN"
local_sha="$(sha256sum "$LOCAL_BIN" | awk '{print $1}')"
echo "local_sha256=${local_sha}"

echo "== Copy Wrapper binary over the 25G link =="
ssh "$WORKER" "mkdir -p '$(dirname "$REMOTE_BIN")'"
scp "$LOCAL_BIN" "${WORKER}:${REMOTE_BIN}.part"

echo "== Verify and activate Wrapper binary =="
remote_sha="$(ssh "$WORKER" "sha256sum '${REMOTE_BIN}.part'" | awk '{print $1}')"
echo "remote_sha256=${remote_sha}"
if [ "$local_sha" != "$remote_sha" ]; then
  die "binary checksum mismatch: local=${local_sha} remote=${remote_sha}"
fi
ssh "$WORKER" \
  "chmod 0755 '${REMOTE_BIN}.part' && mv -f '${REMOTE_BIN}.part' '${REMOTE_BIN}' && ls -lh '${REMOTE_BIN}'"

echo "BRPC_BURST_WRAPPER_BINARY_SHIPPED"
echo "image=${IMAGE}"
echo "worker=${WORKER}"
echo "remote_binary=${REMOTE_BIN}"
echo "sha256=${local_sha}"
