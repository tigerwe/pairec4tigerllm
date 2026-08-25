#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-docker.io/library/pairec-brpc-inference:k8s-arm64-v1}"
LOCAL_BIN="${LOCAL_BIN:-/home/zcx/brpc_rank_burst_wrapper}"
WORKER="${WORKER:-root@192.168.100.11}"
REMOTE_BIN="${REMOTE_BIN:-/home/zcx/bin/brpc_rank_burst_wrapper}"
CONTAINER_NAME="pairec-rank-wrapper-extract-$$"

die() { echo "ERROR: $*" >&2; exit 1; }
cleanup() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

for command in docker scp ssh sha256sum; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image does not exist: $IMAGE"

docker create --name "$CONTAINER_NAME" --entrypoint /bin/true "$IMAGE" >/dev/null
docker cp "${CONTAINER_NAME}:/opt/pairec-brpc/bin/brpc_rank_burst_wrapper" "$LOCAL_BIN"
chmod 0755 "$LOCAL_BIN"
local_sha="$(sha256sum "$LOCAL_BIN" | awk '{print $1}')"

ssh "$WORKER" "mkdir -p '$(dirname "$REMOTE_BIN")'"
scp "$LOCAL_BIN" "${WORKER}:${REMOTE_BIN}.part"
remote_sha="$(ssh "$WORKER" "sha256sum '${REMOTE_BIN}.part'" | awk '{print $1}')"
[[ "$local_sha" = "$remote_sha" ]] || die "binary checksum mismatch: local=$local_sha remote=$remote_sha"
ssh "$WORKER" "chmod 0755 '${REMOTE_BIN}.part' && mv -f '${REMOTE_BIN}.part' '${REMOTE_BIN}'"

echo "BRPC_RANK_BURST_WRAPPER_BINARY_SHIPPED"
echo "worker=$WORKER remote_binary=$REMOTE_BIN sha256=$local_sha"
