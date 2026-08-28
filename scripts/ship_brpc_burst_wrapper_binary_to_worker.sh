#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-docker.io/library/pairec-brpc-inference:k8s-arm64-v1}"
LOCAL_DIR="${LOCAL_DIR:-/home/zcx/pairec-generation-wrapper-bins}"
WORKER="${WORKER:-root@192.168.100.11}"
REMOTE_DIR="${REMOTE_DIR:-/home/zcx/bin}"
CONTAINER_NAME="pairec-brpc-wrapper-extract-$$"
BINARIES=(brpc_burst_wrapper brpc_recommend_client)

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

echo "== Extract generation Wrapper binaries from image =="
mkdir -p "$LOCAL_DIR"
docker create --name "$CONTAINER_NAME" --entrypoint /bin/true "$IMAGE" >/dev/null
ssh "$WORKER" "mkdir -p '$REMOTE_DIR'"
for binary in "${BINARIES[@]}"; do
  local_path="$LOCAL_DIR/$binary"
  remote_path="$REMOTE_DIR/$binary"
  docker cp "${CONTAINER_NAME}:/opt/pairec-brpc/bin/$binary" "$local_path"
  chmod 0755 "$local_path"
  test -x "$local_path" || die "extracted binary is not executable: $local_path"
  local_sha="$(sha256sum "$local_path" | awk '{print $1}')"
  scp "$local_path" "${WORKER}:${remote_path}.part"
  remote_sha="$(ssh "$WORKER" "sha256sum '${remote_path}.part'" | awk '{print $1}')"
  [[ "$local_sha" = "$remote_sha" ]] \
    || die "$binary checksum mismatch: local=$local_sha remote=$remote_sha"
  ssh "$WORKER" \
    "chmod 0755 '${remote_path}.part' && mv -f '${remote_path}.part' '${remote_path}'"
  echo "binary=$binary remote_path=$remote_path sha256=$local_sha"
done

echo "BRPC_BURST_WRAPPER_BINARY_SHIPPED"
echo "image=${IMAGE}"
echo "worker=${WORKER}"
echo "remote_dir=${REMOTE_DIR} count=${#BINARIES[@]}"
