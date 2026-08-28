#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-docker.io/library/pairec-brpc-inference:k8s-arm64-v1}"
LOCAL_DIR="${LOCAL_DIR:-/home/zcx/pairec-rank-burst-bins}"
WORKER="${WORKER:-root@192.168.100.11}"
REMOTE_DIR="${REMOTE_DIR:-/home/zcx/bin}"
CONTAINER_NAME="pairec-rank-wrapper-extract-$$"
BINARIES=(brpc_rank_burst_wrapper brpc_deepfm_rank_adapter brpc_pipeline_client brpc_recommend_client kvc_burst_wrapper)

die() { echo "ERROR: $*" >&2; exit 1; }
cleanup() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

for command in docker scp ssh sha256sum; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image does not exist: $IMAGE"

mkdir -p "$LOCAL_DIR"
ssh "$WORKER" "mkdir -p '$REMOTE_DIR'"
docker create --name "$CONTAINER_NAME" --entrypoint /bin/true "$IMAGE" >/dev/null
for binary in "${BINARIES[@]}"; do
  local_path="$LOCAL_DIR/$binary"
  remote_path="$REMOTE_DIR/$binary"
  docker cp "${CONTAINER_NAME}:/opt/pairec-brpc/bin/$binary" "$local_path"
  chmod 0755 "$local_path"
  if [[ "$binary" = brpc_recommend_client || "$binary" = brpc_pipeline_client ]]; then
    grep -a -q PAIREC_RETURN_CONTROL_V1 "$local_path" \
      || die "$binary from $IMAGE does not support reverse BRPC control"
  fi
  local_sha="$(sha256sum "$local_path" | awk '{print $1}')"
  scp "$local_path" "${WORKER}:${remote_path}.part"
  remote_sha="$(ssh "$WORKER" "sha256sum '${remote_path}.part'" | awk '{print $1}')"
  [[ "$local_sha" = "$remote_sha" ]] \
    || die "$binary checksum mismatch: local=$local_sha remote=$remote_sha"
  ssh "$WORKER" "chmod 0755 '${remote_path}.part' && mv -f '${remote_path}.part' '${remote_path}'"
  if [[ "$binary" = brpc_recommend_client || "$binary" = brpc_pipeline_client ]]; then
    ssh "$WORKER" "grep -a -q PAIREC_RETURN_CONTROL_V1 '${remote_path}'" \
      || die "$binary installed on $WORKER does not support reverse BRPC control"
  fi
  echo "binary=$binary remote_path=$remote_path sha256=$local_sha"
done

echo "BRPC_RANK_BURST_WRAPPER_BINARY_SHIPPED"
echo "BRPC_RANK_BURST_BINARIES_SHIPPED worker=$WORKER remote_dir=$REMOTE_DIR count=${#BINARIES[@]}"
