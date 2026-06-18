#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-docker.io/library/pairec-brpc-gateway:k8s-arm64-v1}"
TAR_PATH="${2:-/home/zcx/pairec-brpc-gateway-k8s-arm64-v1.tar}"
BASE_IMAGE="${BASE_IMAGE:-${3:-zcx-pairec-brpc-sdk:v1}}"

docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -f docker/Dockerfile.brpc.gateway \
  -t "$IMAGE" .

echo "Checking runtime dynamic library dependencies ..."
docker run --rm --entrypoint /bin/bash "$IMAGE" -lc '
  set -euo pipefail
  for bin in \
      /opt/pairec-brpc/bin/brpc_gateway \
      /opt/pairec-brpc/bin/brpc_recommend_client \
      /opt/pairec-brpc/bin/brpc_inference_server; do
    echo "== ldd $bin =="
    ldd "$bin" | tee "/tmp/$(basename "$bin").ldd"
    if grep -q "not found" "/tmp/$(basename "$bin").ldd"; then
      echo "ERROR: missing runtime libraries for $bin" >&2
      exit 1
    fi
  done
'

echo "Built $IMAGE"
echo "Base image: $BASE_IMAGE"
echo "Export with:"
echo "  docker save $IMAGE -o $TAR_PATH"
