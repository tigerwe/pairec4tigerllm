#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-docker.io/library/pairec-brpc-gateway:k8s-arm64-v1}"
TAR_PATH="${2:-/tmp/pairec-brpc-gateway-k8s-arm64-v1.tar}"
BASE_IMAGE="${BASE_IMAGE:-${3:-docker.io/library/zcx-pairec-image:v1.1}}"

docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -f docker/Dockerfile.brpc.gateway \
  -t "$IMAGE" .

echo "Built $IMAGE"
echo "Base image: $BASE_IMAGE"
echo "Export with:"
echo "  docker save $IMAGE -o $TAR_PATH"
