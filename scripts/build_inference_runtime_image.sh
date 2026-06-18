#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-docker.io/library/pairec-inference:k8s-arm64-ds-runtime-v1}"
TAR_PATH="${2:-/home/zcx/pairec-inference-k8s-arm64-ds-runtime-v1.tar}"
BASE_IMAGE="${BASE_IMAGE:-${3:-docker.io/library/zcx-pairec-ds-runtime:v1}}"

docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -f docker/Dockerfile.inference.runtime \
  -t "$IMAGE" .

echo "Built $IMAGE"
echo "Base image: $BASE_IMAGE"
echo "Export with:"
echo "  docker save $IMAGE -o $TAR_PATH"
