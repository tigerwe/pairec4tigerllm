#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-docker.io/library/pairec-inference:k8s-arm64-ds-kv-v1}"
TAR_PATH="${2:-/tmp/pairec-inference-k8s-arm64-ds-kv-v1.tar}"

docker build -f docker/Dockerfile.inference.runtime -t "$IMAGE" .

echo "Built $IMAGE"
echo "Export with:"
echo "  docker save $IMAGE -o $TAR_PATH"
