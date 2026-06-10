#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-docker.io/library/pairec-inference:k8s-arm64-runtime}"

docker build -f docker/Dockerfile.inference.runtime -t "$IMAGE" .

echo "Built $IMAGE"
echo "Export with:"
echo "  docker save $IMAGE -o /tmp/pairec-inference-k8s-arm64-runtime.tar"
