#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-docker.io/library/pairec-server:k8s-arm64-brpc-v1}"
RUNTIME_BASE="${RUNTIME_BASE:-docker.io/library/pairec-server:k8s-arm64-static}"
BUILD_DIR=".docker-build"

echo "Building PaiRec server binary image"
echo "  image:        ${IMAGE}"
echo "  runtime base: ${RUNTIME_BASE}"
echo ""
echo "This path compiles the Go binary on the host and reuses the local runtime"
echo "image, so it does not pull golang/alpine builder images from Docker Hub."

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
trap 'rm -rf "${BUILD_DIR}"' EXIT

GOPROXY=off GOSUMDB=off CGO_ENABLED=0 \
  go build -mod=vendor -ldflags="-s -w" -o "${BUILD_DIR}/pairec-server" ./services/main.go

docker build \
  -f docker/Dockerfile.pairec.binary \
  --build-arg RUNTIME_BASE="${RUNTIME_BASE}" \
  -t "${IMAGE}" \
  .

echo ""
echo "Built ${IMAGE}"
