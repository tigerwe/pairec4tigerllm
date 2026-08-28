#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-docker.io/library/pairec-brpc-inference:k8s-arm64-v1}"
HOST_BIN="${HOST_BIN:-/home/zcx/bin/brpc_return_pressure_sink}"
CONTAINER_NAME="pairec-return-sink-extract-$$"

cleanup() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker image inspect "$IMAGE" >/dev/null 2>&1
mkdir -p "$(dirname "$HOST_BIN")"
docker create --name "$CONTAINER_NAME" --entrypoint /bin/true "$IMAGE" >/dev/null
docker cp "${CONTAINER_NAME}:/opt/pairec-brpc/bin/brpc_return_pressure_sink" "${HOST_BIN}.part"
chmod 0755 "${HOST_BIN}.part"
mv -f "${HOST_BIN}.part" "$HOST_BIN"
sha256sum "$HOST_BIN"
echo "BRPC_RETURN_PRESSURE_SINK_BINARY_SHIPPED path=$HOST_BIN"
