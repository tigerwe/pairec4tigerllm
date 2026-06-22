#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-docker.io/library/pairec-brpc-inference:k8s-arm64-v1}"
TAR_PATH="${2:-/home/zcx/pairec-brpc-inference-k8s-arm64-v1.tar}"
BASE_IMAGE="${BASE_IMAGE:-${3:-zcx-pairec-brpc-sdk:v1}}"
ENABLE_TRTLLM_CPP="${ENABLE_TRTLLM_CPP:-OFF}"
TRTLLM_INCLUDE_DIR="${TRTLLM_INCLUDE_DIR:-}"
TRTLLM_LIBRARY="${TRTLLM_LIBRARY:-}"
TRTLLM_EXTRA_LIBS="${TRTLLM_EXTRA_LIBS:-}"
TRTLLM_CUDA_INCLUDE_DIR="${TRTLLM_CUDA_INCLUDE_DIR:-}"
CUDA_DRIVER_LIBRARY="${CUDA_DRIVER_LIBRARY:-}"

echo "Building brpc inference image"
echo "  image:             $IMAGE"
echo "  base image:        $BASE_IMAGE"
echo "  trtllm cpp:        $ENABLE_TRTLLM_CPP"
if [ "$ENABLE_TRTLLM_CPP" = "ON" ] || [ "$ENABLE_TRTLLM_CPP" = "1" ]; then
  echo "  trtllm include:    ${TRTLLM_INCLUDE_DIR:-<auto>}"
  echo "  trtllm library:    ${TRTLLM_LIBRARY:-<auto>}"
  echo "  trtllm extra libs: ${TRTLLM_EXTRA_LIBS:-<none>}"
  echo "  cuda include:      ${TRTLLM_CUDA_INCLUDE_DIR:-<auto>}"
  echo "  cuda driver lib:   ${CUDA_DRIVER_LIBRARY:-<auto>}"
fi

env -u LD_PRELOAD docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "ENABLE_TRTLLM_CPP=$ENABLE_TRTLLM_CPP" \
  --build-arg "TRTLLM_INCLUDE_DIR=$TRTLLM_INCLUDE_DIR" \
  --build-arg "TRTLLM_LIBRARY=$TRTLLM_LIBRARY" \
  --build-arg "TRTLLM_EXTRA_LIBS=$TRTLLM_EXTRA_LIBS" \
  --build-arg "TRTLLM_CUDA_INCLUDE_DIR=$TRTLLM_CUDA_INCLUDE_DIR" \
  --build-arg "CUDA_DRIVER_LIBRARY=$CUDA_DRIVER_LIBRARY" \
  -f docker/Dockerfile.brpc.gateway \
  -t "$IMAGE" .

echo "Checking runtime dynamic library dependencies ..."
docker run --rm --entrypoint /bin/bash "$IMAGE" -lc '
  set -euo pipefail
  for bin in \
      /opt/pairec-brpc/bin/brpc_inference_server \
      /opt/pairec-brpc/bin/brpc_recommend_client; do
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
