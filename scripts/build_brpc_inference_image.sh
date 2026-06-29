#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-docker.io/library/pairec-brpc-inference:k8s-arm64-v1}"
TAR_PATH="${2:-/home/zcx/pairec-brpc-inference-k8s-arm64-v1.tar}"
BASE_IMAGE="${BASE_IMAGE:-${3:-zcx-pairec-brpc-sdk:v1}}"
ENABLE_TRTLLM_CPP="${ENABLE_TRTLLM_CPP:-OFF}"
ENABLE_DATASYSTEM_KV_PROBE="${ENABLE_DATASYSTEM_KV_PROBE:-OFF}"
TRTLLM_INCLUDE_DIR="${TRTLLM_INCLUDE_DIR:-}"
TRTLLM_LIBRARY="${TRTLLM_LIBRARY:-}"
TRTLLM_PLUGIN_LIBRARY="${TRTLLM_PLUGIN_LIBRARY:-}"
TRTLLM_EXTRA_LIBS="${TRTLLM_EXTRA_LIBS:-}"
TRTLLM_CUDA_INCLUDE_DIR="${TRTLLM_CUDA_INCLUDE_DIR:-}"
CUDA_DRIVER_LIBRARY="${CUDA_DRIVER_LIBRARY:-}"
DATASYSTEM_INCLUDE_DIR="${DATASYSTEM_INCLUDE_DIR:-}"
DATASYSTEM_LIBRARY="${DATASYSTEM_LIBRARY:-}"

echo "Building brpc inference image"
echo "  image:             $IMAGE"
echo "  base image:        $BASE_IMAGE"
echo "  trtllm cpp:        $ENABLE_TRTLLM_CPP"
echo "  ds kv probe:       $ENABLE_DATASYSTEM_KV_PROBE"
if [ "$ENABLE_TRTLLM_CPP" = "ON" ] || [ "$ENABLE_TRTLLM_CPP" = "1" ]; then
  echo "  trtllm include:    ${TRTLLM_INCLUDE_DIR:-<auto>}"
  echo "  trtllm library:    ${TRTLLM_LIBRARY:-<auto>}"
  echo "  trtllm plugin:     ${TRTLLM_PLUGIN_LIBRARY:-<auto>}"
  echo "  trtllm extra libs: ${TRTLLM_EXTRA_LIBS:-<none>}"
  echo "  cuda include:      ${TRTLLM_CUDA_INCLUDE_DIR:-<auto>}"
  echo "  cuda driver lib:   ${CUDA_DRIVER_LIBRARY:-<auto>}"
fi
if [ "$ENABLE_DATASYSTEM_KV_PROBE" = "ON" ] || [ "$ENABLE_DATASYSTEM_KV_PROBE" = "1" ]; then
  echo "  ds include:        ${DATASYSTEM_INCLUDE_DIR:-<auto>}"
  echo "  ds library:        ${DATASYSTEM_LIBRARY:-<auto>}"
fi

env -u LD_PRELOAD docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "ENABLE_TRTLLM_CPP=$ENABLE_TRTLLM_CPP" \
  --build-arg "ENABLE_DATASYSTEM_KV_PROBE=$ENABLE_DATASYSTEM_KV_PROBE" \
  --build-arg "TRTLLM_INCLUDE_DIR=$TRTLLM_INCLUDE_DIR" \
  --build-arg "TRTLLM_LIBRARY=$TRTLLM_LIBRARY" \
  --build-arg "TRTLLM_PLUGIN_LIBRARY=$TRTLLM_PLUGIN_LIBRARY" \
  --build-arg "TRTLLM_EXTRA_LIBS=$TRTLLM_EXTRA_LIBS" \
  --build-arg "TRTLLM_CUDA_INCLUDE_DIR=$TRTLLM_CUDA_INCLUDE_DIR" \
  --build-arg "CUDA_DRIVER_LIBRARY=$CUDA_DRIVER_LIBRARY" \
  --build-arg "DATASYSTEM_INCLUDE_DIR=$DATASYSTEM_INCLUDE_DIR" \
  --build-arg "DATASYSTEM_LIBRARY=$DATASYSTEM_LIBRARY" \
  -f docker/Dockerfile.brpc.gateway \
  -t "$IMAGE" .

echo "Checking runtime dynamic library dependencies ..."
if [ "$ENABLE_TRTLLM_CPP" = "ON" ] || [ "$ENABLE_TRTLLM_CPP" = "1" ]; then
  docker run --rm --entrypoint /bin/bash -e LD_PRELOAD= "$IMAGE" -lc '
    set -euo pipefail
    unset LD_PRELOAD || true

    echo "== ldd /opt/pairec-brpc/bin/brpc_recommend_client =="
    ldd /opt/pairec-brpc/bin/brpc_recommend_client | tee /tmp/brpc_recommend_client.ldd
    if grep -q "not found" /tmp/brpc_recommend_client.ldd; then
      echo "ERROR: missing runtime libraries for brpc_recommend_client" >&2
      exit 1
    fi

    echo "== readelf -d /opt/pairec-brpc/bin/brpc_inference_server =="
    readelf -d /opt/pairec-brpc/bin/brpc_inference_server | grep NEEDED || true
    echo "TRT-LLM server ldd is deferred to the GPU pod because non-GPU build containers can carry placeholder libcuda/libnvidia-ml files."
  '
else
  docker run --rm --entrypoint /bin/bash -e LD_PRELOAD= "$IMAGE" -lc '
    set -euo pipefail
    unset LD_PRELOAD || true
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
fi

echo "Built $IMAGE"
echo "Base image: $BASE_IMAGE"
echo "Export with:"
echo "  docker save $IMAGE -o $TAR_PATH"
