#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-docker.io/library/pairec-brpc-inference:k8s-arm64-v1}"
TAR_PATH="${2:-/home/zcx/pairec-brpc-inference-k8s-arm64-v1.tar}"
BASE_IMAGE="${BASE_IMAGE:-${3:-zcx-pairec-brpc-sdk:v1}}"
ENABLE_TRTLLM_CPP="${ENABLE_TRTLLM_CPP:-OFF}"
ENABLE_DATASYSTEM_KV_PROBE="${ENABLE_DATASYSTEM_KV_PROBE:-OFF}"
ENABLE_BRPC_UB="${ENABLE_BRPC_UB:-OFF}"
TRTLLM_INCLUDE_DIR="${TRTLLM_INCLUDE_DIR:-}"
TRTLLM_LIBRARY="${TRTLLM_LIBRARY:-}"
TRTLLM_PLUGIN_LIBRARY="${TRTLLM_PLUGIN_LIBRARY:-}"
TRTLLM_EXTRA_LIBS="${TRTLLM_EXTRA_LIBS:-}"
TRTLLM_CUDA_INCLUDE_DIR="${TRTLLM_CUDA_INCLUDE_DIR:-}"
CUDA_DRIVER_LIBRARY="${CUDA_DRIVER_LIBRARY:-}"
DATASYSTEM_INCLUDE_DIR="${DATASYSTEM_INCLUDE_DIR:-}"
DATASYSTEM_LIBRARY="${DATASYSTEM_LIBRARY:-}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git rev-parse --short=12 HEAD)}"

die() { echo "ERROR: $*" >&2; exit 1; }

if [ "$ENABLE_DATASYSTEM_KV_PROBE" = "ON" ] || [ "$ENABLE_DATASYSTEM_KV_PROBE" = "1" ]; then
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 \
    || die "DataSystem build base image does not exist locally: $BASE_IMAGE"
  if [ -z "$DATASYSTEM_INCLUDE_DIR" ]; then
    ds_header="$(env -u LD_PRELOAD docker run --rm --entrypoint /bin/bash \
      -e LD_PRELOAD= "$BASE_IMAGE" -lc \
      'find /usr/local /opt /workspace -type f -path "*/datasystem/include/datasystem/kv_client.h" -print -quit 2>/dev/null')"
    [ -n "$ds_header" ] \
      || die "DataSystem C++ header datasystem/kv_client.h is absent from base image $BASE_IMAGE"
    DATASYSTEM_INCLUDE_DIR="${ds_header%/datasystem/kv_client.h}"
  fi
  if [ -z "$DATASYSTEM_LIBRARY" ]; then
    DATASYSTEM_LIBRARY="$(env -u LD_PRELOAD docker run --rm --entrypoint /bin/bash \
      -e LD_PRELOAD= "$BASE_IMAGE" -lc \
      'find /usr/local /opt /workspace \( -type f -o -type l \) -path "*/datasystem/lib/libdatasystem.so" -print -quit 2>/dev/null')"
    [ -n "$DATASYSTEM_LIBRARY" ] \
      || die "libdatasystem.so is absent from base image $BASE_IMAGE"
  fi
fi

echo "Building brpc inference image"
echo "  image:             $IMAGE"
echo "  base image:        $BASE_IMAGE"
echo "  trtllm cpp:        $ENABLE_TRTLLM_CPP"
echo "  ds kv probe:       $ENABLE_DATASYSTEM_KV_PROBE"
echo "  brpc ub:           $ENABLE_BRPC_UB"
echo "  source commit:     $SOURCE_COMMIT"
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
  --build-arg "ENABLE_BRPC_UB=$ENABLE_BRPC_UB" \
  --build-arg "TRTLLM_INCLUDE_DIR=$TRTLLM_INCLUDE_DIR" \
  --build-arg "TRTLLM_LIBRARY=$TRTLLM_LIBRARY" \
  --build-arg "TRTLLM_PLUGIN_LIBRARY=$TRTLLM_PLUGIN_LIBRARY" \
  --build-arg "TRTLLM_EXTRA_LIBS=$TRTLLM_EXTRA_LIBS" \
  --build-arg "TRTLLM_CUDA_INCLUDE_DIR=$TRTLLM_CUDA_INCLUDE_DIR" \
  --build-arg "CUDA_DRIVER_LIBRARY=$CUDA_DRIVER_LIBRARY" \
  --build-arg "DATASYSTEM_INCLUDE_DIR=$DATASYSTEM_INCLUDE_DIR" \
  --build-arg "DATASYSTEM_LIBRARY=$DATASYSTEM_LIBRARY" \
  --build-arg "SOURCE_COMMIT=$SOURCE_COMMIT" \
  -f docker/Dockerfile.brpc.gateway \
  -t "$IMAGE" .

echo "Checking reverse BRPC control protocol support ..."
docker run --rm --entrypoint /bin/bash -e LD_PRELOAD= "$IMAGE" -lc '
  set -euo pipefail
  unset LD_PRELOAD || true
  for binary in brpc_recommend_client brpc_pipeline_client; do
    grep -a -q PAIREC_RETURN_CONTROL_V1 "/opt/pairec-brpc/bin/$binary" || {
      echo "ERROR: $binary does not contain reverse BRPC control support" >&2
      exit 1
    }
  done
  grep -a -q rank-kvc-refresh /opt/pairec-brpc/bin/brpc_pipeline_client
  grep -a -q rank_kvc_business_refresh_complete \
    /opt/pairec-brpc/bin/brpc_rank_burst_wrapper
'

echo "Checking runtime dynamic library dependencies ..."
if [ "$ENABLE_TRTLLM_CPP" = "ON" ] || [ "$ENABLE_TRTLLM_CPP" = "1" ]; then
  docker run --rm --entrypoint /bin/bash -e LD_PRELOAD= "$IMAGE" -lc '
    set -euo pipefail
    unset LD_PRELOAD || true

    for bin in \
        /opt/pairec-brpc/bin/brpc_burst_wrapper \
        /opt/pairec-brpc/bin/brpc_return_pressure_sink \
        /opt/pairec-brpc/bin/brpc_post_rank_hop \
        /opt/pairec-brpc/bin/brpc_rank_burst_wrapper \
        /opt/pairec-brpc/bin/brpc_recommend_client; do
      echo "== ldd $bin =="
      ldd "$bin" | tee "/tmp/$(basename "$bin").ldd"
      if grep -q "not found" "/tmp/$(basename "$bin").ldd"; then
        echo "ERROR: missing runtime libraries for $bin" >&2
        exit 1
      fi
    done
    if test -x /opt/pairec-brpc/bin/kvc_burst_wrapper; then
      ldd /opt/pairec-brpc/bin/kvc_burst_wrapper | tee /tmp/kvc_burst_wrapper.ldd
      ! grep -q "not found" /tmp/kvc_burst_wrapper.ldd
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
        /opt/pairec-brpc/bin/brpc_burst_wrapper \
        /opt/pairec-brpc/bin/brpc_return_pressure_sink \
        /opt/pairec-brpc/bin/brpc_post_rank_hop \
        /opt/pairec-brpc/bin/brpc_rank_burst_wrapper \
        /opt/pairec-brpc/bin/brpc_inference_server \
        /opt/pairec-brpc/bin/brpc_recommend_client; do
      echo "== ldd $bin =="
      ldd "$bin" | tee "/tmp/$(basename "$bin").ldd"
      if grep -q "not found" "/tmp/$(basename "$bin").ldd"; then
        echo "ERROR: missing runtime libraries for $bin" >&2
        exit 1
      fi
    done
    if test -x /opt/pairec-brpc/bin/kvc_burst_wrapper; then
      ldd /opt/pairec-brpc/bin/kvc_burst_wrapper | tee /tmp/kvc_burst_wrapper.ldd
      ! grep -q "not found" /tmp/kvc_burst_wrapper.ldd
    fi
  '
fi

echo "Built $IMAGE"
echo "Base image: $BASE_IMAGE"
echo "Export with:"
echo "  docker save $IMAGE -o $TAR_PATH"
