#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd -P)}"
TRTLLM_DIR="${TRTLLM_DIR:-/home/zcx/TensorRT-LLM}"
RUNTIME_DIR="${RUNTIME_DIR:-/home/zcx/pairec-f19-runtime}"
CONTAINER_REPO_DIR="${CONTAINER_REPO_DIR:-/mnt/pairec-src}"
CONTAINER_TRTLLM_DIR="${CONTAINER_TRTLLM_DIR:-/TensorRT-LLM}"
detected_jobs="$(nproc)"
default_jobs="$detected_jobs"
if (( default_jobs > 32 )); then default_jobs=32; fi
JOBS="${JOBS:-$default_jobs}"
TRT_BUILD_IMAGE="${TRT_BUILD_IMAGE:-zcx-pairec-image:v1.1}"
GATEWAY_BUILD_IMAGE="${GATEWAY_BUILD_IMAGE:-}"
SHOW_HISTORY="${SHOW_HISTORY:-1}"
APPLY_PATCH="${APPLY_PATCH:-1}"

die() { echo "ERROR: $*" >&2; exit 1; }

for command in docker find grep install mktemp sha256sum strings; do
  command -v "$command" >/dev/null || die "missing command: $command"
done
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "JOBS must be a positive integer"
[[ "$SHOW_HISTORY" = 0 || "$SHOW_HISTORY" = 1 ]] || die "SHOW_HISTORY must be 0 or 1"
[[ "$APPLY_PATCH" = 0 || "$APPLY_PATCH" = 1 ]] || die "APPLY_PATCH must be 0 or 1"
test -d "$REPO_DIR/cpp/brpc_gateway" || die "invalid REPO_DIR: $REPO_DIR"
test -d "$TRTLLM_DIR/cpp/build" || die "TensorRT-LLM build tree is missing: $TRTLLM_DIR"

if [[ "$APPLY_PATCH" = 1 ]]; then
  echo "== Apply current F19 attribution patch idempotently =="
  TRTLLM_DIR="$TRTLLM_DIR" \
    bash "$REPO_DIR/scripts/apply_trtllm_datasystem_request_attribution_patch.sh"
fi

grep -Fq PAIREC_DATASYSTEM_REQUEST_ATTRIBUTION_ZERO_INTRUSION_DISABLED_V2 \
  "$TRTLLM_DIR/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp" \
  || die "V2 attribution patch is not present in $TRTLLM_DIR"

if [[ "$SHOW_HISTORY" = 1 ]]; then
  echo "== Previous local build-command evidence (diagnostic only) =="
  history_found=0
  for history_file in /root/.bash_history "$HOME/.bash_history"; do
    [[ -r "$history_file" ]] || continue
    if grep -E 'trtllm-parallel-get-build|f19-gateway-build|host-driver|cudadevrt|cudart_static|pairec-f19-runtime' \
        "$history_file" | tail -40; then
      history_found=1
    fi
  done
  if [[ "$history_found" = 0 ]]; then
    echo "no persisted matching shell-history entry found"
  fi
fi

docker image inspect "$TRT_BUILD_IMAGE" >/dev/null 2>&1 \
  || die "TRT build image is not present in Docker: $TRT_BUILD_IMAGE"
trt_image_arch="$(docker image inspect "$TRT_BUILD_IMAGE" --format '{{.Architecture}}')"
[[ "$trt_image_arch" = arm64 ]] \
  || die "TRT build image architecture must be arm64, got $trt_image_arch"

gateway_image_has_sdk() {
  local image="$1"
  docker image inspect "$image" >/dev/null 2>&1 || return 1
  [[ "$(docker image inspect "$image" --format '{{.Architecture}}')" = arm64 ]] || return 1
  docker run --rm --entrypoint /bin/bash "$image" -lc '
    set -e
    command -v cmake >/dev/null
    command -v c++ >/dev/null
    command -v protoc >/dev/null
    test -f /usr/local/include/brpc/server.h
    test -f /usr/local/include/google/protobuf/message.h
    { test ! -e /TensorRT-LLM || test -d /TensorRT-LLM; }
    find /usr/local/lib /usr/local/lib64 -maxdepth 1 -type f \
      \( -name "libbrpc.so*" -o -name "libbrpc.a" \) -print -quit 2>/dev/null \
      | grep -q .
  ' >/dev/null 2>&1
}

if [[ -n "$GATEWAY_BUILD_IMAGE" ]]; then
  gateway_image_has_sdk "$GATEWAY_BUILD_IMAGE" \
    || die "gateway build image lacks protoc/protobuf/brpc SDK: $GATEWAY_BUILD_IMAGE"
else
  gateway_candidates=(
    "pairec-brpc-inference:k8s-arm64-trtllm-multisequence-kvc-ctx224-v1"
    "pairec-brpc-inference:k8s-arm64-trtllm-parallel-get-ctx224-v1"
    "pairec-brpc-inference:k8s-arm64-trtllm-multisequence-ctx224-v3"
    "pairec-brpc-inference:k8s-arm64-trtllm-native-batch-ctx224-align32-v1"
  )
  for candidate in "${gateway_candidates[@]}"; do
    if gateway_image_has_sdk "$candidate"; then
      GATEWAY_BUILD_IMAGE="$candidate"
      break
    fi
  done
fi
[[ -n "$GATEWAY_BUILD_IMAGE" ]] || die \
  "no local arm64 gateway image contains protoc, protobuf and brpc SDK; set GATEWAY_BUILD_IMAGE"
gateway_image_arch="$(docker image inspect "$GATEWAY_BUILD_IMAGE" --format '{{.Architecture}}')"

cuda_driver_dir="${CUDA_DRIVER_DIR:-}"
if [[ -z "$cuda_driver_dir" ]]; then
  for candidate in /lib64 /usr/lib64 /usr/lib/aarch64-linux-gnu /usr/local/nvidia/lib64; do
    if [[ -f "$candidate/libcuda.so.1" ]]; then
      cuda_driver_dir="$candidate"
      break
    fi
  done
fi
[[ -n "$cuda_driver_dir" && -d "$cuda_driver_dir" \
    && -f "$cuda_driver_dir/libcuda.so.1" ]] \
  || die "host driver directory containing libcuda.so.1 was not found; set CUDA_DRIVER_DIR"

runtime_parent="$(dirname "$RUNTIME_DIR")"
mkdir -p "$runtime_parent"
staging_dir="$(mktemp -d "${RUNTIME_DIR}.staging.XXXXXX")"
cleanup() { rm -rf -- "$staging_dir"; }
trap cleanup EXIT
mkdir -p "$staging_dir/bin" "$staging_dir/lib"

echo "== F19 V2 worker1 dual-container build configuration =="
echo "repo_dir=$REPO_DIR"
echo "trtllm_dir=$TRTLLM_DIR"
echo "runtime_dir=$RUNTIME_DIR"
echo "staging_dir=$staging_dir"
echo "container_repo_dir=$CONTAINER_REPO_DIR"
echo "container_trtllm_dir=$CONTAINER_TRTLLM_DIR"
echo "trt_build_image=$TRT_BUILD_IMAGE"
echo "trt_image_arch=$trt_image_arch"
echo "gateway_build_image=$GATEWAY_BUILD_IMAGE"
echo "gateway_image_arch=$gateway_image_arch"
echo "cuda_driver_dir=$cuda_driver_dir"
echo "jobs=$JOBS detected_jobs=$detected_jobs"
echo "apply_patch=$APPLY_PATCH"

echo "== Stage 1/2: build TensorRT-LLM shared library =="
docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --entrypoint /bin/bash \
  -e JOBS="$JOBS" \
  -e TRTLLM_DIR="$CONTAINER_TRTLLM_DIR" \
  -v "$TRTLLM_DIR:$CONTAINER_TRTLLM_DIR" \
  -v "$staging_dir:/out" \
  -v "$cuda_driver_dir:/host-driver:ro" \
  "$TRT_BUILD_IMAGE" \
  -lc '
    set -euo pipefail
    unset LD_PRELOAD

    for command in cmake c++ install find strings; do
      command -v "$command" >/dev/null || {
        echo "ERROR: TRT build image missing command: $command" >&2
        exit 1
      }
    done

    cuda_static_dir="$(dirname "$(find /usr/local/cuda -type f -name libcudadevrt.a -print -quit)")"
    [[ -f "$cuda_static_dir/libcudadevrt.a" ]] || {
      echo "ERROR: libcudadevrt.a not found in TRT build image" >&2
      exit 1
    }
    [[ -f "$cuda_static_dir/libcudart_static.a" ]] || {
      echo "ERROR: libcudart_static.a not found beside libcudadevrt.a" >&2
      exit 1
    }
    export LIBRARY_PATH="$cuda_static_dir:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="/host-driver:$TRTLLM_DIR/cpp/build/tensorrt_llm:$TRTLLM_DIR/cpp/build/tensorrt_llm/plugins:${LD_LIBRARY_PATH:-}"

    cmake --build "$TRTLLM_DIR/cpp/build" --target tensorrt_llm -j"$JOBS"
    trt_library="$TRTLLM_DIR/cpp/build/tensorrt_llm/libtensorrt_llm.so"
    plugin_library="$TRTLLM_DIR/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so"
    [[ -f "$trt_library" ]] || { echo "ERROR: missing $trt_library" >&2; exit 1; }
    [[ -f "$plugin_library" ]] || { echo "ERROR: missing $plugin_library" >&2; exit 1; }
    grep -Fq datasystem_request_complete < <(strings "$trt_library")
    grep -Fq "\"version\":2" < <(strings "$trt_library")
    install -m 0755 "$trt_library" /out/lib/libtensorrt_llm.so
  '

echo "== Stage 2/2: build BRPC inference gateway =="
docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --entrypoint /bin/bash \
  -e JOBS="$JOBS" \
  -e REPO_DIR="$CONTAINER_REPO_DIR" \
  -e TRTLLM_DIR="$CONTAINER_TRTLLM_DIR" \
  -v "$REPO_DIR:$CONTAINER_REPO_DIR:ro" \
  -v "$TRTLLM_DIR:$CONTAINER_TRTLLM_DIR" \
  -v "$staging_dir:/out" \
  -v "$cuda_driver_dir:/host-driver:ro" \
  "$GATEWAY_BUILD_IMAGE" \
  -lc '
    set -euo pipefail
    unset LD_PRELOAD

    for command in cmake c++ protoc install find strings ldd; do
      command -v "$command" >/dev/null || {
        echo "ERROR: gateway build image missing command: $command" >&2
        exit 1
      }
    done
    [[ -f /usr/local/include/brpc/server.h ]] || {
      echo "ERROR: gateway build image missing brpc headers" >&2
      exit 1
    }

    export LD_LIBRARY_PATH="/host-driver:$TRTLLM_DIR/cpp/build/tensorrt_llm:$TRTLLM_DIR/cpp/build/tensorrt_llm/plugins:${LD_LIBRARY_PATH:-}"
    ds_header="$(find /usr/local -type f -path "*/datasystem/include/datasystem/kv_client.h" -print -quit)"
    ds_include="$(dirname "$(dirname "$ds_header")")"
    ds_library="$(find /usr/local -type f -path "*/datasystem/lib/libdatasystem.so" -print -quit)"
    [[ -f "$ds_include/datasystem/kv_client.h" ]] || {
      echo "ERROR: DataSystem C++ headers not found in gateway build image" >&2
      exit 1
    }
    [[ -f "$ds_library" ]] || {
      echo "ERROR: libdatasystem.so not found in gateway build image" >&2
      exit 1
    }

    trt_library="$TRTLLM_DIR/cpp/build/tensorrt_llm/libtensorrt_llm.so"
    plugin_library="$TRTLLM_DIR/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so"
    [[ -f "$trt_library" ]] || { echo "ERROR: missing $trt_library" >&2; exit 1; }
    [[ -f "$plugin_library" ]] || { echo "ERROR: missing $plugin_library" >&2; exit 1; }

    gateway_build=/tmp/f19-gateway-build-v2
    cmake -E remove_directory "$gateway_build"
    cmake -S "$REPO_DIR/cpp/brpc_gateway" -B "$gateway_build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DPAIREC_ENABLE_TRTLLM_CPP=ON \
      -DPAIREC_ENABLE_DATASYSTEM_KV_PROBE=ON \
      -DTRTLLM_INCLUDE_DIR="$TRTLLM_DIR/cpp/include" \
      -DTRTLLM_LIBRARY="$trt_library" \
      -DTRTLLM_PLUGIN_LIBRARY="$plugin_library" \
      -DTRTLLM_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
      -DCUDA_DRIVER_LIBRARY=/host-driver/libcuda.so.1 \
      -DDATASYSTEM_INCLUDE_DIR="$ds_include" \
      -DDATASYSTEM_LIBRARY="$ds_library"
    cmake --build "$gateway_build" --target brpc_inference_server -j"$JOBS"
    gateway="$gateway_build/brpc_inference_server"
    [[ -x "$gateway" ]] || { echo "ERROR: missing $gateway" >&2; exit 1; }
    grep -Fq output_token_count < <(strings "$gateway")
    grep -Fq runner_ms_per_output_token < <(strings "$gateway")
    if ldd "$gateway" | grep -F "not found"; then
      echo "ERROR: gateway has unresolved runtime dependencies" >&2
      exit 1
    fi
    install -m 0755 "$gateway" /out/bin/brpc_inference_server
  '

echo "== Publish verified F19 V2 runtime =="
test -x "$staging_dir/bin/brpc_inference_server" \
  || die "staged gateway is missing"
test -f "$staging_dir/lib/libtensorrt_llm.so" \
  || die "staged TensorRT-LLM library is missing"
grep -Fq output_token_count < <(strings "$staging_dir/bin/brpc_inference_server")
grep -Fq runner_ms_per_output_token < <(strings "$staging_dir/bin/brpc_inference_server")
grep -Fq datasystem_attribution_ready < <(strings "$staging_dir/lib/libtensorrt_llm.so")

mkdir -p "$RUNTIME_DIR/bin" "$RUNTIME_DIR/lib"
install -m 0755 "$staging_dir/bin/brpc_inference_server" \
  "$RUNTIME_DIR/bin/brpc_inference_server"
install -m 0755 "$staging_dir/lib/libtensorrt_llm.so" \
  "$RUNTIME_DIR/lib/libtensorrt_llm.so"

ls -lh \
  "$RUNTIME_DIR/bin/brpc_inference_server" \
  "$RUNTIME_DIR/lib/libtensorrt_llm.so"
sha256sum \
  "$RUNTIME_DIR/bin/brpc_inference_server" \
  "$RUNTIME_DIR/lib/libtensorrt_llm.so"
grep -F -m1 output_token_count < <(strings "$RUNTIME_DIR/bin/brpc_inference_server")
grep -F -m1 datasystem_attribution_ready < <(strings "$RUNTIME_DIR/lib/libtensorrt_llm.so")
echo "F19_ATTRIBUTION_RUNTIME_BUILD_OK trt_image=$TRT_BUILD_IMAGE gateway_image=$GATEWAY_BUILD_IMAGE runtime_dir=$RUNTIME_DIR"
echo "next=run on master: bash scripts/deploy_f19_datasystem_attribution_overlay.sh apply"
