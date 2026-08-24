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
TRT_BUILD_IMAGE="${TRT_BUILD_IMAGE:-}"
GATEWAY_BUILD_IMAGE="${GATEWAY_BUILD_IMAGE:-}"
SHOW_HISTORY="${SHOW_HISTORY:-1}"
APPLY_PATCH="${APPLY_PATCH:-1}"
APPLY_KVC_BURST_PROXY="${APPLY_KVC_BURST_PROXY:-0}"
BUILD_TRTLLM="${BUILD_TRTLLM:-1}"
TRT_CMAKE_BUILD_TYPE="${TRT_CMAKE_BUILD_TYPE:-Release}"

die() { echo "ERROR: $*" >&2; exit 1; }

for command in docker find grep install mktemp sha256sum strings; do
  command -v "$command" >/dev/null || die "missing command: $command"
done
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "JOBS must be a positive integer"
[[ "$SHOW_HISTORY" = 0 || "$SHOW_HISTORY" = 1 ]] || die "SHOW_HISTORY must be 0 or 1"
[[ "$APPLY_PATCH" = 0 || "$APPLY_PATCH" = 1 ]] || die "APPLY_PATCH must be 0 or 1"
[[ "$APPLY_KVC_BURST_PROXY" = 0 || "$APPLY_KVC_BURST_PROXY" = 1 ]] \
  || die "APPLY_KVC_BURST_PROXY must be 0 or 1"
[[ "$BUILD_TRTLLM" = 0 || "$BUILD_TRTLLM" = 1 ]] || die "BUILD_TRTLLM must be 0 or 1"
[[ "$TRT_CMAKE_BUILD_TYPE" = Release ]] \
  || die "TRT_CMAKE_BUILD_TYPE must be Release for the production inference runtime"
test -d "$REPO_DIR/cpp/brpc_gateway" || die "invalid REPO_DIR: $REPO_DIR"
test -d "$TRTLLM_DIR/cpp/build" || die "TensorRT-LLM build tree is missing: $TRTLLM_DIR"

if [[ "$APPLY_PATCH" = 1 ]]; then
  echo "== Apply current F19 attribution patch idempotently =="
  TRTLLM_DIR="$TRTLLM_DIR" \
    bash "$REPO_DIR/scripts/apply_trtllm_datasystem_request_attribution_patch.sh"
fi

if [[ "$APPLY_KVC_BURST_PROXY" = 1 ]]; then
  echo "== Apply F14 in-process KVC burst proxy patch idempotently =="
  TRTLLM_DIR="$TRTLLM_DIR" \
    bash "$REPO_DIR/scripts/apply_trtllm_kvc_burst_proxy_patch.sh"
fi

grep -Fq PAIREC_DATASYSTEM_REQUEST_ATTRIBUTION_ZERO_INTRUSION_DISABLED_V2 \
  "$TRTLLM_DIR/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp" \
  || die "V2 attribution patch is not present in $TRTLLM_DIR"
grep -Fq PAIREC_TRT_EXECUTOR_PHASE_TIMING_V3 \
  "$TRTLLM_DIR/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp" \
  || die "V3 executor phase timing patch is not present in $TRTLLM_DIR"

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

trt_image_has_toolchain() {
  local image="$1"
  local output
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "TRT candidate rejected: image not found: $image" >&2
    return 1
  fi
  if [[ "$(docker image inspect "$image" --format '{{.Architecture}}')" != arm64 ]]; then
    echo "TRT candidate rejected: image is not arm64: $image" >&2
    return 1
  fi
  if ! output="$(docker run --rm --entrypoint /bin/bash -e LD_PRELOAD= "$image" -lc '
    set -e
    command -v cmake >/dev/null || { echo "missing cmake" >&2; exit 1; }
    command -v c++ >/dev/null || { echo "missing c++" >&2; exit 1; }
    cuda_static_dir=""
    for cuda_root in /usr/local/cuda /usr/local/cuda-*; do
      [[ -e "$cuda_root" ]] || continue
      cuda_devrt="$(find -L "$cuda_root" -type f -name libcudadevrt.a -print -quit 2>/dev/null || true)"
      if [[ -n "$cuda_devrt" ]]; then
        cuda_static_dir="$(dirname "$cuda_devrt")"
        break
      fi
    done
    [[ -n "$cuda_static_dir" ]] \
      || { echo "missing libcudadevrt.a under /usr/local/cuda*" >&2; exit 1; }
    test -f "$cuda_static_dir/libcudadevrt.a"
    test -f "$cuda_static_dir/libcudart_static.a" \
      || { echo "missing libcudart_static.a beside $cuda_static_dir/libcudadevrt.a" >&2; exit 1; }
  ' 2>&1)"; then
    echo "TRT candidate rejected: $image: $output" >&2
    return 1
  fi
}

if [[ "$BUILD_TRTLLM" = 0 ]]; then
  test -f "$RUNTIME_DIR/lib/libtensorrt_llm.so" \
    || die "BUILD_TRTLLM=0 requires $RUNTIME_DIR/lib/libtensorrt_llm.so"
  TRT_BUILD_IMAGE="${TRT_BUILD_IMAGE:-reused-runtime}"
  trt_image_arch="reused"
else
  if [[ -n "$TRT_BUILD_IMAGE" ]]; then
    trt_image_has_toolchain "$TRT_BUILD_IMAGE" \
      || die "TRT build image lacks CUDA static runtime or compiler toolchain: $TRT_BUILD_IMAGE"
  else
    trt_candidates=(
      "zcx-pairec-trtllm-brpc-sdk:parallel-get-ctx224-v1"
      "zcx-pairec-image:v1.1"
    )
    for candidate in "${trt_candidates[@]}"; do
      if trt_image_has_toolchain "$candidate"; then
        TRT_BUILD_IMAGE="$candidate"
        break
      fi
    done
  fi
  [[ -n "$TRT_BUILD_IMAGE" ]] || die \
    "no local arm64 TRT image contains compiler, libcudadevrt.a and libcudart_static.a; set TRT_BUILD_IMAGE"
  trt_image_arch="$(docker image inspect "$TRT_BUILD_IMAGE" --format '{{.Architecture}}')"
fi

gateway_image_has_sdk() {
  local image="$1"
  local output
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "gateway candidate rejected: image not found: $image" >&2
    return 1
  fi
  if [[ "$(docker image inspect "$image" --format '{{.Architecture}}')" != arm64 ]]; then
    echo "gateway candidate rejected: image is not arm64: $image" >&2
    return 1
  fi
  if ! output="$(docker run --rm --entrypoint /bin/bash -e LD_PRELOAD= "$image" -lc '
    set -e
    command -v cmake >/dev/null || { echo "missing cmake" >&2; exit 1; }
    command -v c++ >/dev/null || { echo "missing c++" >&2; exit 1; }
    { test -f /usr/local/include/brpc/server.h || test -f /usr/include/brpc/server.h; } \
      || { echo "missing brpc/server.h" >&2; exit 1; }
    { test -f /usr/local/include/google/protobuf/message.h \
      || test -f /usr/include/google/protobuf/message.h; } \
      || { echo "missing protobuf/message.h" >&2; exit 1; }
    brpc_library_found=0
    for root in /usr/local/lib /usr/local/lib64 /usr/lib64 /usr/lib; do
      test -d "$root" || continue
      if find "$root" -maxdepth 1 \( -type f -o -type l \) \
          \( -name "libbrpc.so*" -o -name "libbrpc.a" \) -print -quit \
          | grep -q .; then
        brpc_library_found=1
        break
      fi
    done
    test "$brpc_library_found" = 1 || { echo "missing libbrpc" >&2; exit 1; }
  ' 2>&1)"; then
    echo "gateway candidate rejected: $image: $output" >&2
    return 1
  fi
}

if [[ -n "$GATEWAY_BUILD_IMAGE" ]]; then
  gateway_image_has_sdk "$GATEWAY_BUILD_IMAGE" \
    || die "gateway build image lacks protobuf/brpc C++ SDK: $GATEWAY_BUILD_IMAGE"
else
  gateway_candidates=(
    "zcx-pairec-trtllm-brpc-sdk:parallel-get-ctx224-v1"
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
  "no local arm64 gateway image contains protobuf and brpc C++ SDK; set GATEWAY_BUILD_IMAGE"
gateway_image_arch="$(docker image inspect "$GATEWAY_BUILD_IMAGE" --format '{{.Architecture}}')"

cuda_driver_file="${CUDA_DRIVER_LIBRARY:-}"
if [[ -z "$cuda_driver_file" && -n "${CUDA_DRIVER_DIR:-}" ]]; then
  cuda_driver_file="$CUDA_DRIVER_DIR/libcuda.so.1"
fi
if [[ -z "$cuda_driver_file" ]]; then
  for candidate in /lib64 /usr/lib64 /usr/lib/aarch64-linux-gnu /usr/local/nvidia/lib64; do
    if [[ -f "$candidate/libcuda.so.1" ]]; then
      cuda_driver_file="$candidate/libcuda.so.1"
      break
    fi
  done
fi
[[ -n "$cuda_driver_file" && -f "$cuda_driver_file" ]] \
  || die "host libcuda.so.1 was not found; set CUDA_DRIVER_LIBRARY"

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
echo "cuda_driver_library=$cuda_driver_file"
echo "jobs=$JOBS detected_jobs=$detected_jobs"
echo "apply_patch=$APPLY_PATCH"
echo "build_trtllm=$BUILD_TRTLLM"
echo "trt_cmake_build_type=$TRT_CMAKE_BUILD_TYPE"

if [[ "$BUILD_TRTLLM" = 1 ]]; then
  echo "== Stage 1/2: build TensorRT-LLM shared library =="
docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --entrypoint /bin/bash \
  -e JOBS="$JOBS" \
  -e TRTLLM_DIR="$CONTAINER_TRTLLM_DIR" \
  -e TRT_CMAKE_BUILD_TYPE="$TRT_CMAKE_BUILD_TYPE" \
  -v "$TRTLLM_DIR:$CONTAINER_TRTLLM_DIR" \
  -v "$staging_dir:/out" \
  -v "$cuda_driver_file:/host-driver/libcuda.so.1:ro" \
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

    cuda_static_dir=""
    for cuda_root in /usr/local/cuda /usr/local/cuda-*; do
      [[ -e "$cuda_root" ]] || continue
      cuda_devrt="$(find -L "$cuda_root" -type f -name libcudadevrt.a -print -quit 2>/dev/null || true)"
      if [[ -n "$cuda_devrt" ]]; then
        cuda_static_dir="$(dirname "$cuda_devrt")"
        break
      fi
    done
    [[ -n "$cuda_static_dir" && -f "$cuda_static_dir/libcudadevrt.a" ]] || {
      echo "ERROR: libcudadevrt.a not found in TRT build image" >&2
      exit 1
    }
    [[ -f "$cuda_static_dir/libcudart_static.a" ]] || {
      echo "ERROR: libcudart_static.a not found beside libcudadevrt.a" >&2
      exit 1
    }
    export LIBRARY_PATH="$cuda_static_dir:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$TRTLLM_DIR/cpp/build/tensorrt_llm:$TRTLLM_DIR/cpp/build/tensorrt_llm/plugins:${LD_LIBRARY_PATH:-}"

    cmake -S "$TRTLLM_DIR/cpp" -B "$TRTLLM_DIR/cpp/build" \
      -DCMAKE_BUILD_TYPE="$TRT_CMAKE_BUILD_TYPE"
    grep -Fxq "CMAKE_BUILD_TYPE:STRING=$TRT_CMAKE_BUILD_TYPE" \
      "$TRTLLM_DIR/cpp/build/CMakeCache.txt" || {
        echo "ERROR: TensorRT-LLM build tree is not configured as $TRT_CMAKE_BUILD_TYPE" >&2
        exit 1
      }
    grep -Eq "^CMAKE_(CXX|CUDA)_FLAGS_RELEASE:STRING=.*-O3.*-DNDEBUG" \
      "$TRTLLM_DIR/cpp/build/CMakeCache.txt" || {
        echo "ERROR: Release optimization flags are missing from TensorRT-LLM CMake cache" >&2
        exit 1
      }
    cmake --build "$TRTLLM_DIR/cpp/build" --target tensorrt_llm -j"$JOBS"
    trt_library="$TRTLLM_DIR/cpp/build/tensorrt_llm/libtensorrt_llm.so"
    plugin_library="$TRTLLM_DIR/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so"
    [[ -f "$trt_library" ]] || { echo "ERROR: missing $trt_library" >&2; exit 1; }
    [[ -f "$plugin_library" ]] || { echo "ERROR: missing $plugin_library" >&2; exit 1; }
    grep -Fq datasystem_request_complete < <(strings "$trt_library")
    grep -Fq "\"version\":3" < <(strings "$trt_library")
    grep -Fq phase_timing_complete < <(strings "$trt_library")
    install -m 0755 "$trt_library" /out/lib/libtensorrt_llm.so
  '
else
  echo "== Stage 1/2: reuse verified TensorRT-LLM shared library =="
  install -m 0755 "$RUNTIME_DIR/lib/libtensorrt_llm.so" \
    "$staging_dir/lib/libtensorrt_llm.so"
  grep -Fq datasystem_request_complete \
    < <(strings "$staging_dir/lib/libtensorrt_llm.so")
  grep -Fq phase_timing_complete \
    < <(strings "$staging_dir/lib/libtensorrt_llm.so")
fi

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
  -v "$cuda_driver_file:/host-driver/libcuda.so.1:ro" \
  "$GATEWAY_BUILD_IMAGE" \
  -lc '
    set -euo pipefail
    unset LD_PRELOAD

    for command in cmake c++ install find strings ldd; do
      command -v "$command" >/dev/null || {
        echo "ERROR: gateway build image missing command: $command" >&2
        exit 1
      }
    done
    brpc_header=""
    for candidate in /usr/local/include/brpc/server.h /usr/include/brpc/server.h; do
      if [[ -f "$candidate" ]]; then brpc_header="$candidate"; break; fi
    done
    [[ -n "$brpc_header" ]] || {
      echo "ERROR: gateway build image missing brpc headers" >&2
      exit 1
    }
    brpc_include="${brpc_header%/brpc/server.h}"
    brpc_library=""
    for root in /usr/local/lib /usr/local/lib64 /usr/lib64 /usr/lib; do
      [[ -d "$root" ]] || continue
      candidate="$(find "$root" -maxdepth 1 \( -type f -o -type l \) \
        \( -name "libbrpc.so*" -o -name "libbrpc.a" \) -print -quit)"
      if [[ -n "$candidate" ]]; then brpc_library="$candidate"; break; fi
    done
    [[ -n "$brpc_library" ]] || {
      echo "ERROR: gateway build image missing brpc library" >&2
      exit 1
    }

    export LD_LIBRARY_PATH="$TRTLLM_DIR/cpp/build/tensorrt_llm:$TRTLLM_DIR/cpp/build/tensorrt_llm/plugins:${LD_LIBRARY_PATH:-}"
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
      -DPAIREC_USE_PREGENERATED_PROTO=ON \
      -DBRPC_INCLUDE_DIR="$brpc_include" \
      -DBRPC_LIBRARY="$brpc_library" \
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
    grep -Fq trt_executor_request_complete < <(strings "$gateway")
    if ldd "$gateway" | grep -F "not found"; then
      echo "ERROR: gateway has unresolved runtime dependencies" >&2
      exit 1
    fi
    install -m 0755 "$gateway" /out/bin/brpc_inference_server

    kvc_build=/tmp/f14-kvc-burst-build
    cmake -E remove_directory "$kvc_build"
    cmake -S "$REPO_DIR/cpp/kvc_burst" -B "$kvc_build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DDATASYSTEM_INCLUDE_DIR="$ds_include" \
      -DDATASYSTEM_LIBRARY="$ds_library"
    cmake --build "$kvc_build" --target kvc_burst_wrapper -j"$JOBS"
    install -m 0755 "$kvc_build/kvc_burst_wrapper" /out/bin/kvc_burst_wrapper
  '

echo "== Publish verified F19 V2 runtime =="
test -x "$staging_dir/bin/brpc_inference_server" \
  || die "staged gateway is missing"
test -f "$staging_dir/lib/libtensorrt_llm.so" \
  || die "staged TensorRT-LLM library is missing"
test -x "$staging_dir/bin/kvc_burst_wrapper" \
  || die "staged KVC burst sidecar is missing"
grep -Fq output_token_count < <(strings "$staging_dir/bin/brpc_inference_server")
grep -Fq runner_ms_per_output_token < <(strings "$staging_dir/bin/brpc_inference_server")
grep -Fq trt_executor_request_complete < <(strings "$staging_dir/bin/brpc_inference_server")
grep -Fq datasystem_attribution_ready < <(strings "$staging_dir/lib/libtensorrt_llm.so")
grep -Fq phase_timing_complete < <(strings "$staging_dir/lib/libtensorrt_llm.so")

mkdir -p "$RUNTIME_DIR/bin" "$RUNTIME_DIR/lib"
install -m 0755 "$staging_dir/bin/brpc_inference_server" \
  "$RUNTIME_DIR/bin/brpc_inference_server"
install -m 0755 "$staging_dir/lib/libtensorrt_llm.so" \
  "$RUNTIME_DIR/lib/libtensorrt_llm.so"
install -m 0755 "$staging_dir/bin/kvc_burst_wrapper" \
  "$RUNTIME_DIR/bin/kvc_burst_wrapper"

if [[ "$APPLY_KVC_BURST_PROXY" = 1 ]]; then
  grep -Fq PAIREC_KVC_BURST_PROXY_V6 \
    < <(strings "$RUNTIME_DIR/lib/libtensorrt_llm.so")
fi

ls -lh \
  "$RUNTIME_DIR/bin/brpc_inference_server" \
  "$RUNTIME_DIR/bin/kvc_burst_wrapper" \
  "$RUNTIME_DIR/lib/libtensorrt_llm.so"
sha256sum \
  "$RUNTIME_DIR/bin/brpc_inference_server" \
  "$RUNTIME_DIR/bin/kvc_burst_wrapper" \
  "$RUNTIME_DIR/lib/libtensorrt_llm.so"
grep -F -m1 output_token_count < <(strings "$RUNTIME_DIR/bin/brpc_inference_server")
grep -F -m1 datasystem_attribution_ready < <(strings "$RUNTIME_DIR/lib/libtensorrt_llm.so")
echo "F19_ATTRIBUTION_RUNTIME_BUILD_OK trt_image=$TRT_BUILD_IMAGE gateway_image=$GATEWAY_BUILD_IMAGE runtime_dir=$RUNTIME_DIR"
echo "next=run on master: bash scripts/deploy_f19_datasystem_attribution_overlay.sh apply"
