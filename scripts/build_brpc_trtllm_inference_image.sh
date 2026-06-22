#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-docker.io/library/pairec-brpc-inference:k8s-arm64-trtllm-v1}"
TAR_PATH="${2:-/home/zcx/pairec-brpc-inference-k8s-arm64-trtllm-v1.tar}"
BASE_IMAGE="${BASE_IMAGE:-${3:-zcx-pairec-trtllm-brpc-sdk:v1}}"

export ENABLE_TRTLLM_CPP="${ENABLE_TRTLLM_CPP:-ON}"
export TRTLLM_INCLUDE_DIR="${TRTLLM_INCLUDE_DIR:-/TensorRT-LLM/cpp/include}"
export TRTLLM_LIBRARY="${TRTLLM_LIBRARY:-/TensorRT-LLM/cpp/build/tensorrt_llm/libtensorrt_llm.so}"
export TRTLLM_PLUGIN_LIBRARY="${TRTLLM_PLUGIN_LIBRARY:-/TensorRT-LLM/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so}"
export TRTLLM_CUDA_INCLUDE_DIR="${TRTLLM_CUDA_INCLUDE_DIR:-/usr/local/cuda/include}"
export CUDA_DRIVER_LIBRARY="${CUDA_DRIVER_LIBRARY:-/usr/local/cuda/lib64/stubs/libcuda.so}"

echo "Building native brpc TensorRT-LLM inference image"
echo "  image:      $IMAGE"
echo "  tar:        $TAR_PATH"
echo "  base image: $BASE_IMAGE"
echo
echo "The base image must already contain:"
echo "  - Apache brpc headers/libs"
echo "  - TensorRT-LLM C++ headers"
echo "  - libtensorrt_llm.so and its runtime dependencies"
echo "  - CUDA/TensorRT runtime libraries"
echo

BASE_IMAGE="$BASE_IMAGE" bash scripts/build_brpc_inference_image.sh "$IMAGE" "$TAR_PATH"
