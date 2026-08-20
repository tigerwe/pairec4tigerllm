#!/usr/bin/env bash
set -euo pipefail

TRTLLM_DIR="${TRTLLM_DIR:-${1:-/TensorRT-LLM}}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"$SCRIPT_DIR/apply_trtllm_datasystem_request_attribution_patch.sh" "$TRTLLM_DIR"
python3 "$SCRIPT_DIR/patch_trtllm_kvc_burst_proxy.py" "$TRTLLM_DIR"

grep -q PAIREC_KVC_BURST_PROXY_V4 \
  "$TRTLLM_DIR/cpp/include/tensorrt_llm/batch_manager/kvcOperationProxy.h"
grep -q kvcOperationProxy.cpp \
  "$TRTLLM_DIR/cpp/tensorrt_llm/batch_manager/CMakeLists.txt"
grep -q kvcBurstParallelGetToken \
  "$TRTLLM_DIR/cpp/tensorrt_llm/batch_manager/kvCacheTransferManager.cpp"

echo "TRTLLM_KVC_BURST_PROXY_APPLY_OK dir=$TRTLLM_DIR"
echo "next=cmake --build $TRTLLM_DIR/cpp/build --target tensorrt_llm -j\$(nproc)"
