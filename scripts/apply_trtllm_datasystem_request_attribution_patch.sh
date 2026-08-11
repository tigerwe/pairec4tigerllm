#!/usr/bin/env bash
set -euo pipefail

TRTLLM_DIR="${TRTLLM_DIR:-${1:-/TensorRT-LLM}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHER="$SCRIPT_DIR/patch_trtllm_datasystem_request_attribution.py"

[[ -d "$TRTLLM_DIR/cpp" ]] || {
  echo "ERROR: TensorRT-LLM source tree not found: $TRTLLM_DIR" >&2
  exit 1
}
[[ -f "$PATCHER" ]] || {
  echo "ERROR: patcher not found: $PATCHER" >&2
  exit 1
}

python3 "$PATCHER" "$TRTLLM_DIR"

for path in \
  "$TRTLLM_DIR/cpp/include/tensorrt_llm/batch_manager/datasystemRequestTracker.h" \
  "$TRTLLM_DIR/cpp/tensorrt_llm/batch_manager/datasystemRequestTracker.cpp"; do
  grep -q PAIREC_DATASYSTEM_REQUEST_ATTRIBUTION_V1 "$path"
done
grep -q datasystemRequestTracker.cpp \
  "$TRTLLM_DIR/cpp/tensorrt_llm/batch_manager/CMakeLists.txt"
grep -q beginDataSystemOperation \
  "$TRTLLM_DIR/cpp/tensorrt_llm/batch_manager/kvCacheTransferManager.cpp"
grep -q PAIREC_DATASYSTEM_REQUEST_ATTRIBUTION_ZERO_INTRUSION_DISABLED_V2 \
  "$TRTLLM_DIR/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp"
grep -q PAIREC_TRT_EXECUTOR_PHASE_TIMING_V3 \
  "$TRTLLM_DIR/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp"
grep -q phase_timing_complete \
  "$TRTLLM_DIR/cpp/tensorrt_llm/batch_manager/datasystemRequestTracker.cpp"

echo "TRTLLM_DATASYSTEM_REQUEST_ATTRIBUTION_APPLY_OK dir=$TRTLLM_DIR"
echo "next=cmake --build $TRTLLM_DIR/cpp/build -j\$(nproc)"
