#!/usr/bin/env bash
set -euo pipefail

TRTLLM_DIR="${TRTLLM_DIR:-/home/vivwimp/TensorRT-LLM}"
PATCH_FILE="${PATCH_FILE:-trtllm-datasystem-service-discovery.patch}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_PATH="$REPO_ROOT/$PATCH_FILE"

if [ ! -d "$TRTLLM_DIR" ]; then
  echo "ERROR: TensorRT-LLM directory not found: $TRTLLM_DIR" >&2
  exit 1
fi

if [ ! -f "$PATCH_PATH" ]; then
  echo "ERROR: patch file not found: $PATCH_PATH" >&2
  exit 1
fi

if git -C "$TRTLLM_DIR" apply --check "$PATCH_PATH"; then
  git -C "$TRTLLM_DIR" apply "$PATCH_PATH"
  echo "Applied $PATCH_PATH to $TRTLLM_DIR"
  exit 0
fi

if rg -q "DATASYSTEM_ETCD_ADDRESS|makeDataSystemConnectOptions" \
  "$TRTLLM_DIR/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp"; then
  echo "Patch appears to be already applied in $TRTLLM_DIR"
  exit 0
fi

echo "ERROR: patch does not apply cleanly and target does not look patched." >&2
exit 1
