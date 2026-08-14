#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

APPLY_PATCH=1 \
APPLY_KVC_BURST_PROXY=1 \
BUILD_TRTLLM=${BUILD_TRTLLM:-1} \
  bash "$SCRIPT_DIR/build_f19_attribution_runtime_worker1.sh"

echo "F14_KVC_BURST_RUNTIME_BUILD_OK"
echo "next=run on master: CONCURRENCY=1 bash scripts/deploy_f14_kvc_burst_overlay.sh apply"
