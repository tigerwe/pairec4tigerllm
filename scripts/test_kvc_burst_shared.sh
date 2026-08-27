#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT=${OUTPUT:-/tmp/kvc_burst_shared_test}
PROXY_OUTPUT=${PROXY_OUTPUT:-/tmp/kvc_operation_proxy_test}
RANK_PROXY_OUTPUT=${RANK_PROXY_OUTPUT:-/tmp/kvc_operation_proxy_rank_test}
CXX=${CXX:-g++}

"$CXX" \
  -std=c++17 \
  -Wall \
  -Wextra \
  -Werror \
  -pthread \
  "$REPO_ROOT/cpp/kvc_burst/kvc_burst_shared_test.cpp" \
  -o "$OUTPUT"

"$OUTPUT"

"$CXX" \
  -std=c++17 \
  -Wall \
  -Wextra \
  -Werror \
  -pthread \
  "$REPO_ROOT/cpp/kvc_burst/kvc_operation_proxy.cpp" \
  "$REPO_ROOT/cpp/kvc_burst/kvc_operation_proxy_test.cpp" \
  -o "$PROXY_OUTPUT"

"$PROXY_OUTPUT"

"$CXX" \
  -std=c++17 \
  -Wall \
  -Wextra \
  -Werror \
  -pthread \
  -I"$REPO_ROOT/cpp/kvc_burst" \
  "$REPO_ROOT/cpp/kvc_burst/kvc_operation_proxy.cpp" \
  "$REPO_ROOT/cpp/kvc_burst/kvc_operation_proxy_rank_test.cpp" \
  -o "$RANK_PROXY_OUTPUT"
"$RANK_PROXY_OUTPUT"
