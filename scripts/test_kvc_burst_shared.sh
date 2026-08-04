#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT=${OUTPUT:-/tmp/kvc_burst_shared_test}
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
