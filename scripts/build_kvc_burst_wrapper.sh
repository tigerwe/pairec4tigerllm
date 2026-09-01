#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_DIR=${SOURCE_DIR:-$REPO_ROOT/cpp/kvc_burst}
BUILD_DIR=${BUILD_DIR:-/tmp/pairec-kvc-burst-build}
INSTALL_DIR=${INSTALL_DIR:-/opt/pairec-kvc-burst}
DATASYSTEM_ROOT=${DATASYSTEM_ROOT:-}
BUILD_JOBS=${BUILD_JOBS:-$(nproc)}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ -z "$DATASYSTEM_ROOT" ]]; then
  DATASYSTEM_ROOT=$(python3 - <<'PY'
import os
import yr.datasystem
print(os.path.dirname(yr.datasystem.__file__))
PY
  )
fi

DATASYSTEM_INCLUDE_DIR=${DATASYSTEM_INCLUDE_DIR:-$DATASYSTEM_ROOT/include}
if [[ -z "${DATASYSTEM_LIBRARY:-}" ]]; then
  for candidate in \
    "$DATASYSTEM_ROOT/lib/libdatasystem.so" \
    "$DATASYSTEM_ROOT/libdatasystem.so"; do
    if [[ -f "$candidate" ]]; then
      DATASYSTEM_LIBRARY=$candidate
      break
    fi
  done
fi
DATASYSTEM_LIBRARY=${DATASYSTEM_LIBRARY:-$DATASYSTEM_ROOT/lib/libdatasystem.so}

[[ -f "$DATASYSTEM_INCLUDE_DIR/datasystem/kv_client.h" ]] \
  || die "DataSystem C++ header not found: $DATASYSTEM_INCLUDE_DIR/datasystem/kv_client.h"
[[ -f "$DATASYSTEM_LIBRARY" ]] || die "DataSystem library not found: $DATASYSTEM_LIBRARY"

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DDATASYSTEM_INCLUDE_DIR="$DATASYSTEM_INCLUDE_DIR" \
  -DDATASYSTEM_LIBRARY="$DATASYSTEM_LIBRARY"
cmake --build "$BUILD_DIR" -j"$BUILD_JOBS"
cmake --install "$BUILD_DIR"

for binary in kvc_burst_wrapper kvc_burst_business_probe kvc_ub_integrity_probe; do
  path="$INSTALL_DIR/bin/$binary"
  [[ -x "$path" ]] || die "built binary is missing: $path"
  if ldd "$path" | grep -q 'not found'; then
    ldd "$path"
    die "$binary has unresolved runtime dependencies"
  fi
done

echo "KVC_BURST_BUILD_OK"
echo "wrapper=$INSTALL_DIR/bin/kvc_burst_wrapper"
echo "business_probe=$INSTALL_DIR/bin/kvc_burst_business_probe"
echo "ub_integrity_probe=$INSTALL_DIR/bin/kvc_ub_integrity_probe"
echo "datasystem_root=$DATASYSTEM_ROOT"
