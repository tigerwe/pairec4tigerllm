#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BRPC_ROOT=${BRPC_ROOT:-/home/zcx/workspace/brpc-827}
INSTALL_DIR=${INSTALL_DIR:-/opt/pairec-brpc-ub-probe}
BUILD_JOBS=${BUILD_JOBS:-32}
EXPECTED_BRPC_COMMIT=${EXPECTED_BRPC_COMMIT:-827db2a9be6a3eac0a1ac3666b4a9cf33b976175}
EXPECTED_UBSCOMM_COMMIT=${EXPECTED_UBSCOMM_COMMIT:-9f80dc9fb5f06ba8b5997064c928b89bda266ffd}
SKIP_VERSION_CHECK=${SKIP_VERSION_CHECK:-0}
STAGE_ONLY=${STAGE_ONLY:-0}
PACKAGE_DIR="$BRPC_ROOT/pairec_ub_probe"

fail()
{
    echo "ERROR: $*" >&2
    exit 1
}

[[ -d "$BRPC_ROOT" ]] || fail "bRPC root does not exist: $BRPC_ROOT"
for source in BUILD.bazel minimal_recommend_server.cpp minimal_recommend_client.cpp payload_integrity.h; do
    [[ -f "$REPO_ROOT/cpp/brpc_ub_probe/$source" ]] || fail "missing source: $source"
done
[[ -f "$REPO_ROOT/proto/recommend.proto" ]] || fail "missing proto/recommend.proto"

if [[ "$SKIP_VERSION_CHECK" != 1 ]]; then
    actual_brpc_commit=$(git -C "$BRPC_ROOT" rev-parse HEAD)
    [[ "$actual_brpc_commit" == "$EXPECTED_BRPC_COMMIT" ]] ||
        fail "bRPC commit mismatch: expected $EXPECTED_BRPC_COMMIT, got $actual_brpc_commit"
    grep -Fq "commit = \"$EXPECTED_UBSCOMM_COMMIT\"" "$BRPC_ROOT/local_deps_ext.bzl" ||
        fail "local_deps_ext.bzl does not pin UBSComm commit $EXPECTED_UBSCOMM_COMMIT"
fi

mkdir -p "$PACKAGE_DIR"
install -m 0644 "$REPO_ROOT/cpp/brpc_ub_probe/BUILD.bazel" "$PACKAGE_DIR/BUILD.bazel"
install -m 0644 "$REPO_ROOT/cpp/brpc_ub_probe/minimal_recommend_server.cpp" \
    "$PACKAGE_DIR/minimal_recommend_server.cpp"
install -m 0644 "$REPO_ROOT/cpp/brpc_ub_probe/minimal_recommend_client.cpp" \
    "$PACKAGE_DIR/minimal_recommend_client.cpp"
install -m 0644 "$REPO_ROOT/cpp/brpc_ub_probe/payload_integrity.h" "$PACKAGE_DIR/payload_integrity.h"
install -m 0644 "$REPO_ROOT/proto/recommend.proto" "$PACKAGE_DIR/recommend.proto"

echo "BRPC_UB_RECOMMEND_STAGE_OK package=$PACKAGE_DIR"
if [[ "$STAGE_ONLY" == 1 ]]; then
    exit 0
fi

(
    cd "$BRPC_ROOT"
    bazel build -c opt \
        --jobs="$BUILD_JOBS" \
        --ignore_dev_dependency \
        --check_direct_dependencies=off \
        --define brpc_with_urma=true \
        //pairec_ub_probe:minimal_recommend_server \
        //pairec_ub_probe:minimal_recommend_client
)

mkdir -p "$INSTALL_DIR/bin"
install -m 0755 "$BRPC_ROOT/bazel-bin/pairec_ub_probe/minimal_recommend_server" "$INSTALL_DIR/bin/"
install -m 0755 "$BRPC_ROOT/bazel-bin/pairec_ub_probe/minimal_recommend_client" "$INSTALL_DIR/bin/"

for binary in minimal_recommend_server minimal_recommend_client; do
    if ldd "$INSTALL_DIR/bin/$binary" | grep -q 'not found'; then
        ldd "$INSTALL_DIR/bin/$binary" >&2
        fail "$binary has unresolved shared libraries"
    fi
    nm -C "$INSTALL_DIR/bin/$binary" 2>/dev/null | grep -F '_GLOBAL__sub_I_ubsocket' >/dev/null ||
        fail "$binary does not contain linked UBSocket objects"
done

echo "BRPC_UB_RECOMMEND_BUILD_OK install_dir=$INSTALL_DIR"
