#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BRPC_ROOT=${BRPC_ROOT:-/home/zcx/workspace/brpc-827}
INSTALL_DIR=${INSTALL_DIR:-/opt/pairec-brpc-ub-probe}
BUILD_JOBS=${BUILD_JOBS:-32}
BAZEL_OUTPUT_BASE=${BAZEL_OUTPUT_BASE:-}
BAZEL_LOCKFILE_MODE=${BAZEL_LOCKFILE_MODE:-}
BAZEL_IGNORE_ALL_RC_FILES=${BAZEL_IGNORE_ALL_RC_FILES:-0}
BAZEL_USE_PREINSTALLED_MAKE=${BAZEL_USE_PREINSTALLED_MAKE:-0}
BAZEL_ACTION_LD_LIBRARY_PATH=${BAZEL_ACTION_LD_LIBRARY_PATH:-}
LOCAL_BCR_REGISTRY=${LOCAL_BCR_REGISTRY:-}
LOCAL_SECRET_REGISTRY=${LOCAL_SECRET_REGISTRY:-}
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

registry_uri()
{
    local path=$1
    [[ "$path" == /* ]] || fail "registry path must be absolute: $path"
    printf 'file://%s' "${path%/}"
}

[[ -d "$BRPC_ROOT" ]] || fail "bRPC root does not exist: $BRPC_ROOT"
for source in \
    BUILD.bazel \
    minimal_recommend_server.cpp \
    minimal_recommend_client.cpp \
    payload_integrity.h \
    ubsocket_trace_key_workaround.h; do
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
install -m 0644 "$REPO_ROOT/cpp/brpc_ub_probe/ubsocket_trace_key_workaround.h" \
    "$PACKAGE_DIR/ubsocket_trace_key_workaround.h"
install -m 0644 "$REPO_ROOT/proto/recommend.proto" "$PACKAGE_DIR/recommend.proto"

echo "BRPC_UB_RECOMMEND_STAGE_OK package=$PACKAGE_DIR"
if [[ "$STAGE_ONLY" == 1 ]]; then
    exit 0
fi

bazel_startup_args=()
if [[ "$BAZEL_IGNORE_ALL_RC_FILES" == 1 ]]; then
    bazel_startup_args+=("--ignore_all_rc_files")
fi
if [[ -n "$BAZEL_OUTPUT_BASE" ]]; then
    bazel_startup_args+=("--output_base=$BAZEL_OUTPUT_BASE")
fi

bazel_repository_args=()
if [[ -n "$LOCAL_SECRET_REGISTRY" ]]; then
    [[ -f "$LOCAL_SECRET_REGISTRY/bazel_registry.json" ]] ||
        fail "invalid SecretFlow registry: $LOCAL_SECRET_REGISTRY"
    bazel_repository_args+=("--registry=$(registry_uri "$LOCAL_SECRET_REGISTRY")")
fi
if [[ -n "$LOCAL_BCR_REGISTRY" ]]; then
    [[ -f "$LOCAL_BCR_REGISTRY/bazel_registry.json" ]] ||
        fail "invalid BCR registry: $LOCAL_BCR_REGISTRY"
    bazel_repository_args+=("--registry=$(registry_uri "$LOCAL_BCR_REGISTRY")")
fi
if [[ -n "$BAZEL_LOCKFILE_MODE" ]]; then
    bazel_repository_args+=("--lockfile_mode=$BAZEL_LOCKFILE_MODE")
fi
if [[ "$BAZEL_USE_PREINSTALLED_MAKE" == 1 ]]; then
    command -v make >/dev/null 2>&1 || fail "preinstalled make was requested but not found in PATH"
    command -v pkg-config >/dev/null 2>&1 || fail "preinstalled pkg-config was requested but not found in PATH"
    bazel_repository_args+=(
        "--extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_make_toolchain"
        "--extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_pkgconfig_toolchain"
    )
fi
if [[ -n "$BAZEL_ACTION_LD_LIBRARY_PATH" ]]; then
    bazel_repository_args+=("--action_env=LD_LIBRARY_PATH=$BAZEL_ACTION_LD_LIBRARY_PATH")
fi

(
    cd "$BRPC_ROOT"
    bazel "${bazel_startup_args[@]}" build -c opt \
        --jobs="$BUILD_JOBS" \
        --ignore_dev_dependency \
        --check_direct_dependencies=off \
        --define brpc_with_urma=true \
        "${bazel_repository_args[@]}" \
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
    strings "$INSTALL_DIR/bin/$binary" | grep -F 'MINIMAL_RECOMMEND_UB_TRACE_KEYS_READY' >/dev/null ||
        fail "$binary does not contain the process-local UBSocket trace-key workaround"
done

echo "BRPC_UB_RECOMMEND_BUILD_OK install_dir=$INSTALL_DIR"
