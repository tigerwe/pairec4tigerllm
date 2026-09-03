#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BRPC_ROOT=${BRPC_ROOT:-/home/zcx/workspace/brpc}
INSTALL_DIR=${INSTALL_DIR:-/opt/pairec-brpc-ub-recommend-known-good}
BAZEL_OUTPUT_BASE=${BAZEL_OUTPUT_BASE:-/root/.cache/bazel/_bazel_root/02df77b0294ccdcf08a6a8a39050de3a}
BUILD_JOBS=${BUILD_JOBS:-32}
EXPECTED_BRPC_COMMIT=${EXPECTED_BRPC_COMMIT:-3431fa24bace7ff0ee34c8717422a1905221ec02}
EXPECTED_UBSCOMM_COMMIT=${EXPECTED_UBSCOMM_COMMIT:-9f80dc9fb5f06ba8b5997064c928b89bda266ffd}
EXPECTED_LOCK_SHA256=${EXPECTED_LOCK_SHA256:-6dd4be421ed7552b2070a9991ca006caa940cb47859968c566fcb85bded641f0}
PACKAGE_DIR="$BRPC_ROOT/pairec_ub_probe"

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "$BRPC_ROOT" ]] || fail "missing BRPC_ROOT: $BRPC_ROOT"
[[ -f "$BRPC_ROOT/MODULE.bazel" ]] || fail "missing MODULE.bazel"
[[ -f "$BRPC_ROOT/.bazelrc" ]] || fail "missing .bazelrc"
[[ -f "$BRPC_ROOT/local_deps_ext.bzl" ]] || fail "missing local_deps_ext.bzl"

actual_brpc_commit=$(git -C "$BRPC_ROOT" rev-parse HEAD)
[[ "$actual_brpc_commit" == "$EXPECTED_BRPC_COMMIT" ]] ||
    fail "bRPC commit mismatch: expected $EXPECTED_BRPC_COMMIT, got $actual_brpc_commit"
grep -Fq "commit = \"$EXPECTED_UBSCOMM_COMMIT\"" "$BRPC_ROOT/local_deps_ext.bzl" ||
    fail "UBSComm commit is not $EXPECTED_UBSCOMM_COMMIT"
grep -Fq 'build --define=BRPC_WITH_BORINGSSL=true' "$BRPC_ROOT/.bazelrc" ||
    fail ".bazelrc does not enable BRPC_WITH_BORINGSSL"

[[ -f "$BRPC_ROOT/MODULE.bazel.lock" ]] || fail "missing baseline MODULE.bazel.lock"
actual_lock_sha256=$(sha256sum "$BRPC_ROOT/MODULE.bazel.lock" | awk '{print $1}')
[[ "$actual_lock_sha256" == "$EXPECTED_LOCK_SHA256" ]] ||
    fail "lockfile changed: expected $EXPECTED_LOCK_SHA256, got $actual_lock_sha256"

for source in BUILD.bazel minimal_recommend_server.cpp minimal_recommend_client.cpp payload_integrity.h; do
    [[ -f "$REPO_ROOT/cpp/brpc_ub_probe/$source" ]] || fail "missing probe source: $source"
done
[[ -f "$REPO_ROOT/proto/recommend.proto" ]] || fail "missing recommend.proto"

mkdir -p "$PACKAGE_DIR"
install -m 0644 "$REPO_ROOT/cpp/brpc_ub_probe/BUILD.bazel" "$PACKAGE_DIR/BUILD.bazel"
install -m 0644 "$REPO_ROOT/cpp/brpc_ub_probe/minimal_recommend_server.cpp" "$PACKAGE_DIR/"
install -m 0644 "$REPO_ROOT/cpp/brpc_ub_probe/minimal_recommend_client.cpp" "$PACKAGE_DIR/"
install -m 0644 "$REPO_ROOT/cpp/brpc_ub_probe/payload_integrity.h" "$PACKAGE_DIR/"
install -m 0644 "$REPO_ROOT/proto/recommend.proto" "$PACKAGE_DIR/"

echo "BRPC_KNOWN_GOOD_BASELINE_OK root=$BRPC_ROOT commit=$actual_brpc_commit ubscomm=$EXPECTED_UBSCOMM_COMMIT"
echo "BRPC_KNOWN_GOOD_BORINGSSL_OK lock_sha256=$actual_lock_sha256"

(
    cd "$BRPC_ROOT"
    bazel --output_base="$BAZEL_OUTPUT_BASE" build -c opt \
        --jobs="$BUILD_JOBS" \
        --lockfile_mode=error \
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
    path="$INSTALL_DIR/bin/$binary"
    if ldd "$path" | grep -q 'not found'; then
        ldd "$path" >&2
        fail "$binary has unresolved shared libraries"
    fi
    nm -C "$path" 2>/dev/null | grep -F '_GLOBAL__sub_I_ubsocket' >/dev/null ||
        fail "$binary does not contain linked UBSocket objects"
    strings "$path" | grep -Eq 'boringssl|BoringSSL|local_deps.*boring' ||
        fail "$binary does not contain BoringSSL build evidence"
done

echo "BRPC_KNOWN_GOOD_RECOMMEND_BUILD_OK install_dir=$INSTALL_DIR"
