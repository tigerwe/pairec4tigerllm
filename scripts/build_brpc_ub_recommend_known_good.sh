#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BRPC_ROOT=${BRPC_ROOT:-/home/zcx/workspace/brpc}
INSTALL_DIR=${INSTALL_DIR:-/opt/pairec-brpc-ub-recommend-known-good}
BAZEL_OUTPUT_BASE=${BAZEL_OUTPUT_BASE:-/root/.cache/bazel/_bazel_root/02df77b0294ccdcf08a6a8a39050de3a}
BUILD_JOBS=${BUILD_JOBS:-32}
BAZEL_LOCKFILE_MODE=${BAZEL_LOCKFILE_MODE:-off}
BAZEL_DISABLE_DOWNLOAD=${BAZEL_DISABLE_DOWNLOAD:-1}
LOCAL_BCR_REGISTRY=${LOCAL_BCR_REGISTRY:-/home/zcx/bazel-local-registry/bcr}
LOCAL_SECRET_REGISTRY=${LOCAL_SECRET_REGISTRY:-/home/zcx/bazel-local-registry/secretflow}
OPENSSL_VERSION=${OPENSSL_VERSION:-3.3.2.bcr.1}
EXPECTED_BRPC_COMMIT=${EXPECTED_BRPC_COMMIT:-827db2a9be6a3eac0a1ac3666b4a9cf33b976175}
EXPECTED_UBSCOMM_COMMIT=${EXPECTED_UBSCOMM_COMMIT:-9f80dc9fb5f06ba8b5997064c928b89bda266ffd}
EXPECTED_LOCK_SHA256=${EXPECTED_LOCK_SHA256:-6dd4be421ed7552b2070a9991ca006caa940cb47859968c566fcb85bded641f0}
PACKAGE_DIR="$BRPC_ROOT/pairec_ub_probe"

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "$BRPC_ROOT" ]] || fail "missing BRPC_ROOT: $BRPC_ROOT"
[[ -f "$BRPC_ROOT/MODULE.bazel" ]] || fail "missing MODULE.bazel"
[[ -f "$BRPC_ROOT/.bazelrc" ]] || fail "missing .bazelrc"
[[ -f "$BRPC_ROOT/local_deps_ext.bzl" ]] || fail "missing local_deps_ext.bzl"
[[ -f "$LOCAL_BCR_REGISTRY/bazel_registry.json" ]] || fail "invalid local BCR registry"
for registry_file in \
    bazel_registry.json \
    modules/leveldb/1.23/MODULE.bazel \
    modules/leveldb/1.23/source.json \
    "modules/openssl/$OPENSSL_VERSION/MODULE.bazel" \
    "modules/openssl/$OPENSSL_VERSION/source.json"; do
    [[ -f "$LOCAL_SECRET_REGISTRY/$registry_file" ]] ||
        fail "missing local SecretFlow registry file: $registry_file"
done

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

for source in \
    BUILD.bazel \
    minimal_recommend_server.cpp \
    minimal_recommend_client.cpp \
    payload_integrity.h \
    ubsocket_trace_key_workaround.h; do
    [[ -f "$REPO_ROOT/cpp/brpc_ub_probe/$source" ]] || fail "missing probe source: $source"
done
[[ -f "$REPO_ROOT/proto/recommend.proto" ]] || fail "missing recommend.proto"

mkdir -p "$PACKAGE_DIR"
install -m 0644 "$REPO_ROOT/cpp/brpc_ub_probe/BUILD.bazel" "$PACKAGE_DIR/BUILD.bazel"
install -m 0644 "$REPO_ROOT/cpp/brpc_ub_probe/minimal_recommend_server.cpp" "$PACKAGE_DIR/"
install -m 0644 "$REPO_ROOT/cpp/brpc_ub_probe/minimal_recommend_client.cpp" "$PACKAGE_DIR/"
install -m 0644 "$REPO_ROOT/cpp/brpc_ub_probe/payload_integrity.h" "$PACKAGE_DIR/"
install -m 0644 "$REPO_ROOT/cpp/brpc_ub_probe/ubsocket_trace_key_workaround.h" "$PACKAGE_DIR/"
install -m 0644 "$REPO_ROOT/proto/recommend.proto" "$PACKAGE_DIR/"

echo "BRPC_KNOWN_GOOD_BASELINE_OK root=$BRPC_ROOT commit=$actual_brpc_commit ubscomm=$EXPECTED_UBSCOMM_COMMIT"
echo "BRPC_KNOWN_GOOD_BORINGSSL_OK lock_sha256=$actual_lock_sha256"

git -C "$BRPC_ROOT" diff --quiet -- MODULE.bazel ||
    fail "MODULE.bazel has local changes; restore or preserve them before this build"

module_file="$BRPC_ROOT/MODULE.bazel"
lock_file="$BRPC_ROOT/MODULE.bazel.lock"
module_backup=$(mktemp /tmp/brpc-ub-module.XXXXXX)
lock_backup=$(mktemp /tmp/brpc-ub-lock.XXXXXX)
cp -p "$module_file" "$module_backup"
cp -p "$lock_file" "$lock_backup"

restore_build_metadata()
{
    cp -p "$module_backup" "$module_file"
    cp -p "$lock_backup" "$lock_file"
    rm -f "$module_backup" "$lock_backup"
}
trap restore_build_metadata EXIT INT TERM

secret_registry_uri="file://${LOCAL_SECRET_REGISTRY%/}"
bcr_registry_uri="file://${LOCAL_BCR_REGISTRY%/}"
sed -i \
    -e "s#https://raw.githubusercontent.com/secretflow/bazel-registry/main#$secret_registry_uri#g" \
    -e "s#bazel_dep(name = 'openssl', version = '3.3.2')#bazel_dep(name = 'openssl', version = '$OPENSSL_VERSION')#" \
    "$module_file"

[[ $(grep -Fc "registry = \"$secret_registry_uri\"," "$module_file") == 2 ]] ||
    fail "failed to map both module overrides to the local SecretFlow registry"
grep -Fq "bazel_dep(name = 'openssl', version = '$OPENSSL_VERSION')" "$module_file" ||
    fail "failed to align the temporary OpenSSL module version"

bazel_repository_args=(
    "--lockfile_mode=$BAZEL_LOCKFILE_MODE"
    "--registry=$secret_registry_uri"
    "--registry=$bcr_registry_uri"
)
if [[ "$BAZEL_DISABLE_DOWNLOAD" == 1 ]]; then
    bazel_repository_args+=("--repository_disable_download")
fi

bazel_startup_args=(
    "--ignore_all_rc_files"
    "--output_base=$BAZEL_OUTPUT_BASE"
)

echo "BRPC_KNOWN_GOOD_RC_ISOLATED boring_ssl=1 urma=1"

(
    cd "$BRPC_ROOT"
    bazel "${bazel_startup_args[@]}" build -c opt \
        --jobs="$BUILD_JOBS" \
        --ignore_dev_dependency \
        --check_direct_dependencies=off \
        --define=BRPC_WITH_BORINGSSL=true \
        --define brpc_with_urma=true \
        "${bazel_repository_args[@]}" \
        //pairec_ub_probe:minimal_recommend_server \
        //pairec_ub_probe:minimal_recommend_client \
        //example:echo_c++_server \
        //example:echo_c++_client
)

restore_build_metadata
trap - EXIT INT TERM

post_build_lock_sha256=$(sha256sum "$lock_file" | awk '{print $1}')
post_build_module_status=$(git -C "$BRPC_ROOT" status --short -- MODULE.bazel)
[[ "$post_build_lock_sha256" == "$EXPECTED_LOCK_SHA256" ]] ||
    fail "baseline lockfile was not restored: expected $EXPECTED_LOCK_SHA256, got $post_build_lock_sha256"
[[ -z "$post_build_module_status" ]] || fail "baseline MODULE.bazel was not restored"
echo "BRPC_KNOWN_GOOD_METADATA_RESTORED lock_sha256=$post_build_lock_sha256"

mkdir -p "$INSTALL_DIR/bin"
install -m 0755 "$BRPC_ROOT/bazel-bin/pairec_ub_probe/minimal_recommend_server" "$INSTALL_DIR/bin/"
install -m 0755 "$BRPC_ROOT/bazel-bin/pairec_ub_probe/minimal_recommend_client" "$INSTALL_DIR/bin/"
install -m 0755 "$BRPC_ROOT/bazel-bin/example/echo_c++_server" "$INSTALL_DIR/bin/"
install -m 0755 "$BRPC_ROOT/bazel-bin/example/echo_c++_client" "$INSTALL_DIR/bin/"

for binary in minimal_recommend_server minimal_recommend_client echo_c++_server echo_c++_client; do
    path="$INSTALL_DIR/bin/$binary"
    if ldd "$path" | grep -q 'not found'; then
        ldd "$path" >&2
        fail "$binary has unresolved shared libraries"
    fi
    nm -C "$path" 2>/dev/null | grep -F '_GLOBAL__sub_I_ubsocket' >/dev/null ||
        fail "$binary does not contain linked UBSocket objects"
    if [[ "$binary" == minimal_recommend_* ]]; then
        strings "$path" | grep -F 'MINIMAL_RECOMMEND_UB_TRACE_KEYS_READY' >/dev/null ||
            fail "$binary does not contain the process-local UBSocket trace-key workaround"
    fi
    if strings "$path" | grep -Eq 'boringssl|BoringSSL|local_deps.*boring'; then
        boring_ssl_elf_evidence=present
    else
        boring_ssl_elf_evidence=not_retained
    fi
    echo "BRPC_KNOWN_GOOD_BINARY_OK binary=$binary boring_ssl_config=explicit_define boring_ssl_elf_string=$boring_ssl_elf_evidence"
done

echo "BRPC_KNOWN_GOOD_ECHO_BUILD_OK install_dir=$INSTALL_DIR"
echo "BRPC_KNOWN_GOOD_RECOMMEND_BUILD_OK install_dir=$INSTALL_DIR"
