#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
REPO_ROOT=$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)
ACTION=${ACTION:-client}
BRPC_ROOT=${BRPC_ROOT:-/home/zcx/workspace/brpc-827}
INSTALL_DIR=${INSTALL_DIR:-/opt/pairec-brpc-ub-echo-baseline}
BUILD_JOBS=${BUILD_JOBS:-32}
BAZEL_OUTPUT_BASE=${BAZEL_OUTPUT_BASE:-/root/.cache/bazel/_bazel_root/0947eeff3cdbdab635f34a3b3ff5f6d1}
LOCAL_BCR_REGISTRY=${LOCAL_BCR_REGISTRY:-/home/zcx/bazel-local-registry/bcr}
LOCAL_SECRET_REGISTRY=${LOCAL_SECRET_REGISTRY:-/home/zcx/bazel-local-registry/secretflow}
GCC_TOOLSET_LIB_DIR=${GCC_TOOLSET_LIB_DIR:-/opt/openEuler/gcc-toolset-14/root/usr/lib64}
EXPECTED_BRPC_COMMIT=${EXPECTED_BRPC_COMMIT:-827db2a9be6a3eac0a1ac3666b4a9cf33b976175}
EXPECTED_UBSCOMM_COMMIT=${EXPECTED_UBSCOMM_COMMIT:-9f80dc9fb5f06ba8b5997064c928b89bda266ffd}
SERVER=${SERVER:-141.62.33.105:18200}
PORT=${PORT:-18200}
RUN_SECONDS=${RUN_SECONDS:-8}
LOG_DIR=${LOG_DIR:-/root/brpc-ub-echo-baseline}
NOFILE_LIMIT=${NOFILE_LIMIT:-1048576}
URMA_RUNTIME_LIB_DIR=${URMA_RUNTIME_LIB_DIR:-/usr/lib64}
URMA_PROVIDER_LIB_DIR=${URMA_PROVIDER_LIB_DIR:-/usr/lib64/urma}
URMA_RUNTIME_LD_LIBRARY_PATH=${URMA_RUNTIME_LD_LIBRARY_PATH:-$URMA_RUNTIME_LIB_DIR:$URMA_PROVIDER_LIB_DIR}

fail()
{
    echo "ERROR: $*" >&2
    exit 1
}

require_host_urma()
{
    [[ -r "$URMA_RUNTIME_LIB_DIR/liburma.so" ]] ||
        fail "host URMA runtime is missing: $URMA_RUNTIME_LIB_DIR/liburma.so"
    [[ -d "$URMA_PROVIDER_LIB_DIR" ]] ||
        fail "host URMA provider directory is missing: $URMA_PROVIDER_LIB_DIR"
    ulimit -n "$NOFILE_LIMIT"
    export LD_LIBRARY_PATH="$URMA_RUNTIME_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "urma_runtime_library_path=$URMA_RUNTIME_LD_LIBRARY_PATH"
}

build_echo()
{
    [[ -d "$BRPC_ROOT" ]] || fail "bRPC root does not exist: $BRPC_ROOT"
    [[ -f "$BRPC_ROOT/example/BUILD.bazel" ]] || fail "missing $BRPC_ROOT/example/BUILD.bazel"
    actual_brpc_commit=$(git -C "$BRPC_ROOT" rev-parse HEAD)
    [[ "$actual_brpc_commit" == "$EXPECTED_BRPC_COMMIT" ]] ||
        fail "bRPC commit mismatch: expected $EXPECTED_BRPC_COMMIT, got $actual_brpc_commit"
    grep -Fq "commit = \"$EXPECTED_UBSCOMM_COMMIT\"" "$BRPC_ROOT/local_deps_ext.bzl" ||
        fail "local_deps_ext.bzl does not pin UBSComm commit $EXPECTED_UBSCOMM_COMMIT"
    [[ -r "$GCC_TOOLSET_LIB_DIR/libbfd-2.42.so" ]] ||
        fail "missing GCC toolset runtime: $GCC_TOOLSET_LIB_DIR/libbfd-2.42.so"

    BRPC_ROOT="$BRPC_ROOT" \
    LOCAL_BCR_REGISTRY="$LOCAL_BCR_REGISTRY" \
    LOCAL_SECRET_REGISTRY="$LOCAL_SECRET_REGISTRY" \
    BAZEL_OUTPUT_BASE="$BAZEL_OUTPUT_BASE" \
    RUN_BUILD=0 \
    RUN_MODULE_GRAPH=0 \
    bash "$REPO_ROOT/scripts/build_brpc_ub_recommend_probe_local_registry.sh"

    (
        cd "$BRPC_ROOT"
        bazel --ignore_all_rc_files --output_base="$BAZEL_OUTPUT_BASE" build -c opt \
            --jobs="$BUILD_JOBS" \
            --ignore_dev_dependency \
            --check_direct_dependencies=off \
            --define brpc_with_urma=true \
            --registry="file://$LOCAL_SECRET_REGISTRY" \
            --registry="file://$LOCAL_BCR_REGISTRY" \
            --lockfile_mode=update \
            --extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_make_toolchain \
            --extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_pkgconfig_toolchain \
            --action_env="LD_LIBRARY_PATH=$GCC_TOOLSET_LIB_DIR" \
            //example:echo_c++_server \
            //example:echo_c++_client
    )

    mkdir -p "$INSTALL_DIR/bin"
    install -m 0755 "$BRPC_ROOT/bazel-bin/example/echo_c++_server" "$INSTALL_DIR/bin/"
    install -m 0755 "$BRPC_ROOT/bazel-bin/example/echo_c++_client" "$INSTALL_DIR/bin/"
    install -m 0755 "$SCRIPT_PATH" "$INSTALL_DIR/bin/verify_brpc_ub_echo_baseline.sh"

    for binary in echo_c++_server echo_c++_client; do
        if ldd "$INSTALL_DIR/bin/$binary" | grep -q 'not found'; then
            ldd "$INSTALL_DIR/bin/$binary" >&2
            fail "$binary has unresolved shared libraries"
        fi
    done

    echo "BRPC_UB_ECHO_BUILD_OK install_dir=$INSTALL_DIR"
    echo "node1_copy=$INSTALL_DIR/bin/echo_c++_client,$INSTALL_DIR/bin/verify_brpc_ub_echo_baseline.sh"
}

run_server()
{
    local server_bin=${SERVER_BIN:-$INSTALL_DIR/bin/echo_c++_server}
    [[ -x "$server_bin" ]] || fail "server binary is not executable: $server_bin"
    require_host_urma
    mkdir -p "$LOG_DIR"
    local log_file="$LOG_DIR/echo-server-$(date +%Y%m%d-%H%M%S).log"
    echo "server_log=$log_file"
    "$server_bin" \
        --port="$PORT" \
        --ubsocket_enable=true \
        --ubsocket_use_ub=true \
        --ubsocket_backup_link_enable=false \
        --ubsocket_degrade_enable=false \
        2>&1 | tee "$log_file"
}

run_client()
{
    local client_bin=${CLIENT_BIN:-$INSTALL_DIR/bin/echo_c++_client}
    [[ -x "$client_bin" ]] || fail "client binary is not executable: $client_bin"
    [[ "$RUN_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "RUN_SECONDS must be a positive integer"
    require_host_urma
    mkdir -p "$LOG_DIR"
    local log_file="$LOG_DIR/echo-client-$(date +%Y%m%d-%H%M%S).log"
    echo "client_log=$log_file"

    set +e
    timeout --signal=INT --kill-after=3s "${RUN_SECONDS}s" \
        "$client_bin" \
        --server="$SERVER" \
        --timeout_ms=15000 \
        --max_retry=0 \
        --ubsocket_enable=true \
        --ubsocket_use_ub=true \
        --ubsocket_backup_link_enable=false \
        --ubsocket_degrade_enable=false \
        2>&1 | tee "$log_file"
    client_status=${PIPESTATUS[0]}
    set -e

    if grep -Eq 'invalid bthread_key|Check failed: false|Segmentation fault' "$log_file"; then
        echo "BRPC_UB_ECHO_BASELINE_CRASH log=$log_file" >&2
        exit 1
    fi
    if [[ "$client_status" -ne 0 && "$client_status" -ne 124 && "$client_status" -ne 130 ]]; then
        fail "echo client exited unexpectedly: status=$client_status log=$log_file"
    fi
    grep -Fq 'Received response from' "$log_file" ||
        fail "echo response evidence is missing: $log_file"
    grep -Fq 'bind jetty success' "$log_file" ||
        fail "UB bind evidence is missing: $log_file"

    response_count=$(grep -Fc 'Received response from' "$log_file")
    echo "BRPC_UB_ECHO_BASELINE_PASS responses=$response_count server=$SERVER log=$log_file"
}

case "$ACTION" in
    build)
        build_echo
        ;;
    server)
        run_server
        ;;
    client)
        run_client
        ;;
    *)
        fail "ACTION must be build, server, or client"
        ;;
esac
