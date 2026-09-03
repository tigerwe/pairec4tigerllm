#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
ACTION=${ACTION:-client}
BRPC_ROOT=${BRPC_ROOT:-/home/zcx/workspace/brpc-827}
INSTALL_DIR=${INSTALL_DIR:-/opt/pairec-brpc-ub-echo-baseline}
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
STRICT_BTHREAD_KEY_CHECK=${STRICT_BTHREAD_KEY_CHECK:-0}

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
    (
        cd "$BRPC_ROOT"
        bazel build -c opt //example:echo_c++_server --define brpc_with_urma=true
        bazel build -c opt //example:echo_c++_client --define brpc_with_urma=true
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
    echo "transport_mode=functional_ub backup_link=default degrade=default"
    "$server_bin" \
        --port="$PORT" \
        --ubsocket_enable=true \
        --ubsocket_use_ub=true \
        2>&1 | tee "$log_file"
}

run_client()
{
    local client_bin=${CLIENT_BIN:-$INSTALL_DIR/bin/echo_c++_client}
    [[ -x "$client_bin" ]] || fail "client binary is not executable: $client_bin"
    [[ "$RUN_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "RUN_SECONDS must be a positive integer"
    [[ "$STRICT_BTHREAD_KEY_CHECK" == 0 || "$STRICT_BTHREAD_KEY_CHECK" == 1 ]] ||
        fail "STRICT_BTHREAD_KEY_CHECK must be 0 or 1"
    require_host_urma
    mkdir -p "$LOG_DIR"
    local log_file="$LOG_DIR/echo-client-$(date +%Y%m%d-%H%M%S).log"
    echo "client_log=$log_file"
    echo "transport_mode=functional_ub backup_link=default degrade=default"

    set +e
    timeout --signal=INT --kill-after=3s "${RUN_SECONDS}s" \
        "$client_bin" \
        --server="$SERVER" \
        --timeout_ms=15000 \
        --max_retry=0 \
        --ubsocket_enable=true \
        --ubsocket_use_ub=true \
        2>&1 | tee "$log_file"
    client_status=${PIPESTATUS[0]}
    set -e

    if [[ "$client_status" -eq 139 ]] || grep -Fq 'Segmentation fault' "$log_file"; then
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

    bthread_key_warning_count=$(grep -Fc 'invalid bthread_key_t' "$log_file" || true)
    if [[ "$bthread_key_warning_count" -gt 0 ]]; then
        echo "WARNING: observed $bthread_key_warning_count invalid bthread key diagnostics; accepting completed Echo responses while stability remains unresolved" >&2
        if [[ "$STRICT_BTHREAD_KEY_CHECK" == 1 ]]; then
            fail "strict bthread key check rejected the functional run"
        fi
    fi

    response_count=$(grep -Fc 'Received response from' "$log_file")
    echo "BRPC_UB_ECHO_BASELINE_PASS responses=$response_count server=$SERVER bthread_key_warnings=$bthread_key_warning_count log=$log_file"
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
