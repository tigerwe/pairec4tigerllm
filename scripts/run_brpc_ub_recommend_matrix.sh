#!/usr/bin/env bash
set -euo pipefail

CLIENT_BIN=${CLIENT_BIN:-/opt/pairec-brpc-ub-probe/bin/minimal_recommend_client}
SERVER=${SERVER:-141.62.33.105:18100}
METHOD=${METHOD:-recommend}
PAYLOAD_SIZES=${PAYLOAD_SIZES:-0,1,4096,4097,65536,1048576,3670016}
REQUESTS_PER_SIZE=${REQUESTS_PER_SIZE:-1}
TIMEOUT_MS=${TIMEOUT_MS:-15000}
LOG_DIR=${LOG_DIR:-/root/brpc-ub-recommend/client-log}
OUTPUT_DIR=${OUTPUT_DIR:-/root/brpc-ub-recommend/evidence}
NOFILE_LIMIT=${NOFILE_LIMIT:-1048576}
URMA_RUNTIME_LIB_DIR=${URMA_RUNTIME_LIB_DIR:-/usr/lib64}
URMA_PROVIDER_LIB_DIR=${URMA_PROVIDER_LIB_DIR:-/usr/lib64/urma}
URMA_RUNTIME_LD_LIBRARY_PATH=${URMA_RUNTIME_LD_LIBRARY_PATH:-$URMA_RUNTIME_LIB_DIR:$URMA_PROVIDER_LIB_DIR}
STRICT_BTHREAD_KEY_CHECK=${STRICT_BTHREAD_KEY_CHECK:-0}

[[ -x "$CLIENT_BIN" ]] || { echo "ERROR: client binary is not executable: $CLIENT_BIN" >&2; exit 1; }
[[ -r "$URMA_RUNTIME_LIB_DIR/liburma.so" ]] || {
    echo "ERROR: host URMA runtime is missing: $URMA_RUNTIME_LIB_DIR/liburma.so" >&2
    exit 1
}
[[ "$REQUESTS_PER_SIZE" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: REQUESTS_PER_SIZE must be a positive integer" >&2
    exit 2
}
[[ "$STRICT_BTHREAD_KEY_CHECK" == 0 || "$STRICT_BTHREAD_KEY_CHECK" == 1 ]] || {
    echo "ERROR: STRICT_BTHREAD_KEY_CHECK must be 0 or 1" >&2
    exit 2
}
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
ulimit -n "$NOFILE_LIMIT"

# Keep a DataSystem virtualenv's bundled liburma from taking precedence over the host driver stack.
export LD_LIBRARY_PATH="$URMA_RUNTIME_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_id="MinimalRecommendUb_$(date +%Y%m%d_%H%M%S)_$$"
raw_log="$LOG_DIR/$run_id.log"
evidence_log="$OUTPUT_DIR/$run_id-evidence.log"
echo "urma_runtime_library_path=$URMA_RUNTIME_LD_LIBRARY_PATH"
echo "transport_mode=functional_ub backup_link=default degrade=default"

set +e
"$CLIENT_BIN" \
    --probe_server="$SERVER" \
    --probe_method="$METHOD" \
    --probe_payload_sizes="$PAYLOAD_SIZES" \
    --probe_requests_per_size="$REQUESTS_PER_SIZE" \
    --probe_timeout_ms="$TIMEOUT_MS" \
    --probe_max_retry=0 \
    --probe_expect_echo=true \
    --ubsocket_enable=true \
    --ubsocket_use_ub=true \
    2>&1 | tee "$raw_log"
client_status=${PIPESTATUS[0]}
set -e

[[ "$client_status" -eq 0 ]] || { echo "ERROR: probe client exited $client_status" >&2; exit 1; }
grep -Fq 'MINIMAL_RECOMMEND_UB_MATRIX_PASS' "$raw_log" || {
    echo "ERROR: integrity matrix pass marker is missing" >&2
    exit 1
}
if grep -Fq '"valid":false' "$raw_log"; then
    echo "ERROR: at least one payload integrity check failed" >&2
    exit 1
fi
grep -Fq 'bind jetty success' "$raw_log" || {
    echo "ERROR: UBSComm bind-jetty evidence is missing" >&2
    exit 1
}

bthread_key_warning_count=$(grep -Fc 'invalid bthread_key_t' "$raw_log" || true)
if [[ "$bthread_key_warning_count" -gt 0 ]]; then
    echo "WARNING: observed $bthread_key_warning_count invalid bthread key diagnostics; accepting the completed functional RPC matrix while stability remains unresolved" >&2
    if [[ "$STRICT_BTHREAD_KEY_CHECK" == 1 ]]; then
        echo "ERROR: strict bthread key check rejected the functional run" >&2
        exit 1
    fi
fi

size_count=$(awk -F, '{print NF}' <<<"$PAYLOAD_SIZES")
expected_count=$((size_count * REQUESTS_PER_SIZE))
actual_count=$(grep -Fc '"event":"minimal_recommend_ub_probe"' "$raw_log")
[[ "$actual_count" -eq "$expected_count" ]] || {
    echo "ERROR: expected $expected_count probe events, got $actual_count" >&2
    exit 1
}

grep -E 'bind jetty success|invalid bthread_key_t|"event":"minimal_recommend_ub_probe"|MINIMAL_RECOMMEND_UB_' \
    "$raw_log" >"$evidence_log"
echo "evidence_log=$evidence_log"
echo "MINIMAL_RECOMMEND_UB_EVIDENCE_PASS events=$actual_count server=$SERVER method=$METHOD bthread_key_warnings=$bthread_key_warning_count"
