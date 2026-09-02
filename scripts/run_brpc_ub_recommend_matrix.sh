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

[[ -x "$CLIENT_BIN" ]] || { echo "ERROR: client binary is not executable: $CLIENT_BIN" >&2; exit 1; }
[[ "$REQUESTS_PER_SIZE" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: REQUESTS_PER_SIZE must be a positive integer" >&2
    exit 2
}
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
ulimit -n "$NOFILE_LIMIT"

run_id="MinimalRecommendUb_$(date +%Y%m%d_%H%M%S)_$$"
raw_log="$LOG_DIR/$run_id.log"
evidence_log="$OUTPUT_DIR/$run_id-evidence.log"

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
    --ubsocket_backup_link_enable=false \
    --ubsocket_degrade_enable=false \
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

size_count=$(awk -F, '{print NF}' <<<"$PAYLOAD_SIZES")
expected_count=$((size_count * REQUESTS_PER_SIZE))
actual_count=$(grep -Fc '"event":"minimal_recommend_ub_probe"' "$raw_log")
[[ "$actual_count" -eq "$expected_count" ]] || {
    echo "ERROR: expected $expected_count probe events, got $actual_count" >&2
    exit 1
}

grep -E 'bind jetty success|"event":"minimal_recommend_ub_probe"|MINIMAL_RECOMMEND_UB_' \
    "$raw_log" >"$evidence_log"
echo "evidence_log=$evidence_log"
echo "MINIMAL_RECOMMEND_UB_EVIDENCE_PASS events=$actual_count server=$SERVER method=$METHOD"
