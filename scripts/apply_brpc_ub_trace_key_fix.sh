#!/usr/bin/env bash
# Allocate the customized bRPC UBSocket trace keys through the bthread key API.
set -euo pipefail

BRPC_ROOT=${BRPC_ROOT:-/home/zcx/workspace/brpc-827}
BACKUP_ROOT=${BACKUP_ROOT:-/tmp/brpc-ub-trace-key-fix-backup}

fail()
{
    echo "ERROR: $*" >&2
    exit 1
}

is_fixed()
{
    grep -Fq 'bthread_key_t ubsocket_trace_rpcid_key = INVALID_BTHREAD_KEY;' \
        "$BRPC_ROOT/src/bthread/bthread.cpp" &&
        grep -Fq 'bthread_key_t ubsocket_trace_call_timestamp = INVALID_BTHREAD_KEY;' \
            "$BRPC_ROOT/src/bthread/bthread.cpp" &&
        grep -Fq 'bthread_key_create(&ubsocket_trace_rpcid_key, NULL)' \
            "$BRPC_ROOT/src/brpc/ubsocket_initializer.cpp" &&
        grep -Fq 'bthread_key_create(&ubsocket_trace_call_timestamp, NULL)' \
            "$BRPC_ROOT/src/brpc/ubsocket_initializer.cpp"
}

for command in grep sed cp date mktemp; do
    command -v "$command" >/dev/null 2>&1 || fail "missing command: $command"
done

[[ -f "$BRPC_ROOT/src/bthread/bthread.cpp" ]] ||
    fail "missing bthread source: $BRPC_ROOT/src/bthread/bthread.cpp"
[[ -f "$BRPC_ROOT/src/brpc/ubsocket_initializer.cpp" ]] ||
    fail "missing UBSocket initializer: $BRPC_ROOT/src/brpc/ubsocket_initializer.cpp"

if is_fixed; then
    echo "BRPC_UB_TRACE_KEY_FIX_ALREADY_APPLIED brpc_root=$BRPC_ROOT"
    exit 0
fi

grep -Fq 'bthread_key_t ubsocket_trace_rpcid_key{0, 0};' \
    "$BRPC_ROOT/src/bthread/bthread.cpp" ||
    fail "unexpected rpcid key definition; refusing to patch"
grep -Fq 'bthread_key_t ubsocket_trace_call_timestamp{1, 0};' \
    "$BRPC_ROOT/src/bthread/bthread.cpp" ||
    fail "unexpected timestamp key definition; refusing to patch"
grep -Fq 'ubsocket_set_logger(UBSocketLogger);' \
    "$BRPC_ROOT/src/brpc/ubsocket_initializer.cpp" ||
    fail "unexpected InitializeUBSocket implementation; refusing to patch"
[[ $(grep -Fc 'ubsocket_set_logger(UBSocketLogger);' \
    "$BRPC_ROOT/src/brpc/ubsocket_initializer.cpp") -eq 1 ]] ||
    fail "UBSocket logger anchor is not unique; refusing to patch"

backup_dir="$BACKUP_ROOT/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup_dir/src/bthread" "$backup_dir/src/brpc"
cp -p "$BRPC_ROOT/src/bthread/bthread.cpp" "$backup_dir/src/bthread/"
cp -p "$BRPC_ROOT/src/brpc/ubsocket_initializer.cpp" "$backup_dir/src/brpc/"

insert_file=$(mktemp)
trap 'rm -f "$insert_file"' EXIT
cat >"$insert_file" <<'INSERT'

    // These keys must be allocated by bthread. Hard-coded index/version pairs
    // are not registered keys and fail validation in bthread_setspecific().
    bool rpcid_key_created = false;
    if (ubsocket_trace_rpcid_key == INVALID_BTHREAD_KEY) {
        if (bthread_key_create(&ubsocket_trace_rpcid_key, NULL) != 0) {
            LOG(ERROR) << "Failed to create UBSocket RPC ID trace key";
            return -1;
        }
        rpcid_key_created = true;
    }
    if (ubsocket_trace_call_timestamp == INVALID_BTHREAD_KEY) {
        if (bthread_key_create(&ubsocket_trace_call_timestamp, NULL) != 0) {
            LOG(ERROR) << "Failed to create UBSocket call timestamp trace key";
            if (rpcid_key_created) {
                (void)bthread_key_delete(ubsocket_trace_rpcid_key);
                ubsocket_trace_rpcid_key = INVALID_BTHREAD_KEY;
            }
            return -1;
        }
    }
INSERT

restore_backup()
{
    cp -p "$backup_dir/src/bthread/bthread.cpp" "$BRPC_ROOT/src/bthread/bthread.cpp"
    cp -p "$backup_dir/src/brpc/ubsocket_initializer.cpp" \
        "$BRPC_ROOT/src/brpc/ubsocket_initializer.cpp"
}

if ! sed -i \
    's/^bthread_key_t ubsocket_trace_rpcid_key{0, 0};$/bthread_key_t ubsocket_trace_rpcid_key = INVALID_BTHREAD_KEY;/; s/^bthread_key_t ubsocket_trace_call_timestamp{1, 0};$/bthread_key_t ubsocket_trace_call_timestamp = INVALID_BTHREAD_KEY;/' \
    "$BRPC_ROOT/src/bthread/bthread.cpp"; then
    restore_backup
    fail "failed to replace bthread trace key definitions; backup restored"
fi
if ! sed -i \
    "\\|ubsocket_set_logger(UBSocketLogger);|r $insert_file" \
    "$BRPC_ROOT/src/brpc/ubsocket_initializer.cpp"; then
    restore_backup
    fail "failed to insert bthread trace key allocation; backup restored"
fi

if ! is_fixed; then
    restore_backup
    fail "post-edit source verification failed; backup restored"
fi

echo "BRPC_UB_TRACE_KEY_FIX_APPLIED brpc_root=$BRPC_ROOT backup=$backup_dir"
echo "next=ACTION=build bash scripts/verify_brpc_ub_echo_baseline.sh"
