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

for command in grep patch cp date mktemp; do
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

backup_dir="$BACKUP_ROOT/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup_dir/src/bthread" "$backup_dir/src/brpc"
cp -p "$BRPC_ROOT/src/bthread/bthread.cpp" "$backup_dir/src/bthread/"
cp -p "$BRPC_ROOT/src/brpc/ubsocket_initializer.cpp" "$backup_dir/src/brpc/"

patch_file=$(mktemp)
trap 'rm -f "$patch_file"' EXIT
cat >"$patch_file" <<'PATCH'
diff --git a/src/bthread/bthread.cpp b/src/bthread/bthread.cpp
--- a/src/bthread/bthread.cpp
+++ b/src/bthread/bthread.cpp
@@ -1,5 +1,5 @@
 #ifdef BRPC_WITH_URMA
-bthread_key_t ubsocket_trace_rpcid_key{0, 0};
-bthread_key_t ubsocket_trace_call_timestamp{1, 0};
+bthread_key_t ubsocket_trace_rpcid_key = INVALID_BTHREAD_KEY;
+bthread_key_t ubsocket_trace_call_timestamp = INVALID_BTHREAD_KEY;
 #endif
 namespace bthread {
diff --git a/src/brpc/ubsocket_initializer.cpp b/src/brpc/ubsocket_initializer.cpp
--- a/src/brpc/ubsocket_initializer.cpp
+++ b/src/brpc/ubsocket_initializer.cpp
@@ -1,5 +1,26 @@
     }
     ubsocket_set_log_level(ub_log_level);
     ubsocket_set_logger(UBSocketLogger);
+
+    // These keys must be allocated by bthread. Hard-coded index/version pairs
+    // are not registered keys and fail validation in bthread_setspecific().
+    bool rpcid_key_created = false;
+    if (ubsocket_trace_rpcid_key == INVALID_BTHREAD_KEY) {
+        if (bthread_key_create(&ubsocket_trace_rpcid_key, NULL) != 0) {
+            LOG(ERROR) << "Failed to create UBSocket RPC ID trace key";
+            return -1;
+        }
+        rpcid_key_created = true;
+    }
+    if (ubsocket_trace_call_timestamp == INVALID_BTHREAD_KEY) {
+        if (bthread_key_create(&ubsocket_trace_call_timestamp, NULL) != 0) {
+            LOG(ERROR) << "Failed to create UBSocket call timestamp trace key";
+            if (rpcid_key_created) {
+                (void)bthread_key_delete(ubsocket_trace_rpcid_key);
+                ubsocket_trace_rpcid_key = INVALID_BTHREAD_KEY;
+            }
+            return -1;
+        }
+    }
     /* initialize ubsocket */
     u_init_options_t options;
PATCH

patch --dry-run --batch --forward -p1 -d "$BRPC_ROOT" <"$patch_file" >/dev/null ||
    fail "patch dry-run failed; source differs from the diagnosed local tree"
patch --batch --forward --no-backup-if-mismatch -p1 -d "$BRPC_ROOT" <"$patch_file"

is_fixed || fail "post-patch source verification failed"

echo "BRPC_UB_TRACE_KEY_FIX_APPLIED brpc_root=$BRPC_ROOT backup=$backup_dir"
echo "next=ACTION=build bash scripts/verify_brpc_ub_echo_baseline.sh"
