#!/usr/bin/env bash
# Collect local-source and binary evidence for the UB invalid-bthread-key crash.
set -euo pipefail

BRPC_ROOT=${BRPC_ROOT:-/home/zcx/workspace/brpc-827}
BAZEL_OUTPUT_BASE=${BAZEL_OUTPUT_BASE:-/root/.cache/bazel/_bazel_root/0947eeff3cdbdab635f34a3b3ff5f6d1}
INSTALL_DIR=${INSTALL_DIR:-/opt/pairec-brpc-ub-echo-baseline}
CLIENT_BIN=${CLIENT_BIN:-$INSTALL_DIR/bin/echo_c++_client}
SERVER_BIN=${SERVER_BIN:-$INSTALL_DIR/bin/echo_c++_server}
CLIENT_LOG=${CLIENT_LOG:-}
SERVER_LOG=${SERVER_LOG:-}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/brpc-ub-bthread-key-diagnostic-$(date +%Y%m%d-%H%M%S)}

die()
{
    echo "ERROR: $*" >&2
    exit 1
}

latest_log()
{
    local directory=$1 pattern=$2
    [[ -d "$directory" ]] || return 0
    find "$directory" -maxdepth 1 -type f -name "$pattern" -printf '%T@ %p\n' 2>/dev/null |
        sort -nr | sed -n '1s/^[^ ]* //p'
}

collect_git_state()
{
    local repository=$1 output=$2
    if ! git -C "$repository" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        printf 'NO_GIT_METADATA path=%s\n' "$repository" >"$output"
        return
    fi
    {
        echo "path=$(readlink -f "$repository")"
        echo "head=$(git -C "$repository" rev-parse HEAD 2>/dev/null || echo unavailable)"
        echo '===== commit ====='
        git -C "$repository" show -s \
            --format='commit=%H%nparents=%P%nauthor=%an <%ae>%nauthor_date=%aI%ncommitter=%cn <%ce>%ncommit_date=%cI%nsubject=%s' \
            HEAD 2>/dev/null || true
        echo '===== remotes (informational only) ====='
        git -C "$repository" remote -v 2>/dev/null || true
        echo '===== status ====='
        git -C "$repository" status --short 2>/dev/null || true
        echo '===== containing refs ====='
        git -C "$repository" branch -a --contains HEAD 2>/dev/null || true
    } >"$output"
}

collect_binary()
{
    local binary=$1 prefix=$2
    if [[ ! -x "$binary" ]]; then
        printf 'BINARY_NOT_FOUND path=%s\n' "$binary" >"$OUTPUT_DIR/$prefix-binary.txt"
        return
    fi
    {
        echo "path=$(readlink -f "$binary")"
        sha256sum "$binary"
        file "$binary"
        echo '===== dynamic dependencies ====='
        ldd "$binary" 2>&1 || true
        echo '===== ELF needed/rpath/runpath ====='
        readelf -d "$binary" 2>/dev/null |
            grep -E 'NEEDED|RPATH|RUNPATH' || true
        echo '===== bthread key symbols ====='
        readelf -Ws "$binary" 2>/dev/null |
            grep -E 'bthread_(key_create2?|key_delete|setspecific|getspecific)' || true
        echo '===== UB initialization symbols ====='
        nm -anC "$binary" 2>/dev/null |
            grep -Ei 'InitializeUBSocket|ubsocket_init|umq_init|bthread.*key' |
            sed -n '1,300p' || true
    } >"$OUTPUT_DIR/$prefix-binary.txt"
}

map_crash_addresses()
{
    local binary=$1 log=$2 prefix=$3
    local addresses_file="$OUTPUT_DIR/$prefix-addresses.txt"
    local mapping_file="$OUTPUT_DIR/$prefix-addr2line.txt"
    local disassembly_file="$OUTPUT_DIR/$prefix-disassembly.txt"
    : >"$addresses_file"
    : >"$mapping_file"
    : >"$disassembly_file"
    [[ -x "$binary" && -f "$log" ]] || return 0

    grep -oE '#[0-9]+ 0x[0-9a-fA-F]+' "$log" 2>/dev/null |
        awk '{print $2}' | sort -u >"$addresses_file" || true
    while IFS= read -r address; do
        [[ -n "$address" ]] || continue
        echo "===== $address =====" >>"$mapping_file"
        addr2line -Cfipe "$binary" "$address" >>"$mapping_file" 2>&1 || true

        numeric=$((address))
        start=$((numeric > 96 ? numeric - 96 : 0))
        stop=$((numeric + 64))
        echo "===== $address [$start,$stop] =====" >>"$disassembly_file"
        objdump -dC --start-address="$start" --stop-address="$stop" "$binary" \
            >>"$disassembly_file" 2>&1 || true
    done <"$addresses_file"
}

for command in rg git sha256sum file ldd readelf nm addr2line objdump; do
    command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done
[[ -d "$BRPC_ROOT" ]] || die "bRPC source tree does not exist: $BRPC_ROOT"
[[ -d "$BAZEL_OUTPUT_BASE/external" ]] || die "Bazel external directory does not exist: $BAZEL_OUTPUT_BASE/external"
mkdir -p "$OUTPUT_DIR"

if [[ -z "$CLIENT_LOG" ]]; then
    CLIENT_LOG=$(latest_log /root/brpc-ub-echo-baseline/client-log 'echo-client-*.log')
fi
if [[ -z "$SERVER_LOG" ]]; then
    SERVER_LOG=$(latest_log /root/brpc-ub-echo-baseline/server-log 'echo-server-*.log')
fi

UBS_DIR=$(find -L "$BAZEL_OUTPUT_BASE/external" -mindepth 1 -maxdepth 1 -type d \
    -name '*local_deps*ubsocket*' -print | sort | sed -n '1p')
[[ -n "$UBS_DIR" ]] || die "unable to locate the Bazel UBSocket external repository"
UBS_DIR=$(readlink -f "$UBS_DIR")

{
    echo "collected_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "kernel=$(uname -srvmo)"
    echo "brpc_root=$(readlink -f "$BRPC_ROOT")"
    echo "ubs_dir=$UBS_DIR"
    echo "bazel_output_base=$(readlink -f "$BAZEL_OUTPUT_BASE")"
    echo "client_bin=$CLIENT_BIN"
    echo "server_bin=$SERVER_BIN"
    echo "client_log=${CLIENT_LOG:-not-found}"
    echo "server_log=${SERVER_LOG:-not-found}"
    echo "core_limit=$(ulimit -c)"
    echo "gcc=$(gcc --version 2>/dev/null | sed -n '1p' || echo unavailable)"
    echo "bazel=$(bazel --version 2>/dev/null || echo unavailable)"
} >"$OUTPUT_DIR/environment.txt"

collect_git_state "$BRPC_ROOT" "$OUTPUT_DIR/brpc-git-state.txt"
collect_git_state "$UBS_DIR" "$OUTPUT_DIR/ubsocket-git-state.txt"

git -C "$BRPC_ROOT" diff -- src/brpc src/bthread src/butil local_deps_ext.bzl MODULE.bazel \
    >"$OUTPUT_DIR/brpc-working-tree.diff" 2>&1 || true
git -C "$BRPC_ROOT" diff --stat -- src/brpc src/bthread src/butil local_deps_ext.bzl MODULE.bazel \
    >"$OUTPUT_DIR/brpc-working-tree-stat.txt" 2>&1 || true
if git -C "$UBS_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$UBS_DIR" diff >"$OUTPUT_DIR/ubsocket-working-tree.diff" 2>&1 || true
    git -C "$UBS_DIR" diff --stat >"$OUTPUT_DIR/ubsocket-working-tree-stat.txt" 2>&1 || true
else
    printf 'NO_GIT_METADATA path=%s\n' "$UBS_DIR" >"$OUTPUT_DIR/ubsocket-working-tree.diff"
    : >"$OUTPUT_DIR/ubsocket-working-tree-stat.txt"
fi

source_roots=("$BRPC_ROOT/src/brpc" "$BRPC_ROOT/src/bthread" "$BRPC_ROOT/src/butil")
if [[ -d "$UBS_DIR/src/ubsocket" ]]; then
    source_roots+=("$UBS_DIR/src/ubsocket")
fi
if [[ -d "$UBS_DIR/src/hcom/umq" ]]; then
    source_roots+=("$UBS_DIR/src/hcom/umq")
fi

rg -n -C 5 --glob '*.{c,cc,cpp,h,hpp}' \
    'bthread_(key_create2?|key_delete|setspecific|getspecific)|bthread_key_t' \
    "${source_roots[@]}" >"$OUTPUT_DIR/bthread-key-usage.txt" 2>&1 || true
rg -n -C 5 --glob '*.{c,cc,cpp,h,hpp}' \
    'GlobalInitializeOrDie|InitializeUBSocket|ubsocket_init|umq_init|pthread_once|bthread_once' \
    "${source_roots[@]}" >"$OUTPUT_DIR/initialization-order.txt" 2>&1 || true

collect_binary "$CLIENT_BIN" client
collect_binary "$SERVER_BIN" server

if [[ -n "$CLIENT_LOG" && -f "$CLIENT_LOG" ]]; then
    cp -p "$CLIENT_LOG" "$OUTPUT_DIR/client-crash.log"
fi
if [[ -n "$SERVER_LOG" && -f "$SERVER_LOG" ]]; then
    cp -p "$SERVER_LOG" "$OUTPUT_DIR/server-crash.log"
fi

map_crash_addresses "$CLIENT_BIN" "${CLIENT_LOG:-/nonexistent}" client
map_crash_addresses "$SERVER_BIN" "${SERVER_LOG:-/nonexistent}" server

if command -v coredumpctl >/dev/null 2>&1; then
    coredumpctl --no-pager info "$CLIENT_BIN" >"$OUTPUT_DIR/client-coredump-info.txt" 2>&1 || true
    coredumpctl --no-pager info "$SERVER_BIN" >"$OUTPUT_DIR/server-coredump-info.txt" 2>&1 || true
else
    printf 'coredumpctl unavailable\n' >"$OUTPUT_DIR/client-coredump-info.txt"
    printf 'coredumpctl unavailable\n' >"$OUTPUT_DIR/server-coredump-info.txt"
fi

client_crash=0
server_crash=0
if [[ -f "$OUTPUT_DIR/client-crash.log" ]] &&
    grep -Eq 'invalid bthread_key|bthread_setspecific.*invalid|Segmentation fault' \
        "$OUTPUT_DIR/client-crash.log"; then
    client_crash=1
fi
if [[ -f "$OUTPUT_DIR/server-crash.log" ]] &&
    grep -Eq 'invalid bthread_key|bthread_setspecific.*invalid|Segmentation fault' \
        "$OUTPUT_DIR/server-crash.log"; then
    server_crash=1
fi

{
    if [[ "$client_crash" == 1 || "$server_crash" == 1 ]]; then
        echo "classification=BRPC_UB_BTHREAD_KEY_CRASH_CONFIRMED"
    else
        echo "classification=BRPC_UB_BTHREAD_KEY_EVIDENCE_COLLECTED_NO_MATCHING_LOG"
    fi
    echo "client_crash_evidence=$client_crash"
    echo "server_crash_evidence=$server_crash"
    echo "public_repository_assumption=disabled"
    echo "source_of_truth=local_brpc_and_bazel_external_ubsocket"
    echo "next_primary_artifact=$OUTPUT_DIR/bthread-key-usage.txt"
    echo "next_symbol_artifact=$OUTPUT_DIR/client-addr2line.txt"
    echo "next_initialization_artifact=$OUTPUT_DIR/initialization-order.txt"
    if [[ "$client_crash" == 0 && "$server_crash" == 0 ]]; then
        echo "warning=no_matching_crash_log_was_found; pass CLIENT_LOG or SERVER_LOG explicitly"
    fi
} >"$OUTPUT_DIR/summary.txt"

tar -C "$(dirname "$OUTPUT_DIR")" -czf "$OUTPUT_DIR.tar.gz" "$(basename "$OUTPUT_DIR")"
cat "$OUTPUT_DIR/summary.txt"
echo "diagnostic_dir=$OUTPUT_DIR"
echo "diagnostic_archive=$OUTPUT_DIR.tar.gz"
echo "BRPC_UB_BTHREAD_KEY_DIAGNOSTIC_OK"
