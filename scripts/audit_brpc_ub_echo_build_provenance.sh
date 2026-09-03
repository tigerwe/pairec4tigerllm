#!/usr/bin/env bash
# Read-only provenance audit for the successful and candidate bRPC UB Echo builds.
set -euo pipefail

SUCCESS_ROOT=${SUCCESS_ROOT:-/home/zcx/workspace/brpc-827}
CANDIDATE_ROOT=${CANDIDATE_ROOT:-/home/zcx/workspace/brpc}
SUCCESS_OUTPUT_BASE=${SUCCESS_OUTPUT_BASE:-/root/.cache/bazel/_bazel_root/0947eeff3cdbdab635f34a3b3ff5f6d1}
CANDIDATE_OUTPUT_BASE=${CANDIDATE_OUTPUT_BASE:-/root/.cache/bazel/_bazel_root/02df77b0294ccdcf08a6a8a39050de3a}
SUCCESS_SERVER_BIN=${SUCCESS_SERVER_BIN:-$SUCCESS_ROOT/bazel-bin/example/echo_c++_server}
SUCCESS_CLIENT_BIN=${SUCCESS_CLIENT_BIN:-$SUCCESS_ROOT/bazel-bin/example/echo_c++_client}
CANDIDATE_SERVER_BIN=${CANDIDATE_SERVER_BIN:-/opt/pairec-brpc-ub-recommend-known-good/bin/echo_c++_server}
CANDIDATE_CLIENT_BIN=${CANDIDATE_CLIENT_BIN:-/opt/pairec-brpc-ub-recommend-known-good/bin/echo_c++_client}
SUCCESS_SERVER_LOG=${SUCCESS_SERVER_LOG:-}
SUCCESS_CLIENT_LOG=${SUCCESS_CLIENT_LOG:-}
FAILED_SERVER_LOG=${FAILED_SERVER_LOG:-}
FAILED_CLIENT_LOG=${FAILED_CLIENT_LOG:-}
EXPECTED_BRPC_COMMIT=${EXPECTED_BRPC_COMMIT:-827db2a9be6a3eac0a1ac3666b4a9cf33b976175}
EXPECTED_UBSCOMM_COMMIT=${EXPECTED_UBSCOMM_COMMIT:-9f80dc9fb5f06ba8b5997064c928b89bda266ffd}
SYSTEM_BAZELRC=${SYSTEM_BAZELRC:-/etc/bazel.bazelrc}
USER_BAZELRC=${USER_BAZELRC:-/root/.bazelrc}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/brpc-ub-echo-provenance-$(date +%Y%m%d-%H%M%S)}

fail()
{
    echo "ERROR: $*" >&2
    exit 1
}

for command in git sha256sum stat readelf file ldd nm strings find grep diff awk sed date realpath cp \
    head sort ls cat uname; do
    command -v "$command" >/dev/null 2>&1 || fail "missing command: $command"
done

[[ -d "$SUCCESS_ROOT" ]] || fail "missing successful-build tree: $SUCCESS_ROOT"
[[ -d "$CANDIDATE_ROOT" ]] || fail "missing candidate-build tree: $CANDIDATE_ROOT"
mkdir -p "$OUTPUT_DIR/success" "$OUTPUT_DIR/candidate" "$OUTPUT_DIR/logs"

sha_or_missing()
{
    local path=$1
    if [[ -f "$path" ]]; then
        sha256sum "$path" | awk '{print $1}'
    else
        printf 'MISSING\n'
    fi
}

copy_if_readable()
{
    local source=$1
    local destination=$2
    if [[ -r "$source" ]]; then
        cp -p "$source" "$destination"
    fi
}

record_binary()
{
    local label=$1
    local path=$2
    local out=$3
    {
        echo "label=$label"
        echo "path=$path"
        echo "realpath=$(realpath -m "$path")"
        if [[ ! -f "$path" ]]; then
            echo "state=MISSING"
            return
        fi
        echo "state=PRESENT"
        echo "sha256=$(sha_or_missing "$path")"
        stat -c 'mtime=%y\nsize=%s\nmode=%A' "$path"
        file "$path"
        readelf -n "$path" 2>/dev/null | grep -F 'Build ID:' || true
        echo "-- ldd --"
        ldd "$path" 2>&1 || true
        echo "-- UB/BoringSSL symbols and strings --"
        nm -C "$path" 2>/dev/null |
            grep -Ei 'ubsocket|umq|urma|boringssl|bthread_setspecific|InitializeUBSocket' |
            head -n 200 || true
        strings "$path" 2>/dev/null |
            grep -Ei 'local_deps.*(ubsocket|boring)|BRPC_WITH_(URMA|BORINGSSL)|ubsocket' |
            head -n 200 || true
    } >"$out"
}

record_external_repositories()
{
    local output_base=$1
    local out=$2
    {
        echo "output_base=$output_base"
        echo "realpath=$(realpath -m "$output_base")"
        [[ -d "$output_base" ]] || { echo "state=MISSING"; return; }
        echo "state=PRESENT"
        copy_if_readable "$output_base/command.log" "$out.command.log"
        find -L "$output_base/external" -mindepth 1 -maxdepth 1 -type d \
            \( -iname '*ubsocket*' -o -iname '*ubscomm*' -o -iname '*boring*' \) \
            -print 2>/dev/null | sort | while IFS= read -r repository; do
                echo "repository=$repository"
                if git -C "$repository" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
                    echo "  head=$(git -C "$repository" rev-parse HEAD 2>/dev/null || true)"
                    echo "  status_begin"
                    git -C "$repository" status --short --untracked-files=all 2>/dev/null |
                        sed 's/^/    /' || true
                    echo "  status_end"
                fi
            done
    } >"$out"
}

record_tree()
{
    local label=$1
    local root=$2
    local output_base=$3
    local out_dir=$4
    local relevant=(
        .bazelrc
        MODULE.bazel
        MODULE.bazel.lock
        local_deps_ext.bzl
        BUILD.bazel
        src/bthread/bthread.cpp
        src/bthread/key.cpp
        src/brpc/channel.cpp
        src/brpc/ubsocket_initializer.cpp
        example/BUILD.bazel
        example/echo_c++_client.cpp
        example/echo_c++_server.cpp
    )

    {
        echo "label=$label"
        echo "root=$root"
        echo "realpath=$(realpath -m "$root")"
        echo "head=$(git -C "$root" rev-parse HEAD 2>/dev/null || echo NOT_A_GIT_TREE)"
        echo "branch=$(git -C "$root" branch --show-current 2>/dev/null || true)"
        echo "expected_brpc_commit=$EXPECTED_BRPC_COMMIT"
        echo "expected_ubscomm_commit=$EXPECTED_UBSCOMM_COMMIT"
        echo "-- status --"
        git -C "$root" status --short --untracked-files=all 2>/dev/null || true
        echo "-- relevant file hashes --"
        for relative in "${relevant[@]}"; do
            printf '%s  %s\n' "$(sha_or_missing "$root/$relative")" "$relative"
        done
        echo "-- UBSComm revision declarations --"
        grep -nE 'ubsocket|ubscomm|commit[[:space:]]*=|branch[[:space:]]*=|tag[[:space:]]*=' \
            "$root/local_deps_ext.bzl" 2>/dev/null || true
        echo "-- trace key definitions and allocation --"
        grep -nE 'ubsocket_trace_(rpcid_key|call_timestamp)|bthread_key_create' \
            "$root/src/bthread/bthread.cpp" \
            "$root/src/brpc/ubsocket_initializer.cpp" 2>/dev/null || true
        echo "-- bazel-bin --"
        ls -ld "$root/bazel-bin" 2>/dev/null || true
        echo "-- relevant diff --"
        git -C "$root" diff --no-ext-diff -- "${relevant[@]}" 2>/dev/null || true
    } >"$out_dir/tree-manifest.txt"

    for config in .bazelrc MODULE.bazel MODULE.bazel.lock local_deps_ext.bzl; do
        copy_if_readable "$root/$config" "$out_dir/${config//\//_}"
    done
    record_external_repositories "$output_base" "$out_dir/output-base.txt"
}

record_host_and_toolchain()
{
    local out=$1
    local bazel_path
    bazel_path=$(command -v bazel 2>/dev/null || true)
    {
        echo "captured_at=$(date --iso-8601=seconds)"
        uname -a
        echo "bazel_path=${bazel_path:-MISSING}"
        if [[ -n "$bazel_path" ]]; then
            echo "bazel_realpath=$(realpath -m "$bazel_path")"
            echo "bazel_sha256=$(sha_or_missing "$bazel_path")"
            bazel version 2>&1 || true
        fi
        for compiler in gcc g++ cc c++ ld ar; do
            compiler_path=$(command -v "$compiler" 2>/dev/null || true)
            echo "$compiler.path=${compiler_path:-MISSING}"
            if [[ -n "$compiler_path" ]]; then
                echo "$compiler.realpath=$(realpath -m "$compiler_path")"
                echo "$compiler.sha256=$(sha_or_missing "$compiler_path")"
                "$compiler" --version 2>&1 | head -n 2 || true
            fi
        done
        echo "system_bazelrc=$SYSTEM_BAZELRC"
        echo "system_bazelrc_sha256=$(sha_or_missing "$SYSTEM_BAZELRC")"
        echo "user_bazelrc=$USER_BAZELRC"
        echo "user_bazelrc_sha256=$(sha_or_missing "$USER_BAZELRC")"
        echo "BAZELRC_env=${BAZELRC:-UNSET}"
        echo "CC_env=${CC:-UNSET}"
        echo "CXX_env=${CXX:-UNSET}"
        echo "LD_LIBRARY_PATH_env=${LD_LIBRARY_PATH:-UNSET}"
    } >"$out"
    copy_if_readable "$SYSTEM_BAZELRC" "$OUTPUT_DIR/system.bazelrc"
    copy_if_readable "$USER_BAZELRC" "$OUTPUT_DIR/user.bazelrc"
}

record_log()
{
    local label=$1
    local path=$2
    local out=$3
    {
        echo "label=$label"
        echo "path=${path:-UNSET}"
        if [[ -z "$path" || ! -r "$path" ]]; then
            echo "state=MISSING"
            return
        fi
        echo "state=PRESENT"
        echo "sha256=$(sha_or_missing "$path")"
        stat -c 'mtime=%y\nsize=%s' "$path"
        grep -E 'serving on port|bind jetty success|Received response|invalid bthread_key|Segmentation fault|create Logic UMQ|remote eid|remote jetty|sha256' \
            "$path" || true
    } >"$out"
}

record_host_and_toolchain "$OUTPUT_DIR/host-toolchain.txt"
record_tree success "$SUCCESS_ROOT" "$SUCCESS_OUTPUT_BASE" "$OUTPUT_DIR/success"
record_tree candidate "$CANDIDATE_ROOT" "$CANDIDATE_OUTPUT_BASE" "$OUTPUT_DIR/candidate"
record_binary success_server "$SUCCESS_SERVER_BIN" "$OUTPUT_DIR/success/server-binary.txt"
record_binary success_client "$SUCCESS_CLIENT_BIN" "$OUTPUT_DIR/success/client-binary.txt"
record_binary candidate_server "$CANDIDATE_SERVER_BIN" "$OUTPUT_DIR/candidate/server-binary.txt"
record_binary candidate_client "$CANDIDATE_CLIENT_BIN" "$OUTPUT_DIR/candidate/client-binary.txt"
record_log success_server "$SUCCESS_SERVER_LOG" "$OUTPUT_DIR/logs/success-server.txt"
record_log success_client "$SUCCESS_CLIENT_LOG" "$OUTPUT_DIR/logs/success-client.txt"
record_log failed_server "$FAILED_SERVER_LOG" "$OUTPUT_DIR/logs/failed-server.txt"
record_log failed_client "$FAILED_CLIENT_LOG" "$OUTPUT_DIR/logs/failed-client.txt"

success_head=$(git -C "$SUCCESS_ROOT" rev-parse HEAD 2>/dev/null || true)
candidate_head=$(git -C "$CANDIDATE_ROOT" rev-parse HEAD 2>/dev/null || true)
success_server_sha=$(sha_or_missing "$SUCCESS_SERVER_BIN")
success_client_sha=$(sha_or_missing "$SUCCESS_CLIENT_BIN")
candidate_server_sha=$(sha_or_missing "$CANDIDATE_SERVER_BIN")
candidate_client_sha=$(sha_or_missing "$CANDIDATE_CLIENT_BIN")
success_lock_sha=$(sha_or_missing "$SUCCESS_ROOT/MODULE.bazel.lock")
candidate_lock_sha=$(sha_or_missing "$CANDIDATE_ROOT/MODULE.bazel.lock")

if [[ "$success_server_sha" != MISSING && "$success_server_sha" == "$candidate_server_sha" &&
      "$success_client_sha" != MISSING && "$success_client_sha" == "$candidate_client_sha" ]]; then
    artifact_comparison=IDENTICAL
else
    artifact_comparison=DIFFERENT_OR_MISSING
fi

if [[ "$success_lock_sha" != MISSING && "$success_lock_sha" == "$candidate_lock_sha" ]]; then
    lockfile_comparison=IDENTICAL
else
    lockfile_comparison=DIFFERENT_OR_MISSING
fi

if [[ -n "$SUCCESS_CLIENT_LOG" && -r "$SUCCESS_CLIENT_LOG" ]] &&
   grep -Fq 'bind jetty success' "$SUCCESS_CLIENT_LOG" &&
   grep -Fq 'Received response from' "$SUCCESS_CLIENT_LOG"; then
    success_log_verdict=PASS_MARKERS_UNBOUND_TO_BINARY
else
    success_log_verdict=NOT_BOUND
fi

success_runtime_binding=UNBOUND
if [[ "$success_server_sha" != MISSING && "$success_client_sha" != MISSING &&
      -n "$SUCCESS_SERVER_LOG" && -r "$SUCCESS_SERVER_LOG" &&
      -n "$SUCCESS_CLIENT_LOG" && -r "$SUCCESS_CLIENT_LOG" ]] &&
   grep -Fq "$success_server_sha" "$SUCCESS_SERVER_LOG" &&
   grep -Fq "$success_client_sha" "$SUCCESS_CLIENT_LOG" &&
   grep -Fq 'Received response from' "$SUCCESS_CLIENT_LOG"; then
    success_runtime_binding=SHA256_BOUND_PASS
fi

if [[ -n "$FAILED_CLIENT_LOG" && -r "$FAILED_CLIENT_LOG" ]] &&
   grep -Eq 'invalid bthread_key|bthread_setspecific.*invalid' "$FAILED_CLIENT_LOG"; then
    failed_log_verdict=INVALID_BTHREAD_KEY_CONFIRMED
else
failed_log_verdict=NOT_BOUND
fi

preserved_dir="$OUTPUT_DIR/preserved-success-artifacts"
mkdir -p "$preserved_dir"
if [[ "$success_server_sha" != MISSING ]]; then
    cp -p "$SUCCESS_SERVER_BIN" "$preserved_dir/echo_c++_server.$success_server_sha"
fi
if [[ "$success_client_sha" != MISSING ]]; then
    cp -p "$SUCCESS_CLIENT_BIN" "$preserved_dir/echo_c++_client.$success_client_sha"
fi

config_comparison_file="$OUTPUT_DIR/config-comparison.txt"
{
    printf 'file\tsuccess_sha256\tcandidate_sha256\tverdict\n'
    for relative in .bazelrc MODULE.bazel MODULE.bazel.lock local_deps_ext.bzl \
        src/bthread/bthread.cpp src/bthread/key.cpp src/brpc/channel.cpp \
        src/brpc/ubsocket_initializer.cpp example/BUILD.bazel; do
        success_sha=$(sha_or_missing "$SUCCESS_ROOT/$relative")
        candidate_sha=$(sha_or_missing "$CANDIDATE_ROOT/$relative")
        verdict=DIFFERENT
        [[ "$success_sha" != MISSING && "$success_sha" == "$candidate_sha" ]] && verdict=IDENTICAL
        printf '%s\t%s\t%s\t%s\n' "$relative" "$success_sha" "$candidate_sha" "$verdict"
    done
} >"$config_comparison_file"

lockfile_diff="$OUTPUT_DIR/lockfile-diff.txt"
{
    echo "success=$SUCCESS_ROOT/MODULE.bazel.lock"
    echo "candidate=$CANDIDATE_ROOT/MODULE.bazel.lock"
    echo "success_sha256=$success_lock_sha"
    echo "candidate_sha256=$candidate_lock_sha"
    echo "-- unified diff --"
    if [[ -f "$SUCCESS_ROOT/MODULE.bazel.lock" && -f "$CANDIDATE_ROOT/MODULE.bazel.lock" ]]; then
        diff -u --label success/MODULE.bazel.lock --label candidate/MODULE.bazel.lock \
            "$SUCCESS_ROOT/MODULE.bazel.lock" "$CANDIDATE_ROOT/MODULE.bazel.lock" || true
    else
        echo "diff unavailable: one or both lockfiles are missing"
    fi
} >"$lockfile_diff"

command_log_diff="$OUTPUT_DIR/command-log-diff.txt"
success_command_log="$OUTPUT_DIR/success/output-base.txt.command.log"
candidate_command_log="$OUTPUT_DIR/candidate/output-base.txt.command.log"
{
    echo "success=$SUCCESS_OUTPUT_BASE/command.log"
    echo "candidate=$CANDIDATE_OUTPUT_BASE/command.log"
    echo "-- unified diff --"
    if [[ -f "$success_command_log" && -f "$candidate_command_log" ]]; then
        diff -u --label success/command.log --label candidate/command.log \
            "$success_command_log" "$candidate_command_log" || true
    else
        [[ -f "$success_command_log" ]] || echo "success command.log is missing"
        [[ -f "$candidate_command_log" ]] || echo "candidate command.log is missing"
    fi
} >"$command_log_diff"

recipe="$OUTPUT_DIR/reproduce-success-build.txt"
{
    echo "set -euo pipefail"
    echo
    echo "# Historical 2026-09-01 recipe reconstructed from the project record."
    echo "# This is executable Bash, but the audit itself never executes it or modifies either tree."
    echo "# The audit already preserved the pre-rebuild binaries under:"
    printf '# %q\n' "$preserved_dir"
    echo
    echo 'verify_sha()'
    echo '{'
    echo '    local expected=$1 path=$2 actual'
    echo '    actual=$(sha256sum "$path")'
    echo '    actual=${actual%% *}'
    echo '    test "$actual" = "$expected"'
    echo '}'
    echo
    echo "# Verify the snapshot used for reproduction:"
    printf 'test "$(git -C %q rev-parse HEAD)" = %q\n' "$SUCCESS_ROOT" "$success_head"
    for relative in .bazelrc MODULE.bazel MODULE.bazel.lock local_deps_ext.bzl \
        src/bthread/bthread.cpp src/brpc/channel.cpp src/brpc/ubsocket_initializer.cpp; do
        expected_sha=$(sha_or_missing "$SUCCESS_ROOT/$relative")
        if [[ "$expected_sha" != MISSING ]]; then
            printf 'verify_sha %q %q\n' "$expected_sha" "$SUCCESS_ROOT/$relative"
        fi
    done
    echo
    echo "# Preserve the known-good files above. The differing lockfile is part of the build input."
    echo "# Do not copy the candidate lockfile into this tree before reproduction."
    echo "# Re-run the original recorded build without rc isolation or extra defines:"
    printf 'cd %q\n' "$SUCCESS_ROOT"
    printf 'bazel --output_base=%q build -c opt --define=brpc_with_urma=true //example:echo_c++_server //example:echo_c++_client\n' \
        "$SUCCESS_OUTPUT_BASE"
    echo "# Compare the rebuilt hashes with the preserved successful artifacts before runtime testing."
} >"$recipe"

summary="$OUTPUT_DIR/summary.txt"
{
    echo "classification=BRPC_UB_ECHO_BUILD_PROVENANCE_AUDIT"
    echo "success_root=$SUCCESS_ROOT"
    echo "candidate_root=$CANDIDATE_ROOT"
    echo "success_output_base=$SUCCESS_OUTPUT_BASE"
    echo "candidate_output_base=$CANDIDATE_OUTPUT_BASE"
    echo "success_head=$success_head"
    echo "candidate_head=$candidate_head"
    echo "expected_head=$EXPECTED_BRPC_COMMIT"
    echo "success_server_sha256=$success_server_sha"
    echo "success_client_sha256=$success_client_sha"
    echo "candidate_server_sha256=$candidate_server_sha"
    echo "candidate_client_sha256=$candidate_client_sha"
    echo "artifact_comparison=$artifact_comparison"
    echo "success_lock_sha256=$success_lock_sha"
    echo "candidate_lock_sha256=$candidate_lock_sha"
    echo "lockfile_comparison=$lockfile_comparison"
    echo "success_log_verdict=$success_log_verdict"
    echo "success_runtime_binding=$success_runtime_binding"
    echo "failed_log_verdict=$failed_log_verdict"
    echo "preserved_success_artifacts=$preserved_dir"
    if [[ "$success_runtime_binding" == SHA256_BOUND_PASS ]]; then
        echo "reproduction_confidence=ARTIFACT_SHA256_BOUND_TO_SUCCESS_LOGS"
    else
        echo "reproduction_confidence=INCOMPLETE_SUCCESS_LOGS_NOT_BOUND_TO_ARTIFACT_SHA256"
    fi
    echo "config_comparison=$config_comparison_file"
    echo "lockfile_diff=$lockfile_diff"
    echo "command_log_diff=$command_log_diff"
    echo "reproduction_recipe=$recipe"
    if [[ "$lockfile_comparison" != IDENTICAL && "$artifact_comparison" != IDENTICAL ]]; then
        echo "diagnosis=LOCKFILE_OR_UNRECORDED_BUILD_GRAPH_DIFFERENCE"
        echo "next_action=inspect lockfile-diff and command-log-diff, then bind the successful logs before rebuilding"
    else
        echo "diagnosis=NO_SINGLE_CAUSE_PROVEN"
        echo "next_action=bind the successful logs and inspect all recorded manifests before rebuilding"
    fi
} >"$summary"

cat "$summary"
echo "BRPC_UB_ECHO_PROVENANCE_AUDIT_OK output_dir=$OUTPUT_DIR"
