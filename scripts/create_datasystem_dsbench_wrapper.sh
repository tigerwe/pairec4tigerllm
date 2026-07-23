#!/usr/bin/env bash
set -euo pipefail

DATASYSTEM_DIR="${DATASYSTEM_DIR:-/home/zcx/yuanrong-datasystem-v081}"
BUILD_DIR="${BUILD_DIR:-${DATASYSTEM_DIR}/build-sustained}"
DSBENCH_CPP="${DSBENCH_CPP:-${BUILD_DIR}/dsbench/dsbench_cpp}"
WRAPPER_PATH="${WRAPPER_PATH:-/home/zcx/bin/dsbench-v081-sustained}"
EXPECTED_VERSION="${EXPECTED_VERSION:-0.8.1}"
LIB_PATH="${LIB_PATH:-}"
DEPENDENCY_CACHE="${DEPENDENCY_CACHE:-/home/zcx/.cache/yr-datasystem-opensource}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

discover_lib_path() {
  local -a roots=()
  local candidate
  for candidate in \
    "${DATASYSTEM_DIR}/output-sustained" \
    "$BUILD_DIR" \
    "$DEPENDENCY_CACHE"; do
    [ -d "$candidate" ] && roots+=("$candidate")
  done
  [ "${#roots[@]}" -gt 0 ] || die "no library search roots exist; set LIB_PATH explicitly"

  LIB_PATH="$(find -L "${roots[@]}" -type f \( -name '*.so' -o -name '*.so.*' \) \
    -printf '%h\n' 2>/dev/null | awk '!seen[$0]++' | paste -sd: -)"
  [ -n "$LIB_PATH" ] || die "no shared-library directories found; set LIB_PATH explicitly"
}

verify_binary() {
  [ -x "$DSBENCH_CPP" ] || die "dsbench_cpp is not executable: $DSBENCH_CPP"

  local ldd_output missing version_output help_output
  ldd_output="$(LD_LIBRARY_PATH="$LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    ldd "$DSBENCH_CPP" 2>&1)" || die "ldd failed for $DSBENCH_CPP: ${ldd_output}"
  missing="$(grep 'not found' <<<"$ldd_output" || true)"
  [ -z "$missing" ] || die "dsbench_cpp has unresolved dependencies:\n${missing}"

  version_output="$(LD_LIBRARY_PATH="$LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$DSBENCH_CPP" -v 2>&1)" || die "dsbench_cpp -v failed: ${version_output}"
  grep -Eq "Version:[[:space:]]+${EXPECTED_VERSION}([[:space:]]|$)" <<<"$version_output" \
    || die "expected DataSystem version ${EXPECTED_VERSION}, got:\n${version_output}"

  help_output="$(LD_LIBRARY_PATH="$LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$DSBENCH_CPP" kv --help 2>&1 || true)"
  grep -q -- '--duration_seconds' <<<"$help_output" \
    || die "dsbench_cpp does not expose --duration_seconds"
  grep -q -- '--ready_file' <<<"$help_output" \
    || die "dsbench_cpp does not expose --ready_file"
  grep -q -- '--prepared_file' <<<"$help_output" \
    || die "dsbench_cpp does not expose --prepared_file"
  grep -q -- '--start_file' <<<"$help_output" \
    || die "dsbench_cpp does not expose --start_file"
  grep -q -- '--stats_file' <<<"$help_output" \
    || die "dsbench_cpp does not expose --stats_file"

  printf '%s\n' "$version_output"
  grep -E -- '--duration_seconds|--ready_file|--prepared_file|--start_file|--stats_file' <<<"$help_output"
}

write_wrapper() {
  local wrapper_dir temporary
  wrapper_dir="$(dirname "$WRAPPER_PATH")"
  mkdir -p "$wrapper_dir"
  temporary="${WRAPPER_PATH}.tmp.$$"
  trap 'rm -f "${temporary:-}"' EXIT

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$LIB_PATH"
    printf 'exec %q "$@"\n' "$DSBENCH_CPP"
  } >"$temporary"
  chmod +x "$temporary"
  mv "$temporary" "$WRAPPER_PATH"
  trap - EXIT
}

if [ -z "$LIB_PATH" ]; then
  discover_lib_path
fi

echo "DataSystem dir: $DATASYSTEM_DIR"
echo "dsbench_cpp:    $DSBENCH_CPP"
echo "Wrapper:       $WRAPPER_PATH"
echo "Library dirs:  $(tr ':' '\n' <<<"$LIB_PATH" | wc -l)"

verify_binary
write_wrapper

echo
echo "Wrapper created and verified: $WRAPPER_PATH"
"$WRAPPER_PATH" -v
