#!/usr/bin/env bash
set -euo pipefail

DATASYSTEM_DIR="${DATASYSTEM_DIR:-/home/zcx/workspace/yuanrong-datasystem}"
PATCH_FILE="${PATCH_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/datasystem-dsbench-sustained-pressure.patch}"
OBSERVABILITY_PATCH_FILE="${OBSERVABILITY_PATCH_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/datasystem-dsbench-sustained-observability.patch}"
ARGS_CPP="${DATASYSTEM_DIR}/dsbench/src/kv/kv_args.cpp"
BENCH_CPP="${DATASYSTEM_DIR}/dsbench/src/kv/kv_bench.cpp"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[ -d "$DATASYSTEM_DIR" ] || die "DataSystem directory not found: $DATASYSTEM_DIR"
[ -f "$PATCH_FILE" ] || die "patch file not found: $PATCH_FILE"
[ -f "$OBSERVABILITY_PATCH_FILE" ] || die "observability patch file not found: $OBSERVABILITY_PATCH_FILE"
[ -f "$ARGS_CPP" ] || die "dsbench args source not found: $ARGS_CPP"
[ -f "$BENCH_CPP" ] || die "dsbench source not found: $BENCH_CPP"

echo "DataSystem dir: $DATASYSTEM_DIR"
echo "Patch file:     $PATCH_FILE"
echo "Observe patch:  $OBSERVABILITY_PATCH_FILE"

if ! grep -q 'duration_seconds' "$ARGS_CPP" \
  || ! grep -q 'args_\.durationSeconds' "$BENCH_CPP"; then
  if ! git -C "$DATASYSTEM_DIR" apply --recount --ignore-space-change --ignore-whitespace \
    --check "$PATCH_FILE"; then
    die "sustained dsbench patch does not apply cleanly; verify the DataSystem source version"
  fi
  git -C "$DATASYSTEM_DIR" apply --recount --ignore-space-change --ignore-whitespace "$PATCH_FILE"
else
  echo "dsbench sustained pressure patch already appears to be applied."
fi

if ! grep -q 'prepared_file' "$ARGS_CPP" \
  || ! grep -q 'max_inflight' "$BENCH_CPP" \
  || ! grep -q 'gReadyWorkers.store' "$BENCH_CPP"; then
  if ! git -C "$DATASYSTEM_DIR" apply --recount --ignore-space-change --ignore-whitespace \
    --check "$OBSERVABILITY_PATCH_FILE"; then
    die "dsbench observability patch does not apply cleanly; verify the sustained patch state"
  fi
  git -C "$DATASYSTEM_DIR" apply --recount --ignore-space-change --ignore-whitespace \
    "$OBSERVABILITY_PATCH_FILE"
else
  echo "dsbench sustained observability patch already appears to be applied."
fi

grep -nE 'duration_seconds|prepared_file|stats_file|max_inflight' "$ARGS_CPP" "$BENCH_CPP"
cat <<'EOF'

Patch applied. Rebuild the existing dsbench target with the repository's normal build command,
then verify:

  dsbench_cpp kv --help | grep -E 'duration_seconds|ready_file|prepared_file|start_file|stats_file'
EOF
