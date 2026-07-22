#!/usr/bin/env bash
set -euo pipefail

CANDIDATES="${CANDIDATES:-192,193,194,195,196,197,198,199,200}"
EXPECTED_OFFLOADS="${EXPECTED_OFFLOADS:-3}"
EXPECTED_ONBOARDS="${EXPECTED_ONBOARDS:-2}"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-scripts/benchmark_brpc_kvc_contention.sh}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/tmp/brpc-kvc-cache-shape/${RUN_ID}}"
SELECTED_ENV="${OUT_ROOT}/selected.env"
RESET_INFERENCE_BEFORE_ROUND="${RESET_INFERENCE_BEFORE_ROUND:-1}"
CONFIRM_REPEATS="${CONFIRM_REPEATS:-3}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[ -f "$BENCHMARK_SCRIPT" ] || die "benchmark script not found: $BENCHMARK_SCRIPT"
case "$RESET_INFERENCE_BEFORE_ROUND" in
  0|1) ;;
  *) die "RESET_INFERENCE_BEFORE_ROUND must be 0 or 1" ;;
esac
[[ "$CONFIRM_REPEATS" =~ ^[1-9][0-9]*$ ]] || die "CONFIRM_REPEATS must be positive"

mkdir -p "$OUT_ROOT"
rm -f "$SELECTED_ENV"

IFS=',' read -r -a candidate_values <<<"$CANDIDATES"
for candidate in "${candidate_values[@]}"; do
  candidate="${candidate//[[:space:]]/}"
  [[ "$candidate" =~ ^[1-9][0-9]*$ ]] || die "invalid PRIME_REQUESTS candidate: $candidate"
  candidate_dir="${OUT_ROOT}/prime-${candidate}"
  mkdir -p "$candidate_dir"

  printf '\n== calibrate PRIME_REQUESTS=%s ==\n' "$candidate"
  set +e
  env \
    MODE=baseline \
    REPEATS=1 \
    STRICT_COUNTS=1 \
    EXPECTED_OFFLOADS="$EXPECTED_OFFLOADS" \
    EXPECTED_ONBOARDS="$EXPECTED_ONBOARDS" \
    PRIME_REQUESTS="$candidate" \
    RESET_INFERENCE_BEFORE_ROUND="$RESET_INFERENCE_BEFORE_ROUND" \
    RUN_ID="${RUN_ID}-prime-${candidate}" \
    OUT_DIR="$candidate_dir" \
    bash "$BENCHMARK_SCRIPT" | tee "${candidate_dir}/console.log"
  code="${PIPESTATUS[0]}"
  set -e
  echo "$code" >"${candidate_dir}/exit_code"

  if [ "$code" -eq 0 ]; then
    confirm_dir="${candidate_dir}/confirm"
    mkdir -p "$confirm_dir"
    printf '\n== confirm PRIME_REQUESTS=%s repeats=%s ==\n' "$candidate" "$CONFIRM_REPEATS"
    set +e
    env \
      MODE=baseline \
      REPEATS="$CONFIRM_REPEATS" \
      STRICT_COUNTS=1 \
      EXPECTED_OFFLOADS="$EXPECTED_OFFLOADS" \
      EXPECTED_ONBOARDS="$EXPECTED_ONBOARDS" \
      PRIME_REQUESTS="$candidate" \
      RESET_INFERENCE_BEFORE_ROUND="$RESET_INFERENCE_BEFORE_ROUND" \
      RUN_ID="${RUN_ID}-prime-${candidate}-confirm" \
      OUT_DIR="$confirm_dir" \
      bash "$BENCHMARK_SCRIPT" | tee "${confirm_dir}/console.log"
    confirm_code="${PIPESTATUS[0]}"
    set -e
    echo "$confirm_code" >"${confirm_dir}/exit_code"
    if [ "$confirm_code" -ne 0 ]; then
      echo "cache_shape_confirmation=FAIL prime_requests=${candidate}" >&2
      continue
    fi

    cat >"$SELECTED_ENV" <<EOF
export PRIME_REQUESTS=${candidate}
export EXPECTED_OFFLOADS=${EXPECTED_OFFLOADS}
export EXPECTED_ONBOARDS=${EXPECTED_ONBOARDS}
export STRICT_COUNTS=1
export MATRIX_STRICT_COUNTS=1
export RESET_INFERENCE_BEFORE_ROUND=1
EOF
    echo "cache_shape=PASS prime_requests=${candidate} offloads=${EXPECTED_OFFLOADS} onboards=${EXPECTED_ONBOARDS} confirm_repeats=${CONFIRM_REPEATS}"
    echo "selected_env=${SELECTED_ENV}"
    exit 0
  fi
done

echo "cache_shape=FAIL candidates=${CANDIDATES}" >&2
echo "output_root=${OUT_ROOT}" >&2
exit 1
