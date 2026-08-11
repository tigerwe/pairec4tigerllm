#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
PAIREC_TARGET="${PAIREC_TARGET:-deploy/pairec-brpc-observed}"
BRPC_TARGET="${BRPC_TARGET:-deployment/inference-brpc-trtllm}"
PRIME_REQUESTS="${PRIME_REQUESTS:-195}"
REPEATS="${REPEATS:-3}"
EXPECTED_SETS="${EXPECTED_SETS:-3}"
EXPECTED_GETS="${EXPECTED_GETS:-2}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/f19-datasystem-real-io/${RUN_ID}}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

for value in "$PRIME_REQUESTS" "$REPEATS" "$EXPECTED_SETS" "$EXPECTED_GETS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "counts must be positive integers"
done

[[ -x scripts/deploy_f19_datasystem_attribution_overlay.sh ]] \
  || die "run from the repository root"
[[ -x scripts/benchmark_brpc_kvc_contention.sh ]] \
  || die "missing contention benchmark"

mkdir -p "$OUT_DIR"

echo "== Verify strict F19 runtime =="
NAMESPACE="$NAMESPACE" \
  bash scripts/deploy_f19_datasystem_attribution_overlay.sh verify \
  | tee "$OUT_DIR/runtime-verify.log"

echo "== Validate exact request-level DataSystem I/O =="
echo "cache_shape=reset + prime:${PRIME_REQUESTS} + replay"
echo "expected_exact_set_get=${EXPECTED_SETS}+${EXPECTED_GETS} repeats=${REPEATS}"

MODE=baseline \
REPEATS="$REPEATS" \
STRICT_COUNTS=1 \
EXPECTED_OFFLOADS="$EXPECTED_SETS" \
EXPECTED_ONBOARDS="$EXPECTED_GETS" \
REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION=1 \
PRIME_REQUESTS="$PRIME_REQUESTS" \
RESET_INFERENCE_BEFORE_ROUND=1 \
NAMESPACE="$NAMESPACE" \
PAIREC_TARGET="$PAIREC_TARGET" \
BRPC_TARGET="$BRPC_TARGET" \
OUT_DIR="$OUT_DIR/contention" \
  bash scripts/benchmark_brpc_kvc_contention.sh \
  | tee "$OUT_DIR/contention.log"

python3 - "$OUT_DIR/contention/result.json" "$REPEATS" "$EXPECTED_SETS" "$EXPECTED_GETS" <<'PY'
import json
import pathlib
import sys

path, repeats, expected_sets, expected_gets = sys.argv[1:]
repeats = int(repeats)
expected_sets = int(expected_sets)
expected_gets = int(expected_gets)
result = json.loads(pathlib.Path(path).read_text())
assert result["status"] == "PASS", result
assert result["valid_repeats"] == repeats, result
assert result["require_exact_datasystem_attribution"] is True, result
for row in result["rows"]:
    assert row["exact_attribution_ok"] is True, row
    assert row["exact_completion_count"] == 1, row
    assert row["exact_set_count"] == expected_sets, row
    assert row["exact_get_count"] == expected_gets, row
    assert row["exact_get_failed_count"] == 0, row
    assert row["exact_set_failed_count"] == 0, row
    assert row["exact_pending_count"] == 0, row
    assert row["exact_unknown_count"] == 0, row
print(
    "F19_DATASYSTEM_REAL_IO_OK "
    f"samples={repeats}/{repeats} exact_set_get={expected_sets}+{expected_gets}"
)
PY

echo "output_dir=$OUT_DIR"
