#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PAIRS="${PAIRS:-3}"
REQUESTS="${REQUESTS:-1000}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-5}"
MAX_AVG_OVERHEAD_MS="${MAX_AVG_OVERHEAD_MS:-0.1}"
MAX_P99_OVERHEAD_MS="${MAX_P99_OVERHEAD_MS:-0.5}"
MAX_THROUGHPUT_LOSS_PCT="${MAX_THROUGHPUT_LOSS_PCT:-1.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/f19-attribution-ab/$(date +%Y%m%d-%H%M%S)-$$}"
OBSERVED_SCRIPT="${OBSERVED_SCRIPT:-$SCRIPT_DIR/deploy_and_validate_pairec_brpc_observed.sh}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-$SCRIPT_DIR/summarize_f19_attribution_ab.py}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ "$PAIRS" =~ ^[1-9][0-9]*$ ]] && (( PAIRS >= 3 )) \
  || die "PAIRS must be an integer >= 3"
[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "REQUESTS must be a positive integer"
[[ "$WARMUP_REQUESTS" =~ ^[1-9][0-9]*$ ]] \
  || die "WARMUP_REQUESTS must be a positive integer"
[[ "$COOLDOWN_SECONDS" =~ ^[0-9]+$ ]] || die "COOLDOWN_SECONDS must be non-negative"
test -x "$OBSERVED_SCRIPT" || die "observed validation script is not executable: $OBSERVED_SCRIPT"
test -f "$SUMMARY_SCRIPT" || die "missing A/B summary script: $SUMMARY_SCRIPT"
for command in bash grep python3 seq tee; do
  command -v "$command" >/dev/null || die "missing command: $command"
done

mkdir -p "$OUTPUT_ROOT"
echo "== F19 attribution A/B configuration =="
echo "repo_dir=$REPO_DIR"
echo "pairs=$PAIRS requests_per_run=$REQUESTS warmup_requests=$WARMUP_REQUESTS"
echo "order=disabled,enabled per pair"
echo "cache_preparation=inference rollout + identical warmup"
echo "max_avg_overhead_ms=$MAX_AVG_OVERHEAD_MS"
echo "max_p99_overhead_ms=$MAX_P99_OVERHEAD_MS"
echo "max_throughput_loss_pct=$MAX_THROUGHPUT_LOSS_PCT"
echo "output_root=$OUTPUT_ROOT"

cd "$REPO_DIR"
for pair in $(seq 1 "$PAIRS"); do
  for mode in disabled enabled; do
    if [[ "$mode" = enabled ]]; then
      require_attribution=1
    else
      require_attribution=0
    fi
    round_dir="$OUTPUT_ROOT/pair-${pair}-${mode}"
    mkdir -p "$round_dir"
    echo
    echo "== Pair $pair/$PAIRS mode=$mode =="
    if ! REQUIRE_DATASYSTEM_ATTRIBUTION="$require_attribution" \
      REQUESTS="$REQUESTS" \
      WARMUP_REQUESTS="$WARMUP_REQUESTS" \
      RUN_HTTP_AB=0 \
      BUILD_IMAGES=0 \
      IMPORT_IMAGES=0 \
      FORCE_INFERENCE_RESTART=1 \
      CLIENT_MAX_P99_MS=0 \
      OUTPUT_DIR="$round_dir" \
        bash "$OBSERVED_SCRIPT" 2>&1 | tee "$round_dir/run.log"; then
      die "A/B round failed: pair=$pair mode=$mode output=$round_dir"
    fi
    grep -Fq 'PAIREC_PURE_BRPC_OBSERVABILITY_OK' "$round_dir/run.log" \
      || die "success marker missing: pair=$pair mode=$mode"
    if [[ "$mode" = enabled ]]; then
      grep -Fq 'NATIVE_DATASYSTEM_COMPLETIONS_OK phase=workload' "$round_dir/run.log" \
        || die "native completion marker missing: pair=$pair mode=$mode"
    fi
    if (( COOLDOWN_SECONDS > 0 )); then
      sleep "$COOLDOWN_SECONDS"
    fi
  done
done

echo
echo "== F19 attribution paired summary =="
python3 "$SUMMARY_SCRIPT" \
  --input-root "$OUTPUT_ROOT" \
  --pairs "$PAIRS" \
  --expected-requests "$REQUESTS" \
  --max-avg-overhead-ms "$MAX_AVG_OVERHEAD_MS" \
  --max-p99-overhead-ms "$MAX_P99_OVERHEAD_MS" \
  --max-throughput-loss-pct "$MAX_THROUGHPUT_LOSS_PCT" \
  --output "$OUTPUT_ROOT/summary.json"
echo "summary_json=$OUTPUT_ROOT/summary.json"
echo "F19_ATTRIBUTION_AB_BENCHMARK_COMPLETE"
