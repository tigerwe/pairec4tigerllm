#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-pairec}
REQUESTS=${REQUESTS:-3}
BURST_POOL_SIZE=${BURST_POOL_SIZE:-10000}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/pairec-brpc-wrapper-kvc-matrix/$(date +%Y%m%d-%H%M%S)-n${REQUESTS}}

die() { echo "ERROR: $*" >&2; exit 1; }
[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "REQUESTS must be positive"
mkdir -p "$OUTPUT_DIR"

run_case() {
  local name=$1 wrapper_concurrency=$2 kvc_concurrency=$3 pressure_keys=$4 payload_bytes=$5
  echo "== Combined case=$name Wrapper=c${wrapper_concurrency} KVC=c${kvc_concurrency} =="
  NAMESPACE="$NAMESPACE" REQUESTS="$REQUESTS" \
    WRAPPER_CONCURRENCY="$wrapper_concurrency" \
    BURST_ACTIVE_CONNECTIONS="$wrapper_concurrency" \
    BURST_POOL_SIZE="$BURST_POOL_SIZE" BURST_PAYLOAD_BYTES="$payload_bytes" \
    KVC_CONCURRENCY="$kvc_concurrency" KVC_PRESSURE_KEY_COUNT="$pressure_keys" \
    OUTPUT_DIR="$OUTPUT_DIR/$name" \
    bash scripts/validate_pairec_brpc_wrapper_kvc_combined.sh \
    | tee "$OUTPUT_DIR/$name.console.log"
}

run_case baseline 1 1 0 0
run_case combined 1000 10 9 102400

python3 - "$OUTPUT_DIR/baseline/summary.json" "$OUTPUT_DIR/combined/summary.json" "$OUTPUT_DIR/summary.json" <<'PY'
import json, pathlib, sys
baseline = json.load(open(sys.argv[1]))
combined = json.load(open(sys.argv[2]))
names = (
    "client_e2e_ms", "pairec_total_ms", "vector_recall_ms", "generative_recall_ms",
    "deepfm_rank_ms", "rerank_ms", "front_brpc_ms", "wrapper_total_ms",
    "backend_brpc_ms", "runner_ms", "kvc_business_get_ms", "kvc_pressure_p99_ms",
    "kvc_barrier_ms", "datasystem_get_ms", "datasystem_set_ms",
    "wrapper_pressure_p95_ms", "wrapper_max_active",
)
rows = []
for name in names:
    left = baseline["metrics"][name]
    right = combined["metrics"][name]
    rows.append({"metric": name, "baseline_avg": left["avg"], "combined_avg": right["avg"],
                 "avg_delta": right["avg"] - left["avg"],
                 "baseline_p99": left["p99"], "combined_p99": right["p99"],
                 "p99_delta": right["p99"] - left["p99"]})
result = {
    "classification": "PAIREC_BRPC_WRAPPER_KVC_MATRIX_OK",
    "baseline": baseline,
    "combined": combined,
    "comparison": rows,
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + "\n")
print("metric baseline_avg combined_avg avg_delta baseline_p99 combined_p99 p99_delta")
for row in rows:
    print(f"{row['metric']} {row['baseline_avg']:.3f} {row['combined_avg']:.3f} "
          f"{row['avg_delta']:.3f} {row['baseline_p99']:.3f} "
          f"{row['combined_p99']:.3f} {row['p99_delta']:.3f}")
print(f"summary_json={sys.argv[3]}")
print(result["classification"])
PY

echo "output_dir=$OUTPUT_DIR"
