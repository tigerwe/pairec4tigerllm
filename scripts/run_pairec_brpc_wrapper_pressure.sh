#!/usr/bin/env bash
set -euo pipefail

PHASE="${PHASE:-${1:-smoke}}"
case "$PHASE" in
  smoke) DEFAULT_REQUESTS=3 ;;
  stability) DEFAULT_REQUESTS=100 ;;
  formal) DEFAULT_REQUESTS=1000 ;;
  *) echo "ERROR: PHASE must be smoke, stability, or formal" >&2; exit 1 ;;
esac

REQUESTS="${REQUESTS:-$DEFAULT_REQUESTS}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pairec-brpc-wrapper-pressure/$(date +%Y%m%d-%H%M%S)-${PHASE}-c1000-n${REQUESTS}}"
MAX_RUNNER_P99_DELTA_MS="${MAX_RUNNER_P99_DELTA_MS:-10}"
MAX_FRONT_BRPC_P99_MS="${MAX_FRONT_BRPC_P99_MS:-10}"
BUILD_PAIREC_IMAGE="${BUILD_PAIREC_IMAGE:-1}"
IMPORT_PAIREC_IMAGE="${IMPORT_PAIREC_IMAGE:-1}"

echo "== PaiRec BRPC Wrapper pressure phase =="
echo "phase=$PHASE modes=c1,c1000 requests_per_mode=$REQUESTS warmup_requests=$WARMUP_REQUESTS"
echo "traffic_per_request=1xRecommend+999xHealth payload_bytes_per_health=102400"
echo "output_dir=$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

echo "== Run c1 baseline =="
BURST_CONCURRENCY=1 \
REQUESTS="$REQUESTS" \
WARMUP_REQUESTS="$WARMUP_REQUESTS" \
OUTPUT_DIR="$OUTPUT_DIR/c1" \
BUILD_PAIREC_IMAGE="$BUILD_PAIREC_IMAGE" \
IMPORT_PAIREC_IMAGE="$IMPORT_PAIREC_IMAGE" \
  bash scripts/deploy_and_validate_pairec_brpc_wrapper_full.sh

echo "== Run c1000 pressure =="
BURST_CONCURRENCY=1000 \
REQUESTS="$REQUESTS" \
WARMUP_REQUESTS="$WARMUP_REQUESTS" \
OUTPUT_DIR="$OUTPUT_DIR/c1000" \
BUILD_PAIREC_IMAGE=0 \
IMPORT_PAIREC_IMAGE=0 \
  bash scripts/deploy_and_validate_pairec_brpc_wrapper_full.sh

echo "== Compare c1 baseline with c1000 pressure =="
python3 - "$OUTPUT_DIR/c1/summary.json" "$OUTPUT_DIR/c1000/summary.json" \
  "$OUTPUT_DIR/summary.json" "$MAX_RUNNER_P99_DELTA_MS" \
  "$MAX_FRONT_BRPC_P99_MS" "$PHASE" <<'PY'
import json,pathlib,sys
baseline_path,pressure_path,output,max_runner_delta,max_front,phase=sys.argv[1:]
baseline=json.load(open(baseline_path)); pressure=json.load(open(pressure_path))
assert baseline["concurrency"]==1,baseline
assert pressure["concurrency"]==1000,pressure
assert len(baseline["samples"])==len(pressure["samples"]),(
    len(baseline["samples"]),len(pressure["samples"]))
bm=baseline["metrics"]; pm=pressure["metrics"]
result={
 "classification":"PAIREC_BRPC_WRAPPER_PRESSURE_OK",
 "phase":phase,
 "samples_per_mode":len(pressure["samples"]),
 "baseline":bm,
 "pressure":pm,
 "runner_p99_delta_ms":pm["runner_ms"]["p99"]-bm["runner_ms"]["p99"],
 "client_e2e_p99_delta_ms":pm["client_e2e_ms"]["p99"]-bm["client_e2e_ms"]["p99"],
 "front_brpc_p99_ms":pm["front_brpc_ms"]["p99"],
 "pressure_max_active_p50":pm["max_active_workers"]["p50"],
 "pressure_start_skew_p99_us":pm["start_skew_us"]["p99"],
 "pressure_health_requests":len(pressure["samples"])*999,
 "pressure_payload_bytes":len(pressure["samples"])*999*102400,
}
pathlib.Path(output).write_text(json.dumps(result,indent=2)+"\n")
print("metric c1_p99 c1000_p99 delta_ms")
for name in ("client_e2e_ms","front_brpc_ms","wrapper_total_ms","runner_ms"):
 left=bm[name]["p99"]; right=pm[name]["p99"]
 print(f"{name} {left:.3f} {right:.3f} {right-left:.3f}")
print(f"c1000 max_active_p50={result['pressure_max_active_p50']:.3f}")
print(f"c1000 start_skew_p99_us={result['pressure_start_skew_p99_us']:.3f}")
print(f"c1000 pressure_health_requests={result['pressure_health_requests']}")
print(f"c1000 pressure_payload_bytes={result['pressure_payload_bytes']}")
assert result["runner_p99_delta_ms"]<=float(max_runner_delta),(
    f"runner p99 regression {result['runner_p99_delta_ms']:.3f}ms exceeds {max_runner_delta}ms")
assert result["front_brpc_p99_ms"]<=float(max_front),(
    f"front BRPC p99 {result['front_brpc_p99_ms']:.3f}ms exceeds {max_front}ms")
PY

echo "PAIREC_BRPC_WRAPPER_PRESSURE_OK phase=$PHASE samples_per_mode=$REQUESTS"
echo "PAIREC_BRPC_WRAPPER_PRESSURE_${PHASE^^}_OK"
