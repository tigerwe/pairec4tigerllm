#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-pairec}
REQUESTS=${REQUESTS:-3}
PRIME_REQUESTS=${PRIME_REQUESTS:-20}
BRPC_WRAPPER_CONCURRENCY=${BRPC_WRAPPER_CONCURRENCY:-1000}
BRPC_PRESSURE_PAYLOAD_BYTES=${BRPC_PRESSURE_PAYLOAD_BYTES:-102400}
BUSINESS_PAYLOAD_BYTES=${BUSINESS_PAYLOAD_BYTES:-102400}
COMBINED_KVC_CONCURRENCY=${COMBINED_KVC_CONCURRENCY:-32}
COMBINED_KVC_PRESSURE_KEY_COUNT=${COMBINED_KVC_PRESSURE_KEY_COUNT:-4}
COMBINED_KVC_OBJECT_SIZE=${COMBINED_KVC_OBJECT_SIZE:-3670016}
KVC_PRESSURE_LEAD_US=${KVC_PRESSURE_LEAD_US:-1000}
COMBINED_KVC_INPROCESS_PRESSURE=${COMBINED_KVC_INPROCESS_PRESSURE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/pairec-brpc-wrapper-kvc-factorial/$(date +%Y%m%d-%H%M%S)-n${REQUESTS}}

die() { echo "ERROR: $*" >&2; exit 1; }
[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "REQUESTS must be positive"
[[ "$PRIME_REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "PRIME_REQUESTS must be positive"
[[ "$BRPC_WRAPPER_CONCURRENCY" = 1000 ]] \
  || die "BRPC_WRAPPER_CONCURRENCY must be 1000"
[[ "$BRPC_PRESSURE_PAYLOAD_BYTES" = 102400 ]] \
  || die "factorial matrix requires BRPC_PRESSURE_PAYLOAD_BYTES=102400"
[[ "$BUSINESS_PAYLOAD_BYTES" = 102400 ]] \
  || die "factorial matrix requires BUSINESS_PAYLOAD_BYTES=102400"
[[ "$COMBINED_KVC_CONCURRENCY" = 32 ]] \
  || die "factorial in-process pressure requires COMBINED_KVC_CONCURRENCY=32"
[[ "$COMBINED_KVC_INPROCESS_PRESSURE" = 1 ]] \
  || die "factorial matrix requires COMBINED_KVC_INPROCESS_PRESSURE=1"
mkdir -p "$OUTPUT_DIR"

run_case() {
  local name=$1 wrapper_concurrency=$2 kvc_concurrency=$3 pressure_keys=$4 object_size=$5
  local onboard_min=$6 onboard_max=$7 inprocess_pressure=$8
  echo "== Factorial case=$name Wrapper=c${wrapper_concurrency} KVC=c${kvc_concurrency} =="
  NAMESPACE="$NAMESPACE" REQUESTS="$REQUESTS" PRIME_REQUESTS="$PRIME_REQUESTS" \
    WRAPPER_CONCURRENCY="$wrapper_concurrency" \
    BUSINESS_PAYLOAD_BYTES="$BUSINESS_PAYLOAD_BYTES" \
    BRPC_PRESSURE_PAYLOAD_BYTES="$BRPC_PRESSURE_PAYLOAD_BYTES" \
    KVC_CONCURRENCY="$kvc_concurrency" KVC_PRESSURE_KEY_COUNT="$pressure_keys" \
    KVC_OBJECT_SIZE="$object_size" KVC_PRESSURE_LEAD_US="$KVC_PRESSURE_LEAD_US" \
    KVC_INPROCESS_PRESSURE="$inprocess_pressure" \
    EXPECTED_ONBOARDS_MIN="$onboard_min" EXPECTED_ONBOARDS_MAX="$onboard_max" \
    OUTPUT_DIR="$OUTPUT_DIR/$name" \
    bash scripts/validate_pairec_brpc_wrapper_kvc_combined.sh \
    | tee "$OUTPUT_DIR/$name.console.log"
}

# 2x2 factorial: A=no pressure, B=Wrapper only, C=KVC only, D=both.
run_case baseline 1 1 0 3670016 2 2 0
run_case brpc_only "$BRPC_WRAPPER_CONCURRENCY" 1 0 3670016 \
  2 2 0
run_case kvc_only 1 "$COMBINED_KVC_CONCURRENCY" \
  "$COMBINED_KVC_PRESSURE_KEY_COUNT" "$COMBINED_KVC_OBJECT_SIZE" 2 2 \
  "$COMBINED_KVC_INPROCESS_PRESSURE"
run_case combined "$BRPC_WRAPPER_CONCURRENCY" "$COMBINED_KVC_CONCURRENCY" \
  "$COMBINED_KVC_PRESSURE_KEY_COUNT" "$COMBINED_KVC_OBJECT_SIZE" \
  2 2 "$COMBINED_KVC_INPROCESS_PRESSURE"

python3 - "$OUTPUT_DIR/baseline/summary.json" "$OUTPUT_DIR/brpc_only/summary.json" \
  "$OUTPUT_DIR/kvc_only/summary.json" "$OUTPUT_DIR/combined/summary.json" \
  "$OUTPUT_DIR/summary.json" "$BRPC_PRESSURE_PAYLOAD_BYTES" \
  "$BUSINESS_PAYLOAD_BYTES" <<'PY'
import json
import pathlib
import sys

baseline = json.load(open(sys.argv[1]))
brpc_only = json.load(open(sys.argv[2]))
kvc_only = json.load(open(sys.argv[3]))
combined = json.load(open(sys.argv[4]))
output = pathlib.Path(sys.argv[5])
brpc_payload_bytes = int(sys.argv[6])
business_payload_bytes = int(sys.argv[7])
cases = {
    "baseline": baseline,
    "brpc_only": brpc_only,
    "kvc_only": kvc_only,
    "combined": combined,
}

expected_shapes = {
    "baseline": (1, 1),
    "brpc_only": (1000, 1),
    "kvc_only": (1, 32),
    "combined": (1000, 32),
}
for name, (wrapper_concurrency, kvc_concurrency) in expected_shapes.items():
    case = cases[name]
    assert case["wrapper_concurrency"] == wrapper_concurrency, (name, case)
    assert case["kvc_concurrency"] == kvc_concurrency, (name, case)
    assert case["classification"].endswith("_OK"), (name, case["classification"])
    assert case["business_payload_bytes"] == business_payload_bytes, (name, case)
    assert case["brpc_pressure_payload_bytes"] == brpc_payload_bytes, (name, case)
    assert case["brpc_pressure_model"] == "synchronized_one_shot_burst", (name, case)
    assert case["embedded_burst_concurrency"] == wrapper_concurrency, (name, case)
    pressure_expected = name in {"brpc_only", "combined"}
    assert case["brpc_pressure_enabled"] is pressure_expected, (name, case)
    assert case["brpc_pressure_concurrency"] == (
        wrapper_concurrency - 1), (name, case)

names = (
    "client_e2e_ms", "client_e2e_adjusted_ms", "pairec_total_ms",
    "pairec_total_adjusted_ms",
    "vector_recall_ms", "generative_recall_ms", "deepfm_rank_ms", "rerank_ms",
    "front_brpc_ms", "wrapper_total_ms", "wrapper_total_adjusted_ms",
    "backend_brpc_ms", "runner_ms",
    "runner_adjusted_ms", "brpc_pressure_ready_wait_ms", "brpc_pressure_qps",
    "brpc_pressure_gbps", "brpc_pressure_max_active",
    "brpc_pressure_first_round_completed", "brpc_pressure_second_round_started",
    "brpc_pressure_active_at_ready", "wrapper_external_active_health_at_start",
    "wrapper_external_max_active_health", "wrapper_external_health_calls_at_start",
    "wrapper_external_health_calls_during_recommend",
    "wrapper_external_health_payload_bytes_during_recommend",
    "wrapper_external_health_calls_during_backend",
    "wrapper_external_health_payload_bytes_during_backend",
    "brpc_pressure_drain_ms", "brpc_pressure_drain_quiet_ms",
    "brpc_pressure_stop_signal_delay_ms", "kvc_business_get_ms",
    "kvc_business_get_1_ms",
    "kvc_business_get_2_ms", "kvc_pressure_p99_ms", "kvc_barrier_ms",
    "kvc_pressure_first_wait_ms", "kvc_pressure_lead_wait_ms",
    "kvc_coordination_wait_ms", "kvc_pressure_tail_after_stop_ms",
    "kvc_pressure_stop_signal_delay_ms", "kvc_pressure_stop_to_first_add_token_ms",
    "kvc_add_token_window_ms", "kvc_pressure_add_token_overlap_ms",
    "kvc_pressure_active_at_stop", "kvc_pressure_active_at_second_get_start",
    "kvc_pressure_completed_at_second_get_start", "kvc_pressure_completed_at_stop",
    "kvc_pressure_completed_gets", "kvc_pressure_completions_after_stop",
    "kvc_pressure_active_at_first_add_token", "kvc_pressure_active_at_last_add_token",
    "kvc_add_token_observation_count", "kvc_pressure_inflight_at_business_start",
    "datasystem_get_ms", "datasystem_set_ms", "native_add_token_ms",
    "native_kv_update_ms", "native_executor_queue_ms", "native_add_sequence_ms",
    "native_prefill_gap_ms", "native_decode_gap_ms", "native_finalization_gap_ms",
    "native_remove_sequence_ms", "native_lifecycle_ms", "native_accounted_ms",
    "wrapper_pressure_p95_ms", "wrapper_max_active",
)

rows = []
for metric in names:
    averages = {name: float(case["metrics"][metric]["avg"])
                for name, case in cases.items()}
    p99s = {name: float(case["metrics"][metric]["p99"])
            for name, case in cases.items()}
    rows.append({
        "metric": metric,
        "averages": averages,
        "p99s": p99s,
        "effects_avg": {
            "brpc": averages["brpc_only"] - averages["baseline"],
            "kvc": averages["kvc_only"] - averages["baseline"],
            "combined": averages["combined"] - averages["baseline"],
            "interaction": (averages["combined"] - averages["brpc_only"]
                            - averages["kvc_only"] + averages["baseline"]),
        },
        "effects_p99": {
            "brpc": p99s["brpc_only"] - p99s["baseline"],
            "kvc": p99s["kvc_only"] - p99s["baseline"],
            "combined": p99s["combined"] - p99s["baseline"],
            "interaction": (p99s["combined"] - p99s["brpc_only"]
                            - p99s["kvc_only"] + p99s["baseline"]),
        },
    })

result = {
    "classification": "PAIREC_BRPC_WRAPPER_KVC_FACTORIAL_OK",
    "design": "2x2_factorial",
    "pressure_profiles": {
        "brpc": {
            "model": "synchronized_one_shot_burst",
            "total_concurrency": 1000,
            "business_recommend_requests": 1,
            "pressure_health_requests": 999,
            "health_payload_bytes": brpc_payload_bytes,
            "business_payload_bytes": business_payload_bytes,
            "preconnected_session_pool": 10000,
            "selected_sessions": 1000,
            "release": "single_barrier",
            "coverage": "one_shot_concurrent_with_business_recommend",
            "stop_policy": "none_pressure_calls_finish_once",
            "backend_forwarded": False,
        },
        "kvc": {
            "concurrency": 32,
            "business_lanes": 1,
            "pressure_lanes": 31,
            "pressure_engine": "inprocess-shared-client",
            "object_size_bytes": combined["kvc_object_size_bytes"],
            "pressure_key_count": combined["kvc_pressure_key_count"],
            "stop_policy": "nonblocking_after_second_business_get",
        },
    },
    "cases": cases,
    # Preserve the two most commonly consumed top-level case names.
    "baseline": baseline,
    "combined": combined,
    "comparison": rows,
}
output.write_text(json.dumps(result, indent=2) + "\n")

print("metric baseline brpc_only kvc_only combined brpc_delta kvc_delta combined_delta interaction")
for row in rows:
    avg = row["averages"]
    effect = row["effects_avg"]
    print(f"{row['metric']} {avg['baseline']:.3f} {avg['brpc_only']:.3f} "
          f"{avg['kvc_only']:.3f} {avg['combined']:.3f} {effect['brpc']:.3f} "
          f"{effect['kvc']:.3f} {effect['combined']:.3f} {effect['interaction']:.3f}")
print(f"summary_json={output}")
print(result["classification"])
PY

echo "output_dir=$OUTPUT_DIR"
