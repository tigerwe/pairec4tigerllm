#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pairec-post-rank-two-hop-b-n3-$(date +%Y%m%d-%H%M%S)}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git rev-parse --short=12 HEAD)}"
HOP_BUILD_IMAGE="${HOP_BUILD_IMAGE:-docker.io/library/pairec-brpc-inference:post-rank-two-hop-${SOURCE_COMMIT}}"
HOP_RUNTIME_IMAGE="${HOP_RUNTIME_IMAGE:-docker.io/library/pairec-brpc-inference:k8s-arm64-v1}"
BASE_IMAGE="${BASE_IMAGE:-pairec-brpc-inference:k8s-arm64-v1}"
BUILD_HOP_IMAGE="${BUILD_HOP_IMAGE:-1}"
BUILD_PAIREC_IMAGE="${BUILD_PAIREC_IMAGE:-1}"
IMPORT_PAIREC_IMAGE="${IMPORT_PAIREC_IMAGE:-1}"
GENERATION_BURST_POOL_SIZE="${GENERATION_BURST_POOL_SIZE:-2000}"
PRIME_REQUESTS="${PRIME_REQUESTS:-20}"
MAX_E2E_MS="${MAX_E2E_MS:-550}"
MAX_POST_RANK_MS="${MAX_POST_RANK_MS:-120}"
MEASURED_REQUESTS="${MEASURED_REQUESTS:-3}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ "$GENERATION_BURST_POOL_SIZE" = 2000 ]] || die "B profile fixes generation pool_size=2000"
[[ "$BUILD_HOP_IMAGE" = 0 || "$BUILD_HOP_IMAGE" = 1 ]] || die "BUILD_HOP_IMAGE must be 0 or 1"
[[ "$MEASURED_REQUESTS" = 3 ]] || die "short post-rank validation fixes measured requests=3"
mkdir -p "$OUTPUT_DIR"

if [[ "$BUILD_HOP_IMAGE" = 1 ]]; then
  echo "== Build post-rank two-hop BRPC binary image =="
  BASE_IMAGE="$BASE_IMAGE" SOURCE_COMMIT="$SOURCE_COMMIT" \
    ENABLE_TRTLLM_CPP=OFF ENABLE_DATASYSTEM_KV_PROBE=OFF \
    bash scripts/build_brpc_inference_image.sh "$HOP_BUILD_IMAGE" \
      "$OUTPUT_DIR/post-rank-two-hop.tar" "$BASE_IMAGE" \
    | tee "$OUTPUT_DIR/hop-build.log"
fi

echo "== Deploy fixed master-local Hop-2 then Hop-1 =="
NAMESPACE="$NAMESPACE" SOURCE_COMMIT="$SOURCE_COMMIT" \
  BUILD_IMAGE="$HOP_BUILD_IMAGE" RUNTIME_IMAGE="$HOP_RUNTIME_IMAGE" \
  OUTPUT_DIR="$OUTPUT_DIR/post-rank-infrastructure" \
  bash scripts/deploy_post_rank_two_hop.sh | tee "$OUTPUT_DIR/post-rank-deploy.log"

echo "== Deploy Rank BRPC c1000 + Rank KVC c32 x 8MiB =="
NAMESPACE="$NAMESPACE" RANK_KVC_CONCURRENCY=32 RANK_KVC_PRESSURE_KEY_COUNT=4 \
  OUTPUT_DIR="$OUTPUT_DIR/rank-infrastructure" \
  bash scripts/deploy_deepfm_rank_burst_worker1.sh | tee "$OUTPUT_DIR/rank-infrastructure.log"

echo "== One excluded B warmup, strict drain, then measured B n3 =="
env \
  NAMESPACE="$NAMESPACE" REQUESTS="$MEASURED_REQUESTS" WARMUP_REQUESTS=1 PRIME_REQUESTS="$PRIME_REQUESTS" \
  OUTPUT_DIR="$OUTPUT_DIR/run" BUILD_PAIREC_IMAGE="$BUILD_PAIREC_IMAGE" \
  IMPORT_PAIREC_IMAGE="$IMPORT_PAIREC_IMAGE" \
  WRAPPER_CONCURRENCY=1000 BURST_POOL_SIZE="$GENERATION_BURST_POOL_SIZE" \
  BURST_ACTIVE_CONNECTIONS=1000 BUSINESS_PAYLOAD_BYTES=102400 \
  BRPC_PRESSURE_PAYLOAD_BYTES=102400 \
  KVC_CONCURRENCY=32 KVC_PRESSURE_KEY_COUNT=4 KVC_OBJECT_SIZE=3670016 \
  KVC_PRESSURE_LEAD_US=1000 KVC_INPROCESS_PRESSURE=1 \
  EXPECTED_ONBOARDS_MIN=2 EXPECTED_ONBOARDS_MAX=2 \
  RANK_DEPLOYMENT=deepfm-rank-burst-wrapper RANK_SERVICE=deepfm-rank-burst-wrapper \
  RANK_PORT=18213 RANK_ENDPOINT_OVERRIDE=192.168.100.11:18213 RANK_TIMEOUT_MS=1500 \
  RANK_BURST_ENABLED=1 RANK_BURST_CONCURRENCY=1000 RANK_BURST_POOL_SIZE=1000 \
  RANK_BUSINESS_PAYLOAD_BYTES=102400 RANK_BURST_PAYLOAD_BYTES=102400 \
  RANK_BURST_PRESSURE_TIMEOUT_MS=5000 \
  RANK_KVC_ENABLED=1 RANK_KVC_CONCURRENCY=32 RANK_KVC_OBJECT_SIZE=8388608 \
  POST_RANK_HOPS_ENABLED=1 POST_RANK_HOP1_ENDPOINT=192.168.100.12:18311 \
  POST_RANK_TIMEOUT_MS=1500 POST_RANK_BURST_CONCURRENCY=1000 \
  POST_RANK_BURST_POOL_SIZE=1000 POST_RANK_PAYLOAD_BYTES=102400 \
  POST_RANK_PRESSURE_TIMEOUT_MS=5000 E2E_TIMEOUT_MS=5000 \
  POST_RANK_PRESSURE_START_QUORUM=950 POST_RANK_PRESSURE_START_TIMEOUT_MS=250 \
  bash scripts/validate_pairec_brpc_wrapper_kvc_combined.sh \
  | tee "$OUTPUT_DIR/run.console.log"

python3 - "$OUTPUT_DIR/run/summary.json" "$OUTPUT_DIR/run/post-rank-pairec.log" \
  "$OUTPUT_DIR/run/post-rank-hop1.log" "$OUTPUT_DIR/result.json" \
  "$MAX_E2E_MS" "$MAX_POST_RANK_MS" <<'PY'
import json,pathlib,sys
summary=json.load(open(sys.argv[1])); samples=summary["samples"]
assert len(samples)==3,len(samples)
def events(path):
    result=[]
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        pos=line.find("{")
        if pos<0: continue
        try: item=json.loads(line[pos:])
        except json.JSONDecodeError: continue
        if item.get("request_id"): result.append(item)
    return result
pairec=events(sys.argv[2]); hop1=events(sys.argv[3])
def one(source,name,rid):
    items=[item for item in source if item.get("event")==name and item.get("request_id")==rid]
    assert len(items)==1,(name,items)
    return items[0]
rows=[]
for sample in samples:
    rid=sample["request_id"]
    chain=one(pairec,"post_rank_two_hop_complete",rid)
    outer=one(pairec,"pairec_post_rank_hop1_brpc_burst_business_complete",rid)
    inner=one(hop1,"pairec_post_rank_hop2_brpc_burst_business_complete",rid)
    outer_complete=one(pairec,"pairec_post_rank_hop1_brpc_burst_complete",rid)
    inner_complete=one(hop1,"pairec_post_rank_hop2_brpc_burst_complete",rid)
    row={
      "request_id":rid, "client_e2e_ms":sample["client_e2e_ms"],
      "pairec_total_ms":sample["pairec_total_ms"],
      "generative_recall_ms":sample["generative_recall_ms"],
      "deepfm_rank_ms":sample["deepfm_rank_ms"],
      "post_rank_two_hop_ms":chain["client_total_ms"],
      "post_rank_hop1_business_ms":outer["business_client_wall_ms"],
      "post_rank_hop1_front_brpc_ms":outer["front_brpc_estimate_ms"],
      "post_rank_hop1_marker_wait_ms":outer["marker_wait_ms"],
      "post_rank_hop2_front_brpc_ms":inner["front_brpc_estimate_ms"],
      "post_rank_hop2_marker_wait_ms":inner["marker_wait_ms"],
      "post_rank_hop1_pressure_success":outer_complete["pressure_success"],
      "post_rank_hop2_pressure_success":inner_complete["pressure_success"],
      "candidate_count":chain["candidate_count"], "candidate_sha256":chain["candidate_sha256"],
    }
    assert row["candidate_count"]==50,row
    assert outer["pressure_started_at_business_start"]>=950,outer
    assert inner["pressure_started_at_business_start"]>=950,inner
    assert outer_complete["burst_valid"] is True and outer_complete["pressure_errors"]==0,outer_complete
    assert inner_complete["burst_valid"] is True and inner_complete["pressure_errors"]==0,inner_complete
    assert outer_complete["pressure_overlap_business"] > 0,outer_complete
    assert inner_complete["pressure_overlap_business"] > 0,inner_complete
    assert row["client_e2e_ms"] <= float(sys.argv[5]),row
    assert row["post_rank_two_hop_ms"] <= float(sys.argv[6]),row
    rows.append(row)
def metric(name):
    values=sorted(float(row[name]) for row in rows)
    return {"avg_ms":sum(values)/len(values),"p50_ms":values[1],"p95_ms":values[-1],"max_ms":values[-1]}
result={"classification":"PAIREC_POST_RANK_TWO_HOP_B_N3_OK","requests":3,
        "metrics":{"client_e2e":metric("client_e2e_ms"),
                   "post_rank_two_hop":metric("post_rank_two_hop_ms"),
                   "post_rank_hop1_front_brpc":metric("post_rank_hop1_front_brpc_ms"),
                   "post_rank_hop2_front_brpc":metric("post_rank_hop2_front_brpc_ms"),
                   "post_rank_hop1_marker_wait":metric("post_rank_hop1_marker_wait_ms"),
                   "post_rank_hop2_marker_wait":metric("post_rank_hop2_marker_wait_ms")},
        "samples":rows}
pathlib.Path(sys.argv[4]).write_text(json.dumps(result,indent=2)+"\n")
print(f"classification={result['classification']}")
print(f"requests={result['requests']}")
for name,value in result["metrics"].items(): print(f"{name}={value}")
PY

echo "PAIREC_POST_RANK_TWO_HOP_B_N3_OK"
echo "output_dir=$OUTPUT_DIR"
