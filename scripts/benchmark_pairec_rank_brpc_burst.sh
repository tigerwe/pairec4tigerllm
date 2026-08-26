#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
REQUESTS="${REQUESTS:-1}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-0}"
BUILD_PAIREC_IMAGE="${BUILD_PAIREC_IMAGE:-1}"
IMPORT_PAIREC_IMAGE="${IMPORT_PAIREC_IMAGE:-1}"
DEPLOY_RANK_INFRA="${DEPLOY_RANK_INFRA:-1}"
RANK_BUSINESS_PAYLOAD_BYTES="${RANK_BUSINESS_PAYLOAD_BYTES:-102400}"
RANK_PRESSURE_PAYLOAD_BYTES="${RANK_PRESSURE_PAYLOAD_BYTES:-102400}"
RANK_TIMEOUT_MS="${RANK_TIMEOUT_MS:-1000}"
RANK_PRESSURE_TIMEOUT_MS="${RANK_PRESSURE_TIMEOUT_MS:-5000}"
COMPLETION_TIMEOUT_SECONDS="${COMPLETION_TIMEOUT_SECONDS:-30}"
PAIREC_DEPLOYMENT="pairec-brpc-observed-wrapper"
RANK_DEPLOYMENT="deepfm-rank-burst-wrapper"
INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-inference-brpc-trtllm}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pairec-rank-brpc-burst/$(date +%Y%m%d-%H%M%S)-n${REQUESTS}}"

die() { echo "ERROR: $*" >&2; exit 1; }
ready_pod() {
  kubectl -n "$NAMESPACE" get pods -l "app=$1" -o json | python3 -c '
import json,sys
pods=[]
for pod in json.load(sys.stdin).get("items",[]):
 status=pod.get("status",{}); cs=status.get("containerStatuses",[])
 if status.get("phase")=="Running" and cs and all(x.get("ready") for x in cs):
  pods.append((pod["metadata"].get("creationTimestamp",""),pod["metadata"]["name"]))
assert pods,"no ready pod for app="+sys.argv[1]
print(max(pods)[1])
' "$1"
}

[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || die "REQUESTS must be positive"
[[ "$WARMUP_REQUESTS" = 0 ]] \
  || die "Rank burst A/B requires WARMUP_REQUESTS=0 to prevent asynchronous warmup pressure tail from contaminating measured requests"
for flag in "$BUILD_PAIREC_IMAGE" "$IMPORT_PAIREC_IMAGE" "$DEPLOY_RANK_INFRA"; do
  [[ "$flag" = 0 || "$flag" = 1 ]] || die "boolean flags must be 0 or 1"
done
for command in kubectl python3; do command -v "$command" >/dev/null || die "missing command: $command"; done
mkdir -p "$OUTPUT_DIR"

if [[ "$DEPLOY_RANK_INFRA" = 1 ]]; then
  OUTPUT_DIR="$OUTPUT_DIR/infrastructure" bash scripts/deploy_deepfm_rank_burst_worker1.sh \
    | tee "$OUTPUT_DIR.infrastructure.log"
else
  test -s "$OUTPUT_DIR/infrastructure/cpu-isolation.json" \
    || die "DEPLOY_RANK_INFRA=0 requires existing $OUTPUT_DIR/infrastructure/cpu-isolation.json"
fi
CPU_ISOLATION_VALID="$(python3 -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["valid"])).lower())' \
  "$OUTPUT_DIR/infrastructure/cpu-isolation.json")"
echo "cpu_isolation_valid=$CPU_ISOLATION_VALID"
if [[ "$CPU_ISOLATION_VALID" != true ]]; then
  echo "WARNING: CPU isolation is not established; Rank latency results may include same-node CPU scheduling contention" >&2
fi

run_case() {
  local name="$1" concurrency="$2" build="$3" import_image="$4"
  local case_dir="$OUTPUT_DIR/$name" started pairec_pod rank_pod inference_pod deadline
  mkdir -p "$case_dir"
  started="$(date --iso-8601=seconds)"
  echo "== Rank BRPC case=$name concurrency=$concurrency =="
  env \
    OUTPUT_DIR="$case_dir" \
    REQUESTS="$REQUESTS" WARMUP_REQUESTS="$WARMUP_REQUESTS" \
    QUALIFICATION_REQUESTS=0 \
    BUILD_PAIREC_IMAGE="$build" IMPORT_PAIREC_IMAGE="$import_image" \
    BURST_CONCURRENCY=1 BURST_POOL_SIZE=1 BURST_ACTIVE_CONNECTIONS=1 \
    BUSINESS_PAYLOAD_BYTES=0 \
    RANK_DEPLOYMENT="$RANK_DEPLOYMENT" RANK_SERVICE="$RANK_DEPLOYMENT" RANK_PORT=18213 \
    RANK_TIMEOUT_MS="$RANK_TIMEOUT_MS" \
    RANK_BUSINESS_PAYLOAD_BYTES="$RANK_BUSINESS_PAYLOAD_BYTES" \
    RANK_BURST_ENABLED=1 RANK_BURST_CONCURRENCY="$concurrency" \
    RANK_BURST_POOL_SIZE="$concurrency" \
    RANK_BURST_PAYLOAD_BYTES="$RANK_PRESSURE_PAYLOAD_BYTES" \
    RANK_BURST_PRECONNECT=1 \
    RANK_BURST_PRESSURE_TIMEOUT_MS="$RANK_PRESSURE_TIMEOUT_MS" \
    bash scripts/deploy_and_validate_pairec_brpc_wrapper_full.sh \
    | tee "$case_dir/full-chain.console.log"

  pairec_pod="$(ready_pod "$PAIREC_DEPLOYMENT")"
  rank_pod="$(ready_pod "$RANK_DEPLOYMENT")"
  inference_pod="$(ready_pod "$INFERENCE_DEPLOYMENT")"
  deadline=$((SECONDS + COMPLETION_TIMEOUT_SECONDS))
  while true; do
    kubectl -n "$NAMESPACE" logs "$pairec_pod" -c pairec --since-time="$started" --timestamps \
      >"$case_dir/pairec-rank.log"
    kubectl -n "$NAMESPACE" logs "$rank_pod" -c rank-burst-wrapper --since-time="$started" --timestamps \
      >"$case_dir/rank-wrapper.log"
    kubectl -n "$NAMESPACE" logs "$inference_pod" -c brpc-inference --since-time="$started" --timestamps \
      >"$case_dir/inference.log"
    if python3 - "$case_dir/requests.tsv" "$case_dir/pairec-rank.log" <<'PY'
import csv,json,pathlib,sys
ids={row["request_id"] for row in csv.DictReader(open(sys.argv[1]),delimiter="\t")}
found=set()
for line in pathlib.Path(sys.argv[2]).read_text(errors="replace").splitlines():
 pos=line.find("{")
 if pos<0: continue
 try: event=json.loads(line[pos:])
 except json.JSONDecodeError: continue
 if event.get("event")=="pairec_rank_brpc_burst_complete": found.add(event.get("request_id"))
if not ids<=found:
 raise SystemExit(1)
PY
    then
      break
    fi
    (( SECONDS < deadline )) || die "timed out waiting for Rank pressure completion case=$name"
    sleep 0.2
  done
  python3 scripts/summarize_pairec_rank_brpc_burst.py \
    --case "$case_dir" --concurrency "$concurrency" \
    --business-bytes "$RANK_BUSINESS_PAYLOAD_BYTES" \
    --pressure-bytes "$RANK_PRESSURE_PAYLOAD_BYTES" \
    --output "$case_dir/rank-summary.json" \
    | tee "$case_dir/rank-summary.txt"
}

run_case baseline 1 "$BUILD_PAIREC_IMAGE" "$IMPORT_PAIREC_IMAGE"
run_case pressure 1000 0 0

python3 - "$OUTPUT_DIR/baseline/rank-summary.json" "$OUTPUT_DIR/pressure/rank-summary.json" \
  "$OUTPUT_DIR/summary.json" <<'PY'
import json,pathlib,sys
baseline,pressure=(json.load(open(path)) for path in sys.argv[1:3])
assert baseline["valid"] and pressure["valid"]
names=sorted(set(baseline["metrics"]) & set(pressure["metrics"]))
deltas={name:pressure["metrics"][name]["avg"]-baseline["metrics"][name]["avg"] for name in names}
result={"classification":"PAIREC_RANK_BRPC_BURST_AB_OK","valid":True,
        "requests_per_case":baseline["requests"],"baseline":baseline,
        "pressure":pressure,"average_deltas":deltas}
pathlib.Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+"\n")
print("metric baseline_avg pressure_avg delta")
for name in names:
 print(name,f'{baseline["metrics"][name]["avg"]:.3f}',
       f'{pressure["metrics"][name]["avg"]:.3f}',f'{deltas[name]:+.3f}')
print(result["classification"])
PY

echo "output_dir=$OUTPUT_DIR"
