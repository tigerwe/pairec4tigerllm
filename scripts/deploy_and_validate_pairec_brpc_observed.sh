#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
INFERENCE_SERVICE="${INFERENCE_SERVICE:-inference-brpc-trtllm}"
INFERENCE_PORT="${INFERENCE_PORT:-18100}"
MILVUS_SERVICE="${MILVUS_SERVICE:-milvus-standalone}"
DEEPFM_MODEL_ROLE="${DEEPFM_MODEL_ROLE:-engineering}"
REQUESTS="${REQUESTS:-1000}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
USER_ID="${USER_ID:-1}"
SCENE_ID="${SCENE_ID:-home_feed}"
SIZE="${SIZE:-10}"
BUILD_IMAGES="${BUILD_IMAGES:-1}"
IMPORT_IMAGES="${IMPORT_IMAGES:-1}"
RUN_HTTP_AB="${RUN_HTTP_AB:-1}"
HTTP_BASELINE_SERVICE="${HTTP_BASELINE_SERVICE:-pairec-multi-recall-rank}"
HTTP_BASELINE_REQUESTS="${HTTP_BASELINE_REQUESTS:-100}"
REQUIRE_DATASYSTEM_ATTRIBUTION="${REQUIRE_DATASYSTEM_ATTRIBUTION:-0}"
PAIREC_IMAGE="${PAIREC_IMAGE:-docker.io/library/pairec-server:k8s-arm64-brpc-v1}"
ADAPTER_IMAGE="${ADAPTER_IMAGE:-docker.io/library/pairec-brpc-inference:k8s-arm64-v1}"
ADAPTER_BASE_IMAGE="${ADAPTER_BASE_IMAGE:-zcx-pairec-brpc-sdk:v1}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-configs/pairec_config.brpc_observed.json}"
ADAPTER_MANIFEST="${ADAPTER_MANIFEST:-k8s/deployment-pipeline-brpc-adapters.yaml}"
PAIREC_MANIFEST="${PAIREC_MANIFEST:-k8s/deployment-pairec-brpc-observed.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pairec-brpc-observed/$(date +%Y%m%d-%H%M%S)}"
PYMILVUS_RUNTIME_DIR="${PYMILVUS_RUNTIME_DIR:-/home/zcx/pairec-python-runtime}"

die() { echo "ERROR: $*" >&2; exit 1; }
for value in "$REQUESTS" "$HTTP_BASELINE_REQUESTS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "request counts must be positive integers"
done
[[ "$WARMUP_REQUESTS" =~ ^[0-9]+$ ]] || die "WARMUP_REQUESTS must be non-negative"
for value in "$BUILD_IMAGES" "$IMPORT_IMAGES" "$RUN_HTTP_AB" "$REQUIRE_DATASYSTEM_ATTRIBUTION"; do
  [[ "$value" = 0 || "$value" = 1 ]] || die "boolean flags must be 0 or 1"
done
for command in kubectl python3 curl; do
  command -v "$command" >/dev/null || die "missing command: $command"
done
test -f "$CONFIG_TEMPLATE" || die "missing config template"
test -f "$ADAPTER_MANIFEST" || die "missing adapter manifest"
test -f "$PAIREC_MANIFEST" || die "missing PaiRec manifest"
test -d "$PYMILVUS_RUNTIME_DIR/pymilvus" || \
  die "missing injected pymilvus runtime: $PYMILVUS_RUNTIME_DIR (run scripts/export_pymilvus_runtime_from_container.sh)"
mkdir -p "$OUTPUT_DIR"

INFERENCE_IP="$(kubectl -n "$NAMESPACE" get service "$INFERENCE_SERVICE" -o jsonpath='{.spec.clusterIP}')"
MILVUS_IP="$(kubectl -n "$NAMESPACE" get service "$MILVUS_SERVICE" -o jsonpath='{.spec.clusterIP}')"
[[ -n "$INFERENCE_IP" && -n "$MILVUS_IP" ]] || die "dependency ClusterIP is empty"
INFERENCE_ENDPOINT="${INFERENCE_IP}:${INFERENCE_PORT}"

echo "== Build and import code images =="
if [[ "$BUILD_IMAGES" = 1 ]]; then
  BASE_IMAGE="$ADAPTER_BASE_IMAGE" \
    bash scripts/build_brpc_gateway_image.sh "$ADAPTER_IMAGE" "$OUTPUT_DIR/adapter.tar" "$ADAPTER_BASE_IMAGE"
  bash scripts/build_pairec_binary_image.sh "$PAIREC_IMAGE"
fi
if [[ "$IMPORT_IMAGES" = 1 ]]; then
  for image in "$ADAPTER_IMAGE" "$PAIREC_IMAGE"; do
    docker image inspect "$image" >/dev/null || die "Docker image missing: $image"
    docker save "$image" | ctr -n k8s.io images import -
  done
fi

echo "== Render and deploy BRPC adapters =="
python3 - "$ADAPTER_MANIFEST" "$OUTPUT_DIR/adapters.yaml" "$MILVUS_IP" \
  "$ADAPTER_IMAGE" "$DEEPFM_MODEL_ROLE" "$PYMILVUS_RUNTIME_DIR" <<'PY'
import pathlib, sys
source, target, milvus_ip, adapter_image, model_role, pymilvus_runtime = sys.argv[1:]
text = pathlib.Path(source).read_text()
text = text.replace("__MILVUS_IP__", milvus_ip)
text = text.replace("__PYMILVUS_RUNTIME_DIR__", pymilvus_runtime)
text = text.replace("docker.io/library/pairec-brpc-inference:k8s-arm64-v1", adapter_image)
text = text.replace("{name: DEEPFM_MODEL_ROLE, value: engineering}",
                    "{name: DEEPFM_MODEL_ROLE, value: " + model_role + "}")
assert "__" not in text
pathlib.Path(target).write_text(text)
PY
kubectl apply -f "$OUTPUT_DIR/adapters.yaml"
for deployment in vector-recall-brpc deepfm-rank-brpc; do
  kubectl -n "$NAMESPACE" rollout status "deployment/$deployment" --timeout=10m
done
VECTOR_IP="$(kubectl -n "$NAMESPACE" get service vector-recall-brpc -o jsonpath='{.spec.clusterIP}')"
RANK_IP="$(kubectl -n "$NAMESPACE" get service deepfm-rank-brpc -o jsonpath='{.spec.clusterIP}')"
[[ -n "$VECTOR_IP" && -n "$RANK_IP" ]] || die "adapter ClusterIP is empty"

echo "== Verify adapter contracts =="
for specification in "vector-recall-brpc:vector:18201" "deepfm-rank-brpc:rank:18211"; do
  IFS=: read -r deployment service port <<<"$specification"
  pod="$(kubectl -n "$NAMESPACE" get pod -l "app=$deployment" -o jsonpath='{.items[0].metadata.name}')"
  kubectl -n "$NAMESPACE" exec "$pod" -c adapter -- \
    /opt/pairec-brpc/bin/brpc_pipeline_client \
      "--server=127.0.0.1:${port}" "--service=${service}" --timeout_ms=1000
done

echo "== Render and deploy isolated PaiRec =="
python3 - "$CONFIG_TEMPLATE" "$OUTPUT_DIR/pairec_config.json" \
  "$INFERENCE_ENDPOINT" "$DEEPFM_MODEL_ROLE" \
  "${VECTOR_IP}:18201" "${RANK_IP}:18211" <<'PY'
import json, pathlib, sys
source, target, inference, role, vector_endpoint, rank_endpoint = sys.argv[1:]
text = pathlib.Path(source).read_text()
text = text.replace("__INFERENCE_ENDPOINT__", inference)
text = text.replace("__DEEPFM_MODEL_ROLE__", role)
text = text.replace("__VECTOR_ENDPOINT__", vector_endpoint)
text = text.replace("__RANK_ENDPOINT__", rank_endpoint)
assert "__" not in text
config = json.loads(text)
recalls = {entry["Name"]: json.loads(entry["RecallAlgo"])
           for entry in config["RecallConfs"]}
assert recalls["generative_recall"]["protocol"] == "brpc"
assert recalls["generative_recall"]["brpc_fallback_to_http"] is False
assert recalls["generative_recall"]["max_retries"] == 0
assert recalls["milvus_recall"]["protocol"] == "brpc"
assert recalls["milvus_recall"]["brpc_endpoint"] == vector_endpoint
rank = config["UserDefineConfs"]["DeepFMRankSorts"][0]
assert rank["protocol"] == "brpc" and "server_url" not in rank
assert rank["brpc_endpoint"] == rank_endpoint
pathlib.Path(target).write_text(json.dumps(config, indent=2) + "\n")
print("PAIREC_PURE_BRPC_CONFIG_OK")
PY
kubectl -n "$NAMESPACE" create configmap pairec-config-brpc-observed \
  --from-file="pairec_config.json=$OUTPUT_DIR/pairec_config.json" \
  --dry-run=client -o yaml | kubectl apply -f -
python3 - "$PAIREC_MANIFEST" "$OUTPUT_DIR/pairec.yaml" \
  "$INFERENCE_IP" "$INFERENCE_PORT" "$VECTOR_IP" "$RANK_IP" <<'PY'
import pathlib, sys
source, target, host, port, vector_host, rank_host = sys.argv[1:]
text = pathlib.Path(source).read_text()
text = text.replace("__INFERENCE_HOST__", host).replace("__INFERENCE_PORT__", port)
text = text.replace("__VECTOR_HOST__", vector_host).replace("__RANK_HOST__", rank_host)
assert "__" not in text
pathlib.Path(target).write_text(text)
PY
kubectl apply -f "$OUTPUT_DIR/pairec.yaml"
kubectl -n "$NAMESPACE" set image deployment/pairec-brpc-observed "pairec=$PAIREC_IMAGE"
kubectl -n "$NAMESPACE" set env deployment/pairec-brpc-observed \
  "PAIREC_REQUIRE_DATASYSTEM_ATTRIBUTION=$REQUIRE_DATASYSTEM_ATTRIBUTION"
kubectl -n "$NAMESPACE" rollout restart deployment/pairec-brpc-observed
kubectl -n "$NAMESPACE" rollout status deployment/pairec-brpc-observed --timeout=5m

PAIREC_POD="$(kubectl -n "$NAMESPACE" get pod -l app=pairec-brpc-observed -o jsonpath='{.items[0].metadata.name}')"
PAIREC_IP="$(kubectl -n "$NAMESPACE" get service pairec-brpc-observed -o jsonpath='{.spec.clusterIP}')"
PAIREC_URL="http://${PAIREC_IP}:18080/api/recommend"

collect_cpu_stat() {
  local deployment="$1" container="$2" output="$3"
  local pod
  pod="$(kubectl -n "$NAMESPACE" get pod -l "app=$deployment" -o jsonpath='{.items[0].metadata.name}')"
  kubectl -n "$NAMESPACE" exec "$pod" -c "$container" -- /bin/sh -ec \
    'if test -f /sys/fs/cgroup/cpu.stat; then cat /sys/fs/cgroup/cpu.stat; else cat /sys/fs/cgroup/cpu/cpu.stat; fi' \
    >"$output"
}

RESOURCE_TARGETS=(
  "pairec-brpc-observed:pairec"
  "vector-recall-brpc:adapter"
  "vector-recall-brpc:backend"
  "deepfm-rank-brpc:adapter"
  "deepfm-rank-brpc:backend"
)
for tuple in "${RESOURCE_TARGETS[@]}"; do
  IFS=: read -r deployment container <<<"$tuple"
  collect_cpu_stat "$deployment" "$container" "$OUTPUT_DIR/${deployment}-${container}.cpu.before"
done

run_requests() {
  local url="$1" count="$2" directory="$3"
  mkdir -p "$directory"
  printf 'index\te2e_ms\trequest_id\n' >"$directory/requests.tsv"
  for index in $(seq 1 "$count"); do
    response="$directory/response-${index}.json"
    seconds="$(curl --noproxy '*' -sS --connect-timeout 2 --max-time 10 \
      "$url" -H 'Content-Type: application/json' \
      -d "{\"scene_id\":\"$SCENE_ID\",\"uid\":\"$USER_ID\",\"size\":$SIZE}" \
      -o "$response" -w '%{time_total}')"
    read -r request_id item_count < <(python3 - "$response" "$SIZE" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
size = int(sys.argv[2])
assert data.get("code") == 200, data
items = data.get("items", [])
assert len(items) == size and len({item["item_id"] for item in items}) == size, data
assert any(item.get("retrieve_id") == "generative_recall" for item in items), data
print(data["request_id"], len(items))
PY
)
    e2e_ms="$(python3 -c 'import sys; print(round(float(sys.argv[1])*1000, 3))' "$seconds")"
    printf '%s\t%s\t%s\n' "$index" "$e2e_ms" "$request_id" >>"$directory/requests.tsv"
  done
}

echo "== Warm up PaiRec =="
if (( WARMUP_REQUESTS > 0 )); then
  run_requests "$PAIREC_URL" "$WARMUP_REQUESTS" "$OUTPUT_DIR/warmup"
fi

echo "== Run pure BRPC observed workload: $REQUESTS requests =="
since="$(date --iso-8601=seconds)"
run_requests "$PAIREC_URL" "$REQUESTS" "$OUTPUT_DIR/brpc"
kubectl -n "$NAMESPACE" logs "$PAIREC_POD" --since-time="$since" >"$OUTPUT_DIR/brpc/pairec.log"

summary_args=(
  --log "$OUTPUT_DIR/brpc/pairec.log"
  --requests-tsv "$OUTPUT_DIR/brpc/requests.tsv"
  --expected "$REQUESTS"
  --output "$OUTPUT_DIR/brpc/summary.json"
)
if [[ "$REQUIRE_DATASYSTEM_ATTRIBUTION" = 1 ]]; then
  summary_args+=(--require-datasystem-attribution)
fi
python3 scripts/summarize_pairec_pipeline_trace.py "${summary_args[@]}"

echo "== Verify metrics and absence of fallback =="
curl --noproxy '*' -fsS "http://${PAIREC_IP}:18080/metrics" >"$OUTPUT_DIR/metrics.txt"
grep -q 'pairec_pipeline_span_duration_seconds' "$OUTPUT_DIR/metrics.txt"
grep -q 'pairec_pipeline_trace_total' "$OUTPUT_DIR/metrics.txt"
grep -q 'pairec_pipeline_service_phase_duration_seconds' "$OUTPUT_DIR/metrics.txt"
! grep -Eqi 'fallback.to.http|protocol=http|http fallback' "$OUTPUT_DIR/brpc/pairec.log" \
  || die "HTTP fallback marker detected"

if [[ "$RUN_HTTP_AB" = 1 ]]; then
  HTTP_IP="$(kubectl -n "$NAMESPACE" get service "$HTTP_BASELINE_SERVICE" -o jsonpath='{.spec.clusterIP}')"
  echo "== Run retained HTTP baseline: $HTTP_BASELINE_REQUESTS requests =="
  run_requests "http://${HTTP_IP}:18080/api/recommend" "$HTTP_BASELINE_REQUESTS" "$OUTPUT_DIR/http"
  python3 - "$OUTPUT_DIR/http/requests.tsv" "$OUTPUT_DIR/brpc/requests.tsv" "$OUTPUT_DIR/ab.json" <<'PY'
import csv, json, math, statistics, sys
def values(path):
    return [float(row["e2e_ms"]) for row in csv.DictReader(open(path), delimiter="\t")]
def pct(data, q):
    data = sorted(data); pos = (len(data)-1)*q; lo = math.floor(pos); hi = math.ceil(pos)
    return data[lo] if lo == hi else data[lo]*(hi-pos)+data[hi]*(pos-lo)
http, brpc = values(sys.argv[1]), values(sys.argv[2])
result = {"http": {"samples": len(http), "p50_ms": pct(http,.5), "p95_ms": pct(http,.95), "p99_ms": pct(http,.99)},
          "brpc": {"samples": len(brpc), "p50_ms": pct(brpc,.5), "p95_ms": pct(brpc,.95), "p99_ms": pct(brpc,.99)}}
result["brpc_minus_http_p99_ms"] = result["brpc"]["p99_ms"] - result["http"]["p99_ms"]
open(sys.argv[3], "w").write(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
fi

echo "== Resource gates =="
for tuple in "${RESOURCE_TARGETS[@]}"; do
  IFS=: read -r deployment container <<<"$tuple"
  collect_cpu_stat "$deployment" "$container" "$OUTPUT_DIR/${deployment}-${container}.cpu.after"
  python3 - "$OUTPUT_DIR/${deployment}-${container}.cpu.before" \
    "$OUTPUT_DIR/${deployment}-${container}.cpu.after" "$deployment/$container" <<'PY'
import pathlib, sys
def parse(path):
    result = {}
    for line in pathlib.Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) == 2:
            result[fields[0]] = int(fields[1])
    return result
before, after = parse(sys.argv[1]), parse(sys.argv[2])
periods = after.get("nr_periods", 0) - before.get("nr_periods", 0)
throttled = after.get("nr_throttled", 0) - before.get("nr_throttled", 0)
ratio = 100.0 * throttled / max(periods, 1)
print(f"resource deployment={sys.argv[3]} periods={periods} throttled={throttled} throttled_period_pct={ratio:.3f}")
assert ratio <= 5.0, f"CPU throttled period gate failed: {ratio:.3f}%"
PY
done
for deployment in pairec-brpc-observed vector-recall-brpc deepfm-rank-brpc; do
  pod="$(kubectl -n "$NAMESPACE" get pod -l "app=$deployment" -o jsonpath='{.items[0].metadata.name}')"
  restarts="$(kubectl -n "$NAMESPACE" get pod "$pod" -o jsonpath='{.status.containerStatuses[*].restartCount}')"
  [[ "$restarts" =~ ^(0[[:space:]]*)+$ ]] || die "$deployment restart count is not zero: $restarts"
  if kubectl -n "$NAMESPACE" get events \
    --field-selector "involvedObject.kind=Pod,involvedObject.name=$pod" | \
    grep -Eq 'OOMKilled|BackOff|Unhealthy'; then
    die "resource or health failure event detected for $pod"
  fi
done

echo "== Summary =="
echo "classification=PAIREC_PURE_BRPC_OBSERVABILITY_OK"
echo "requests=$REQUESTS"
echo "output_dir=$OUTPUT_DIR"
echo "datasystem_attribution_required=$REQUIRE_DATASYSTEM_ATTRIBUTION"
echo "PAIREC_PURE_BRPC_OBSERVABILITY_OK"
