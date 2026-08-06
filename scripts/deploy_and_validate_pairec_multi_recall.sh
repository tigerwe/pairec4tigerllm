#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
DEPLOYMENT="${DEPLOYMENT:-pairec-multi-recall}"
CONFIGMAP="${CONFIGMAP:-pairec-config-multi-recall}"
INFERENCE_SERVICE="${INFERENCE_SERVICE:-inference-brpc-trtllm}"
INFERENCE_PORT="${INFERENCE_PORT:-18100}"
DSSM_HOST="${DSSM_HOST:-141.61.91.189}"
DSSM_PORT="${DSSM_PORT:-18200}"
USER_ID="${USER_ID:-1}"
SCENE_ID="${SCENE_ID:-home_feed}"
RESULT_SIZE="${RESULT_SIZE:-50}"
SMOKE_REQUESTS="${SMOKE_REQUESTS:-3}"
STABILITY_REQUESTS="${STABILITY_REQUESTS:-100}"
RUN_STABILITY="${RUN_STABILITY:-1}"
IMAGE="${IMAGE:-docker.io/library/pairec-server:k8s-arm64-brpc-v1}"
RUNTIME_BASE="${RUNTIME_BASE:-docker.io/library/pairec-server:k8s-arm64-static}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"
IMPORT_IMAGE="${IMPORT_IMAGE:-1}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-5m}"
SERVICE_READY_TIMEOUT_SECONDS="${SERVICE_READY_TIMEOUT_SECONDS:-60}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-configs/pairec_config.multi_recall_ip.json}"
MANIFEST="${MANIFEST:-k8s/deployment-pairec-multi-recall.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pairec-multi-recall/$(date +%Y%m%d-%H%M%S)}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

is_bool() {
  [[ "$1" = "0" || "$1" = "1" ]]
}

for value in "$SMOKE_REQUESTS" "$STABILITY_REQUESTS" "$RESULT_SIZE"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "request counts and RESULT_SIZE must be positive integers"
done
[[ "$SERVICE_READY_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] \
  || die "SERVICE_READY_TIMEOUT_SECONDS must be a positive integer"
test "$RESULT_SIZE" = "50" || die "RESULT_SIZE must remain 50 for the confirmed 2+48 contract"
is_bool "$RUN_STABILITY" || die "RUN_STABILITY must be 0 or 1"
is_bool "$BUILD_IMAGE" || die "BUILD_IMAGE must be 0 or 1"
is_bool "$IMPORT_IMAGE" || die "IMPORT_IMAGE must be 0 or 1"

for command in kubectl curl python3; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: ${command}"
done
test -f "$CONFIG_TEMPLATE" || die "missing config template: ${CONFIG_TEMPLATE}"
test -f "$MANIFEST" || die "missing manifest: ${MANIFEST}"

mkdir -p "$OUTPUT_DIR"
DSSM_URL="http://${DSSM_HOST}:${DSSM_PORT}"

echo "== Resolve numeric dependencies =="
INFERENCE_IP="$(kubectl -n "$NAMESPACE" get service "$INFERENCE_SERVICE" -o jsonpath='{.spec.clusterIP}')"
test -n "$INFERENCE_IP" && test "$INFERENCE_IP" != "None" \
  || die "service/${INFERENCE_SERVICE} has no ClusterIP"
INFERENCE_ENDPOINT="${INFERENCE_IP}:${INFERENCE_PORT}"
echo "inference_endpoint=${INFERENCE_ENDPOINT}"
echo "dssm_url=${DSSM_URL}"

echo
echo "== Dependency preflight =="
curl --noproxy '*' -fsS --connect-timeout 2 --max-time 5 "${DSSM_URL}/health" \
  -o "${OUTPUT_DIR}/dssm-health.json"
python3 - "${OUTPUT_DIR}/dssm-health.json" <<'PY'
import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert data.get("code") == 200, data
assert data.get("status") == "healthy", data
assert data.get("milvus") is True, data
print("DSSM_MILVUS_HEALTH_OK")
PY

curl --noproxy '*' -fsS --connect-timeout 2 --max-time 5 "${DSSM_URL}/recall" \
  -H 'Content-Type: application/json' \
  -d "{\"user_id\":\"${USER_ID}\",\"topk\":50}" \
  -o "${OUTPUT_DIR}/dssm-recall.json"
python3 - "${OUTPUT_DIR}/dssm-recall.json" <<'PY'
import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert data.get("code") == 200, data
assert data.get("source") == "milvus", data
assert len(data.get("items", [])) >= 48, data
print(f"DSSM_MILVUS_RECALL_OK items={len(data['items'])} latency_ms={data.get('latency_ms')}")
PY

python3 - "$INFERENCE_IP" "$INFERENCE_PORT" <<'PY'
import socket, sys
with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=3):
    pass
print("INFERENCE_TCP_OK")
PY

if [[ "$BUILD_IMAGE" = "1" ]]; then
  echo
  echo "== Build latest PaiRec image =="
  command -v docker >/dev/null 2>&1 || die "docker is required when BUILD_IMAGE=1"
  RUNTIME_BASE="$RUNTIME_BASE" bash scripts/build_pairec_binary_image.sh "$IMAGE"
fi

if [[ "$IMPORT_IMAGE" = "1" ]]; then
  echo
  echo "== Import PaiRec image into k8s.io containerd =="
  command -v docker >/dev/null 2>&1 || die "docker is required when IMPORT_IMAGE=1"
  command -v ctr >/dev/null 2>&1 || die "ctr is required when IMPORT_IMAGE=1"
  IMAGE_TAR="${OUTPUT_DIR}/pairec-multi-recall-image.tar"
  docker save -o "$IMAGE_TAR" "$IMAGE"
  ctr -n k8s.io images import "$IMAGE_TAR"
  rm -f "$IMAGE_TAR"
fi

echo
echo "== Render strict numeric-address configuration =="
RENDERED_CONFIG="${OUTPUT_DIR}/pairec_config.json"
python3 - "$CONFIG_TEMPLATE" "$RENDERED_CONFIG" "$INFERENCE_ENDPOINT" "$DSSM_URL" <<'PY'
import json, pathlib, sys
source, target, inference, dssm = sys.argv[1:]
text = pathlib.Path(source).read_text(encoding="utf-8")
text = text.replace("__INFERENCE_ENDPOINT__", inference)
text = text.replace("__DSSM_RECALL_URL__", dssm)
if "__" in text:
    raise SystemExit("unresolved placeholder remains in rendered config")
config = json.loads(text)
recalls = {item["Name"]: item for item in config["RecallConfs"]}
gen = json.loads(recalls["generative_recall"]["RecallAlgo"])
milvus = json.loads(recalls["milvus_recall"]["RecallAlgo"])
multi = json.loads(recalls["multi_recall_2_48"]["RecallAlgo"])
assert config["SceneConfs"]["home_feed"]["default"]["RecallNames"] == ["multi_recall_2_48"]
assert config["SortNames"]["home_feed"] == []
assert gen["protocol"] == "brpc" and gen["brpc_endpoint"] == inference
assert gen["brpc_fallback_to_http"] is False and gen["max_retries"] == 0
assert milvus["server_url"] == dssm and milvus["timeout_ms"] == 300
assert multi["primary_quota"] == 2 and multi["total_limit"] == 50
pathlib.Path(target).write_text(json.dumps(config, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
print("MULTI_RECALL_CONFIG_OK")
PY

kubectl -n "$NAMESPACE" create configmap "$CONFIGMAP" \
  --from-file="pairec_config.json=${RENDERED_CONFIG}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo
echo "== Deploy isolated PaiRec multi-recall instance =="
kubectl apply -f "$MANIFEST"
kubectl -n "$NAMESPACE" set image "deployment/${DEPLOYMENT}" "pairec=${IMAGE}"
kubectl -n "$NAMESPACE" set env "deployment/${DEPLOYMENT}" \
  "INFERENCE_ENDPOINT=${INFERENCE_ENDPOINT}" \
  "DSSM_HEALTH_URL=${DSSM_URL}/health" \
  "NO_PROXY=127.0.0.1,localhost,${INFERENCE_IP},${DSSM_HOST}" \
  "no_proxy=127.0.0.1,localhost,${INFERENCE_IP},${DSSM_HOST}"
kubectl -n "$NAMESPACE" rollout restart "deployment/${DEPLOYMENT}"
kubectl -n "$NAMESPACE" rollout status "deployment/${DEPLOYMENT}" --timeout="$ROLLOUT_TIMEOUT"

POD="$(kubectl -n "$NAMESPACE" get pod -l "app=${DEPLOYMENT}" \
  -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{"\n"}{end}' | tail -1)"
test -n "$POD" || die "no running pod for deployment/${DEPLOYMENT}"
kubectl -n "$NAMESPACE" get pod "$POD" -o wide
kubectl -n "$NAMESPACE" logs "$POD" >"${OUTPUT_DIR}/startup.log" 2>&1
grep -F 'Registering GenerativeRecall: generative_recall' "${OUTPUT_DIR}/startup.log" >/dev/null \
  || die "GenerativeRecall registration evidence missing"
grep -F 'Registering MilvusRecall: milvus_recall' "${OUTPUT_DIR}/startup.log" >/dev/null \
  || die "MilvusRecall registration evidence missing"
grep -F 'Registering QuotaMultiRecall: multi_recall_2_48' "${OUTPUT_DIR}/startup.log" >/dev/null \
  || die "QuotaMultiRecall registration evidence missing"
kubectl -n "$NAMESPACE" exec "$POD" -- cat /app/configs/pairec_config.json \
  >"${OUTPUT_DIR}/mounted-config.json"
python3 - "${OUTPUT_DIR}/mounted-config.json" "$INFERENCE_ENDPOINT" "$DSSM_URL" <<'PY'
import json, pathlib, sys
config = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
recalls = {item["Name"]: json.loads(item["RecallAlgo"]) for item in config["RecallConfs"]}
assert recalls["generative_recall"]["brpc_endpoint"] == sys.argv[2]
assert recalls["milvus_recall"]["server_url"] == sys.argv[3]
assert recalls["multi_recall_2_48"]["primary_quota"] == 2
assert recalls["multi_recall_2_48"]["total_limit"] == 50
print("MOUNTED_MULTI_RECALL_CONFIG_OK")
PY

SERVICE_IP="$(kubectl -n "$NAMESPACE" get service "$DEPLOYMENT" -o jsonpath='{.spec.clusterIP}')"
test -n "$SERVICE_IP" && test "$SERVICE_IP" != "None" \
  || die "service/${DEPLOYMENT} has no ClusterIP"
PAIREC_URL="http://${SERVICE_IP}:18080/api/recommend"

echo
echo "== Wait for PaiRec Service endpoint =="
service_deadline="$((SECONDS + SERVICE_READY_TIMEOUT_SECONDS))"
service_ready=0
while (( SECONDS < service_deadline )); do
  ready_addresses="$(kubectl -n "$NAMESPACE" get endpoints "$DEPLOYMENT" \
    -o jsonpath='{range .subsets[*].addresses[*]}{.ip}{"\n"}{end}' 2>/dev/null || true)"
  if [[ -n "$ready_addresses" ]]; then
    if ping_response="$(curl --noproxy '*' -fsS --connect-timeout 1 --max-time 2 \
        "http://${SERVICE_IP}:18080/ping" 2>/dev/null)" \
      && grep -q success <<<"$ping_response"; then
      service_ready=1
      break
    fi
  fi
  sleep 1
done
if [[ "$service_ready" != "1" ]]; then
  kubectl -n "$NAMESPACE" get service "$DEPLOYMENT" -o wide || true
  kubectl -n "$NAMESPACE" get endpoints "$DEPLOYMENT" -o yaml || true
  kubectl -n "$NAMESPACE" get pod -l "app=${DEPLOYMENT}" -o wide || true
  die "service/${DEPLOYMENT} did not become reachable within ${SERVICE_READY_TIMEOUT_SECONDS}s"
fi
ready_addresses_csv="$(tr '\n' ',' <<<"$ready_addresses" | sed 's/,$//')"
echo "PAIREC_MULTI_RECALL_SERVICE_READY endpoint=${SERVICE_IP}:18080 addresses=${ready_addresses_csv}"

pod_state() {
  kubectl -n "$NAMESPACE" get pod "$POD" -o json | python3 -c '
import json, sys
pod = json.load(sys.stdin)
statuses = pod.get("status", {}).get("containerStatuses", [])
assert pod.get("status", {}).get("phase") == "Running", pod.get("status", {})
assert statuses and all(s.get("ready") for s in statuses), statuses
print("uid=" + pod["metadata"]["uid"])
print("restarts=" + str(sum(s.get("restartCount", 0) for s in statuses)))
'
}

run_phase() {
  local phase="$1"
  local requests="$2"
  local phase_dir="${OUTPUT_DIR}/${phase}"
  local started_at
  mkdir -p "$phase_dir"
  started_at="$(date --iso-8601=seconds)"
  pod_state >"${phase_dir}/pod.before"
  printf 'index\thttp_code\te2e_ms\trequest_id\tgenerative_count\tmilvus_count\n' \
    >"${phase_dir}/requests.tsv"

  echo
  echo "== Run ${phase}: ${requests} requests =="
  for index in $(seq 1 "$requests"); do
    local response_file="${phase_dir}/response-${index}.json"
    local metrics_file="${phase_dir}/curl-${index}.txt"
    curl --noproxy '*' -sS --connect-timeout 3 --max-time 10 "$PAIREC_URL" \
      -H 'Content-Type: application/json' \
      -d "{\"scene_id\":\"${SCENE_ID}\",\"uid\":\"${USER_ID}\",\"size\":${RESULT_SIZE}}" \
      -o "$response_file" \
      -w '%{http_code}\t%{time_total}\n' >"$metrics_file"
    local http_code e2e_seconds response_fields
    read -r http_code e2e_seconds <"$metrics_file"
    response_fields="$(python3 - "$response_file" <<'PY'
import collections, json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
counts = collections.Counter(item.get("retrieve_id", "") for item in data.get("items", []))
print(data.get("code"), data.get("request_id", ""), len(data.get("items", [])),
      counts["generative_recall"], counts["milvus_recall"])
PY
)"
    local response_code request_id item_count generative_count milvus_count
    read -r response_code request_id item_count generative_count milvus_count <<<"$response_fields"
    test "$http_code" = "200" || die "${phase} request ${index}: HTTP=${http_code}"
    test "$response_code" = "200" || die "${phase} request ${index}: code=${response_code}"
    test "$item_count" = "$RESULT_SIZE" || die "${phase} request ${index}: items=${item_count}"
    test "$generative_count" = "2" || die "${phase} request ${index}: generative_count=${generative_count}"
    test "$milvus_count" = "$((RESULT_SIZE - 2))" \
      || die "${phase} request ${index}: milvus_count=${milvus_count}"
    local e2e_ms
    e2e_ms="$(python3 -c 'import sys; print(f"{float(sys.argv[1]) * 1000:.3f}")' "$e2e_seconds")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$index" "$http_code" "$e2e_ms" "$request_id" "$generative_count" "$milvus_count" \
      >>"${phase_dir}/requests.tsv"
  done

  kubectl -n "$NAMESPACE" logs "$POD" --since-time="$started_at" >"${phase_dir}/pairec.log" 2>&1
  pod_state >"${phase_dir}/pod.after"
  cmp -s "${phase_dir}/pod.before" "${phase_dir}/pod.after" \
    || die "${phase}: pod identity, readiness, or restart count changed"

  python3 - "${phase_dir}/requests.tsv" "${phase_dir}/pairec.log" "${phase_dir}/summary.json" <<'PY'
import csv, json, pathlib, re, statistics, sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
log_lines = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace").splitlines()

def fields(line):
    return dict(re.findall(r"([A-Za-z_]+)=([^\s]+)", line))

generative_ms = []
milvus_ms = []
merge_ms = []
for row in rows:
    rid = row["request_id"]
    matching = [line for line in log_lines if rid in line]
    gen = [line for line in matching if "module=GenerativeRecall" in line and "from=inference" in line]
    milvus = [line for line in matching if "module=MilvusRecall" in line and "source=milvus" in line]
    merged = [line for line in matching if "module=QuotaMultiRecall" in line]
    assert gen, f"missing GenerativeRecall inference trace for {rid}"
    assert milvus, f"missing MilvusRecall source=milvus trace for {rid}"
    assert merged, f"missing QuotaMultiRecall trace for {rid}"
    gf, mf, qf = fields(gen[-1]), fields(milvus[-1]), fields(merged[-1])
    assert gf.get("protocol") == "brpc", (rid, gf)
    assert int(gf.get("count", "0")) >= 2, (rid, gf)
    assert int(mf.get("count", "0")) >= 48, (rid, mf)
    assert qf.get("primary_selected") == "2", (rid, qf)
    assert qf.get("secondary_selected") == "48", (rid, qf)
    assert qf.get("final_count") == "50", (rid, qf)
    assert qf.get("degraded") == "false", (rid, qf)
    generative_ms.append(float(gf["cost"]))
    milvus_ms.append(float(mf["service_ms"]))
    merge_ms.append(float(qf["cost"]))
    bad = ("from=cache", "fallback to HTTP", "brpc request failed", "Segmentation", "core dumped", "Out of memory")
    assert not any(marker in line for marker in bad for line in matching), (rid, matching)

latencies = [float(row["e2e_ms"]) for row in rows]
def percentile(values, p):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)

def metric(values):
    return {
        "avg_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values),
    }

summary = {
    "samples": len(rows),
    "valid": len(rows),
    "e2e": metric(latencies),
    "generative_recall": metric(generative_ms),
    "milvus_recall": metric(milvus_ms),
    "multi_recall": metric(merge_ms),
    "generative_per_request": 2,
    "milvus_per_request": 48,
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print("metric samples avg_ms p50_ms p95_ms p99_ms max_ms")
for name in ("e2e", "generative_recall", "milvus_recall", "multi_recall"):
    item = summary[name]
    print(f"{name} {len(rows)} {item['avg_ms']:.3f} {item['p50_ms']:.3f} "
          f"{item['p95_ms']:.3f} {item['p99_ms']:.3f} {item['max_ms']:.3f}")
PY
  echo "PAIREC_MULTI_RECALL_PHASE_OK phase=${phase} samples=${requests}"
}

run_phase smoke "$SMOKE_REQUESTS"
if [[ "$RUN_STABILITY" = "1" ]]; then
  run_phase stability "$STABILITY_REQUESTS"
fi

echo
echo "== Summary =="
echo "classification=PAIREC_MULTI_RECALL_2_48_OK"
echo "status=PASS"
echo "deployment=${DEPLOYMENT}"
echo "endpoint=${PAIREC_URL}"
echo "inference_endpoint=${INFERENCE_ENDPOINT}"
echo "dssm_url=${DSSM_URL}"
echo "user_id=${USER_ID}"
echo "output_dir=${OUTPUT_DIR}"
echo "PAIREC_MULTI_RECALL_VALIDATION_COMPLETE"
