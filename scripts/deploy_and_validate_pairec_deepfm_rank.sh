#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
DEPLOYMENT="${DEPLOYMENT:-pairec-multi-recall-rank}"
CONFIGMAP="${CONFIGMAP:-pairec-config-multi-recall-rank}"
INFERENCE_SERVICE="${INFERENCE_SERVICE:-inference-brpc-trtllm}"
INFERENCE_PORT="${INFERENCE_PORT:-18100}"
DSSM_HOST="${DSSM_HOST:-141.61.91.189}"
DSSM_PORT="${DSSM_PORT:-18200}"
DEEPFM_HOST="${DEEPFM_HOST:-141.61.91.189}"
DEEPFM_PORT="${DEEPFM_PORT:-18210}"
DEEPFM_CONTAINER="${DEEPFM_CONTAINER:-deepfm-rank}"
USER_ID="${USER_ID:-1}"
SCENE_ID="${SCENE_ID:-home_feed}"
SMOKE_REQUESTS="${SMOKE_REQUESTS:-3}"
STABILITY_REQUESTS="${STABILITY_REQUESTS:-100}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
RUN_STABILITY="${RUN_STABILITY:-1}"
RUN_FAILURE_INJECTION="${RUN_FAILURE_INJECTION:-1}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"
IMPORT_IMAGE="${IMPORT_IMAGE:-1}"
IMAGE="${IMAGE:-docker.io/library/pairec-server:k8s-arm64-brpc-v1}"
RUNTIME_BASE="${RUNTIME_BASE:-docker.io/library/pairec-server:k8s-arm64-static}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-configs/pairec_config.multi_recall_rank_ip.json}"
MANIFEST="${MANIFEST:-k8s/deployment-pairec-multi-recall-rank.yaml}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-5m}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pairec-deepfm-rank/$(date +%Y%m%d-%H%M%S)}"

die() { echo "ERROR: $*" >&2; exit 1; }
for value in "$SMOKE_REQUESTS" "$STABILITY_REQUESTS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "request counts must be positive integers"
done
[[ "$WARMUP_REQUESTS" =~ ^[0-9]+$ ]] || die "WARMUP_REQUESTS must be non-negative"
for value in "$RUN_STABILITY" "$RUN_FAILURE_INJECTION" "$BUILD_IMAGE" "$IMPORT_IMAGE"; do
  [[ "$value" = 0 || "$value" = 1 ]] || die "boolean flags must be 0 or 1"
done
for command in kubectl curl python3; do
  command -v "$command" >/dev/null || die "missing command: $command"
done
if [[ "$BUILD_IMAGE" = 1 || "$IMPORT_IMAGE" = 1 || "$RUN_FAILURE_INJECTION" = 1 ]]; then
  command -v docker >/dev/null || die "docker is required for the selected workflow"
fi
if [[ "$IMPORT_IMAGE" = 1 ]]; then
  command -v ctr >/dev/null || die "ctr is required when IMPORT_IMAGE=1"
fi
test -f "$CONFIG_TEMPLATE" || die "missing config template: $CONFIG_TEMPLATE"
test -f "$MANIFEST" || die "missing manifest: $MANIFEST"
mkdir -p "$OUTPUT_DIR"

INFERENCE_IP="$(kubectl -n "$NAMESPACE" get service "$INFERENCE_SERVICE" -o jsonpath='{.spec.clusterIP}')"
INFERENCE_ENDPOINT="${INFERENCE_IP}:${INFERENCE_PORT}"
DSSM_URL="http://${DSSM_HOST}:${DSSM_PORT}"
DEEPFM_URL="http://${DEEPFM_HOST}:${DEEPFM_PORT}"

echo "== Dependency preflight =="
curl --noproxy '*' -fsS --connect-timeout 2 --max-time 5 "$DSSM_URL/health" \
  -o "$OUTPUT_DIR/dssm-health.json"
curl --noproxy '*' -fsS --connect-timeout 2 --max-time 5 "$DEEPFM_URL/health" \
  -o "$OUTPUT_DIR/deepfm-health.json"
python3 - "$OUTPUT_DIR/dssm-health.json" "$OUTPUT_DIR/deepfm-health.json" <<'PY'
import json, sys
dssm, rank = (json.load(open(path)) for path in sys.argv[1:])
assert dssm.get("code") == 200 and dssm.get("milvus") is True, dssm
assert rank.get("code") == 200 and rank.get("status") == "healthy", rank
assert rank.get("expected_candidates") == 50, rank
print("DEEPFM_DEPENDENCY_PREFLIGHT_OK model_version=" + rank["model_version"])
PY

echo "== Validate strict Rank Service protocol =="
python3 - "$DEEPFM_URL" "$USER_ID" "$OUTPUT_DIR" <<'PY'
import json, pathlib, sys, urllib.request
base, user_id, out_dir = sys.argv[1:]

def post(payload):
    request = urllib.request.Request(base + "/rank", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=2) as response:
        return json.load(response)

valid = {"request_id": "rank-contract-valid", "user_id": user_id,
         "items": [{"item_id": str(i)} for i in range(1, 51)]}
results = []
for index in range(100):
    request_payload = {**valid, "request_id": f"rank-contract-{index}"}
    result = post(request_payload)
    assert result.get("code") == 200 and len(result.get("items", [])) == 50, result
    assert {x["item_id"] for x in result["items"]} == {str(i) for i in range(1, 51)}
    assert result.get("model_version")
    results.append(result)
assert len({result["model_version"] for result in results}) == 1
for name, payload in {
    "short": {**valid, "request_id": "rank-short", "items": valid["items"][:-1]},
    "duplicate": {**valid, "request_id": "rank-duplicate",
                  "items": valid["items"][:-1] + [valid["items"][0]]},
}.items():
    invalid = post(payload)
    assert invalid.get("code") == 400 and invalid.get("items") == [], (name, invalid)
pathlib.Path(out_dir, "rank-contract.json").write_text(json.dumps(results[-1], indent=2))
print("DEEPFM_RANK_PROTOCOL_OK calls=100 model_version=" + results[-1]["model_version"])
PY

if [[ "$BUILD_IMAGE" = 1 ]]; then
  RUNTIME_BASE="$RUNTIME_BASE" bash scripts/build_pairec_binary_image.sh "$IMAGE"
fi
if [[ "$IMPORT_IMAGE" = 1 ]]; then
  IMAGE_TAR="$OUTPUT_DIR/pairec-rank-image.tar"
  docker save -o "$IMAGE_TAR" "$IMAGE"
  ctr -n k8s.io images import "$IMAGE_TAR"
  rm -f "$IMAGE_TAR"
fi

echo "== Render rank configuration =="
RENDERED_CONFIG="$OUTPUT_DIR/pairec_config.json"
python3 - "$CONFIG_TEMPLATE" "$RENDERED_CONFIG" "$INFERENCE_ENDPOINT" "$DSSM_URL" "$DEEPFM_URL" <<'PY'
import json, pathlib, sys
source, target, inference, dssm, rank = sys.argv[1:]
text = pathlib.Path(source).read_text()
for key, value in {
    "__INFERENCE_ENDPOINT__": inference,
    "__DSSM_RECALL_URL__": dssm,
    "__DEEPFM_RANK_URL__": rank,
}.items():
    text = text.replace(key, value)
assert "__" not in text
config = json.loads(text)
assert config["SortNames"]["home_feed"] == ["deepfm_rank_sort"]
ranker = config["UserDefineConfs"]["DeepFMRankSorts"]
assert ranker == [{"name": "deepfm_rank_sort", "server_url": rank,
                   "timeout_ms": 100, "expected_candidates": 50}], ranker
pathlib.Path(target).write_text(json.dumps(config, indent=2) + "\n")
print("DEEPFM_RANK_CONFIG_OK")
PY
kubectl -n "$NAMESPACE" create configmap "$CONFIGMAP" \
  --from-file="pairec_config.json=$RENDERED_CONFIG" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "== Deploy isolated PaiRec rank experiment =="
kubectl apply -f "$MANIFEST"
kubectl -n "$NAMESPACE" set image "deployment/$DEPLOYMENT" "pairec=$IMAGE"
kubectl -n "$NAMESPACE" set env "deployment/$DEPLOYMENT" \
  "INFERENCE_ENDPOINT=$INFERENCE_ENDPOINT" \
  "DSSM_HEALTH_URL=$DSSM_URL/health" \
  "DEEPFM_HEALTH_URL=$DEEPFM_URL/health" \
  "NO_PROXY=127.0.0.1,localhost,$INFERENCE_IP,$DSSM_HOST,$DEEPFM_HOST" \
  "no_proxy=127.0.0.1,localhost,$INFERENCE_IP,$DSSM_HOST,$DEEPFM_HOST"
kubectl -n "$NAMESPACE" rollout restart "deployment/$DEPLOYMENT"
kubectl -n "$NAMESPACE" rollout status "deployment/$DEPLOYMENT" --timeout="$ROLLOUT_TIMEOUT"
POD="$(kubectl -n "$NAMESPACE" get pod -l "app=$DEPLOYMENT" \
  -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{"\n"}{end}' | tail -1)"
test -n "$POD" || die "no Running Pod for deployment/$DEPLOYMENT"
POD_IP="$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.status.podIP}')"
SERVICE_IP="$(kubectl -n "$NAMESPACE" get service "$DEPLOYMENT" -o jsonpath='{.spec.clusterIP}')"
PAIREC_URL="http://${SERVICE_IP}:18080/api/recommend"
for _ in $(seq 1 60); do
  curl --noproxy '*' -fsS --connect-timeout 1 --max-time 2 \
    "http://${SERVICE_IP}:18080/ping" >/dev/null 2>&1 && break
  sleep 1
done
curl --noproxy '*' -fsS --connect-timeout 2 --max-time 3 \
  "http://${SERVICE_IP}:18080/ping" | grep -q success || die "PaiRec service is not reachable"

run_phase() {
  local phase="$1" requests="$2" size="$3" phase_dir="$OUTPUT_DIR/$phase"
  mkdir -p "$phase_dir"
  local since
  since="$(date --iso-8601=seconds)"
  printf 'index\te2e_ms\trequest_id\n' >"$phase_dir/requests.tsv"
  for index in $(seq 1 "$requests"); do
    local response="$phase_dir/response-$index.json" metrics="$phase_dir/curl-$index.txt"
    curl --noproxy '*' -sS --connect-timeout 3 --max-time 10 "$PAIREC_URL" \
      -H 'Content-Type: application/json' \
      -d "{\"scene_id\":\"$SCENE_ID\",\"uid\":\"$USER_ID\",\"size\":$size}" \
      -o "$response" -w '%{http_code}\t%{time_total}\n' >"$metrics"
    python3 - "$response" "$size" <<'PY'
import json, math, sys
data = json.load(open(sys.argv[1]))
size = int(sys.argv[2])
assert data.get("code") == 200, data
items = data.get("items", [])
assert len(items) == size and len({x["item_id"] for x in items}) == size, data
scores = [x["score"] for x in items]
assert all(math.isfinite(x) for x in scores), scores
assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), scores
PY
    read -r http_code seconds <"$metrics"
    [[ "$http_code" = 200 ]] || die "$phase request $index HTTP=$http_code"
    request_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["request_id"])' "$response")"
    e2e_ms="$(python3 -c 'import sys; print(f"{float(sys.argv[1])*1000:.3f}")' "$seconds")"
    printf '%s\t%s\t%s\n' "$index" "$e2e_ms" "$request_id" >>"$phase_dir/requests.tsv"
  done
  kubectl -n "$NAMESPACE" logs "$POD" --since-time="$since" >"$phase_dir/pairec.log"
  python3 - "$phase_dir/requests.tsv" "$phase_dir/pairec.log" "$phase_dir/summary.json" <<'PY'
import csv, json, pathlib, re, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
request_ids = {row["request_id"] for row in rows}
events = []
log_text = pathlib.Path(sys.argv[2]).read_text(errors="replace")
log_lines = log_text.splitlines()
for line in log_lines:
    if '"event":"deepfm_rank_complete"' not in line:
        continue
    try:
        event = json.loads(line[line.index("{"):])
    except Exception:
        continue
    if event.get("request_id") in request_ids:
        events.append(event)
assert len(events) == len(rows), (len(events), len(rows))
assert all(event["candidate_count"] == 50 for event in events), events
versions = {event["model_version"] for event in events}
assert len(versions) == 1, versions
assert any(event["reordered"] for event in events), "DeepFM never changed recall order"
assert '"event":"deepfm_rank_error"' not in log_text

def fields(line):
    return dict(re.findall(r"([A-Za-z_]+)=([^\s]+)", line))

multi_recall_ms = []
for request_id in request_ids:
    matching = [line for line in log_lines if request_id in line]
    merged = [line for line in matching if "module=QuotaMultiRecall" in line]
    assert merged, "missing QuotaMultiRecall trace for " + request_id
    values = fields(merged[-1])
    assert values.get("primary_minimum") == "1", values
    assert 1 <= int(values.get("primary_selected", "0")) <= 2, values
    assert values.get("final_count") == "50" and values.get("degraded") == "false", values
    multi_recall_ms.append(float(values["cost"]))

def metrics(name, values):
    values = sorted(values)
    def percentile(p):
        index = (len(values)-1)*p
        lo, hi = int(index), min(int(index)+1, len(values)-1)
        return values[lo] + (values[hi]-values[lo])*(index-lo)
    return {f"{name}_avg_ms": statistics.mean(values),
            f"{name}_p50_ms": percentile(.5),
            f"{name}_p95_ms": percentile(.95),
            f"{name}_p99_ms": percentile(.99)}

summary = {"samples": len(rows), "model_version": next(iter(versions))}
summary.update(metrics("e2e", [float(row["e2e_ms"]) for row in rows]))
summary.update(metrics("multi_recall", multi_recall_ms))
summary.update(metrics("rank_client", [float(event["client_total_ms"]) for event in events]))
summary.update(metrics("rank_service", [float(event["service_total_ms"]) for event in events]))
pathlib.Path(sys.argv[3]).write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, sort_keys=True))
PY
}

if (( WARMUP_REQUESTS > 0 )); then
  for _ in $(seq 1 "$WARMUP_REQUESTS"); do
    curl --noproxy '*' -fsS --max-time 10 "$PAIREC_URL" -H 'Content-Type: application/json' \
      -d "{\"scene_id\":\"$SCENE_ID\",\"uid\":\"$USER_ID\",\"size\":10}" >/dev/null
  done
fi
run_phase smoke-50 "$SMOKE_REQUESTS" 50
run_phase smoke-10 "$SMOKE_REQUESTS" 10
if [[ "$RUN_STABILITY" = 1 ]]; then
  run_phase stability-10 "$STABILITY_REQUESTS" 10
fi

if [[ "$RUN_FAILURE_INJECTION" = 1 ]]; then
  echo "== Failure injection: rank service unavailable =="
  rank_stopped=0
  restore_rank() {
    if [[ "$rank_stopped" = 1 ]]; then
      docker start "$DEEPFM_CONTAINER" >/dev/null 2>&1 || true
    fi
  }
  trap restore_rank EXIT
  docker stop --time 10 "$DEEPFM_CONTAINER" >/dev/null
  rank_stopped=1
  for _ in $(seq 1 20); do
    ready="$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.status.containerStatuses[0].ready}')"
    [[ "$ready" = false ]] && break
    sleep 1
  done
  [[ "$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.status.containerStatuses[0].ready}')" = false ]] \
    || die "PaiRec remained Ready after rank service stopped"
  failure_http="$(curl --noproxy '*' -sS --connect-timeout 2 --max-time 10 \
    "http://${POD_IP}:18080/api/recommend" -H 'Content-Type: application/json' \
    -d "{\"scene_id\":\"$SCENE_ID\",\"uid\":\"$USER_ID\",\"size\":10}" \
    -o "$OUTPUT_DIR/failure-response.json" -w '%{http_code}')"
  [[ "$failure_http" = 200 ]] || die "failure response HTTP=$failure_http"
  python3 - "$OUTPUT_DIR/failure-response.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert data.get("code") == 500, data
assert data.get("msg") == "deepfm rank failed", data
assert data.get("size") == 0 and data.get("items") == [], data
print("DEEPFM_RANK_FAIL_CLOSED_OK")
PY
  docker start "$DEEPFM_CONTAINER" >/dev/null
  rank_stopped=0
  kubectl -n "$NAMESPACE" wait --for=condition=Ready "pod/$POD" --timeout=90s
  trap - EXIT
fi

restarts="$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.status.containerStatuses[0].restartCount}')"
[[ "$restarts" = 0 ]] || die "PaiRec restarted $restarts times"
echo "PAIREC_DEEPFM_RANK_VALIDATION_OK"
echo "output_dir=$OUTPUT_DIR"
