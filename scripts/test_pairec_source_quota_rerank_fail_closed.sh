#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
DEPLOYMENT="${DEPLOYMENT:-pairec-brpc-observed}"
CONFIGMAP="${CONFIGMAP:-pairec-config-brpc-observed}"
SERVICE="${SERVICE:-pairec-brpc-observed}"
USER_ID="${USER_ID:-1}"
SCENE_ID="${SCENE_ID:-home_feed}"
SIZE="${SIZE:-10}"
FAULT_MINIMUM_GENERATIVE="${FAULT_MINIMUM_GENERATIVE:-3}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-5m}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pairec-f20-rerank-fault/$(date +%Y%m%d-%H%M%S)}"

die() { echo "ERROR: $*" >&2; exit 1; }

for command in kubectl python3 curl; do
  command -v "$command" >/dev/null || die "missing command: $command"
done
[[ "$SIZE" =~ ^[1-9][0-9]*$ ]] || die "SIZE must be a positive integer"
[[ "$FAULT_MINIMUM_GENERATIVE" =~ ^[1-9][0-9]*$ ]] \
  || die "FAULT_MINIMUM_GENERATIVE must be a positive integer"
mkdir -p "$OUTPUT_DIR"

ORIGINAL_CONFIG="$OUTPUT_DIR/pairec_config.original.json"
FAULT_CONFIG="$OUTPUT_DIR/pairec_config.fault.json"
FAULT_RESPONSE="$OUTPUT_DIR/fault-response.json"
RESTORED_RESPONSE="$OUTPUT_DIR/restored-response.json"
FAULT_LOG="$OUTPUT_DIR/fault-pairec.log"
restore_needed=0

apply_config() {
  local config_file="$1"
  kubectl -n "$NAMESPACE" create configmap "$CONFIGMAP" \
    --from-file="pairec_config.json=$config_file" \
    --dry-run=client -o yaml | kubectl apply -f -
}

restart_and_wait() {
  kubectl -n "$NAMESPACE" rollout restart "deployment/$DEPLOYMENT"
  kubectl -n "$NAMESPACE" rollout status "deployment/$DEPLOYMENT" \
    "--timeout=$ROLLOUT_TIMEOUT"
}

restore_config() {
  echo "== Restore original rerank configuration =="
  apply_config "$ORIGINAL_CONFIG"
  restart_and_wait
  restore_needed=0
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$restore_needed" = 1 ]]; then
    echo "== Emergency restore after failure ==" >&2
    if ! apply_config "$ORIGINAL_CONFIG" || ! restart_and_wait; then
      echo "ERROR: automatic ConfigMap restore failed; restore from $ORIGINAL_CONFIG" >&2
      status=1
    fi
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "== F20 fail-closed injection target =="
echo "namespace=$NAMESPACE deployment=$DEPLOYMENT configmap=$CONFIGMAP service=$SERVICE"
echo "user_id=$USER_ID scene_id=$SCENE_ID size=$SIZE fault_minimum_generative=$FAULT_MINIMUM_GENERATIVE"
echo "output_dir=$OUTPUT_DIR"

kubectl -n "$NAMESPACE" get configmap "$CONFIGMAP" \
  -o jsonpath='{.data.pairec_config\.json}' >"$ORIGINAL_CONFIG"
python3 -m json.tool "$ORIGINAL_CONFIG" >/dev/null

python3 - "$ORIGINAL_CONFIG" "$FAULT_CONFIG" "$FAULT_MINIMUM_GENERATIVE" <<'PY'
import json
import pathlib
import sys

source, target, injected_minimum = sys.argv[1], sys.argv[2], int(sys.argv[3])
config = json.loads(pathlib.Path(source).read_text())
reranks = config.get("UserDefineConfs", {}).get("RerankConfs", [])
assert len(reranks) == 1, f"expected exactly one rerank config, got {len(reranks)}"
rerank = reranks[0]
assert rerank.get("enabled") is True, "source quota rerank is not enabled"
assert rerank.get("name") == "source_quota_tail", rerank
original_maximum = int(rerank["max_generative"])
assert injected_minimum > original_maximum, (
    f"fault quota {injected_minimum} must exceed configured maximum {original_maximum}"
)
rerank["minimum_generative"] = injected_minimum
rerank["max_generative"] = injected_minimum
pathlib.Path(target).write_text(json.dumps(config, indent=2) + "\n")
print(
    "F20_FAULT_CONFIG_OK",
    f"original_maximum={original_maximum}",
    f"injected_minimum={injected_minimum}",
)
PY

echo "== Apply fault configuration =="
restore_needed=1
apply_config "$FAULT_CONFIG"
restart_and_wait

SERVICE_IP="$(kubectl -n "$NAMESPACE" get service "$SERVICE" \
  -o jsonpath='{.spec.clusterIP}')"
[[ -n "$SERVICE_IP" && "$SERVICE_IP" != "None" ]] || die "service has no ClusterIP"
PAIREC_URL="http://${SERVICE_IP}:18080/api/recommend"

echo "== Verify fail-closed response =="
curl --noproxy '*' -sS --connect-timeout 2 --max-time "$REQUEST_TIMEOUT_SECONDS" \
  "$PAIREC_URL" -H 'Content-Type: application/json' \
  -d "{\"scene_id\":\"$SCENE_ID\",\"uid\":\"$USER_ID\",\"size\":$SIZE}" \
  -o "$FAULT_RESPONSE"

REQUEST_ID="$(python3 - "$FAULT_RESPONSE" <<'PY'
import json
import sys

response = json.load(open(sys.argv[1]))
assert response.get("code") == 500, response
assert response.get("msg") == "rerank failed", response
assert response.get("size") == 0, response
assert response.get("items") == [], response
request_id = response.get("request_id")
assert isinstance(request_id, str) and request_id, response
print(request_id)
PY
)"
echo "F20_RERANK_FAIL_CLOSED_RESPONSE_OK request_id=$REQUEST_ID"

PAIREC_POD="$(kubectl -n "$NAMESPACE" get pod -l "app=$DEPLOYMENT" \
  --sort-by=.metadata.creationTimestamp -o name | tail -1)"
kubectl -n "$NAMESPACE" logs "$PAIREC_POD" --since=5m >"$FAULT_LOG"

python3 - "$FAULT_LOG" "$REQUEST_ID" <<'PY'
import json
import pathlib
import sys

log_path, request_id = sys.argv[1], sys.argv[2]
events = []
for line in pathlib.Path(log_path).read_text(errors="replace").splitlines():
    start = line.find("{")
    if start < 0:
        continue
    try:
        event = json.loads(line[start:])
    except json.JSONDecodeError:
        continue
    if event.get("request_id") == request_id:
        events.append(event)

rerank_event = next(
    (event for event in events if event.get("event") == "source_quota_rerank_complete"),
    None,
)
assert rerank_event is not None, "missing source_quota_rerank_complete event"
assert rerank_event.get("status") == "error", rerank_event
assert "missing generative candidates" in rerank_event.get("error", ""), rerank_event

pipeline = next(
    (event for event in events if event.get("event") == "pipeline_trace_complete"),
    None,
)
assert pipeline is not None, "missing pipeline_trace_complete event"
assert pipeline.get("status") == "error", pipeline
rerank_span = next(
    (span for span in pipeline.get("spans", []) if span.get("name") == "rerank"),
    None,
)
assert rerank_span is not None and rerank_span.get("status") == "error", rerank_span
print("F20_RERANK_FAIL_CLOSED_TRACE_OK", f"request_id={request_id}")
PY

restore_config

echo "== Verify successful response after restore =="
curl --noproxy '*' -sS --connect-timeout 2 --max-time "$REQUEST_TIMEOUT_SECONDS" \
  "$PAIREC_URL" -H 'Content-Type: application/json' \
  -d "{\"scene_id\":\"$SCENE_ID\",\"uid\":\"$USER_ID\",\"size\":$SIZE}" \
  -o "$RESTORED_RESPONSE"

python3 - "$RESTORED_RESPONSE" "$SIZE" <<'PY'
import json
import sys

response = json.load(open(sys.argv[1]))
size = int(sys.argv[2])
assert response.get("code") == 200, response
items = response.get("items", [])
assert len(items) == size, response
assert len({item.get("item_id") for item in items}) == size, response
sources = [item.get("retrieve_id") for item in items]
generative = sources.count("generative_recall")
assert 1 <= generative <= min(2, size), response
assert sources == ["milvus_recall"] * (size - generative) + ["generative_recall"] * generative, response
print("F20_RERANK_RESTORE_OK", f"request_id={response.get('request_id')}")
PY

echo "classification=F20_SOURCE_QUOTA_RERANK_FAIL_CLOSED_OK"
echo "output_dir=$OUTPUT_DIR"
echo "F20_SOURCE_QUOTA_RERANK_FAIL_CLOSED_COMPLETE"
