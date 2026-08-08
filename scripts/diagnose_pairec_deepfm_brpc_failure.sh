#!/usr/bin/env bash
# Collect and classify a fail-closed DeepFM BRPC ranking failure without changing workloads.
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
PAIREC_DEPLOYMENT="${PAIREC_DEPLOYMENT:-pairec-brpc-observed}"
RANK_DEPLOYMENT="${RANK_DEPLOYMENT:-deepfm-rank-brpc}"
REQUEST_ID="${REQUEST_ID:-${1:-}}"
LOG_SINCE="${LOG_SINCE:-15m}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/pairec-deepfm-brpc-diagnostic}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/$(date +%Y%m%d-%H%M%S)}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

matches() {
  local pattern="$1" file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -qi "$pattern" "$file"
  else
    grep -Eqi "$pattern" "$file"
  fi
}

classify() {
  local combined="$1"
  if matches 'expected [0-9]+ candidates, got [0-9]+|items must contain exactly [0-9]+ candidates' "$combined"; then
    echo "RANK_CANDIDATE_COUNT_MISMATCH"
  elif matches 'rank model_role mismatch' "$combined"; then
    echo "RANK_MODEL_ROLE_MISMATCH"
  elif matches 'rank score_unique_count must be positive|non-finite score' "$combined"; then
    echo "RANK_SCORE_CONTRACT_FAILURE"
  elif matches 'rank item count mismatch|unknown item_id|duplicate (candidate|item_id)' "$combined"; then
    echo "RANK_ITEM_CONTRACT_FAILURE"
  elif matches 'DeepFM scoring failed|deepfm scoring failed|Traceback \(most recent call last\)' "$combined"; then
    echo "RANK_BACKEND_MODEL_FAILURE"
  elif matches 'rank service code=400|backend HTTP status=400|items must be an array|request_id is required|user_id must be' "$combined"; then
    echo "RANK_BACKEND_REQUEST_CONTRACT_FAILURE"
  elif matches 'call BRPC rank service.*(timeout|timed out|deadline|ERPCTIMEDOUT)|backend.*(timeout|timed out)|EHOSTDOWN' "$combined"; then
    echo "RANK_BRPC_OR_BACKEND_TIMEOUT"
  elif matches 'call BRPC rank service.*(connection refused|unavailable|failed to connect|ECONNREFUSED)' "$combined"; then
    echo "RANK_BRPC_UNREACHABLE"
  elif matches 'rank request_id mismatch|rank model_version is empty' "$combined"; then
    echo "RANK_RESPONSE_IDENTITY_FAILURE"
  elif matches 'deepfm_rank_error|deepfm rank failed' "$combined"; then
    echo "UNKNOWN_DEEPFM_RANK_FAILURE"
  else
    echo "NO_DEEPFM_RANK_FAILURE_FOUND"
  fi
}

if [[ "${1:-}" == "--classify-file" ]]; then
  [[ -n "${2:-}" && -f "$2" ]] || die "usage: $0 --classify-file LOG_FILE"
  classify "$2"
  exit 0
fi

for command in kubectl python3; do
  command -v "$command" >/dev/null 2>&1 || die "$command is not installed"
done
mkdir -p "$OUTPUT_DIR"

latest_pod() {
  local deployment="$1"
  kubectl -n "$NAMESPACE" get pod -l "app=${deployment}" \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1:].metadata.name}'
}

PAIREC_POD="$(latest_pod "$PAIREC_DEPLOYMENT")"
RANK_POD="$(latest_pod "$RANK_DEPLOYMENT")"
[[ -n "$PAIREC_POD" ]] || die "no pod found for app=${PAIREC_DEPLOYMENT}"
[[ -n "$RANK_POD" ]] || die "no pod found for app=${RANK_DEPLOYMENT}"

kubectl -n "$NAMESPACE" logs "$PAIREC_POD" --since="$LOG_SINCE" \
  >"$OUTPUT_DIR/pairec.log" 2>&1 || true
kubectl -n "$NAMESPACE" logs "$RANK_POD" -c adapter --since="$LOG_SINCE" \
  >"$OUTPUT_DIR/rank-adapter.log" 2>&1 || true
kubectl -n "$NAMESPACE" logs "$RANK_POD" -c backend --since="$LOG_SINCE" \
  >"$OUTPUT_DIR/rank-backend.log" 2>&1 || true
kubectl -n "$NAMESPACE" get deployment "$PAIREC_DEPLOYMENT" "$RANK_DEPLOYMENT" -o yaml \
  >"$OUTPUT_DIR/deployments.yaml" 2>&1 || true
kubectl -n "$NAMESPACE" get pod "$PAIREC_POD" "$RANK_POD" -o yaml \
  >"$OUTPUT_DIR/pods.yaml" 2>&1 || true
{
  kubectl -n "$NAMESPACE" get service "$PAIREC_DEPLOYMENT" "$RANK_DEPLOYMENT" -o wide
  kubectl -n "$NAMESPACE" get endpoints "$PAIREC_DEPLOYMENT" "$RANK_DEPLOYMENT" -o wide
} >"$OUTPUT_DIR/network.txt" 2>&1 || true
kubectl -n "$NAMESPACE" get configmap pairec-config-brpc-observed \
  -o jsonpath='{.data.pairec_config\.json}' \
  >"$OUTPUT_DIR/pairec_config.json" 2>&1 || true

if [[ -z "$REQUEST_ID" ]]; then
  REQUEST_ID="$(python3 - "$OUTPUT_DIR/pairec.log" <<'PY'
import json, re, sys

latest = ""
for line in open(sys.argv[1], errors="replace"):
    if "deepfm_rank_error" not in line:
        continue
    start = line.find("{")
    if start >= 0:
        try:
            event = json.loads(line[start:])
            if event.get("event") == "deepfm_rank_error" and event.get("request_id"):
                latest = str(event["request_id"])
                continue
        except json.JSONDecodeError:
            pass
    match = re.search(r'request[_ ]?[Ii]d[=:" ]+([0-9a-zA-Z-]+)', line)
    if match:
        latest = match.group(1)
print(latest)
PY
)"
fi

if [[ -n "$REQUEST_ID" ]]; then
  grep -F "$REQUEST_ID" "$OUTPUT_DIR/pairec.log" \
    >"$OUTPUT_DIR/request.log" 2>/dev/null || true
else
  : >"$OUTPUT_DIR/request.log"
fi

{
  cat "$OUTPUT_DIR/request.log"
  cat "$OUTPUT_DIR/pairec.log"
  cat "$OUTPUT_DIR/rank-adapter.log"
  cat "$OUTPUT_DIR/rank-backend.log"
  cat "$OUTPUT_DIR/network.txt"
} >"$OUTPUT_DIR/combined.log"

set +e
kubectl -n "$NAMESPACE" exec "$RANK_POD" -c adapter -- \
  /opt/pairec-brpc/bin/brpc_pipeline_client \
    --server=127.0.0.1:18211 --service=rank --timeout_ms=1000 \
  >"$OUTPUT_DIR/rank-health.txt" 2>&1
HEALTH_STATUS=$?
set -e

CLASSIFICATION="$(classify "$OUTPUT_DIR/request.log")"
if [[ "$CLASSIFICATION" == "UNKNOWN_DEEPFM_RANK_FAILURE" ||
      "$CLASSIFICATION" == "NO_DEEPFM_RANK_FAILURE_FOUND" ]]; then
  secondary_classification="$(classify "$OUTPUT_DIR/rank-backend.log")"
  if [[ "$secondary_classification" != "NO_DEEPFM_RANK_FAILURE_FOUND" ]]; then
    CLASSIFICATION="$secondary_classification"
  elif [[ "$CLASSIFICATION" == "NO_DEEPFM_RANK_FAILURE_FOUND" ]]; then
    CLASSIFICATION="$(classify "$OUTPUT_DIR/combined.log")"
  fi
fi
NEXT_ACTION="inspect ${OUTPUT_DIR}/request.log and rank backend logs"
case "$CLASSIFICATION" in
  RANK_CANDIDATE_COUNT_MISMATCH)
    NEXT_ACTION="inspect QuotaMultiRecall final_count and duplicate_count for request ${REQUEST_ID:-unknown}"
    ;;
  RANK_BRPC_OR_BACKEND_TIMEOUT)
    NEXT_ACTION="compare the 100ms PaiRec timeout and 80ms adapter backend timeout; verify whether the backend completed POST /rank after the client timed out"
    ;;
  RANK_BACKEND_MODEL_FAILURE)
    NEXT_ACTION="inspect the Python traceback in ${OUTPUT_DIR}/rank-backend.log before changing timeouts"
    ;;
  RANK_BACKEND_REQUEST_CONTRACT_FAILURE|RANK_ITEM_CONTRACT_FAILURE|RANK_RESPONSE_IDENTITY_FAILURE|RANK_SCORE_CONTRACT_FAILURE)
    NEXT_ACTION="compare the protobuf JSON bridge fields with the Python /rank request and response contract"
    ;;
  RANK_MODEL_ROLE_MISMATCH)
    NEXT_ACTION="make DEEPFM_MODEL_ROLE and required_model_role identical"
    ;;
  RANK_BRPC_UNREACHABLE)
    NEXT_ACTION="inspect ${OUTPUT_DIR}/network.txt and the rank Pod container states"
    ;;
esac

cat >"$OUTPUT_DIR/result.env" <<EOF
namespace=$NAMESPACE
pairec_pod=$PAIREC_POD
rank_pod=$RANK_POD
request_id=${REQUEST_ID:-UNKNOWN}
rank_health_status=$HEALTH_STATUS
classification=$CLASSIFICATION
output_dir=$OUTPUT_DIR
EOF

echo "== DeepFM BRPC diagnostic target =="
echo "namespace=$NAMESPACE pairec_pod=$PAIREC_POD rank_pod=$RANK_POD"
echo "request_id=${REQUEST_ID:-UNKNOWN} log_since=$LOG_SINCE"
echo
echo "== Request evidence =="
if [[ -s "$OUTPUT_DIR/request.log" ]]; then
  tail -40 "$OUTPUT_DIR/request.log"
else
  echo "REQUEST_SPECIFIC_LOG_NOT_FOUND"
fi
echo
echo "== Rank adapter health =="
cat "$OUTPUT_DIR/rank-health.txt"
echo
echo "== Rank backend tail =="
tail -40 "$OUTPUT_DIR/rank-backend.log"
echo
echo "== Diagnosis =="
echo "classification=$CLASSIFICATION"
echo "rank_health_status=$HEALTH_STATUS"
echo "next_action=$NEXT_ACTION"
echo "output_dir=$OUTPUT_DIR"
echo "PAIREC_DEEPFM_BRPC_DIAGNOSTIC_COMPLETE"

[[ "$CLASSIFICATION" != "NO_DEEPFM_RANK_FAILURE_FOUND" ]]
