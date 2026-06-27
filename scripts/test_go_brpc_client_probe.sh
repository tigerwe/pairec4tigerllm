#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
INFERENCE_SERVICE="${INFERENCE_SERVICE:-inference-brpc-trtllm}"
BRPC_PORT="${BRPC_PORT:-18100}"
ENDPOINT="${ENDPOINT:-}"
USER_ID="${USER_ID:-go_brpc_probe}"
TOPK="${TOPK:-10}"
REQUESTS="${REQUESTS:-1}"
TIMEOUT_MS="${TIMEOUT_MS:-5000}"
MAX_RETRIES="${MAX_RETRIES:-1}"
PAYLOAD_BYTES="${PAYLOAD_BYTES:-0}"
HISTORY_SOURCE="${HISTORY_SOURCE:-synthetic}"
UIDS="${UIDS:-}"
USER_FEATURES_PATH="${USER_FEATURES_PATH:-data/user_features.json}"
SEMANTIC_MAP_PATH="${SEMANTIC_MAP_PATH:-data/tenrec/processed/semantic_id_map.json}"
HISTORY_MAX_LENGTH="${HISTORY_MAX_LENGTH:-20}"
VARY_USER_ID="${VARY_USER_ID:-true}"

if [ -z "$ENDPOINT" ]; then
  service_ip="$(kubectl -n "$NAMESPACE" get svc "$INFERENCE_SERVICE" -o jsonpath='{.spec.clusterIP}')"
  if [ -z "$service_ip" ] || [ "$service_ip" = "None" ]; then
    echo "ERROR: service ${NAMESPACE}/${INFERENCE_SERVICE} has no ClusterIP" >&2
    exit 1
  fi
  ENDPOINT="${service_ip}:${BRPC_PORT}"
fi

echo "Go brpc client probe"
echo "  endpoint: ${ENDPOINT}"
echo "  user_id:  ${USER_ID}"
echo "  topk:     ${TOPK}"
echo "  requests: ${REQUESTS}"
echo "  payload:  ${PAYLOAD_BYTES} bytes"
echo "  history:  ${HISTORY_SOURCE}"
echo

GOPROXY="${GOPROXY:-off}" \
GOSUMDB="${GOSUMDB:-off}" \
go run -mod=vendor ./scripts/probe_go_brpc_client.go \
  --endpoint="$ENDPOINT" \
  --method=health \
  --requests="$REQUESTS" \
  --payload_bytes="$PAYLOAD_BYTES" \
  --timeout_ms="$TIMEOUT_MS" \
  --max_retries="$MAX_RETRIES"

GOPROXY="${GOPROXY:-off}" \
GOSUMDB="${GOSUMDB:-off}" \
go run -mod=vendor ./scripts/probe_go_brpc_client.go \
  --endpoint="$ENDPOINT" \
  --method=recommend \
  --user_id="$USER_ID" \
  --topk="$TOPK" \
  --requests="$REQUESTS" \
  --payload_bytes="$PAYLOAD_BYTES" \
  --timeout_ms="$TIMEOUT_MS" \
  --max_retries="$MAX_RETRIES" \
  --history_source="$HISTORY_SOURCE" \
  --uids="$UIDS" \
  --user_features_path="$USER_FEATURES_PATH" \
  --semantic_map_path="$SEMANTIC_MAP_PATH" \
  --history_max_length="$HISTORY_MAX_LENGTH" \
  --vary_user_id="$VARY_USER_ID"
