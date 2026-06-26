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
echo

GOPROXY="${GOPROXY:-off}" \
GOSUMDB="${GOSUMDB:-off}" \
go run -mod=vendor ./scripts/probe_go_brpc_client.go \
  --endpoint="$ENDPOINT" \
  --method=health \
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
  --timeout_ms="$TIMEOUT_MS" \
  --max_retries="$MAX_RETRIES"
