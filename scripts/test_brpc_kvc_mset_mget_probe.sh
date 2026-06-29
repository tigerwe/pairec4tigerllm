#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
INFERENCE_SERVICE="${INFERENCE_SERVICE:-inference-brpc-kvc-probe}"
TARGET="${TARGET:-deployment/${INFERENCE_SERVICE}}"
CONTAINER="${CONTAINER:-brpc-kvc-probe}"
REQUESTS="${REQUESTS:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
TOPK="${TOPK:-1}"
TIMEOUT_MS="${TIMEOUT_MS:-30000}"
MAX_RETRIES="${MAX_RETRIES:-1}"
PAYLOAD_BYTES="${PAYLOAD_BYTES:-0}"
HISTORY_SOURCE="${HISTORY_SOURCE:-synthetic}"
SINCE="${SINCE:-5m}"

echo "brpc DataSystem KVC MSet/MGet probe"
echo "  namespace: ${NAMESPACE}"
echo "  service:   ${INFERENCE_SERVICE}"
echo "  target:    ${TARGET}"
echo "  container: ${CONTAINER}"
echo "  requests:  ${REQUESTS}"
echo "  conc:      ${CONCURRENCY}"
echo

NAMESPACE="$NAMESPACE" \
INFERENCE_SERVICE="$INFERENCE_SERVICE" \
REQUESTS="$REQUESTS" \
CONCURRENCY="$CONCURRENCY" \
TOPK="$TOPK" \
TIMEOUT_MS="$TIMEOUT_MS" \
MAX_RETRIES="$MAX_RETRIES" \
PAYLOAD_BYTES="$PAYLOAD_BYTES" \
HISTORY_SOURCE="$HISTORY_SOURCE" \
bash scripts/test_go_brpc_client_probe.sh

echo
echo "== Recent KVC MSet/MGet probe logs =="
kubectl -n "$NAMESPACE" logs "$TARGET" -c "$CONTAINER" --since="$SINCE" \
  | grep "method=KvcMSetMGetProbe" \
  | tail -50 || true
