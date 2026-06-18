#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
TARGET="${TARGET:-deployment/inference-brpc-native}"
CONTAINER="${CONTAINER:-brpc-inference}"
SERVER="${SERVER:-127.0.0.1:18100}"
REQUESTS="${REQUESTS:-1}"
TOPK="${TOPK:-5}"

kubectl -n "$NAMESPACE" exec "$TARGET" -c "$CONTAINER" -- \
  /opt/pairec-brpc/bin/brpc_recommend_client \
  --server="$SERVER" \
  --method=health \
  --requests=1

kubectl -n "$NAMESPACE" exec "$TARGET" -c "$CONTAINER" -- \
  /opt/pairec-brpc/bin/brpc_recommend_client \
  --server="$SERVER" \
  --method=recommend \
  --requests="$REQUESTS" \
  --topk="$TOPK" \
  --print_raw_json=0
