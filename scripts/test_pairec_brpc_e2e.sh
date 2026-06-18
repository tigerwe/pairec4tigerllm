#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
TARGET="${TARGET:-deployment/pairec}"
PAIREC_CONTAINER="${PAIREC_CONTAINER:-pairec}"
UID_VALUE="${UID_VALUE:-6312}"
SIZE="${SIZE:-10}"
SCENE_ID="${SCENE_ID:-home_feed}"

echo "== brpc proxy health =="
kubectl -n "$NAMESPACE" exec "$TARGET" -c "$PAIREC_CONTAINER" -- \
  curl -fsS http://127.0.0.1:18090/health
echo

echo "== pairec health =="
kubectl -n "$NAMESPACE" exec "$TARGET" -c "$PAIREC_CONTAINER" -- \
  curl -fsS http://127.0.0.1:18080/ping
echo

echo "== pairec recommend through brpc proxy =="
response="$(
  kubectl -n "$NAMESPACE" exec "$TARGET" -c "$PAIREC_CONTAINER" -- \
    curl -fsS -X POST http://127.0.0.1:18080/api/recommend \
      -H 'Content-Type: application/json' \
      -d "{\"uid\":\"${UID_VALUE}\",\"size\":${SIZE},\"scene_id\":\"${SCENE_ID}\"}"
)"
echo "$response"
echo "$response" | grep -q '"code":200'
echo "pairec brpc e2e ok uid=${UID_VALUE} size=${SIZE} scene=${SCENE_ID}"
