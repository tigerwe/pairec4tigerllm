#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
MANIFEST="${MANIFEST:-k8s/deployment-brpc-pressure-target-188.yaml}"
DEPLOYMENT="${DEPLOYMENT:-brpc-pressure-target}"
ENDPOINT="${ENDPOINT:-192.168.100.11:18101}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-300s}"
PROBE_BIN="${PROBE_BIN:-/tmp/probe-go-brpc-client}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[ -f "$MANIFEST" ] || die "manifest not found: $MANIFEST"
command -v kubectl >/dev/null 2>&1 || die "kubectl is required"
command -v go >/dev/null 2>&1 || die "go is required"

kubectl apply -f "$MANIFEST"
kubectl -n "$NAMESPACE" rollout status "deployment/${DEPLOYMENT}" --timeout="$ROLLOUT_TIMEOUT"
kubectl -n "$NAMESPACE" get pod -l "app=${DEPLOYMENT}" -o wide

GOPROXY="${GOPROXY:-off}" GOSUMDB="${GOSUMDB:-off}" \
  go build -mod=vendor -o "$PROBE_BIN" ./scripts/probe_go_brpc_client.go

"$PROBE_BIN" \
  --endpoint="$ENDPOINT" \
  --method=health \
  --requests=1 \
  --concurrency=1 \
  --timeout_ms=5000 \
  --max_retries=0

echo "BRPC pressure target is ready: ${ENDPOINT}"
