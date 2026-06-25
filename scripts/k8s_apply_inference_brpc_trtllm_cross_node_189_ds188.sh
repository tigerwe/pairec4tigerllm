#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
DEPLOYMENT="${DEPLOYMENT:-inference-brpc-trtllm}"
MANIFEST="${MANIFEST:-k8s/deployment-inference-brpc-trtllm-cross-node-189-ds188.yaml}"

kubectl apply -f k8s/namespace.yaml
kubectl apply -f "$MANIFEST"
kubectl -n "$NAMESPACE" rollout status "deploy/${DEPLOYMENT}" --timeout=10m
kubectl -n "$NAMESPACE" get pods -l app="$DEPLOYMENT" -o wide

echo
echo "Expected route:"
echo "  188 PaiRec -> brpc/TCP -> 189 inference pod -> 188 DataSystem worker 141.61.91.188:18481"
echo
echo "Verify fixed DataSystem endpoint:"
echo "  kubectl -n ${NAMESPACE} logs deploy/${DEPLOYMENT} -c brpc-inference --tail=160 | grep -E 'with host endpoint|with ServiceDiscovery|Init KvCache'"
echo
echo "Roll back to the default deployment:"
echo "  bash scripts/k8s_apply_inference_brpc_trtllm.sh"
