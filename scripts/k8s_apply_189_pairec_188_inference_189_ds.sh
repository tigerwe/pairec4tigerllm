#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-inference-brpc-trtllm}"
PAIREC_DEPLOYMENT="${PAIREC_DEPLOYMENT:-pairec}"
INFERENCE_MANIFEST="${INFERENCE_MANIFEST:-k8s/deployment-inference-brpc-trtllm-cross-node-188-ds189.yaml}"
PAIREC_MANIFEST="${PAIREC_MANIFEST:-k8s/deployment-pairec-brpc-master.yaml}"
APPLY_CONFIGMAP="${APPLY_CONFIGMAP:-1}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-15m}"

echo "Applying cross-node topology:"
echo "  189 PaiRec -> brpc/TCP -> 188 inference -> 189 DataSystem"
echo

kubectl apply -f k8s/namespace.yaml

if [ "$APPLY_CONFIGMAP" = "1" ]; then
  kubectl apply -f k8s/configmap-brpc.yaml
fi

kubectl apply -f "$INFERENCE_MANIFEST"
kubectl apply -f "$PAIREC_MANIFEST"

echo
echo "== Wait for inference rollout =="
kubectl -n "$NAMESPACE" rollout status "deploy/${INFERENCE_DEPLOYMENT}" --timeout="$WAIT_TIMEOUT"

echo
echo "== Wait for PaiRec rollout =="
kubectl -n "$NAMESPACE" rollout status "deploy/${PAIREC_DEPLOYMENT}" --timeout=5m

echo
echo "== Pods =="
kubectl -n "$NAMESPACE" get pods -o wide | grep -E 'pairec|inference-brpc-trtllm|datasystem-pool' || true

echo
echo "Expected route:"
echo "  189 PaiRec -> brpc/TCP -> 188 inference pod -> 189 DataSystem worker 141.61.91.189:18481"
echo
echo "Verify inference fixed DataSystem endpoint:"
echo "  kubectl -n ${NAMESPACE} logs deploy/${INFERENCE_DEPLOYMENT} -c brpc-inference --tail=160 | grep -E 'with host endpoint|with ServiceDiscovery|Init KvCache|Rank 0 is using GPU'"
echo
echo "Verify PaiRec request:"
echo "  kubectl -n ${NAMESPACE} exec deploy/${PAIREC_DEPLOYMENT} -- wget -q -O - --header='Content-Type: application/json' --post-data='{\"scene_id\":\"home_feed\",\"uid\":\"6312\",\"size\":10}' http://127.0.0.1:18080/api/recommend"
