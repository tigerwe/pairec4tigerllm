#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
MASTER_NODE_NAME="${MASTER_NODE_NAME:-master}"
REMOVE_MASTER_GPU_LABEL="${REMOVE_MASTER_GPU_LABEL:-1}"
DELETE_MASTER_NVIDIA_PLUGIN_PODS="${DELETE_MASTER_NVIDIA_PLUGIN_PODS:-1}"
DELETE_INFERENCE_PODS="${DELETE_INFERENCE_PODS:-1}"

echo "Cleaning failed master/189 GPU inference attempt"
echo "  namespace: ${NAMESPACE}"
echo "  master node: ${MASTER_NODE_NAME}"
echo

echo "== Scale inference-brpc-trtllm to 0 =="
kubectl -n "$NAMESPACE" scale deploy/inference-brpc-trtllm --replicas=0 --ignore-not-found=true

if [ "$DELETE_INFERENCE_PODS" = "1" ]; then
  echo
  echo "== Delete current inference-brpc-trtllm pods =="
  kubectl -n "$NAMESPACE" delete pod -l app=inference-brpc-trtllm --force --grace-period=0 --ignore-not-found=true
fi

if [ "$REMOVE_MASTER_GPU_LABEL" = "1" ]; then
  echo
  echo "== Remove pairec/gpu label from ${MASTER_NODE_NAME} =="
  kubectl label node "$MASTER_NODE_NAME" pairec/gpu- || true
fi

if [ "$DELETE_MASTER_NVIDIA_PLUGIN_PODS" = "1" ]; then
  echo
  echo "== Delete NVIDIA device plugin pods on ${MASTER_NODE_NAME} =="
  mapfile -t master_plugin_pods < <(
    kubectl -n kube-system get pods -o wide --no-headers \
      | awk -v node="$MASTER_NODE_NAME" 'tolower($1) ~ /nvidia/ && $7 == node {print "pod/" $1}'
  )
  if [ "${#master_plugin_pods[@]}" -gt 0 ]; then
    kubectl -n kube-system delete "${master_plugin_pods[@]}" --force --grace-period=0
  else
    echo "no NVIDIA device plugin pods currently scheduled on ${MASTER_NODE_NAME}"
  fi
fi

echo
echo "== Current inference pods =="
kubectl -n "$NAMESPACE" get pods -l app=inference-brpc-trtllm -o wide || true

echo
echo "== Current NVIDIA device plugin pods =="
kubectl -n kube-system get pods -o wide | grep -i nvidia || true

echo
echo "== Node labels =="
kubectl get node "$MASTER_NODE_NAME" --show-labels

echo
echo "Cleanup done."
