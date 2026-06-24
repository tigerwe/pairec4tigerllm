#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
DELETE_LEGACY_DATASYSTEM="${DELETE_LEGACY_DATASYSTEM:-1}"

kubectl apply -f k8s/namespace.yaml

if [ "$DELETE_LEGACY_DATASYSTEM" = "1" ]; then
  kubectl -n "$NAMESPACE" delete deployment datasystem --ignore-not-found=true
fi

kubectl apply -f k8s/deployment-datasystem-pool-hostnetwork.yaml
kubectl -n "$NAMESPACE" rollout status deploy/datasystem-pool-etcd --timeout=5m
kubectl -n "$NAMESPACE" rollout status daemonset/datasystem-pool-worker --timeout=5m
kubectl -n "$NAMESPACE" get pods -l app.kubernetes.io/part-of=datasystem-pool -o wide
