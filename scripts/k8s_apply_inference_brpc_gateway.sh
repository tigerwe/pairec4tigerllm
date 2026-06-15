#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment-inference-brpc-image.yaml

kubectl -n "$NAMESPACE" rollout status deploy/inference --timeout=15m
kubectl -n "$NAMESPACE" get pods -l app=inference -o wide
kubectl -n "$NAMESPACE" get svc inference -o wide
