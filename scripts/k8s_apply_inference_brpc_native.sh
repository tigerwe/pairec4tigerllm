#!/usr/bin/env bash
set -euo pipefail

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment-inference-brpc-native.yaml
kubectl -n pairec rollout status deploy/inference-brpc-native --timeout=5m
kubectl -n pairec get pods -l app=inference-brpc-native -o wide
