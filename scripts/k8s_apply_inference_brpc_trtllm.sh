#!/usr/bin/env bash
set -euo pipefail

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment-inference-brpc-trtllm.yaml
kubectl -n pairec rollout status deploy/inference-brpc-trtllm --timeout=10m
kubectl -n pairec get pods -l app=inference-brpc-trtllm -o wide
