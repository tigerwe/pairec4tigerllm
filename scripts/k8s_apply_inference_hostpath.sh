#!/usr/bin/env bash
set -euo pipefail

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment-inference-hostpath.yaml
kubectl -n pairec rollout status deploy/inference --timeout=10m
kubectl -n pairec get pods -l app=inference -o wide
