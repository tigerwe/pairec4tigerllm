#!/usr/bin/env bash
set -euo pipefail

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap-pairec-brpc.yaml
kubectl apply -f k8s/deployment-pairec-brpc-hostpath.yaml
kubectl -n pairec rollout status deploy/pairec --timeout=5m
kubectl -n pairec get pods -l app=pairec -o wide
