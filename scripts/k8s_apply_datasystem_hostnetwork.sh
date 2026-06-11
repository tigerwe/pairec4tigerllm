#!/usr/bin/env bash
set -euo pipefail

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment-datasystem-hostnetwork.yaml
kubectl -n pairec rollout status deploy/datasystem --timeout=5m
kubectl -n pairec get pods -l app=datasystem -o wide
