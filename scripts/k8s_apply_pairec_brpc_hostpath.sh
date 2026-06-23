#!/usr/bin/env bash
set -euo pipefail

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap-brpc.yaml
kubectl apply -f k8s/deployment-pairec-hostpath.yaml
kubectl -n pairec rollout restart deploy/pairec
kubectl -n pairec rollout status deploy/pairec --timeout=5m
kubectl -n pairec get pods -l app=pairec -o wide
