#!/usr/bin/env bash
set -euo pipefail

kubectl apply -f k8s/namespace.yaml
kubectl -n pairec delete job trt-engine-paged-fmha --ignore-not-found
kubectl apply -f k8s/job-build-trt-engine-paged-fmha.yaml
kubectl -n pairec wait --for=condition=complete job/trt-engine-paged-fmha --timeout=45m
kubectl -n pairec logs job/trt-engine-paged-fmha
