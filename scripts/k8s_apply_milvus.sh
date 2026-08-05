#!/usr/bin/env bash
# scripts/k8s_apply_milvus.sh
#
# 部署 Milvus standalone 到 master 并验证.
# 用法: bash scripts/k8s_apply_milvus.sh
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
DEPLOY="${DEPLOY:-milvus-standalone}"

kubectl apply -f k8s/deployment-milvus-standalone.yaml
kubectl -n "$NAMESPACE" rollout status "deploy/${DEPLOY}" --timeout=300s

echo "== milvus pod =="
kubectl -n "$NAMESPACE" get pods -o wide -l app=milvus-standalone

echo "== healthz =="
kubectl -n "$NAMESPACE" exec "deploy/${DEPLOY}" -- \
  curl -sf http://127.0.0.1:9091/healthz && echo

echo "== service =="
kubectl -n "$NAMESPACE" get svc milvus-standalone

echo "下一步: port-forward 后运行灌库脚本:"
echo "  kubectl -n $NAMESPACE port-forward svc/milvus-standalone 19530:19530 &"
echo "  python3 scripts/load_item_embeddings_to_milvus.py --milvus_host 127.0.0.1"
