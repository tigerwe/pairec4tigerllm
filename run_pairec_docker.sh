#!/bin/bash
# pairec Go 服务一键构建 + 运行
# 用法: bash run_pairec_docker.sh [build|run|stop]
set -e

IMAGE="pairec-server:latest"
DATA_DIR="${PWD}/data"

case "${1:-run}" in
    build)
        echo "=== Building pairec-server image ==="
        docker build -f docker/Dockerfile.pairec -t "$IMAGE" .
        echo "Done: $IMAGE"
        ;;
    run)
        echo "=== Starting pairec-server ==="
        echo "Inference: localhost:18000 (must be running)"
        echo "Pairec:    localhost:18080"
        echo "Data dir:  $DATA_DIR"
        echo ""
        docker run -d --name pairec-server --network host \
            -v "$DATA_DIR:/data" \
            "$IMAGE"
        echo "Container started. Check logs: docker logs -f pairec-server"
        ;;
    stop)
        docker stop pairec-server 2>/dev/null || true
        docker rm pairec-server 2>/dev/null || true
        echo "Stopped."
        ;;
    logs)
        docker logs -f pairec-server
        ;;
    *)
        echo "Usage: bash run_pairec_docker.sh [build|run|stop|logs]"
        ;;
esac
