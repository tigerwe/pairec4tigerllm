#!/bin/sh
# pairec Go 推荐服务启动脚本

set -e

CONFIG_PATH="${CONFIG_PATH:-/app/configs/pairec_config.json}"
PORT="${PORT:-8080}"

echo "========================================="
echo "PaiRec4TigerLLM Recommendation Service"
echo "========================================="
echo "Config path: $CONFIG_PATH"
echo "Port:        $PORT"
echo ""

# 检查配置文件
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found at $CONFIG_PATH"
    exit 1
fi

# 检查推理服务是否可达（如果配置了健康检查等待）
INFERENCE_URL="${INFERENCE_URL:-}"
if [ -n "$INFERENCE_URL" ]; then
    echo "Waiting for inference service at $INFERENCE_URL ..."
    for i in $(seq 1 60); do
        if curl -fsS "${INFERENCE_URL}/health" > /dev/null 2>&1; then
            echo "Inference service ready after ${i}s"
            break
        fi
        sleep 1
    done
fi

echo ""
echo "Starting pairec server..."
exec /app/pairec-server
