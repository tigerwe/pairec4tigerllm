#!/bin/bash
# pairec Go 推荐服务启动脚本
# 在容器启动时将环境变量注入配置文件

set -e

CONFIG_PATH="${CONFIG_PATH:-/app/configs/pairec_config.json}"
INFERENCE_HOST="${INFERENCE_HOST:-localhost}"
INFERENCE_PORT="${INFERENCE_PORT:-18000}"
PAIREC_PORT="${PAIREC_PORT:-18080}"
PAIREC_ALSO_LOG_TO_STDERR="${PAIREC_ALSO_LOG_TO_STDERR:-true}"

echo "========================================="
echo "PaiRec4TigerLLM Recommendation Service"
echo "========================================="
echo "Config path:     $CONFIG_PATH"
echo "Inference:       ${INFERENCE_HOST}:${INFERENCE_PORT}"
echo "Pairec port:     $PAIREC_PORT"
echo "Log to stderr:   $PAIREC_ALSO_LOG_TO_STDERR"
echo ""

# 检查配置文件
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found at $CONFIG_PATH"
    exit 1
fi

# === 将环境变量注入配置文件 ===
# 替换 server_url 为实际推理服务地址
TMP_CONFIG=$(mktemp)
sed \
    -e "s|\\\\\"server_url\\\\\"[[:space:]]*:[[:space:]]*\\\\\"[^\\\\\"]*\\\\\"|\\\\\"server_url\\\\\":\\\\\"http://${INFERENCE_HOST}:${INFERENCE_PORT}\\\\\"|g" \
    -e "s|\"server_url\"[[:space:]]*:[[:space:]]*\"[^\"]*\"|\"server_url\":\"http://${INFERENCE_HOST}:${INFERENCE_PORT}\"|g" \
    "$CONFIG_PATH" > "$TMP_CONFIG"
CONFIG_PATH="$TMP_CONFIG"
export CONFIG_PATH

echo "Injected inference URL: http://${INFERENCE_HOST}:${INFERENCE_PORT}"

# 等待推理服务就绪
echo "Waiting for inference service at http://${INFERENCE_HOST}:${INFERENCE_PORT} ..."
for i in $(seq 1 120); do
    if curl -fsS "http://${INFERENCE_HOST}:${INFERENCE_PORT}/health" > /dev/null 2>&1; then
        echo "Inference service ready after ${i}s"
        break
    fi
    sleep 1
done

echo ""
echo "Starting pairec server on :${PAIREC_PORT}..."
exec /app/pairec-server
