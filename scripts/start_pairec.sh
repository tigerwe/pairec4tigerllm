#!/bin/bash
# 启动 pairec 推荐服务

set -e

echo "========================================="
echo "Starting PaiRec Service"
echo "========================================="

# 默认参数
CONFIG_PATH="${CONFIG_PATH:-./configs/pairec_config.json}"
PORT="${PORT:-8080}"

echo "Configuration:"
echo "  Config path: $CONFIG_PATH"
echo "  Port: $PORT"
echo ""

# 检查配置文件
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

cd "$(dirname "$0")/.."

# 启动服务
go run services/main.go \
    --config "$CONFIG_PATH" \
    --port "$PORT"
