#!/bin/bash
# 启动 pairec 推荐服务

set -e

echo "========================================="
echo "Starting PaiRec Service"
echo "========================================="

# 默认参数
CONFIG_PATH="${CONFIG_PATH:-./configs/pairec_config.json}"
PAIREC_ALSO_LOG_TO_STDERR="${PAIREC_ALSO_LOG_TO_STDERR:-true}"

echo "Configuration:"
echo "  Config path: $CONFIG_PATH"
echo "  Also log to stderr: $PAIREC_ALSO_LOG_TO_STDERR"
echo ""

# 检查配置文件
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

cd "$(dirname "$0")/../services"

# 检查配置文件路径（如果是相对路径，转换为相对于项目根目录）
if [[ ! "$CONFIG_PATH" = /* ]]; then
    CONFIG_PATH="../$CONFIG_PATH"
fi
export CONFIG_PATH

# 启动服务
go run -mod=vendor main.go \
    --config "$CONFIG_PATH" \
    --alsologtostderr="$PAIREC_ALSO_LOG_TO_STDERR"
