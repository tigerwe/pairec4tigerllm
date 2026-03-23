#!/bin/bash
# 启动 TensorRT-LLM 推理服务

set -e

echo "========================================="
echo "Starting TensorRT-LLM Inference Server"
echo "========================================="

# 默认参数
MODEL_PATH="${MODEL_PATH:-./checkpoints/decoder/decoder_best.pt}"
PORT="${PORT:-18000}"
DEVICE="${DEVICE:-cuda}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"

echo "Configuration:"
echo "  Model path: $MODEL_PATH"
echo "  Port: $PORT"
echo "  Device: $DEVICE"
echo "  Max batch size: $MAX_BATCH_SIZE"
echo "  Max seq len: $MAX_SEQ_LEN"
echo ""

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

cd "$(dirname "$0")/.."

# 启动服务
python inference/trt_llm/server.py \
    --model_path "$MODEL_PATH" \
    --port "$PORT" \
    --device "$DEVICE" \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN"
