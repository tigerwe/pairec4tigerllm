#!/bin/sh
# 推理服务启动脚本 (Qwen3 + GPT2 双 backbone)

set -e

MODEL_PATH="${MODEL_PATH:-/app/checkpoints/decoder/decoder_best.pt}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
BACKBONE="${BACKBONE:-qwen3}"
QWEN3_MODEL_PATH="${QWEN3_MODEL_PATH:-/app/models/Qwen3-0.6B}"
USE_TRT_LLM="${USE_TRT_LLM:-false}"

echo "========================================="
echo "PaiRec4TigerLLM Inference Service"
echo "========================================="
echo "Backbone:        $BACKBONE"
echo "Qwen3 path:      $QWEN3_MODEL_PATH"
echo "Model path:      $MODEL_PATH"
echo "Port:            $PORT"
echo "Device:          $DEVICE"
echo "Max batch size:  $MAX_BATCH_SIZE"
echo "Max seq len:     $MAX_SEQ_LEN"
echo "Use TRT-LLM:     $USE_TRT_LLM"
echo ""

# 等待模型文件就绪
if [ ! -f "$MODEL_PATH" ]; then
    echo "WARNING: Model file not found at $MODEL_PATH"
    echo "Waiting for model file..."
    for i in $(seq 1 60); do
        if [ -f "$MODEL_PATH" ]; then
            echo "Model file found after ${i}s"
            break
        fi
        sleep 1
    done
    if [ ! -f "$MODEL_PATH" ]; then
        echo "ERROR: Model file still not found after 60s, exiting"
        exit 1
    fi
fi

# 检查语义 ID 映射文件
SEMANTIC_MAP="/app/data/tenrec/processed/semantic_id_map.json"
if [ ! -f "$SEMANTIC_MAP" ]; then
    echo "WARNING: Semantic ID map not found at $SEMANTIC_MAP"
fi

# 构建启动参数
ARGS="--model_path $MODEL_PATH --port $PORT --device $DEVICE"
ARGS="$ARGS --max_batch_size $MAX_BATCH_SIZE --max_seq_len $MAX_SEQ_LEN"

if [ "$BACKBONE" = "qwen3" ]; then
    ARGS="$ARGS --qwen3_model_path $QWEN3_MODEL_PATH"
    echo "Using Qwen3 prompt-mode backend"
else
    echo "Using GPT2 backend"
fi

if [ "$USE_TRT_LLM" = "true" ]; then
    ARGS="$ARGS --use_trt_llm"
    echo "TensorRT-LLM enabled"
fi

echo ""
echo "Starting inference server..."
exec python /app/inference/trt_llm/server.py $ARGS
