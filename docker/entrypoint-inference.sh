#!/bin/sh
# 推理服务启动脚本 (Qwen3 + TensorRT-LLM + DataSystem)

set -e

MODEL_PATH="${MODEL_PATH:-/app/checkpoints/decoder/decoder_best.pt}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
BACKBONE="${BACKBONE:-qwen3}"
QWEN3_MODEL_PATH="${QWEN3_MODEL_PATH:-/app/models/Qwen3-0.6B}"
USE_TRT_LLM="${USE_TRT_LLM:-false}"
TRT_ENGINE_DIR="${TRT_ENGINE_DIR:-}"
DATASYSTEM_HOST="${DATASYSTEM_HOST:-}"
DATASYSTEM_PORT="${DATASYSTEM_PORT:-31501}"
TRT_MAX_KV_TOKENS="${TRT_MAX_KV_TOKENS:-2048}"
TRT_SCHEDULER_POLICY="${TRT_SCHEDULER_POLICY:-max_utilization}"
TRT_MAX_INPUT_LEN="${TRT_MAX_INPUT_LEN:-64}"
TRT_NUM_SAMPLES="${TRT_NUM_SAMPLES:-8}"
TRT_RESULT_CACHE_ENABLED="${TRT_RESULT_CACHE_ENABLED:-1}"

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
echo "TRT engine dir:  ${TRT_ENGINE_DIR:-<disabled>}"
echo "DataSystem:      ${DATASYSTEM_HOST:-<disabled>}:${DATASYSTEM_PORT}"
echo "TRT KV tokens:   $TRT_MAX_KV_TOKENS"
echo "TRT samples:     $TRT_NUM_SAMPLES"
echo "Result cache:    $TRT_RESULT_CACHE_ENABLED"
echo "LD_PRELOAD:      ${LD_PRELOAD:-<unset>}"
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

if [ -n "$TRT_ENGINE_DIR" ]; then
    if [ ! -d "$TRT_ENGINE_DIR" ]; then
        echo "ERROR: TRT engine dir not found at $TRT_ENGINE_DIR"
        exit 1
    fi
    ARGS="$ARGS --trt_engine_dir $TRT_ENGINE_DIR"
fi

if [ -n "$DATASYSTEM_HOST" ]; then
    ARGS="$ARGS --datasystem_host $DATASYSTEM_HOST --datasystem_port $DATASYSTEM_PORT"
fi

ARGS="$ARGS --trt_max_kv_tokens $TRT_MAX_KV_TOKENS"
ARGS="$ARGS --trt_scheduler_policy $TRT_SCHEDULER_POLICY"
ARGS="$ARGS --trt_max_input_len $TRT_MAX_INPUT_LEN"
ARGS="$ARGS --trt_num_samples $TRT_NUM_SAMPLES"
ARGS="$ARGS --trt_result_cache_enabled $TRT_RESULT_CACHE_ENABLED"

echo ""
echo "Starting inference server..."
exec python -m inference.trt_llm.server $ARGS
