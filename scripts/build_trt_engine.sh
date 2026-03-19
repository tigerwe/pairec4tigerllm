#!/bin/bash
# TensorRT 引擎构建脚本

set -e

echo "========================================="
echo "Building TensorRT Engine"
echo "========================================="

# 默认参数
ONNX_PATH="${ONNX_PATH:-./exported/decoder/decoder_step.onnx}"
OUTPUT_PATH="${OUTPUT_PATH:-./exported/decoder/decoder.engine}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
FP16="${FP16:-true}"

echo "Configuration:"
echo "  ONNX path: $ONNX_PATH"
echo "  Output path: $OUTPUT_PATH"
echo "  Max batch size: $MAX_BATCH_SIZE"
echo "  Max seq len: $MAX_SEQ_LEN"
echo "  FP16: $FP16"
echo ""

# 检查 ONNX 文件是否存在
if [ ! -f "$ONNX_PATH" ]; then
    echo "Error: ONNX file not found at $ONNX_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT_PATH")"

cd "$(dirname "$0")/.."

# 使用 TensorRT-LLM 构建引擎
# 注意：需要已安装 TensorRT-LLM
python inference/trt_llm/build_engine.py \
    --onnx_path "$ONNX_PATH" \
    --output_path "$OUTPUT_PATH" \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    $(if [ "$FP16" = "true" ]; then echo "--fp16"; fi)

echo ""
echo "TensorRT engine built successfully!"
echo "Engine saved to: $OUTPUT_PATH"
