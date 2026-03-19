#!/bin/bash
# 一键启动全流程

set -e

echo "========================================="
echo "PaiRec4TigerLLM - Full Pipeline"
echo "========================================="

# 检查依赖
check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed"
        exit 1
    fi
}

echo "Checking dependencies..."
check_dependency python
check_dependency go
echo "All dependencies are available"
echo ""

# 步骤 1: 数据预处理
echo "Step 1: Data Preprocessing"
echo "-----------------------------------------"
if [ ! -f "./data/tenrec/processed/train_sequences.json" ]; then
    echo "Running data preprocessing..."
    python scripts/preprocess_data.py
else
    echo "Data already preprocessed, skipping..."
fi
echo ""

# 步骤 2: 启动 TensorRT-LLM 推理服务（后台）
echo "Step 2: Starting TensorRT-LLM Inference Server"
echo "-----------------------------------------"
if pgrep -f "trt_llm/server.py" > /dev/null; then
    echo "Inference server already running"
else
    echo "Starting inference server on port 8000..."
    ./scripts/start_trt_server.sh &
    INFERENCE_PID=$!
    
    # 等待服务启动
    echo "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null; then
            echo "Inference server is ready!"
            break
        fi
        sleep 1
    done
fi
echo ""

# 步骤 3: 启动 pairec 服务
echo "Step 3: Starting PaiRec Service"
echo "-----------------------------------------"
echo "Starting pairec service on port 8080..."
./scripts/start_pairec.sh

# 清理（当 pairec 服务停止时）
if [ ! -z "$INFERENCE_PID" ]; then
    echo "Stopping inference server..."
    kill $INFERENCE_PID 2>/dev/null || true
fi

echo ""
echo "========================================="
echo "All services stopped"
echo "========================================="
