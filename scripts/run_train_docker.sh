#!/bin/bash
# ============================================
# PaiRec4TigerLLM - 训练容器一键脚本
# ============================================
# 用法:
#   构建:  bash scripts/run_train_docker.sh build
#   训练:  bash scripts/run_train_docker.sh train
#   进入:  bash scripts/run_train_docker.sh shell
#   导出:  bash scripts/run_train_docker.sh export
# ============================================

set -e

IMAGE="pairec-train:latest"
CONTAINER="pairec-train"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

build() {
    echo "=== 构建训练镜像 ==="
    docker build -f docker/Dockerfile.train -t "$IMAGE" "$PROJECT_DIR"
    echo "构建完成: $IMAGE"
}

export_image() {
    local out="pairec-train.tar.gz"
    echo "=== 导出镜像 ==="
    docker save "$IMAGE" | gzip > "$out"
    echo "导出完成: $out ($(du -h "$out" | cut -f1))"
    echo ""
    echo "迁移到目标机器:"
    echo "  scp $out user@188:/path/"
    echo "  ssh user@188 'gunzip -c /path/$out | docker load'"
}

train() {
    echo "=== 启动训练 ==="
    docker run --rm --gpus all \
        --name "$CONTAINER" \
        -v "$PROJECT_DIR:/workspace/pairec4tigerllm" \
        -v "$HOME/models/Qwen3-0.6B:/workspace/models/Qwen3-0.6B:ro" \
        -v "$(dirname "$PROJECT_DIR")/data:/workspace/data:ro" \
        -v "$PROJECT_DIR/checkpoints:/workspace/checkpoints" \
        -w /workspace/pairec4tigerllm \
        -e PYTHONPATH=/workspace/pairec4tigerllm \
        "$IMAGE" \
        python -m training.decoder.train \
            --backbone qwen3 \
            --qwen3_model_path /workspace/models/Qwen3-0.6B \
            --train_data /workspace/data/tenrec/processed/train_sequences.json \
            --num_epochs 10 --batch_size 4 --learning_rate 5e-5 \
            --lora_rank 8 --lora_alpha 16 \
            --checkpoint_dir /workspace/checkpoints/decoder_qwen3
}

shell() {
    echo "=== 进入容器 Shell ==="
    docker run --rm -it --gpus all \
        --name "$CONTAINER" \
        -v "$PROJECT_DIR:/workspace/pairec4tigerllm" \
        -v "$HOME/models/Qwen3-0.6B:/workspace/models/Qwen3-0.6B:ro" \
        -v "$(dirname "$PROJECT_DIR")/data:/workspace/data:ro" \
        -v "$PROJECT_DIR/checkpoints:/workspace/checkpoints" \
        -w /workspace/pairec4tigerllm \
        -e PYTHONPATH=/workspace/pairec4tigerllm \
        "$IMAGE" /bin/bash
}

case "${1:-}" in
    build)   build ;;
    train)   train ;;
    shell)   shell ;;
    export)  export_image ;;
    *)
        echo "Usage: $0 {build|train|shell|export}"
        echo ""
        echo "  build   - 构建训练镜像"
        echo "  train   - 启动训练 (前台)"
        echo "  shell   - 进入容器交互 Shell"
        echo "  export  - 导出镜像为 tar.gz (用于迁移到 188)"
        exit 1
        ;;
esac
