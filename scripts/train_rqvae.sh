#!/bin/bash
# RQ-VAE 训练脚本

set -e

echo "========================================="
echo "Training RQ-VAE Model"
echo "========================================="

# 默认参数
DATA_PATH="${DATA_PATH:-./data/tenrec/item_features.npy}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/rqvae}"
LOG_DIR="${LOG_DIR:-./logs/rqvae}"

# 模型参数
INPUT_DIM="${INPUT_DIM:-1}"
EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
NUM_QUANTIZERS="${NUM_QUANTIZERS:-4}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
HIDDEN_DIMS="${HIDDEN_DIMS:-256 128}"

# 训练参数
NUM_EPOCHS="${NUM_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
DEVICE="${DEVICE:-cuda}"

echo "Configuration:"
echo "  Data path: $DATA_PATH"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Log dir: $LOG_DIR"
echo "  Device: $DEVICE"
echo ""

# 创建目录
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# 运行训练
cd "$(dirname "$0")/.."

python -m training.rqvae.train \
    --data_path "$DATA_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_dir "$LOG_DIR" \
    --input_dim "$INPUT_DIM" \
    --embedding_dim "$EMBEDDING_DIM" \
    --num_quantizers "$NUM_QUANTIZERS" \
    --codebook_size "$CODEBOOK_SIZE" \
    --hidden_dims $HIDDEN_DIMS \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --device "$DEVICE"

echo ""
echo "RQ-VAE training completed!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
