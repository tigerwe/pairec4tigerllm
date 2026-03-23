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
INPUT_DIM="${INPUT_DIM:-14}"
EMBEDDING_DIM="${EMBEDDING_DIM:-32}"           # 64->32，降低复杂度
NUM_QUANTIZERS="${NUM_QUANTIZERS:-3}"          # 4->3，减少层数
CODEBOOK_SIZE="${CODEBOOK_SIZE:-128}"          # 256->128，减少码本大小
HIDDEN_DIMS="${HIDDEN_DIMS:-128 64}"

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
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

# 提示：如果设置了 CUDA_VISIBLE_DEVICES，PyTorch 会将可见的 GPU 映射为 cuda:0, cuda:1, ...
# 例如：CUDA_VISIBLE_DEVICES=4 时，cuda:0 实际上对应物理 GPU 4
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Note: CUDA_VISIBLE_DEVICES is set. Physical GPU $CUDA_VISIBLE_DEVICES is mapped as cuda:0"
fi

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
