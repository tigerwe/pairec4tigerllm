#!/bin/bash
# Decoder (GPT2) 训练脚本

set -e

echo "========================================="
echo "Training Decoder Model"
echo "========================================="

# 默认参数
TRAIN_DATA="${TRAIN_DATA:-./data/tenrec/processed/train_sequences.json}"
VAL_DATA="${VAL_DATA:-./data/tenrec/processed/test_sequences.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/decoder}"
LOG_DIR="${LOG_DIR:-./logs/decoder}"

# 模型参数
VOCAB_SIZE="${VOCAB_SIZE:-256}"
NUM_QUANTIZERS="${NUM_QUANTIZERS:-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-8}"
FFN_DIM="${FFN_DIM:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-50}"

# 训练参数
NUM_EPOCHS="${NUM_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
DEVICE="${DEVICE:-cuda}"

echo "Configuration:"
echo "  Train data: $TRAIN_DATA"
echo "  Val data: $VAL_DATA"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Log dir: $LOG_DIR"
echo "  Device: $DEVICE"
echo ""

# 创建目录
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# 运行训练
cd "$(dirname "$0")/.."

python -m training.decoder.train \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_dir "$LOG_DIR" \
    --vocab_size "$VOCAB_SIZE" \
    --num_quantizers "$NUM_QUANTIZERS" \
    --embedding_dim "$EMBEDDING_DIM" \
    --num_layers "$NUM_LAYERS" \
    --num_heads "$NUM_HEADS" \
    --ffn_dim "$FFN_DIM" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --device "$DEVICE"

echo ""
echo "Decoder training completed!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
