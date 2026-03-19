# -*- coding: utf-8 -*-
"""
Decoder 训练脚本.

提供生成式召回模型的训练流程.
"""

import os
import argparse
import json
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from .model import GenerativeDecoder


class SequenceDataset(Dataset):
    """序列数据集.

    将用户交互序列转换为训练样本.
    """

    def __init__(
        self,
        sequences: List[List[List[int]]],
        max_seq_len: int = 50,
        pad_token_id: int = 0
    ):
        """初始化数据集.

        Args:
            sequences: 语义 ID 序列列表，每个序列是 [seq_len, num_quantizers]
            max_seq_len: 最大序列长度
            pad_token_id: 填充 token ID
        """
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取一个训练样本.

        Returns:
            input_ids: 输入序列 [max_seq_len, num_quantizers]
            labels: 目标标签 [max_seq_len, num_quantizers]
            attention_mask: 注意力掩码 [max_seq_len]
        """
        seq = self.sequences[idx]
        seq_len = len(seq)
        num_quantizers = len(seq[0]) if seq_len > 0 else 1

        # 截断或填充
        if seq_len > self.max_seq_len:
            seq = seq[-self.max_seq_len:]
            seq_len = self.max_seq_len

        # 创建输入和标签（输入是前 n-1 个，标签是后 n-1 个）
        input_seq = seq[:-1] if len(seq) > 1 else seq
        label_seq = seq[1:] if len(seq) > 1 else seq

        # 填充
        pad_length = self.max_seq_len - len(input_seq)

        # 输入填充
        input_ids = input_seq + [[self.pad_token_id] * num_quantizers] * pad_length
        labels = label_seq + [[self.pad_token_id] * num_quantizers] * (pad_length + 1)
        labels = labels[:self.max_seq_len]  # 确保长度一致

        # 注意力掩码
        attention_mask = [1] * len(input_seq) + [0] * pad_length

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.float)
        )


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """批处理函数.

    Args:
        batch: 样本列表

    Returns:
        批处理后的张量
    """
    input_ids, labels, attention_masks = zip(*batch)
    return (
        torch.stack(input_ids),
        torch.stack(labels),
        torch.stack(attention_masks)
    )


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int = 10
) -> Dict[str, float]:
    """计算评估指标.

    Args:
        logits: 模型输出 [batch_size, seq_len, num_quantizers, vocab_size]
        labels: 目标标签 [batch_size, seq_len, num_quantizers]
        attention_mask: 注意力掩码 [batch_size, seq_len]
        k: Top-k 评估

    Returns:
        指标字典
    """
    batch_size, seq_len, num_quantizers, vocab_size = logits.shape

    # 重塑
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    mask_flat = attention_mask.unsqueeze(-1).expand(-1, -1, num_quantizers).reshape(-1)

    # 只考虑有效位置
    valid_indices = mask_flat.bool()
    logits_flat = logits_flat[valid_indices]
    labels_flat = labels_flat[valid_indices]

    if len(labels_flat) == 0:
        return {'accuracy': 0.0, f'hit@{k}': 0.0}

    # 预测
    predictions = logits_flat.argmax(dim=-1)

    # 准确率
    accuracy = (predictions == labels_flat).float().mean().item()

    # Hit@K
    _, top_k_indices = logits_flat.topk(k, dim=-1)
    hits = (top_k_indices == labels_flat.unsqueeze(-1)).any(dim=-1).float().mean().item()

    return {
        'accuracy': accuracy,
        f'hit@{k}': hits
    }


def train_decoder(
    model: GenerativeDecoder,
    train_sequences: List[List[List[int]]],
    val_sequences: Optional[List[List[List[int]]]] = None,
    num_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    max_seq_len: int = 50,
    device: str = 'cuda',
    checkpoint_dir: str = './checkpoints',
    log_dir: str = './logs',
    save_interval: int = 5,
    patience: int = 10,
    warmup_steps: int = 1000
) -> GenerativeDecoder:
    """训练 Decoder 模型.

    Args:
        model: Decoder 模型
        train_sequences: 训练序列
        val_sequences: 验证序列
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        weight_decay: 权重衰减
        max_seq_len: 最大序列长度
        device: 训练设备
        checkpoint_dir: 检查点目录
        log_dir: 日志目录
        save_interval: 保存间隔
        patience: 早停耐心值
        warmup_steps: 预热步数

    Returns:
        训练后的模型
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Training on device: {device}")
    print(f"Train sequences: {len(train_sequences)}")
    if val_sequences:
        print(f"Val sequences: {len(val_sequences)}")

    # 数据集
    train_dataset = SequenceDataset(train_sequences, max_seq_len, model.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95)
    )

    # 学习率调度器（带预热）
    total_steps = len(train_loader) * num_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # TensorBoard
    writer = SummaryWriter(log_dir)

    # 训练状态
    best_val_loss = float('inf')
    epochs_no_improve = 0
    global_step = 0

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (input_ids, labels, attention_mask) in enumerate(progress_bar):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            # 前向传播
            logits, loss = model(input_ids, attention_mask, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # 记录
            epoch_losses.append(loss.item())
            global_step += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"
            })

            # 写入 TensorBoard
            if global_step % 10 == 0:
                writer.add_scalar('Train/loss', loss.item(), global_step)
                writer.add_scalar('Train/lr', scheduler.get_last_lr()[0], global_step)

        # 平均训练损失
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar('Epoch/train_loss', avg_train_loss, epoch)

        # 验证
        if val_sequences is not None:
            model.eval()
            val_losses = []
            val_metrics_list = []

            val_dataset = SequenceDataset(val_sequences, max_seq_len, model.pad_token_id)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )

            with torch.no_grad():
                for input_ids, labels, attention_mask in val_loader:
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    attention_mask = attention_mask.to(device)

                    logits, loss = model(input_ids, attention_mask, labels)

                    val_losses.append(loss.item())

                    # 计算指标
                    metrics = compute_metrics(logits, labels, attention_mask, k=10)
                    val_metrics_list.append(metrics)

            avg_val_loss = sum(val_losses) / len(val_losses)

            # 平均指标
            avg_metrics = {
                key: sum(m[key] for m in val_metrics_list) / len(val_metrics_list)
                for key in val_metrics_list[0].keys()
            }

            print(f"Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}, "
                  f"Accuracy: {avg_metrics['accuracy']:.4f}, "
                  f"Hit@10: {avg_metrics['hit@10']:.4f}")

            writer.add_scalar('Epoch/val_loss', avg_val_loss, epoch)
            writer.add_scalar('Epoch/val_accuracy', avg_metrics['accuracy'], epoch)
            writer.add_scalar('Epoch/val_hit@10', avg_metrics['hit@10'], epoch)

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0

                # 保存最佳模型
                best_model_path = os.path.join(checkpoint_dir, 'decoder_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'config': {
                        'vocab_size': model.vocab_size,
                        'num_quantizers': model.num_quantizers,
                        'embedding_dim': model.embedding_dim,
                        'num_layers': len(model.transformer_blocks),
                        'num_heads': model.transformer_blocks[0].attention.num_heads,
                        'max_seq_len': model.max_seq_len
                    }
                }, best_model_path)
                print(f"Best model saved to {best_model_path}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # 定期保存
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'decoder_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    writer.close()
    print("\nTraining completed!")

    return model


def main():
    """命令行入口."""
    parser = argparse.ArgumentParser(description='Train Decoder model')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training sequences (.json file)')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation sequences')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/decoder',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs/decoder',
                        help='Directory for tensorboard logs')
    parser.add_argument('--vocab_size', type=int, default=256,
                        help='Vocabulary size')
    parser.add_argument('--num_quantizers', type=int, default=4,
                        help='Number of quantization layers')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--ffn_dim', type=int, default=1024,
                        help='FFN dimension')
    parser.add_argument('--max_seq_len', type=int, default=50,
                        help='Maximum sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载数据
    print(f"Loading training data from {args.train_data}")
    with open(args.train_data, 'r') as f:
        train_sequences = json.load(f)

    val_sequences = None
    if args.val_data:
        print(f"Loading validation data from {args.val_data}")
        with open(args.val_data, 'r') as f:
            val_sequences = json.load(f)

    # 创建模型
    model = GenerativeDecoder(
        vocab_size=args.vocab_size,
        num_quantizers=args.num_quantizers,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        max_seq_len=args.max_seq_len
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    train_decoder(
        model=model,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_seq_len=args.max_seq_len,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )


if __name__ == '__main__':
    main()
