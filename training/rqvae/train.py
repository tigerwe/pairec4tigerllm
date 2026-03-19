# -*- coding: utf-8 -*-
"""
RQ-VAE 训练脚本.

提供 RQ-VAE 模型的训练流程，包括数据加载、训练循环和模型保存.
"""

import os
import argparse
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .model import RQVAE


def train_rqvae(
    model: RQVAE,
    train_data: torch.Tensor,
    val_data: Optional[torch.Tensor] = None,
    num_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = 'cuda',
    checkpoint_dir: str = './checkpoints',
    log_dir: str = './logs',
    save_interval: int = 10,
    patience: int = 10
) -> RQVAE:
    """训练 RQ-VAE 模型.

    Args:
        model: RQ-VAE 模型实例
        train_data: 训练数据 [num_items, feature_dim]
        val_data: 验证数据（可选）
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        weight_decay: 权重衰减
        device: 训练设备
        checkpoint_dir: 检查点保存目录
        log_dir: 日志目录
        save_interval: 保存间隔（轮数）
        patience: 早停耐心值

    Returns:
        训练后的模型
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 设备设置
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_data = train_data.to(device)
    if val_data is not None:
        val_data = val_data.to(device)

    print(f"Training on device: {device}")
    print(f"Train data shape: {train_data.shape}")

    # 数据加载器
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 2, verbose=True
    )

    # TensorBoard
    writer = SummaryWriter(log_dir)

    # 训练状态
    best_val_loss = float('inf')
    epochs_no_improve = 0
    global_step = 0

    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (batch_data,) in enumerate(progress_bar):
            # 前向传播
            reconstructed, quantized, ids_distributions, ids_indices = model(batch_data)

            # 计算损失
            loss, loss_dict = model.compute_loss(
                batch_data, reconstructed, quantized, ids_distributions
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 记录损失
            epoch_losses.append(loss_dict['total_loss'])
            global_step += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'recon': f"{loss_dict['reconstruction_loss']:.4f}"
            })

            # 写入 TensorBoard（每 10 步）
            if global_step % 10 == 0:
                for key, value in loss_dict.items():
                    writer.add_scalar(f'Train/{key}', value, global_step)

        # 计算平均训练损失
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")

        # 验证
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                reconstructed, quantized, ids_distributions, _ = model(val_data)
                val_loss, val_loss_dict = model.compute_loss(
                    val_data, reconstructed, quantized, ids_distributions
                )

            avg_val_loss = val_loss_dict['total_loss']
            print(f"Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}")

            # 写入 TensorBoard
            writer.add_scalar('Val/total_loss', avg_val_loss, epoch)
            writer.add_scalar('Val/reconstruction_loss', val_loss_dict['reconstruction_loss'], epoch)

            # 学习率调度
            scheduler.step(avg_val_loss)

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0

                # 保存最佳模型
                best_model_path = os.path.join(checkpoint_dir, 'rqvae_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, best_model_path)
                print(f"Best model saved to {best_model_path}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # 定期保存检查点
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'rqvae_epoch_{epoch + 1}.pt')
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
    parser = argparse.ArgumentParser(description='Train RQ-VAE model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data (.npy file)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/rqvae',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs/rqvae',
                        help='Directory for tensorboard logs')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Input feature dimension')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--num_quantizers', type=int, default=4,
                        help='Number of quantization layers')
    parser.add_argument('--codebook_size', type=int, default=256,
                        help='Codebook size per layer')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128],
                        help='Hidden layer dimensions')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation data ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 加载数据
    print(f"Loading data from {args.data_path}")
    data = torch.from_numpy(np.load(args.data_path)).float()

    # 划分训练集和验证集
    num_val = int(len(data) * args.val_ratio)
    num_train = len(data) - num_val

    indices = torch.randperm(len(data))
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_data = data[train_indices]
    val_data = data[val_indices] if num_val > 0 else None

    print(f"Train samples: {len(train_data)}, Val samples: {num_val}")

    # 创建模型
    model = RQVAE(
        input_dim=args.input_dim,
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        num_quantizers=args.num_quantizers,
        codebook_size=args.codebook_size
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    train_rqvae(
        model=model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )


if __name__ == '__main__':
    import numpy as np
    main()
