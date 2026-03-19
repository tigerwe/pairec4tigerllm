# -*- coding: utf-8 -*-
"""
RQ-VAE 模型定义.

基于 Residual Quantized Variational Autoencoder 的语义 ID 生成模型.
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualVectorQuantizer(nn.Module):
    """残差向量量化器.

    使用多个码本层级量化，生成语义 ID.

    Attributes:
        num_layers: 量化层数
        codebook_size: 每层码本大小
        embedding_dim: 嵌入维度
        codebooks: 码本列表
    """

    def __init__(
        self,
        num_layers: int = 4,
        codebook_size: int = 256,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        temperature: float = 0.5
    ):
        """初始化量化器.

        Args:
            num_layers: 量化层数（决定语义 ID 长度）
            codebook_size: 每层码本大小（如 256 对应 1 字节）
            embedding_dim: 嵌入维度
            commitment_cost: 承诺损失系数
            temperature: Gumbel-Softmax 温度参数
        """
        super().__init__()
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.temperature = temperature

        # 创建多层码本
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, embedding_dim)
            for _ in range(num_layers)
        ])

        # 初始化码本
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """前向传播.

        Args:
            z: 输入向量 [batch_size, embedding_dim]

        Returns:
            量化后的向量、语义 ID 分布列表、语义 ID 索引列表
        """
        residual = z
        quantized = torch.zeros_like(z)
        ids_distributions = []
        ids_indices = []

        for layer_idx, codebook in enumerate(self.codebooks):
            # 计算与码本向量的距离
            distances = torch.cdist(residual.unsqueeze(1), codebook.weight.unsqueeze(0))
            distances = distances.squeeze(1)

            # Gumbel-Softmax 采样
            logits = -distances / self.temperature
            probs = F.softmax(logits, dim=-1)

            # 训练时使用 Gumbel-Softmax，推理时使用 argmax
            if self.training:
                ids = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
                quant = torch.matmul(ids, codebook.weight)
                ids_idx = ids.argmax(dim=-1)
            else:
                ids_idx = distances.argmin(dim=-1)
                quant = codebook(ids_idx)
                ids = F.one_hot(ids_idx, self.codebook_size).float()

            quantized = quantized + quant
            residual = residual - quant
            ids_distributions.append(probs)
            ids_indices.append(ids_idx)

        return quantized, ids_distributions, ids_indices

    def get_semantic_ids(self, z: torch.Tensor) -> torch.Tensor:
        """获取语义 ID.

        Args:
            z: 输入向量 [batch_size, embedding_dim]

        Returns:
            语义 ID [batch_size, num_layers]
        """
        residual = z
        semantic_ids = []

        for codebook in self.codebooks:
            distances = torch.cdist(residual.unsqueeze(1), codebook.weight.unsqueeze(0))
            distances = distances.squeeze(1)
            ids = distances.argmin(dim=-1)
            semantic_ids.append(ids)

            quant = codebook(ids)
            residual = residual - quant

        return torch.stack(semantic_ids, dim=1)

    def compute_loss(
        self,
        z: torch.Tensor,
        quantized: torch.Tensor,
        ids_distributions: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, dict]:
        """计算损失.

        Args:
            z: 原始向量
            quantized: 量化后的向量
            ids_distributions: ID 分布列表

        Returns:
            总损失、损失字典
        """
        # 承诺损失：使编码器输出接近量化后向量
        commitment_loss = F.mse_loss(z, quantized.detach())

        # 码本损失：使码本向量接近编码器输出
        codebook_loss = F.mse_loss(quantized, z.detach())

        # 总损失
        total_loss = commitment_loss + self.commitment_cost * codebook_loss

        loss_dict = {
            'commitment_loss': commitment_loss.item(),
            'codebook_loss': codebook_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict


class Encoder(nn.Module):
    """编码器.

    将物品特征编码为连续向量.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 64
    ):
        """初始化编码器.

        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出嵌入维度
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播.

        Args:
            x: 输入特征 [batch_size, input_dim]

        Returns:
            编码向量 [batch_size, output_dim]
        """
        return self.network(x)


class Decoder(nn.Module):
    """解码器.

    将量化后的向量解码为原始特征.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 256],
        output_dim: int = 1
    ):
        """初始化解码器.

        Args:
            input_dim: 输入嵌入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出特征维度
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # 输出归一化到 [0, 1]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播.

        Args:
            x: 量化向量 [batch_size, input_dim]

        Returns:
            重建特征 [batch_size, output_dim]
        """
        return self.network(x)


class RQVAE(nn.Module):
    """RQ-VAE 完整模型.

    结合编码器、量化器和解码器的端到端模型.
    """

    def __init__(
        self,
        input_dim: int = 1,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [256, 128],
        num_quantizers: int = 4,
        codebook_size: int = 256,
        commitment_cost: float = 0.25,
        temperature: float = 0.5
    ):
        """初始化 RQ-VAE.

        Args:
            input_dim: 输入特征维度
            embedding_dim: 嵌入维度
            hidden_dims: 编码器隐藏层维度
            num_quantizers: 量化器层数
            codebook_size: 每层码本大小
            commitment_cost: 承诺损失系数
            temperature: Gumbel-Softmax 温度
        """
        super().__init__()

        self.encoder = Encoder(input_dim, hidden_dims, embedding_dim)
        self.quantizer = ResidualVectorQuantizer(
            num_layers=num_quantizers,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            temperature=temperature
        )
        self.decoder = Decoder(embedding_dim, list(reversed(hidden_dims)), input_dim)

        self.embedding_dim = embedding_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """前向传播.

        Args:
            x: 输入特征 [batch_size, input_dim]

        Returns:
            重建特征、量化向量、ID 分布、ID 索引
        """
        # 编码
        z = self.encoder(x)

        # 量化
        quantized, ids_distributions, ids_indices = self.quantizer(z)

        # 解码
        reconstructed = self.decoder(quantized)

        return reconstructed, quantized, ids_distributions, ids_indices

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码为连续向量.

        Args:
            x: 输入特征

        Returns:
            编码向量
        """
        return self.encoder(x)

    def get_semantic_ids(self, x: torch.Tensor) -> torch.Tensor:
        """获取语义 ID.

        Args:
            x: 输入特征 [batch_size, input_dim]

        Returns:
            语义 ID [batch_size, num_quantizers]
        """
        z = self.encoder(x)
        return self.quantizer.get_semantic_ids(z)

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor,
        quantized: torch.Tensor,
        ids_distributions: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, dict]:
        """计算总损失.

        Args:
            x: 原始输入
            reconstructed: 重建输出
            quantized: 量化向量
            ids_distributions: ID 分布列表

        Returns:
            总损失、损失字典
        """
        # 重建损失
        reconstruction_loss = F.mse_loss(reconstructed, x)

        # 量化损失
        z = self.encoder(x)
        quantization_loss, quant_loss_dict = self.quantizer.compute_loss(
            z, quantized, ids_distributions
        )

        # 总损失
        total_loss = reconstruction_loss + quantization_loss

        loss_dict = {
            'reconstruction_loss': reconstruction_loss.item(),
            **quant_loss_dict,
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict
