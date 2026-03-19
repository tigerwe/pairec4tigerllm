# -*- coding: utf-8 -*-
"""
Generative Decoder 模型定义.

基于 GPT2 架构的 Decoder-only 生成式推荐模型.
使用语义 ID 作为输入，自回归生成下一个物品.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """因果自注意力层.

    确保模型只能看到当前位置之前的信息.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        """初始化注意力层.

        Args:
            embedding_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout 率
            max_seq_len: 最大序列长度
        """
        super().__init__()
        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Q, K, V 投影
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        # 输出投影
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 因果掩码（下三角矩阵）
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播.

        Args:
            x: 输入 [batch_size, seq_len, embedding_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]

        Returns:
            输出 [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = x.shape

        # 计算 Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 重塑为多头格式 [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用因果掩码
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # 应用额外的注意力掩码（用于 padding）
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax 和 Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 应用注意力到 V
        attn_output = torch.matmul(attn_weights, v)

        # 重塑回原始格式
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embedding_dim)

        # 输出投影
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        return output


class TransformerBlock(nn.Module):
    """Transformer 块.

    包含层归一化、因果自注意力和前馈网络.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        """初始化 Transformer 块.

        Args:
            embedding_dim: 嵌入维度
            num_heads: 注意力头数
            ffn_dim: 前馈网络维度
            dropout: Dropout 率
            max_seq_len: 最大序列长度
        """
        super().__init__()

        # 层归一化（Pre-LN）
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

        # 注意力
        self.attention = CausalSelfAttention(
            embedding_dim, num_heads, dropout, max_seq_len
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播.

        Args:
            x: 输入 [batch_size, seq_len, embedding_dim]
            attention_mask: 注意力掩码

        Returns:
            输出 [batch_size, seq_len, embedding_dim]
        """
        # 自注意力（带残差连接）
        x = x + self.attention(self.ln1(x), attention_mask)

        # 前馈网络（带残差连接）
        x = x + self.ffn(self.ln2(x))

        return x


class GenerativeDecoder(nn.Module):
    """生成式解码器.

    基于 GPT2 的 Decoder-only 架构，用于生成推荐.
    输入是语义 ID 序列，输出是下一个物品的分布.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        num_quantizers: int = 4,
        embedding_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        """初始化生成式解码器.

        Args:
            vocab_size: 词汇表大小（codebook_size）
            num_quantizers: 量化器层数（语义 ID 长度）
            embedding_dim: 嵌入维度
            num_layers: Transformer 层数
            num_heads: 注意力头数
            ffn_dim: 前馈网络维度
            max_seq_len: 最大序列长度
            dropout: Dropout 率
            pad_token_id: 填充 token ID
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_quantizers = num_quantizers
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # 每个量化层有独立的嵌入表
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
            for _ in range(num_quantizers)
        ])

        # 位置编码
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer 层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, ffn_dim, dropout, max_seq_len)
            for _ in range(num_layers)
        ])

        # 最终层归一化
        self.ln_f = nn.LayerNorm(embedding_dim)

        # 输出头（每个量化层一个）
        self.output_heads = nn.ModuleList([
            nn.Linear(embedding_dim, vocab_size, bias=False)
            for _ in range(num_quantizers)
        ])

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        semantic_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播.

        Args:
            semantic_ids: 语义 ID 序列 [batch_size, seq_len, num_quantizers]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签（用于计算损失）[batch_size, seq_len, num_quantizers]

        Returns:
            logits: 输出 logits [batch_size, seq_len, num_quantizers, vocab_size]
            loss: 损失值（如果提供了 labels）
        """
        batch_size, seq_len, _ = semantic_ids.shape

        # 构建输入嵌入
        # 对每个位置，将 num_quantizers 个 token 的嵌入相加
        x = torch.zeros(batch_size, seq_len, self.embedding_dim, device=semantic_ids.device)
        for i, embedding_layer in enumerate(self.token_embeddings):
            x = x + embedding_layer(semantic_ids[:, :, i])

        # 添加位置编码
        positions = torch.arange(seq_len, device=semantic_ids.device).unsqueeze(0)
        x = x + self.position_embedding(positions)

        x = self.dropout(x)

        # Transformer 层
        for block in self.transformer_blocks:
            x = block(x, attention_mask)

        # 最终层归一化
        x = self.ln_f(x)

        # 输出 logits（每个量化层一个）
        logits_list = []
        for head in self.output_heads:
            logits_list.append(head(x))

        # 堆叠为 [batch_size, seq_len, num_quantizers, vocab_size]
        logits = torch.stack(logits_list, dim=2)

        # 计算损失
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels, attention_mask)

        return logits, loss

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算交叉熵损失.

        Args:
            logits: 模型输出 [batch_size, seq_len, num_quantizers, vocab_size]
            labels: 目标标签 [batch_size, seq_len, num_quantizers]
            attention_mask: 注意力掩码 [batch_size, seq_len]

        Returns:
            平均损失
        """
        batch_size, seq_len, num_quantizers, vocab_size = logits.shape

        # 重塑为 [batch_size * seq_len * num_quantizers, vocab_size]
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        # 创建掩码（忽略 padding 位置）
        if attention_mask is not None:
            # 扩展到每个 quantizer
            mask = attention_mask.unsqueeze(-1).expand(-1, -1, num_quantizers)
            mask_flat = mask.reshape(-1)
        else:
            mask_flat = torch.ones(batch_size * seq_len * num_quantizers, device=logits.device)

        # 只计算非 padding 位置的损失
        valid_indices = mask_flat.bool()
        logits_flat = logits_flat[valid_indices]
        labels_flat = labels_flat[valid_indices]

        loss = F.cross_entropy(logits_flat, labels_flat)

        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """生成推荐.

        Args:
            input_ids: 输入语义 ID 序列 [batch_size, seq_len, num_quantizers]
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-k 采样
            attention_mask: 注意力掩码

        Returns:
            生成的语义 ID [batch_size, max_new_tokens, num_quantizers]
        """
        self.eval()

        batch_size = input_ids.shape[0]
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # 截断到最大长度
            if generated.shape[1] > self.max_seq_len:
                generated = generated[:, -self.max_seq_len:]

            # 前向传播
            logits, _ = self.forward(generated, attention_mask)

            # 取最后一个位置的 logits
            next_token_logits = logits[:, -1, :, :]  # [batch_size, num_quantizers, vocab_size]

            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Top-k 采样
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.shape[-1]))
                next_token_logits[next_token_logits < v[:, :, [-1]]] = float('-inf')

            # 采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(
                probs.view(-1, self.vocab_size), num_samples=1
            ).view(batch_size, self.num_quantizers)

            # 添加到序列
            next_tokens = next_tokens.unsqueeze(1)  # [batch_size, 1, num_quantizers]
            generated = torch.cat([generated, next_tokens], dim=1)

        # 返回新生成的部分
        return generated[:, -max_new_tokens:, :]

    @torch.no_grad()
    def beam_search(
        self,
        input_ids: torch.Tensor,
        beam_width: int = 5,
        max_length: int = 10,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Beam Search 生成.

        Args:
            input_ids: 输入语义 ID 序列 [batch_size, seq_len, num_quantizers]
            beam_width: Beam 宽度
            max_length: 最大生成长度
            attention_mask: 注意力掩码

        Returns:
            生成的语义 ID [batch_size * beam_width, max_length, num_quantizers]
            对应的分数 [batch_size * beam_width]
        """
        self.eval()

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 扩展输入为 beam_width 份
        input_ids = input_ids.unsqueeze(1).repeat(1, beam_width, 1, 1)
        input_ids = input_ids.view(batch_size * beam_width, -1, self.num_quantizers)

        # 初始化分数
        scores = torch.zeros(batch_size * beam_width, device=device)

        # 对每个 batch 分别进行 beam search
        for step in range(max_length):
            # 前向传播
            logits, _ = self.forward(input_ids, attention_mask)

            # 取最后一个位置的对数概率
            next_token_logits = logits[:, -1, :, :]  # [batch_size * beam_width, num_quantizers, vocab_size]
            log_probs = F.log_softmax(next_token_logits, dim=-1)

            # 简化的 beam search：对每个 quantizer 独立处理
            # 实际实现中可能需要更复杂的联合搜索

            # 这里使用 greedy 作为简化
            next_tokens = log_probs.argmax(dim=-1)  # [batch_size * beam_width, num_quantizers]
            next_tokens = next_tokens.unsqueeze(1)  # [batch_size * beam_width, 1, num_quantizers]

            # 更新输入
            input_ids = torch.cat([input_ids, next_tokens], dim=1)

        return input_ids[:, -max_length:, :], scores
