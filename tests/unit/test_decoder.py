#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decoder 单元测试."""

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from training.decoder.model import GenerativeDecoder, CausalSelfAttention, TransformerBlock


class TestDecoder(unittest.TestCase):
    """测试 Decoder 模型."""

    def setUp(self):
        """测试前准备."""
        self.batch_size = 2
        self.seq_len = 10
        self.vocab_size = 256
        self.num_quantizers = 4
        self.embedding_dim = 256
        self.num_layers = 6
        self.num_heads = 8
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_causal_self_attention(self):
        """测试因果自注意力."""
        attention = CausalSelfAttention(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            max_seq_len=512
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)
        output = attention(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embedding_dim))

    def test_transformer_block(self):
        """测试 Transformer 块."""
        block = TransformerBlock(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            ffn_dim=1024,
            max_seq_len=512
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)
        output = block(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embedding_dim))

    def test_decoder_forward(self):
        """测试 Decoder 前向传播."""
        model = GenerativeDecoder(
            vocab_size=self.vocab_size,
            num_quantizers=self.num_quantizers,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            max_seq_len=512
        )
        
        # 输入: [batch_size, seq_len, num_quantizers]
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len, self.num_quantizers))
        logits, loss = model(input_ids)
        
        # 输出: [batch_size, seq_len, num_quantizers, vocab_size]
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.num_quantizers, self.vocab_size))
        self.assertIsNone(loss)  # 没有提供 labels

    def test_decoder_forward_with_labels(self):
        """测试带标签的 Decoder 前向传播."""
        model = GenerativeDecoder(
            vocab_size=self.vocab_size,
            num_quantizers=self.num_quantizers,
            embedding_dim=self.embedding_dim,
            num_layers=2,  # 减少层数加速测试
            num_heads=self.num_heads,
            max_seq_len=512
        )
        
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len, self.num_quantizers))
        labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len, self.num_quantizers))
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        
        logits, loss = model(input_ids, attention_mask, labels)
        
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.num_quantizers, self.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0)

    def test_generate(self):
        """测试生成功能."""
        model = GenerativeDecoder(
            vocab_size=self.vocab_size,
            num_quantizers=self.num_quantizers,
            embedding_dim=self.embedding_dim,
            num_layers=2,  # 减少层数加速测试
            num_heads=self.num_heads,
            max_seq_len=512
        )
        
        # 输入序列
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, 5, self.num_quantizers))
        
        # 生成
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=3,
            temperature=1.0
        )
        
        # 输出: [batch_size, max_new_tokens, num_quantizers]
        self.assertEqual(generated.shape, (self.batch_size, 3, self.num_quantizers))

    def test_model_parameters(self):
        """测试模型参数."""
        model = GenerativeDecoder(
            vocab_size=self.vocab_size,
            num_quantizers=self.num_quantizers,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads
        )
        
        # 检查是否有参数
        params = list(model.parameters())
        self.assertGreater(len(params), 0)
        
        # 检查参数总数
        total_params = sum(p.numel() for p in params)
        self.assertGreater(total_params, 0)

    def test_device_consistency(self):
        """测试设备一致性."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        model = GenerativeDecoder(
            vocab_size=self.vocab_size,
            num_quantizers=self.num_quantizers,
            embedding_dim=self.embedding_dim,
            num_layers=2,
            num_heads=self.num_heads
        ).to(self.device)
        
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len, self.num_quantizers))
        input_ids = input_ids.to(self.device)
        
        logits, _ = model(input_ids)
        
        self.assertEqual(logits.device.type, 'cuda')


if __name__ == '__main__':
    unittest.main()
