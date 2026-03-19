#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ-VAE 单元测试."""

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from training.rqvae.model import RQVAE, ResidualVectorQuantizer, Encoder, Decoder


class TestRQVAE(unittest.TestCase):
    """测试 RQ-VAE 模型."""

    def setUp(self):
        """测试前准备."""
        self.batch_size = 4
        self.input_dim = 1
        self.embedding_dim = 64
        self.num_quantizers = 4
        self.codebook_size = 256
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_encoder(self):
        """测试编码器."""
        encoder = Encoder(
            input_dim=self.input_dim,
            hidden_dims=[256, 128],
            output_dim=self.embedding_dim
        )
        
        x = torch.randn(self.batch_size, self.input_dim)
        z = encoder(x)
        
        self.assertEqual(z.shape, (self.batch_size, self.embedding_dim))

    def test_decoder(self):
        """测试解码器."""
        decoder = Decoder(
            input_dim=self.embedding_dim,
            hidden_dims=[128, 256],
            output_dim=self.input_dim
        )
        
        z = torch.randn(self.batch_size, self.embedding_dim)
        reconstructed = decoder(z)
        
        self.assertEqual(reconstructed.shape, (self.batch_size, self.input_dim))
        # 输出应该在 [0, 1] 之间（Sigmoid）
        self.assertTrue(torch.all(reconstructed >= 0))
        self.assertTrue(torch.all(reconstructed <= 1))

    def test_quantizer(self):
        """测试量化器."""
        quantizer = ResidualVectorQuantizer(
            num_layers=self.num_quantizers,
            codebook_size=self.codebook_size,
            embedding_dim=self.embedding_dim
        )
        
        z = torch.randn(self.batch_size, self.embedding_dim)
        quantized, ids_distributions, ids_indices = quantizer(z)
        
        self.assertEqual(quantized.shape, (self.batch_size, self.embedding_dim))
        self.assertEqual(len(ids_distributions), self.num_quantizers)
        self.assertEqual(len(ids_indices), self.num_quantizers)
        
        # 每个分布应该是 [batch_size, codebook_size]
        for dist in ids_distributions:
            self.assertEqual(dist.shape, (self.batch_size, self.codebook_size))

    def test_rqvae_forward(self):
        """测试 RQ-VAE 前向传播."""
        model = RQVAE(
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            num_quantizers=self.num_quantizers,
            codebook_size=self.codebook_size
        )
        
        x = torch.randn(self.batch_size, self.input_dim)
        reconstructed, quantized, ids_distributions, ids_indices = model(x)
        
        self.assertEqual(reconstructed.shape, (self.batch_size, self.input_dim))
        self.assertEqual(quantized.shape, (self.batch_size, self.embedding_dim))

    def test_get_semantic_ids(self):
        """测试语义 ID 生成."""
        model = RQVAE(
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            num_quantizers=self.num_quantizers,
            codebook_size=self.codebook_size
        )
        
        x = torch.randn(self.batch_size, self.input_dim)
        semantic_ids = model.get_semantic_ids(x)
        
        self.assertEqual(semantic_ids.shape, (self.batch_size, self.num_quantizers))
        # 每个 ID 应该在 [0, codebook_size) 范围内
        self.assertTrue(torch.all(semantic_ids >= 0))
        self.assertTrue(torch.all(semantic_ids < self.codebook_size))

    def test_model_on_device(self):
        """测试模型在指定设备上运行."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        model = RQVAE(
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            num_quantizers=self.num_quantizers,
            codebook_size=self.codebook_size
        ).to(self.device)
        
        x = torch.randn(self.batch_size, self.input_dim).to(self.device)
        reconstructed, _, _, _ = model(x)
        
        self.assertEqual(reconstructed.device.type, 'cuda')


if __name__ == '__main__':
    unittest.main()
