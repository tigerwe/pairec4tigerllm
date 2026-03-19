#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端集成测试.

验证从数据预处理到模型推理的完整流程.
"""

import unittest
import os
import sys
import tempfile
import json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from training.rqvae.model import RQVAE
from training.decoder.model import GenerativeDecoder
from data.utils.data_loader import TenrecDataLoader
from data.utils.preprocessor import TenrecPreprocessor


class TestEndToEnd(unittest.TestCase):
    """端到端测试."""

    @classmethod
    def setUpClass(cls):
        """测试类准备."""
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cls.temp_dir = tempfile.mkdtemp()
        
        print(f"\n使用设备: {cls.device}")
        print(f"临时目录: {cls.temp_dir}")

    def test_01_rqvae_training_pipeline(self):
        """测试 RQ-VAE 训练流程."""
        print("\n[测试] RQ-VAE 训练流程")
        
        # 创建模拟数据
        num_items = 100
        input_dim = 1
        
        data = torch.randn(num_items, input_dim)
        
        # 创建模型
        model = RQVAE(
            input_dim=input_dim,
            embedding_dim=64,
            num_quantizers=4,
            codebook_size=256
        )
        
        # 简单训练几步
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for step in range(5):
            optimizer.zero_grad()
            reconstructed, quantized, ids_distributions, _ = model(data)
            loss, _ = model.compute_loss(data, reconstructed, quantized, ids_distributions)
            loss.backward()
            optimizer.step()
            
            print(f"  Step {step + 1}/5, Loss: {loss.item():.4f}")
        
        # 验证模型可以生成语义 ID
        model.eval()
        with torch.no_grad():
            semantic_ids = model.get_semantic_ids(data[:10])
        
        self.assertEqual(semantic_ids.shape, (10, 4))
        
        # 保存模型
        model_path = os.path.join(self.temp_dir, 'rqvae_test.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'input_dim': input_dim,
                'embedding_dim': 64,
                'num_quantizers': 4,
                'codebook_size': 256
            }
        }, model_path)
        
        print(f"  模型已保存: {model_path}")
        self.assertTrue(os.path.exists(model_path))

    def test_02_decoder_training_pipeline(self):
        """测试 Decoder 训练流程."""
        print("\n[测试] Decoder 训练流程")
        
        # 创建模拟序列数据
        num_sequences = 20
        seq_len = 10
        vocab_size = 256
        num_quantizers = 4
        
        sequences = []
        for _ in range(num_sequences):
            seq = []
            for _ in range(seq_len):
                sem_id = [np.random.randint(0, vocab_size) for _ in range(num_quantizers)]
                seq.append(sem_id)
            sequences.append(seq)
        
        # 创建模型
        model = GenerativeDecoder(
            vocab_size=vocab_size,
            num_quantizers=num_quantizers,
            embedding_dim=128,
            num_layers=2,  # 减少层数加速测试
            num_heads=4,
            max_seq_len=50
        )
        
        # 准备训练数据
        batch_size = 4
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len, num_quantizers))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len, num_quantizers))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # 训练几步
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for step in range(5):
            optimizer.zero_grad()
            logits, loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            
            print(f"  Step {step + 1}/5, Loss: {loss.item():.4f}")
        
        # 验证生成
        model.eval()
        with torch.no_grad():
            generated = model.generate(input_ids[:1], max_new_tokens=3)
        
        self.assertEqual(generated.shape, (1, 3, num_quantizers))
        
        # 保存模型
        model_path = os.path.join(self.temp_dir, 'decoder_test.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'vocab_size': vocab_size,
                'num_quantizers': num_quantizers,
                'embedding_dim': 128,
                'num_layers': 2,
                'num_heads': 4,
                'max_seq_len': 50,
                'pad_token_id': 0
            }
        }, model_path)
        
        print(f"  模型已保存: {model_path}")
        self.assertTrue(os.path.exists(model_path))

    def test_03_semantic_id_mapping(self):
        """测试语义 ID 映射流程."""
        print("\n[测试] 语义 ID 映射")
        
        # 模拟物品 ID
        item_ids = list(range(1000, 1100))
        
        # 创建语义 ID 映射
        semantic_id_map = {}
        for item_id in item_ids:
            semantic_id_map[item_id] = [
                item_id % 256,
                (item_id // 256) % 256,
                (item_id // 65536) % 256,
                (item_id // 16777216) % 256,
            ]
        
        # 保存映射
        map_path = os.path.join(self.temp_dir, 'semantic_id_map.json')
        with open(map_path, 'w') as f:
            json.dump(semantic_id_map, f)
        
        # 加载映射
        with open(map_path, 'r') as f:
            loaded_map = json.load(f)
        
        # 验证
        self.assertEqual(len(loaded_map), len(item_ids))
        
        # 验证每个映射
        for item_id in item_ids:
            key = str(item_id)
            self.assertIn(key, loaded_map)
            self.assertEqual(len(loaded_map[key]), 4)
        
        print(f"  映射文件: {map_path}")
        print(f"  映射数量: {len(loaded_map)}")

    def test_04_inference_pipeline(self):
        """测试推理流程."""
        print("\n[测试] 推理流程")
        
        # 创建简单的推理模型
        model = GenerativeDecoder(
            vocab_size=256,
            num_quantizers=4,
            embedding_dim=128,
            num_layers=2,
            num_heads=4,
            max_seq_len=50
        )
        model.eval()
        
        # 模拟用户历史
        user_history = [
            [100, 50, 25, 10],
            [101, 51, 26, 11],
            [102, 52, 27, 12]
        ]
        
        # 转换为张量
        input_ids = torch.tensor([user_history], dtype=torch.long)
        
        # 推理
        with torch.no_grad():
            # 生成推荐
            generated = model.generate(input_ids, max_new_tokens=5, temperature=1.0)
        
        # 验证输出
        self.assertEqual(generated.shape[0], 1)  # batch_size
        self.assertEqual(generated.shape[1], 5)  # max_new_tokens
        self.assertEqual(generated.shape[2], 4)  # num_quantizers
        
        # 验证生成的 ID 在有效范围内
        self.assertTrue(torch.all(generated >= 0))
        self.assertTrue(torch.all(generated < 256))
        
        print(f"  输入历史长度: {len(user_history)}")
        print(f"  生成长度: {generated.shape[1]}")
        print(f"  生成的语义 ID 示例: {generated[0, 0].tolist()}")

    def test_05_model_save_load(self):
        """测试模型保存和加载."""
        print("\n[测试] 模型保存和加载")
        
        # 创建并保存模型
        original_model = RQVAE(
            input_dim=1,
            embedding_dim=64,
            num_quantizers=4,
            codebook_size=256
        )
        
        model_path = os.path.join(self.temp_dir, 'model_checkpoint.pt')
        
        # 保存
        torch.save({
            'model_state_dict': original_model.state_dict(),
            'config': {
                'input_dim': 1,
                'embedding_dim': 64,
                'num_quantizers': 4,
                'codebook_size': 256
            }
        }, model_path)
        
        # 加载
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']
        
        loaded_model = RQVAE(
            input_dim=config['input_dim'],
            embedding_dim=config['embedding_dim'],
            num_quantizers=config['num_quantizers'],
            codebook_size=config['codebook_size']
        )
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 验证参数一致
        for p1, p2 in zip(original_model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
        
        print(f"  检查点: {model_path}")
        print("  模型加载成功，参数一致")

    @classmethod
    def tearDownClass(cls):
        """测试类清理."""
        import shutil
        
        # 清理临时目录
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
            print(f"\n清理临时目录: {cls.temp_dir}")


if __name__ == '__main__':
    unittest.main()
