#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据预处理脚本.

将 Tenrec 原始数据预处理为模型训练所需格式.
支持 RQ-VAE 语义 ID 生成（如果模型可用）.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.utils.data_loader import TenrecDataLoader
from data.utils.preprocessor import TenrecPreprocessor


def load_rqvae_model(model_path: str, device: str = 'cuda'):
    """加载 RQ-VAE 模型.
    
    Args:
        model_path: 模型路径
        device: 设备
        
    Returns:
        模型或 None
    """
    try:
        import torch
        from training.rqvae.model import RQVAE
        
        if not os.path.exists(model_path):
            print(f"   RQ-VAE 模型不存在: {model_path}")
            return None
        
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['config']
        
        model = RQVAE(
            input_dim=config['input_dim'],
            embedding_dim=config['embedding_dim'],
            hidden_dims=config.get('hidden_dims', [256, 128]),
            num_quantizers=config['num_quantizers'],
            codebook_size=config['codebook_size']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"   RQ-VAE 模型加载成功: {model_path}")
        return model
    except Exception as e:
        print(f"   加载 RQ-VAE 模型失败: {e}")
        return None


def generate_semantic_ids(
    item_ids: List[int],
    preprocessor: TenrecPreprocessor,
    rqvae_model=None,
    item_features: np.ndarray = None
) -> Dict[int, List[int]]:
    """生成语义 ID 映射.
    
    Args:
        item_ids: 原始物品 ID 列表
        preprocessor: 预处理器
        rqvae_model: RQ-VAE 模型（可选）
        item_features: 物品特征（用于 RQ-VAE）
        
    Returns:
        原始物品 ID -> 语义 ID 的映射
    """
    semantic_id_map = {}
    
    if rqvae_model is not None and item_features is not None:
        # 使用 RQ-VAE 生成语义 ID
        print("   使用 RQ-VAE 生成语义 ID...")
        import torch
        
        with torch.no_grad():
            features = torch.from_numpy(item_features).float().to(rqvae_model.device)
            semantic_ids = rqvae_model.get_semantic_ids(features)
            semantic_ids = semantic_ids.cpu().numpy()
        
        for i, item_id in enumerate(item_ids):
            transformed_id = preprocessor.transform_item_id(item_id)
            if 0 <= transformed_id < len(semantic_ids):
                semantic_id_map[item_id] = semantic_ids[transformed_id].tolist()
    else:
        # 使用简化方式生成语义 ID（哈希）
        print("   使用哈希方式生成语义 ID（临时方案）...")
        print("   提示: 训练 RQ-VAE 后重新运行可获得更优语义 ID")
        
        for item_id in item_ids:
            # 4 层语义 ID，每层 0-255
            semantic_id_map[item_id] = [
                item_id % 256,
                (item_id // 256) % 256,
                (item_id // 65536) % 256,
                (item_id // 16777216) % 256,
            ]
    
    return semantic_id_map


def preprocess_tenrec_data(
    input_path: str,
    output_dir: str,
    test_ratio: float = 0.2,
    min_sequence_length: int = 5,
    max_seq_len: int = 50,
    rqvae_model_path: str = None,
    device: str = 'cuda'
) -> None:
    """预处理 Tenrec 数据.
    
    Args:
        input_path: 输入数据路径
        output_dir: 输出目录
        test_ratio: 测试集比例
        min_sequence_length: 最小序列长度
        max_seq_len: 最大序列长度
        rqvae_model_path: RQ-VAE 模型路径（可选）
        device: 设备
    """
    print("=" * 60)
    print("Tenrec Data Preprocessing")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    print("\n1. Loading data...")
    loader = TenrecDataLoader(input_path, cache_dir=os.path.join(output_dir, 'cache'))
    df = loader.load(use_cache=True)

    stats = loader.get_statistics()
    print(f"   Interactions: {stats['num_interactions']}")
    print(f"   Users: {stats['num_users']}")
    print(f"   Items: {stats['num_items']}")
    print(f"   Sparsity: {stats['sparsity']:.4f}")

    # 2. 数据预处理
    print("\n2. Preprocessing...")
    preprocessor = TenrecPreprocessor()
    preprocessor.fit(loader.item_ids)

    # 保存词汇表
    vocab_path = os.path.join(output_dir, 'vocab.json')
    preprocessor.save_vocab(vocab_path)

    # 3. 创建用户序列
    print("\n3. Creating user sequences...")
    user_sequences = loader.get_all_user_sequences()

    # 过滤短序列
    filtered_sequences = {
        uid: seq for uid, seq in user_sequences.items()
        if len(seq) >= min_sequence_length
    }
    print(f"   Filtered sequences: {len(filtered_sequences)}")

    # 4. 划分训练集和测试集
    print("\n4. Splitting train/test...")
    user_ids = list(filtered_sequences.keys())
    train_users, test_users = train_test_split(
        user_ids, test_size=test_ratio, random_state=42
    )

    train_sequences = {uid: filtered_sequences[uid] for uid in train_users}
    test_sequences = {uid: filtered_sequences[uid] for uid in test_users}

    print(f"   Train users: {len(train_sequences)}")
    print(f"   Test users: {len(test_sequences)}")

    # 5. 创建 RQ-VAE 训练数据
    print("\n5. Creating RQ-VAE training data...")
    item_features = preprocessor.create_rqvae_training_data(user_sequences)

    # 保存 RQ-VAE 训练数据
    rqvae_data_path = os.path.join(output_dir, 'rqvae_train_data.npy')
    np.save(rqvae_data_path, item_features)
    print(f"   Saved to: {rqvae_data_path}")

    # 保存物品 ID 映射
    item_ids_array = np.array(list(preprocessor.item_id_map.keys()))
    item_ids_path = os.path.join(output_dir, 'item_ids.npy')
    np.save(item_ids_path, item_ids_array)
    print(f"   Saved item IDs to: {item_ids_path}")

    # 6. 生成语义 ID 映射
    print("\n6. Generating semantic ID mapping...")
    rqvae_model = None
    if rqvae_model_path:
        rqvae_model = load_rqvae_model(rqvae_model_path, device)
    
    semantic_id_map = generate_semantic_ids(
        list(preprocessor.item_id_map.keys()),
        preprocessor,
        rqvae_model,
        item_features if rqvae_model is not None else None
    )
    
    # 保存语义 ID 映射
    semantic_map_path = os.path.join(output_dir, 'semantic_id_map.json')
    with open(semantic_map_path, 'w', encoding='utf-8') as f:
        json.dump(semantic_id_map, f, ensure_ascii=False, indent=2)
    print(f"   Saved semantic ID map to: {semantic_map_path}")
    
    # 保存为 CSV 格式（便于查看）
    csv_data = []
    for item_id, sem_ids in semantic_id_map.items():
        csv_data.append({
            'item_id': item_id,
            'semantic_id': '-'.join(map(str, sem_ids)),
            **{f'code_{i}': sem_ids[i] for i in range(len(sem_ids))}
        })
    
    df_semantic = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, 'semantic_id_map.csv')
    df_semantic.to_csv(csv_path, index=False)
    print(f"   Saved CSV to: {csv_path}")

    # 7. 创建 Decoder 训练序列
    print("\n7. Creating Decoder training sequences...")

    def create_semantic_sequences(sequences: Dict[int, List[int]]) -> List[List[List[int]]]:
        """将物品 ID 序列转换为语义 ID 序列."""
        semantic_sequences = []

        for uid, seq in sequences.items():
            if len(seq) < 2:
                continue

            semantic_seq = []
            for item_id in seq:
                # 查询语义 ID 映射
                sem_id = semantic_id_map.get(item_id)
                if sem_id is None:
                    # 回退到哈希方式
                    sem_id = [
                        item_id % 256,
                        (item_id // 256) % 256,
                        (item_id // 65536) % 256,
                        (item_id // 16777216) % 256,
                    ]
                semantic_seq.append(sem_id)

            semantic_sequences.append(semantic_seq)

        return semantic_sequences

    train_semantic_seqs = create_semantic_sequences(train_sequences)
    test_semantic_seqs = create_semantic_sequences(test_sequences)

    # 保存 Decoder 训练数据
    train_seq_path = os.path.join(output_dir, 'train_sequences.json')
    test_seq_path = os.path.join(output_dir, 'test_sequences.json')

    with open(train_seq_path, 'w') as f:
        json.dump(train_semantic_seqs, f)
    print(f"   Saved train sequences to: {train_seq_path}")

    with open(test_seq_path, 'w') as f:
        json.dump(test_semantic_seqs, f)
    print(f"   Saved test sequences to: {test_seq_path}")

    # 8. 保存数据统计信息
    print("\n8. Saving statistics...")
    stats_dict = {
        'num_interactions': int(stats['num_interactions']),
        'num_users': int(stats['num_users']),
        'num_items': int(stats['num_items']),
        'sparsity': float(stats['sparsity']),
        'num_train_users': len(train_sequences),
        'num_test_users': len(test_sequences),
        'num_train_sequences': len(train_semantic_seqs),
        'num_test_sequences': len(test_semantic_seqs),
        'num_quantizers': 4,  # 4 层语义 ID
        'codebook_size': 256,
        'max_seq_len': max_seq_len,
        'rqvae_model_used': rqvae_model is not None
    }

    stats_path = os.path.join(output_dir, 'statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"   Saved statistics to: {stats_path}")

    print("\n" + "=" * 60)
    print("Preprocessing completed!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # 打印下一步指引
    print("\n下一步:")
    if rqvae_model is None:
        print("  1. 训练 RQ-VAE: ./scripts/train_rqvae.sh")
        print("  2. 重新预处理数据以获取更优语义 ID")
        print("  3. 训练 Decoder: ./scripts/train_decoder.sh")
    else:
        print("  1. 训练 Decoder: ./scripts/train_decoder.sh")
        print("  2. 导出模型: python training/decoder/export.py ...")
        print("  3. 启动服务: ./scripts/run_all.sh")


def main():
    """命令行入口."""
    parser = argparse.ArgumentParser(description='Preprocess Tenrec data')
    parser.add_argument('--input', type=str,
                        default='/home/vivwimp/Tenrec/ctr_data_1M.csv',
                        help='Input data path')
    parser.add_argument('--output', type=str,
                        default='./data/tenrec/processed',
                        help='Output directory')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test set ratio')
    parser.add_argument('--min_sequence_length', type=int, default=5,
                        help='Minimum sequence length')
    parser.add_argument('--max_seq_len', type=int, default=50,
                        help='Maximum sequence length')
    parser.add_argument('--rqvae_model', type=str,
                        default='./checkpoints/rqvae/rqvae_best.pt',
                        help='RQ-VAE model path (optional)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    preprocess_tenrec_data(
        input_path=args.input,
        output_dir=args.output,
        test_ratio=args.test_ratio,
        min_sequence_length=args.min_sequence_length,
        max_seq_len=args.max_seq_len,
        rqvae_model_path=args.rqvae_model,
        device=args.device
    )


if __name__ == '__main__':
    main()
