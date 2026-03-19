# -*- coding: utf-8 -*-
"""
Tenrec 数据预处理器.

提供物品 ID 重映射、负采样、数据增强等功能.
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


class TenrecPreprocessor:
    """Tenrec 数据预处理器.

    负责物品 ID 重映射、用户序列构造、负采样等预处理操作.

    Attributes:
        item_id_map: 原始物品 ID -> 连续物品 ID 映射
        reverse_item_id_map: 连续物品 ID -> 原始物品 ID 映射
        num_items: 物品总数
    """

    def __init__(self):
        """初始化预处理器."""
        self.item_id_map: Dict[int, int] = {}
        self.reverse_item_id_map: Dict[int, int] = {}
        self.num_items: int = 0

    def fit(self, item_ids: np.ndarray) -> 'TenrecPreprocessor':
        """构建物品 ID 映射.

        Args:
            item_ids: 原始物品 ID 数组

        Returns:
            self
        """
        unique_items = sorted(set(item_ids))
        self.item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
        self.reverse_item_id_map = {new_id: old_id for old_id, new_id in self.item_id_map.items()}
        self.num_items = len(unique_items)

        print(f"Fitted preprocessor: {self.num_items} unique items")
        return self

    def transform_item_id(self, item_id: int) -> int:
        """转换物品 ID.

        Args:
            item_id: 原始物品 ID

        Returns:
            连续物品 ID
        """
        return self.item_id_map.get(item_id, 0)

    def inverse_transform_item_id(self, item_id: int) -> int:
        """反向转换物品 ID.

        Args:
            item_id: 连续物品 ID

        Returns:
            原始物品 ID
        """
        return self.reverse_item_id_map.get(item_id, 0)

    def transform_sequence(self, sequence: List[int]) -> List[int]:
        """转换序列中的物品 ID.

        Args:
            sequence: 原始物品 ID 序列

        Returns:
            连续物品 ID 序列
        """
        return [self.transform_item_id(item_id) for item_id in sequence]

    def save_vocab(self, filepath: str) -> None:
        """保存词汇表.

        Args:
            filepath: 保存路径
        """
        vocab_data = {
            'item_id_map': self.item_id_map,
            'reverse_item_id_map': self.reverse_item_id_map,
            'num_items': self.num_items
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"Vocab saved to: {filepath}")

    def load_vocab(self, filepath: str) -> 'TenrecPreprocessor':
        """加载词汇表.

        Args:
            filepath: 词汇表文件路径

        Returns:
            self
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        self.item_id_map = {int(k): v for k, v in vocab_data['item_id_map'].items()}
        self.reverse_item_id_map = {int(k): v for k, v in vocab_data['reverse_item_id_map'].items()}
        self.num_items = vocab_data['num_items']
        print(f"Vocab loaded from: {filepath}, {self.num_items} items")
        return self

    def create_training_sequences(
        self,
        user_sequences: Dict[int, List[int]],
        max_length: int = 20,
        stride: int = 1
    ) -> List[Tuple[List[int], int]]:
        """创建训练序列.

        使用滑动窗口从用户序列中生成 (输入序列, 目标物品) 训练样本.

        Args:
            user_sequences: 用户交互序列字典
            max_length: 最大序列长度
            stride: 滑动窗口步长

        Returns:
            训练样本列表，每个样本为 (input_sequence, target_item) 元组
        """
        training_samples = []

        for user_id, sequence in user_sequences.items():
            # 转换物品 ID
            transformed_seq = self.transform_sequence(sequence)

            if len(transformed_seq) < 2:
                continue

            # 滑动窗口生成样本
            for i in range(0, len(transformed_seq) - 1, stride):
                # 输入序列
                input_seq = transformed_seq[max(0, i - max_length + 1):i + 1]
                # 目标物品
                target = transformed_seq[i + 1]

                training_samples.append((input_seq, target))

        print(f"Created {len(training_samples)} training samples")
        return training_samples

    def negative_sampling(
        self,
        positive_items: Set[int],
        num_negatives: int,
        exclude_items: Optional[Set[int]] = None
    ) -> List[int]:
        """负采样.

        Args:
            positive_items: 正样本物品集合（连续 ID）
            num_negatives: 负样本数量
            exclude_items: 需要排除的物品集合

        Returns:
            负样本物品 ID 列表
        """
        exclude = set(positive_items)
        if exclude_items:
            exclude.update(exclude_items)

        negatives = []
        max_attempts = num_negatives * 10
        attempts = 0

        while len(negatives) < num_negatives and attempts < max_attempts:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in exclude and neg_item not in negatives:
                negatives.append(neg_item)
            attempts += 1

        return negatives

    def create_rqvae_training_data(
        self,
        user_sequences: Dict[int, List[int]],
        item_features: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """创建 RQ-VAE 训练数据.

        RQ-VAE 用于学习物品的语义表示，输入是物品的特征向量.
        对于 Tenrec 数据，我们使用物品的交互频率作为特征.

        Args:
            user_sequences: 用户交互序列
            item_features: 物品特征数据框

        Returns:
            物品特征矩阵
        """
        # 统计每个物品的交互次数
        item_counts = np.zeros(self.num_items)
        for sequence in user_sequences.values():
            for item_id in sequence:
                transformed_id = self.transform_item_id(item_id)
                if 0 <= transformed_id < self.num_items:
                    item_counts[transformed_id] += 1

        # 构建特征矩阵（可以扩展更多特征）
        features = item_counts.reshape(-1, 1)

        # 归一化
        if features.max() > 0:
            features = features / features.max()

        print(f"Created RQ-VAE training data: {features.shape}")
        return features

    def augment_sequence(
        self,
        sequence: List[int],
        mask_ratio: float = 0.2,
        max_mask_num: int = 3
    ) -> List[int]:
        """数据增强：随机 mask.

        Args:
            sequence: 原始序列
            mask_ratio: mask 比例
            max_mask_num: 最大 mask 数量

        Returns:
            增强后的序列
        """
        if len(sequence) < 3:
            return sequence

        seq = sequence.copy()
        num_to_mask = min(max_mask_num, max(1, int(len(seq) * mask_ratio)))

        # 随机选择 mask 位置（不 mask 最后一个）
        mask_positions = np.random.choice(
            len(seq) - 1, size=num_to_mask, replace=False
        )

        # 使用特殊 ID 0 作为 mask
        for pos in mask_positions:
            seq[pos] = 0

        return seq

    def pad_sequence(
        self,
        sequence: List[int],
        max_length: int,
        pad_value: int = 0
    ) -> Tuple[List[int], int]:
        """序列填充.

        Args:
            sequence: 原始序列
            max_length: 目标长度
            pad_value: 填充值

        Returns:
            (填充后的序列, 原始长度) 元组
        """
        original_length = len(sequence)

        if len(sequence) >= max_length:
            return sequence[-max_length:], max_length
        else:
            padded = [pad_value] * (max_length - len(sequence)) + sequence
            return padded, original_length
