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
        # 转换 numpy int64 为 Python int，以便 JSON 序列化
        vocab_data = {
            'item_id_map': {int(k): int(v) for k, v in self.item_id_map.items()},
            'reverse_item_id_map': {int(k): int(v) for k, v in self.reverse_item_id_map.items()},
            'num_items': int(self.num_items)
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
        df: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """创建 RQ-VAE 训练数据.

        RQ-VAE 用于学习物品的语义表示，输入是物品的特征向量.
        从原始数据中提取丰富的物品特征.

        Args:
            user_sequences: 用户交互序列
            df: 原始数据框（包含 click, like, follow, share 等字段）

        Returns:
            物品特征矩阵 [num_items, feature_dim]
        """
        print("Extracting item features for RQ-VAE...")
        
        # ==================== 基础交互统计 ====================
        # 1. 每个物品的交互次数（总点击次数）
        item_click_count = np.zeros(self.num_items)
        # 2. 每个物品被多少不同用户交互过
        item_user_count = np.zeros(self.num_items)
        # 3. 每个物品在序列中的平均位置
        item_position_sum = np.zeros(self.num_items)
        item_position_count = np.zeros(self.num_items)
        
        for user_id, sequence in user_sequences.items():
            unique_items = set(sequence)
            for pos, item_id in enumerate(sequence):
                transformed_id = self.transform_item_id(item_id)
                if 0 <= transformed_id < self.num_items:
                    item_click_count[transformed_id] += 1
                    item_position_sum[transformed_id] += pos
                    item_position_count[transformed_id] += 1
            
            for item_id in unique_items:
                transformed_id = self.transform_item_id(item_id)
                if 0 <= transformed_id < self.num_items:
                    item_user_count[transformed_id] += 1
        
        # 避免除零
        item_position_count = np.maximum(item_position_count, 1)
        item_avg_position = item_position_sum / item_position_count

        # ==================== 从原始数据提取高级特征 ====================
        if df is not None and len(df) > 0:
            # 确保 item_id 可以映射
            df = df.copy()
            df['item_id_transformed'] = df['item_id'].map(self.item_id_map)
            df_valid = df[df['item_id_transformed'].notna() & 
                         (df['item_id_transformed'] >= 0) & 
                         (df['item_id_transformed'] < self.num_items)].copy()
            df_valid['item_id_transformed'] = df_valid['item_id_transformed'].astype(int)
            
            # 按 item_id 聚合统计
            item_stats = df_valid.groupby('item_id_transformed').agg({
                'click': ['sum', 'count'],
                'follow': 'sum',
                'like': 'sum',
                'share': 'sum',
                'video_category': 'first',
                'watching_times': ['mean', 'max'],
                'gender': 'mean',  # 男性比例
                'age': 'mean',
            }).reset_index()
            
            # 展平列名
            item_stats.columns = ['item_id', 'click_sum', 'exposure_count', 
                                  'follow_sum', 'like_sum', 'share_sum',
                                  'video_category', 'watching_times_mean', 'watching_times_max',
                                  'male_ratio', 'avg_user_age']
            
            # 填充缺失值
            item_stats = item_stats.fillna(0)
            
            # 创建特征数组
            item_like_count = np.zeros(self.num_items)
            item_follow_count = np.zeros(self.num_items)
            item_share_count = np.zeros(self.num_items)
            item_watching_times_mean = np.zeros(self.num_items)
            item_male_ratio = np.zeros(self.num_items)
            item_avg_age = np.zeros(self.num_items)
            
            # 填充统计值
            for _, row in item_stats.iterrows():
                idx = int(row['item_id'])
                if 0 <= idx < self.num_items:
                    item_like_count[idx] = row['like_sum']
                    item_follow_count[idx] = row['follow_sum']
                    item_share_count[idx] = row['share_sum']
                    item_watching_times_mean[idx] = row['watching_times_mean']
                    item_male_ratio[idx] = row['male_ratio']
                    item_avg_age[idx] = row['avg_user_age']
        else:
            # 如果没有原始数据框，使用基础统计作为替代
            item_like_count = item_click_count * 0.1  # 估计值
            item_follow_count = item_click_count * 0.05
            item_share_count = item_click_count * 0.02
            item_watching_times_mean = np.ones(self.num_items) * 0.5
            item_male_ratio = np.ones(self.num_items) * 0.5
            item_avg_age = np.ones(self.num_items) * 25

        # ==================== 计算派生特征 ====================
        # 点击率（互动率）
        item_ctr = np.divide(item_click_count, np.maximum(item_user_count, 1))
        # 点赞率
        item_like_rate = np.divide(item_like_count, np.maximum(item_click_count, 1))
        # 分享率
        item_share_rate = np.divide(item_share_count, np.maximum(item_click_count, 1))
        # 关注转化率
        item_follow_rate = np.divide(item_follow_count, np.maximum(item_click_count, 1))
        
        # 热度指数（综合交互）
        item_heat_score = (
            item_click_count * 1.0 +
            item_like_count * 2.0 +
            item_follow_count * 3.0 +
            item_share_count * 5.0
        )

        # ==================== 构建特征矩阵 ====================
        # 选择有意义的特征（共14维）
        feature_list = [
            # 基础统计（4维）
            item_click_count,                    # 1. 点击次数
            item_user_count,                     # 2. 独立用户数
            np.log1p(item_click_count),          # 3. log点击次数
            np.log1p(item_user_count),           # 4. log用户数
            
            # 互动统计（4维）
            item_like_count,                     # 5. 点赞数
            item_follow_count,                   # 6. 关注数
            item_share_count,                    # 7. 分享数
            item_heat_score,                     # 8. 热度指数
            
            # 转化率（4维）
            item_ctr,                            # 9. 点击率
            item_like_rate,                      # 10. 点赞率
            item_share_rate,                     # 11. 分享率
            item_follow_rate,                    # 12. 关注率
            
            # 用户画像（2维）
            item_male_ratio,                     # 13. 男性比例
            item_avg_age,                        # 14. 平均用户年龄
        ]
        
        features = np.stack(feature_list, axis=1)

        # ==================== 归一化 ====================
        # 每列单独归一化到 [0, 1]
        feature_min = features.min(axis=0, keepdims=True)
        feature_max = features.max(axis=0, keepdims=True)
        feature_range = feature_max - feature_min
        feature_range[feature_range == 0] = 1  # 避免除零
        features = (features - feature_min) / feature_range

        print(f"Created RQ-VAE training data: {features.shape}")
        print(f"  Feature dim: {features.shape[1]}")
        print(f"  Feature stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
        
        # 打印特征示例
        print(f"\n  Sample features (first 3 items):")
        for i in range(min(3, len(features))):
            print(f"    Item {i}: click={item_click_count[i]:.0f}, users={item_user_count[i]:.0f}, "
                  f"ctr={item_ctr[i]:.3f}, like_rate={item_like_rate[i]:.3f}")
        
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
