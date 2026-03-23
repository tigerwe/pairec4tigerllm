# -*- coding: utf-8 -*-
"""Tenrec 数据加载器 - 优化版本.

使用 groupby 替代 filter，大幅提升性能.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class TenrecDataLoader:
    """Tenrec 数据集加载器 - 优化版本.

    负责加载 ctr_data_1M.csv 文件，解析用户行为数据，
    并构建训练和测试数据集.

    Attributes:
        data_path: 数据集文件路径
        cache_dir: 缓存目录
        df: 加载的数据框
        user_ids: 用户 ID 列表
        item_ids: 物品 ID 列表
    """

    # CSV 列名定义
    REQUIRED_COLUMNS = [
        'user_id', 'item_id', 'click', 'follow', 'like', 'share',
        'video_category', 'watching_times', 'gender', 'age',
        'hist_1', 'hist_2', 'hist_3', 'hist_4', 'hist_5',
        'hist_6', 'hist_7', 'hist_8', 'hist_9', 'hist_10'
    ]

    HISTORY_COLUMNS = [f'hist_{i}' for i in range(1, 11)]

    def __init__(self, data_path: str, cache_dir: Optional[str] = None):
        """初始化数据加载器.

        Args:
            data_path: ctr_data_1M.csv 文件路径
            cache_dir: 缓存目录，用于存储预处理后的数据
        """
        self.data_path = data_path
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(data_path), 'cache'
        )
        self.df: Optional[pd.DataFrame] = None
        self.user_ids: Optional[np.ndarray] = None
        self.item_ids: Optional[np.ndarray] = None
        self.user_interactions: Optional[Dict[int, List[int]]] = None

        os.makedirs(self.cache_dir, exist_ok=True)

    def load(self, use_cache: bool = True) -> pd.DataFrame:
        """加载数据集.

        Args:
            use_cache: 是否使用缓存

        Returns:
            加载的数据框

        Raises:
            FileNotFoundError: 数据文件不存在
            ValueError: 数据格式不正确
        """
        cache_file = os.path.join(self.cache_dir, 'tenrec_processed.pkl')

        # 尝试从缓存加载
        if use_cache and os.path.exists(cache_file):
            print(f"Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.df = cached_data['df']
                self.user_ids = cached_data['user_ids']
                self.item_ids = cached_data['item_ids']
                self.user_interactions = cached_data['user_interactions']
                return self.df

        # 从 CSV 加载
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        print(f"Loading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)

        # 验证列名
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # 构建用户和物品 ID 列表
        self.user_ids = self.df['user_id'].unique()
        self.item_ids = self.df['item_id'].unique()

        # 构建用户交互序列 - 使用优化版本
        self._build_user_interactions_optimized()

        # 保存到缓存
        if use_cache:
            self._save_cache(cache_file)

        print(f"Data loaded: {len(self.df)} interactions, "
              f"{len(self.user_ids)} users, {len(self.item_ids)} items")

        return self.df

    def _build_user_interactions_optimized(self) -> None:
        """构建用户交互序列 - 优化版本.

        使用 groupby 替代 filter，时间复杂度从 O(n*m) 降到 O(n).
        """
        print("Building user interaction sequences (optimized)...")
        
        self.user_interactions = {}
        
        # 优化 1: 使用 groupby 替代 filter
        # 原始: for user_id in user_ids: df[df['user_id'] == user_id]  # O(n*m)
        # 优化: df.groupby('user_id')  # O(n)
        print("  Grouping data by user_id...")
        grouped = self.df.groupby('user_id', sort=False)
        
        total_users = len(grouped)
        print(f"  Processing {total_users} users...")
        
        # 优化 2: 预取历史列，避免重复查找
        history_cols = self.HISTORY_COLUMNS
        
        # 优化 3: 批量处理，减少 Python 循环开销
        for idx, (user_id, user_data) in enumerate(grouped, 1):
            if idx % 10000 == 0:
                print(f"  Processed {idx}/{total_users} users ({idx/total_users*100:.1f}%)")
            
            # 获取第一条记录的历史特征（所有记录的历史特征相同）
            first_row = user_data.iloc[0]
            
            # 收集用户历史点击（从 hist_1 到 hist_10）
            history = []
            for col in history_cols:
                val = first_row[col]
                if pd.notna(val) and val > 0:
                    history.append(int(val))
            
            # 添加当前交互（如果点击了）
            # 优化 4: 使用向量化操作替代 iterrows
            clicked_items = user_data.loc[user_data['click'] == 1, 'item_id'].tolist()
            history.extend([int(x) for x in clicked_items])
            
            if len(history) > 0:
                self.user_interactions[user_id] = history
        
        print(f"Built interactions for {len(self.user_interactions)} users")

    def _save_cache(self, cache_file: str) -> None:
        """保存数据到缓存.

        Args:
            cache_file: 缓存文件路径
        """
        cache_data = {
            'df': self.df,
            'user_ids': self.user_ids,
            'item_ids': self.item_ids,
            'user_interactions': self.user_interactions
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Cache saved to: {cache_file}")

    def get_user_sequence(self, user_id: int) -> List[int]:
        """获取用户的交互序列.

        Args:
            user_id: 用户 ID

        Returns:
            物品 ID 列表（按时间顺序）
        """
        if self.user_interactions is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self.user_interactions.get(user_id, [])

    def get_all_user_sequences(self) -> Dict[int, List[int]]:
        """获取所有用户的交互序列.

        Returns:
            用户 ID -> 交互序列 的字典
        """
        if self.user_interactions is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self.user_interactions

    def split_train_test(
        self,
        test_ratio: float = 0.2,
        min_sequence_length: int = 5,
        random_state: int = 42
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """划分训练集和测试集.

        对每个用户，保留最后一部分作为测试集，其余作为训练集.

        Args:
            test_ratio: 测试集比例
            min_sequence_length: 最小序列长度
            random_state: 随机种子

        Returns:
            (train_sequences, test_sequences) 元组
        """
        if self.user_interactions is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        train_sequences = {}
        test_sequences = {}

        for user_id, sequence in self.user_interactions.items():
            if len(sequence) < min_sequence_length:
                continue

            # 按时间顺序划分
            test_size = max(1, int(len(sequence) * test_ratio))
            train_seq = sequence[:-test_size]
            test_seq = sequence[-test_size:]

            if len(train_seq) > 0:
                train_sequences[user_id] = train_seq
            if len(test_seq) > 0:
                test_sequences[user_id] = test_seq

        print(f"Train users: {len(train_sequences)}, "
              f"Test users: {len(test_sequences)}")

        return train_sequences, test_sequences

    def get_item_features(self) -> pd.DataFrame:
        """获取物品特征.

        Returns:
            物品特征数据框
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        # 提取物品特征（去重）
        item_features = self.df.groupby('item_id').agg({
            'video_category': 'first',
        }).reset_index()

        return item_features

    def get_statistics(self) -> Dict:
        """获取数据集统计信息.

        Returns:
            统计信息字典
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        stats = {
            'num_interactions': len(self.df),
            'num_users': len(self.user_ids),
            'num_items': len(self.item_ids),
            'sparsity': 1 - len(self.df) / (len(self.user_ids) * len(self.item_ids)),
            'click_ratio': self.df['click'].mean(),
            'avg_sequence_length': np.mean([
                len(seq) for seq in self.user_interactions.values()
            ]) if self.user_interactions else 0
        }

        return stats
