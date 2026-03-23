#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成用户特征文件 (优化版).

从 Tenrec 数据生成 pairec 所需的用户特征文件 (user_features.json).
使用 groupby 优化，时间复杂度从 O(n²) 降到 O(n).
"""

import os
import sys
import json
import argparse
from typing import Dict, List

import pandas as pd
import numpy as np


def generate_user_features_fast(
    data_path: str,
    output_path: str,
    history_feature_name: str = 'click_history',
    max_history_length: int = 20
) -> None:
    """生成用户特征文件 (优化版).
    
    Args:
        data_path: Tenrec 数据路径
        output_path: 输出文件路径
        history_feature_name: 历史特征字段名
        max_history_length: 最大历史长度
    """
    print(f"加载数据: {data_path}")
    
    # 只加载需要的列，节省内存
    usecols = ['user_id', 'item_id', 'click', 'gender', 'age'] + [f'hist_{i}' for i in range(1, 11)]
    df = pd.read_csv(data_path, usecols=usecols)
    
    print(f"数据加载完成: {len(df)} 条交互记录")
    print("生成用户特征...")
    
    user_features = {}
    
    # 使用 groupby 替代循环过滤，效率提升 1000 倍以上
    for user_id, user_data in df.groupby('user_id'):
        if len(user_data) == 0:
            continue
        
        # 取该用户的第一行获取基本特征
        first_row = user_data.iloc[0]
        
        # 构建历史点击序列
        history = []
        
        # 从 hist_1 到 hist_10 读取历史
        for i in range(1, 11):
            col = f'hist_{i}'
            if col in user_data.columns:
                val = first_row[col]
                if pd.notna(val) and val > 0:
                    history.append(int(val))
        
        # 添加当前点击的物品
        clicked_items = user_data[user_data['click'] == 1]['item_id'].tolist()
        history.extend([int(x) for x in clicked_items])
        
        # 去重并保持顺序（最新的在前面）
        seen = set()
        unique_history = []
        for item in reversed(history):
            if item not in seen and len(unique_history) < max_history_length:
                seen.add(item)
                unique_history.append(item)
        unique_history.reverse()
        
        # 构建用户特征字典
        user_features[str(int(user_id))] = {
            'user_id': str(int(user_id)),
            'gender': int(first_row['gender']) if pd.notna(first_row['gender']) else 0,
            'age': int(first_row['age']) if pd.notna(first_row['age']) else 0,
            history_feature_name: ','.join(map(str, unique_history)),
        }
    
    # 保存为 JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(user_features, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 用户特征已保存: {output_path}")
    print(f"  用户数: {len(user_features)}")
    print(f"  特征字段: user_id, gender, age, {history_feature_name}")
    
    # 打印示例
    if user_features:
        sample_user = list(user_features.keys())[0]
        print(f"\n示例用户 {sample_user}:")
        print(json.dumps(user_features[sample_user], indent=2))


def main():
    """命令行入口."""
    parser = argparse.ArgumentParser(description='Generate user features (fast version)')
    parser.add_argument('--input', type=str,
                        default='./data/tenrec/ctr_data_1M.csv',
                        help='Input data path')
    parser.add_argument('--output', type=str,
                        default='./data/user_features.json',
                        help='Output file path')
    parser.add_argument('--history_feature_name', type=str,
                        default='click_history',
                        help='History feature field name')
    parser.add_argument('--max_history_length', type=int,
                        default=20,
                        help='Maximum history length')
    
    args = parser.parse_args()
    
    generate_user_features_fast(
        data_path=args.input,
        output_path=args.output,
        history_feature_name=args.history_feature_name,
        max_history_length=args.max_history_length
    )


if __name__ == '__main__':
    main()
