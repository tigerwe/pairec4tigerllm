#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据集采样脚本.

从大型数据集中采样出适合实验的小数据集.
支持多种采样策略，保证数据质量和用户行为完整性.
"""

import os
import sys
import argparse
from typing import Optional

import pandas as pd
import numpy as np
from tqdm import tqdm


def sample_by_users(
    input_path: str,
    output_path: str,
    num_users: int = 10000,
    min_interactions_per_user: int = 5,
    random_seed: int = 42
) -> None:
    """按用户采样（推荐，保持用户行为完整性）.
    
    策略：
    1. 筛选出有足够交互行为的活跃用户
    2. 随机采样指定数量的用户
    3. 保留这些用户的全部交互记录
    
    Args:
        input_path: 原始数据路径
        output_path: 输出数据路径
        num_users: 采样用户数量
        min_interactions_per_user: 每个用户最少交互数
        random_seed: 随机种子
    """
    print(f"加载原始数据: {input_path}")
    
    # 流式读取，只读取必要列进行用户统计
    print("统计用户信息...")
    user_counts = {}
    chunk_iter = pd.read_csv(input_path, usecols=['user_id'], chunksize=100000)
    
    for chunk in tqdm(chunk_iter, desc="统计用户交互数"):
        for user_id in chunk['user_id']:
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
    
    # 筛选活跃用户
    active_users = [uid for uid, count in user_counts.items() 
                    if count >= min_interactions_per_user]
    print(f"总用户数: {len(user_counts)}, 活跃用户(>={min_interactions_per_user}条): {len(active_users)}")
    
    # 随机采样用户
    np.random.seed(random_seed)
    if len(active_users) <= num_users:
        sampled_users = set(active_users)
        print(f"警告: 活跃用户不足 {num_users}，实际使用 {len(active_users)} 个用户")
    else:
        sampled_users = set(np.random.choice(active_users, num_users, replace=False))
    
    print(f"采样用户数: {len(sampled_users)}")
    
    # 第二次遍历，提取采样用户的数据
    print("提取采样用户数据...")
    sampled_chunks = []
    chunk_iter = pd.read_csv(input_path, chunksize=100000)
    
    for chunk in tqdm(chunk_iter, desc="提取数据"):
        mask = chunk['user_id'].isin(sampled_users)
        if mask.any():
            sampled_chunks.append(chunk[mask])
    
    # 合并并保存
    result_df = pd.concat(sampled_chunks, ignore_index=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    
    # 打印统计
    actual_users = result_df['user_id'].nunique()
    actual_items = result_df['item_id'].nunique()
    print(f"\n✓ 采样完成！")
    print(f"  输出文件: {output_path}")
    print(f"  总记录数: {len(result_df):,}")
    print(f"  用户数: {actual_users:,}")
    print(f"  物品数: {actual_items:,}")
    print(f"  平均每个用户交互数: {len(result_df) / actual_users:.1f}")


def sample_by_rows(
    input_path: str,
    output_path: str,
    num_rows: int = 100000,
    random_seed: int = 42
) -> None:
    """按行数随机采样（简单快速，但可能破坏用户行为完整性）.
    
    Args:
        input_path: 原始数据路径
        output_path: 输出数据路径
        num_rows: 采样行数
        random_seed: 随机种子
    """
    print(f"按行数采样: {num_rows} 行")
    
    # 先统计总行数
    print("统计总行数...")
    total_rows = sum(1 for _ in open(input_path)) - 1  # 减去表头
    print(f"总行数: {total_rows:,}")
    
    # 随机选择要保留的行号
    np.random.seed(random_seed)
    keep_rows = set(np.random.choice(total_rows, min(num_rows, total_rows), replace=False))
    
    # 流式读取并采样
    print("采样数据...")
    sampled_chunks = []
    chunk_iter = pd.read_csv(input_path, chunksize=50000)
    
    row_idx = 0
    for chunk in tqdm(chunk_iter, total=total_rows//50000 + 1):
        chunk_size = len(chunk)
        chunk_indices = set(range(row_idx, row_idx + chunk_size))
        keep_in_chunk = list(chunk_indices & keep_rows)
        
        if keep_in_chunk:
            # 调整索引到 chunk 内
            local_indices = [i - row_idx for i in keep_in_chunk]
            sampled_chunks.append(chunk.iloc[local_indices])
        
        row_idx += chunk_size
    
    # 合并并保存
    result_df = pd.concat(sampled_chunks, ignore_index=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    
    print(f"\n✓ 采样完成！")
    print(f"  输出文件: {output_path}")
    print(f"  总记录数: {len(result_df):,}")
    print(f"  用户数: {result_df['user_id'].nunique():,}")
    print(f"  物品数: {result_df['item_id'].nunique():,}")


def sample_stratified(
    input_path: str,
    output_path: str,
    num_users: int = 10000,
    min_interactions: int = 5,
    max_interactions: int = 500,
    random_seed: int = 42
) -> None:
    """分层采样（保证不同活跃度用户的比例）.
    
    Args:
        input_path: 原始数据路径
        output_path: 输出数据路径
        num_users: 采样用户数量
        min_interactions: 最少交互数
        max_interactions: 最多交互数（防止超级用户 dominating）
        random_seed: 随机种子
    """
    print(f"分层采样 {num_users} 个用户...")
    
    # 统计用户交互数
    print("统计用户活跃度...")
    user_counts = {}
    chunk_iter = pd.read_csv(input_path, usecols=['user_id'], chunksize=100000)
    
    for chunk in tqdm(chunk_iter, desc="统计"):
        for user_id in chunk['user_id']:
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
    
    # 筛选满足条件的用户
    valid_users = {uid: count for uid, count in user_counts.items()
                   if min_interactions <= count <= max_interactions}
    
    # 按交互数分层
    low_activity = [uid for uid, count in valid_users.items() if count < 20]
    mid_activity = [uid for uid, count in valid_users.items() if 20 <= count < 100]
    high_activity = [uid for uid, count in valid_users.items() if count >= 100]
    
    print(f"低活跃度用户(5-19): {len(low_activity)}")
    print(f"中活跃度用户(20-99): {len(mid_activity)}")
    print(f"高活跃度用户(100+): {len(high_activity)}")
    
    # 按比例采样
    np.random.seed(random_seed)
    total_valid = len(valid_users)
    
    n_low = int(num_users * len(low_activity) / total_valid)
    n_mid = int(num_users * len(mid_activity) / total_valid)
    n_high = num_users - n_low - n_mid
    
    sampled_users = set()
    sampled_users.update(np.random.choice(low_activity, min(n_low, len(low_activity)), replace=False))
    sampled_users.update(np.random.choice(mid_activity, min(n_mid, len(mid_activity)), replace=False))
    sampled_users.update(np.random.choice(high_activity, min(n_high, len(high_activity)), replace=False))
    
    print(f"实际采样: 低活跃 {min(n_low, len(low_activity))}, 中活跃 {min(n_mid, len(mid_activity))}, 高活跃 {min(n_high, len(high_activity))}")
    
    # 提取数据
    print("提取数据...")
    sampled_chunks = []
    chunk_iter = pd.read_csv(input_path, chunksize=100000)
    
    for chunk in tqdm(chunk_iter, desc="提取"):
        mask = chunk['user_id'].isin(sampled_users)
        if mask.any():
            sampled_chunks.append(chunk[mask])
    
    result_df = pd.concat(sampled_chunks, ignore_index=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    
    print(f"\n✓ 采样完成！")
    print(f"  输出文件: {output_path}")
    print(f"  总记录数: {len(result_df):,}")
    print(f"  用户数: {result_df['user_id'].nunique():,}")
    print(f"  物品数: {result_df['item_id'].nunique():,}")


def main():
    """命令行入口."""
    parser = argparse.ArgumentParser(description='Sample dataset from large CSV')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--strategy', type=str, default='by_users',
                        choices=['by_users', 'by_rows', 'stratified'],
                        help='Sampling strategy')
    parser.add_argument('--num_users', type=int, default=10000,
                        help='Number of users to sample (for by_users/stratified)')
    parser.add_argument('--num_rows', type=int, default=100000,
                        help='Number of rows to sample (for by_rows)')
    parser.add_argument('--min_interactions', type=int, default=5,
                        help='Minimum interactions per user')
    parser.add_argument('--max_interactions', type=int, default=500,
                        help='Maximum interactions per user (stratified only)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    if args.strategy == 'by_users':
        sample_by_users(
            input_path=args.input,
            output_path=args.output,
            num_users=args.num_users,
            min_interactions_per_user=args.min_interactions,
            random_seed=args.seed
        )
    elif args.strategy == 'by_rows':
        sample_by_rows(
            input_path=args.input,
            output_path=args.output,
            num_rows=args.num_rows,
            random_seed=args.seed
        )
    elif args.strategy == 'stratified':
        sample_stratified(
            input_path=args.input,
            output_path=args.output,
            num_users=args.num_users,
            min_interactions=args.min_interactions,
            max_interactions=args.max_interactions,
            random_seed=args.seed
        )


if __name__ == '__main__':
    main()
