#!/usr/bin/env python3
"""检查语义 ID 的多样性."""

import json
import sys

def check_semantic_ids(mapping_path: str):
    """检查语义 ID 映射是否有足够的多样性."""
    with open(mapping_path, 'r') as f:
        semantic_map = json.load(f)
    
    print(f"总物品数: {len(semantic_map)}")
    
    # 统计唯一语义 ID 数量
    unique_ids = set()
    for item_id, sem_id in semantic_map.items():
        unique_ids.add(tuple(sem_id))
    
    print(f"唯一语义 ID 数: {len(unique_ids)}")
    print(f"多样性比例: {len(unique_ids) / len(semantic_map) * 100:.2f}%")
    
    # 检查是否有重复的
    if len(unique_ids) == 1:
        print("\n❌ 严重：所有物品的语义 ID 都相同！")
        print(f"   重复的 ID: {list(unique_ids)[0]}")
        return False
    elif len(unique_ids) < len(semantic_map) * 0.1:
        print("\n⚠️ 警告：语义 ID 多样性不足（<10%）")
        return False
    else:
        print("\n✅ 语义 ID 多样性良好")
        
    # 显示前10个示例
    print("\n前10个物品的语义 ID:")
    for i, (item_id, sem_id) in enumerate(list(semantic_map.items())[:10]):
        print(f"  Item {item_id}: {sem_id}")
    
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        path = './data/tenrec/processed/semantic_id_map.json'
    else:
        path = sys.argv[1]
    
    success = check_semantic_ids(path)
    sys.exit(0 if success else 1)
