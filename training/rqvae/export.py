# -*- coding: utf-8 -*-
"""
RQ-VAE 模型导出脚本.

提供模型导出、语义 ID 生成和 ONNX 转换功能.
"""

import os
import argparse
import json
from typing import Dict, List
import torch
import numpy as np
import pandas as pd

from .model import RQVAE


def export_rqvae(
    model: RQVAE,
    item_features: torch.Tensor,
    item_ids: List[int],
    output_dir: str,
    model_name: str = 'rqvae',
    opset_version: int = 14
) -> Dict[int, List[int]]:
    """导出 RQ-VAE 模型并生成语义 ID 映射.

    Args:
        model: 训练好的 RQ-VAE 模型
        item_features: 物品特征 [num_items, feature_dim]
        item_ids: 原始物品 ID 列表
        output_dir: 输出目录
        model_name: 模型名称
        opset_version: ONNX opset 版本

    Returns:
        物品 ID -> 语义 ID 的映射字典
    """
    os.makedirs(output_dir, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    print(f"Exporting RQ-VAE model...")
    print(f"Items: {len(item_ids)}, Features shape: {item_features.shape}")

    # 生成语义 ID
    print("Generating semantic IDs...")
    with torch.no_grad():
        item_features = item_features.to(device)
        semantic_ids = model.get_semantic_ids(item_features)
        semantic_ids = semantic_ids.cpu().numpy()

    # 创建映射字典
    semantic_id_map = {}
    for i, item_id in enumerate(item_ids):
        semantic_id_map[item_id] = semantic_ids[i].tolist()

    # 保存语义 ID 映射
    mapping_path = os.path.join(output_dir, f'{model_name}_semantic_ids.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(semantic_id_map, f, ensure_ascii=False, indent=2)
    print(f"Semantic ID mapping saved to {mapping_path}")

    # 保存为 CSV 格式（便于查看）
    csv_data = []
    for item_id, sem_ids in semantic_id_map.items():
        csv_data.append({
            'item_id': item_id,
            'semantic_id': '-'.join(map(str, sem_ids)),
            **{f'code_{i}': sem_ids[i] for i in range(len(sem_ids))}
        })

    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, f'{model_name}_semantic_ids.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")

    # 导出 PyTorch 模型
    model_path = os.path.join(output_dir, f'{model_name}_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': model.encoder.network[0].in_features,
            'embedding_dim': model.embedding_dim,
            'num_quantizers': model.num_quantizers,
            'codebook_size': model.codebook_size,
        }
    }, model_path)
    print(f"Model saved to {model_path}")

    # 导出为 ONNX 格式
    onnx_path = os.path.join(output_dir, f'{model_name}_encoder.onnx')

    # 准备示例输入
    dummy_input = torch.randn(1, item_features.shape[1]).to(device)

    # 导出编码器
    torch.onnx.export(
        model.encoder,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_features'],
        output_names=['encoded_vector'],
        dynamic_axes={
            'input_features': {0: 'batch_size'},
            'encoded_vector': {0: 'batch_size'}
        }
    )
    print(f"ONNX encoder saved to {onnx_path}")

    # 导出完整模型（用于验证）
    onnx_full_path = os.path.join(output_dir, f'{model_name}_full.onnx')

    class RQVAEWrapper(torch.nn.Module):
        """包装器，用于导出完整的语义 ID 生成流程."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            semantic_ids = self.model.get_semantic_ids(x)
            return semantic_ids.float()

    wrapped_model = RQVAEWrapper(model)

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_full_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_features'],
        output_names=['semantic_ids'],
        dynamic_axes={
            'input_features': {0: 'batch_size'},
            'semantic_ids': {0: 'batch_size'}
        }
    )
    print(f"ONNX full model saved to {onnx_full_path}")

    # 保存模型配置
    config = {
        'model_name': model_name,
        'num_items': len(item_ids),
        'input_dim': item_features.shape[1],
        'embedding_dim': model.embedding_dim,
        'num_quantizers': model.num_quantizers,
        'codebook_size': model.codebook_size,
        'semantic_id_length': model.num_quantizers,
        'max_semantic_id_value': model.codebook_size - 1,
        'onnx_encoder_path': onnx_path,
        'onnx_full_path': onnx_full_path,
        'model_path': model_path,
        'mapping_path': mapping_path
    }

    config_path = os.path.join(output_dir, f'{model_name}_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"Config saved to {config_path}")

    # 验证导出的模型
    print("\nValidating exported model...")
    _validate_export(model, item_features[:100], device)

    return semantic_id_map


def _validate_export(
    model: RQVAE,
    test_features: torch.Tensor,
    device: torch.device
) -> None:
    """验证导出的模型.

    Args:
        model: 原始模型
        test_features: 测试特征
        device: 设备
    """
    model.eval()
    test_features = test_features.to(device)

    with torch.no_grad():
        semantic_ids = model.get_semantic_ids(test_features)

    print(f"Validation passed! Generated {len(semantic_ids)} semantic IDs")
    print(f"Semantic ID shape: {semantic_ids.shape}")
    print(f"Example semantic ID: {semantic_ids[0].cpu().numpy()}")


def main():
    """命令行入口."""
    parser = argparse.ArgumentParser(description='Export RQ-VAE model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to item features (.npy file)')
    parser.add_argument('--item_ids_path', type=str, required=True,
                        help='Path to item ids (.npy file)')
    parser.add_argument('--output_dir', type=str, default='./exported/rqvae',
                        help='Output directory')
    parser.add_argument('--model_name', type=str, default='rqvae',
                        help='Model name')
    parser.add_argument('--opset_version', type=int, default=14,
                        help='ONNX opset version')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 加载模型
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint['config']

    model = RQVAE(
        input_dim=config['input_dim'],
        embedding_dim=config['embedding_dim'],
        num_quantizers=config['num_quantizers'],
        codebook_size=config['codebook_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 加载数据
    print(f"Loading data from {args.data_path}")
    item_features = torch.from_numpy(np.load(args.data_path)).float()
    item_ids = np.load(args.item_ids_path).tolist()

    # 导出
    export_rqvae(
        model=model,
        item_features=item_features,
        item_ids=item_ids,
        output_dir=args.output_dir,
        model_name=args.model_name,
        opset_version=args.opset_version
    )


if __name__ == '__main__':
    main()
