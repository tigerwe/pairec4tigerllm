# -*- coding: utf-8 -*-
"""
Decoder 模型导出脚本.

提供模型导出为 ONNX 格式，用于 TensorRT-LLM 部署.
"""

import os
import argparse
import json
from typing import Dict
import torch
import numpy as np

from .model import GenerativeDecoder


class DecoderExportWrapper(torch.nn.Module):
    """Decoder 导出包装器.

    用于导出为 ONNX 格式，简化输入输出.
    """

    def __init__(self, model: GenerativeDecoder):
        super().__init__()
        self.model = model

    def forward(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        """前向传播.

        Args:
            semantic_ids: 语义 ID 序列 [batch_size, seq_len, num_quantizers]

        Returns:
            logits: 输出 logits [batch_size, seq_len, num_quantizers, vocab_size]
        """
        logits, _ = self.model(semantic_ids)
        return logits


class DecoderGenerationWrapper(torch.nn.Module):
    """Decoder 生成包装器.

    用于导出生成推理图，支持单步生成.
    """

    def __init__(self, model: GenerativeDecoder):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """单步前向传播.

        Args:
            input_ids: 输入序列 [batch_size, seq_len, num_quantizers]

        Returns:
            next_token_logits: 下一个 token 的 logits [batch_size, num_quantizers, vocab_size]
        """
        logits, _ = self.model(input_ids)
        # 只返回最后一个位置的 logits
        return logits[:, -1, :, :]


def export_decoder(
    model: GenerativeDecoder,
    output_dir: str,
    model_name: str = 'decoder',
    opset_version: int = 14,
    max_batch_size: int = 32,
    max_seq_len: int = 512
) -> Dict:
    """导出 Decoder 模型.

    Args:
        model: 训练好的 Decoder 模型
        output_dir: 输出目录
        model_name: 模型名称
        opset_version: ONNX opset 版本
        max_batch_size: 最大批次大小
        max_seq_len: 最大序列长度

    Returns:
        导出配置字典
    """
    os.makedirs(output_dir, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    print(f"Exporting Decoder model...")
    print(f"Vocab size: {model.vocab_size}, Quantizers: {model.num_quantizers}")

    # 1. 导出完整模型（用于训练验证）
    onnx_full_path = os.path.join(output_dir, f'{model_name}_full.onnx')

    wrapper = DecoderExportWrapper(model)
    dummy_input = torch.randint(
        0, model.vocab_size,
        (1, 10, model.num_quantizers),
        dtype=torch.long,
        device=device
    )

    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_full_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    print(f"Full model exported to {onnx_full_path}")

    # 2. 导出单步生成模型（用于推理）
    onnx_step_path = os.path.join(output_dir, f'{model_name}_step.onnx')

    step_wrapper = DecoderGenerationWrapper(model)
    dummy_step_input = torch.randint(
        0, model.vocab_size,
        (1, max_seq_len, model.num_quantizers),
        dtype=torch.long,
        device=device
    )

    torch.onnx.export(
        step_wrapper,
        dummy_step_input,
        onnx_step_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['next_logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'next_logits': {0: 'batch_size'}
        }
    )
    print(f"Step model exported to {onnx_step_path}")

    # 3. 保存 PyTorch 模型
    model_path = os.path.join(output_dir, f'{model_name}_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': model.vocab_size,
            'num_quantizers': model.num_quantizers,
            'embedding_dim': model.embedding_dim,
            'num_layers': len(model.transformer_blocks),
            'num_heads': model.transformer_blocks[0].attention.num_heads,
            'ffn_dim': model.transformer_blocks[0].ffn[0].out_features,
            'max_seq_len': model.max_seq_len,
            'pad_token_id': model.pad_token_id
        }
    }, model_path)
    print(f"PyTorch model saved to {model_path}")

    # 4. 保存模型配置（用于 TensorRT-LLM）
    config = {
        'model_name': model_name,
        'architecture': 'gpt2_decoder',
        'vocab_size': model.vocab_size,
        'num_quantizers': model.num_quantizers,
        'embedding_dim': model.embedding_dim,
        'num_layers': len(model.transformer_blocks),
        'num_heads': model.transformer_blocks[0].attention.num_heads,
        'ffn_dim': model.transformer_blocks[0].ffn[0].out_features,
        'max_seq_len': max_seq_len,
        'max_batch_size': max_batch_size,
        'pad_token_id': model.pad_token_id,
        'onnx_full_path': onnx_full_path,
        'onnx_step_path': onnx_step_path,
        'model_path': model_path,
        'tensorrt_engine_path': os.path.join(output_dir, f'{model_name}.engine')
    }

    config_path = os.path.join(output_dir, f'{model_name}_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"Config saved to {config_path}")

    # 5. 生成 TensorRT-LLM 构建脚本
    build_script_path = os.path.join(output_dir, 'build_trt_engine.py')
    _generate_trt_build_script(config, build_script_path)
    print(f"TensorRT build script saved to {build_script_path}")

    return config


def _generate_trt_build_script(config: Dict, output_path: str) -> None:
    """生成 TensorRT-LLM 引擎构建脚本.

    调用 inference/trt_llm/build_engine.py 进行引擎构建.

    Args:
        config: 模型配置
        output_path: 输出路径
    """
    # 使用项目提供的完整构建脚本
    script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRT-LLM 1.0.0 引擎构建脚本.

自动生成的脚本，调用项目统一的构建入口.
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from inference.trt_llm.build_engine import main as build_engine_main

if __name__ == '__main__':
    # 调用统一的构建入口
    build_engine_main()
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    # 添加执行权限
    os.chmod(output_path, 0o755)
    
    print(f"TensorRT-LLM 1.0.0 构建脚本已生成: {output_path}")
    print("使用方法:")
    print(f"  python {output_path} \\")
    print(f"    --checkpoint_path ./checkpoints/decoder/decoder_best.pt \\")
    print(f"    --output_path {config['tensorrt_engine_path']} \\")
    print(f"    --dtype float16 \\")
    print(f"    --use_gpt_attention_plugin \\")
    print(f"    --use_gemm_plugin")


def validate_export(model: GenerativeDecoder, device: torch.device) -> None:
    """验证导出的模型.

    Args:
        model: 原始模型
        device: 设备
    """
    model.eval()

    # 测试输入
    test_input = torch.randint(
        0, model.vocab_size,
        (2, 10, model.num_quantizers),
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        logits, _ = model(test_input)

    print(f"Validation passed!")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: [2, 10, {model.num_quantizers}, {model.vocab_size}]")


def main():
    """命令行入口."""
    parser = argparse.ArgumentParser(description='Export Decoder model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./exported/decoder',
                        help='Output directory')
    parser.add_argument('--model_name', type=str, default='decoder',
                        help='Model name')
    parser.add_argument('--opset_version', type=int, default=14,
                        help='ONNX opset version')
    parser.add_argument('--max_batch_size', type=int, default=32,
                        help='Maximum batch size')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 加载模型
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint['config']

    model = GenerativeDecoder(
        vocab_size=config['vocab_size'],
        num_quantizers=config['num_quantizers'],
        embedding_dim=config['embedding_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        max_seq_len=config['max_seq_len']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 验证
    validate_export(model, device)

    # 导出
    export_decoder(
        model=model,
        output_dir=args.output_dir,
        model_name=args.model_name,
        opset_version=args.opset_version,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len
    )


if __name__ == '__main__':
    main()
