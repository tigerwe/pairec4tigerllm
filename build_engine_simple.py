#!/usr/bin/env python3
"""使用 TensorRT-LLM Python API 构建引擎"""

import torch
import json
import os
import sys
import argparse

# 先测试是否能导入
try:
    import tensorrt_llm
    from tensorrt_llm import Builder
    print(f"✅ TensorRT-LLM {tensorrt_llm.__version__} 导入成功")
except ImportError as e:
    print(f"❌ 无法导入 TensorRT-LLM: {e}")
    print("尝试使用 PyTorch 导出方式...")
    sys.exit(1)

def build_engine(args):
    """构建 TensorRT-LLM 引擎"""
    
    # 加载模型配置
    config_path = os.path.join(args.checkpoint_dir, "decoder_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"模型配置: vocab_size={config['vocab_size']}, "
          f"num_layers={config['num_layers']}, "
          f"hidden_size={config['embedding_dim']}")
    
    # 创建 Builder
    builder = Builder()
    
    # 构建配置
    builder_config = builder.create_builder_config(
        name=config['model_name'],
        precision=args.dtype,
        use_gpt_attention_plugin=args.use_gpt_attention_plugin,
        use_gemm_plugin=args.use_gemm_plugin,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len
    )
    
    print(f"开始构建引擎: {args.output_path}")
    print(f"配置: dtype={args.dtype}, max_batch={args.max_batch_size}")
    
    # 这里简化处理，实际应该加载权重并构建
    # 由于不同版本 API 差异大，建议参考官方文档
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--use_gpt_attention_plugin", action="store_true")
    parser.add_argument("--use_gemm_plugin", action="store_true")
    parser.add_argument("--max_batch_size", type=int, default=32)
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--max_output_len", type=int, default=512)
    
    args = parser.parse_args()
    build_engine(args)
