#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TensorRT-LLM 引擎构建脚本.

为 Decoder 模型构建 TensorRT 推理引擎.
适配 TensorRT-LLM 1.0.0 + CUDA 12.2 + L40S (Ada Lovelace)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_tensorrt_llm() -> tuple[bool, str]:
    """检查 TensorRT-LLM 是否安装及版本.
    
    Returns:
        (是否安装, 版本号)
    """
    try:
        import tensorrt_llm
        version = getattr(tensorrt_llm, "__version__", "unknown")
        
        # 检查版本是否 >= 1.0.0
        major_version = int(version.split('.')[0]) if version != "unknown" else 0
        if major_version < 1:
            logger.warning(f"TensorRT-LLM 版本 {version} 较旧，建议使用 1.0.0+")
        
        return True, version
    except ImportError:
        return False, "not_installed"


def build_gpt_decoder_engine(
    checkpoint_path: str,
    output_path: str,
    max_batch_size: int = 32,
    max_seq_len: int = 512,
    max_input_len: int = 256,
    max_output_len: int = 256,
    vocab_size: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    hidden_size: int = 256,
    ffn_hidden_size: int = 1024,
    num_quantizers: int = 4,
    dtype: str = "float16",
    remove_input_padding: bool = True,
    use_gpt_attention_plugin: bool = True,
    use_gemm_plugin: bool = True,
    use_weight_only: bool = False,
    weight_only_precision: str = "int8",
    parallel_build: bool = False,
    verbose: bool = False
) -> bool:
    """使用 TensorRT-LLM 1.0.0 构建 GPT Decoder 引擎.
    
    Args:
        checkpoint_path: PyTorch 检查点路径
        output_path: 输出引擎路径
        max_batch_size: 最大批次大小
        max_seq_len: 最大序列长度
        max_input_len: 最大输入长度
        max_output_len: 最大输出长度
        vocab_size: 词汇表大小
        num_layers: Transformer 层数
        num_heads: 注意力头数
        hidden_size: 隐藏层维度
        ffn_hidden_size: FFN 隐藏层维度
        num_quantizers: 量化器数量（语义 ID 层数）
        dtype: 数据类型 (float16/bfloat16/float32)
        remove_input_padding: 移除输入填充
        use_gpt_attention_plugin: 使用 GPT Attention 插件
        use_gemm_plugin: 使用 GEMM 插件
        use_weight_only: 仅使用权重量化
        weight_only_precision: 权重量化精度 (int8/int4)
        parallel_build: 并行构建
        verbose: 详细日志
        
    Returns:
        构建是否成功
    """
    installed, version = check_tensorrt_llm()
    if not installed:
        logger.error("TensorRT-LLM 未安装")
        logger.info("安装命令: pip install tensorrt-llm==1.0.0 --extra-index-url https://pypi.nvidia.com")
        return False
    
    logger.info(f"TensorRT-LLM 版本: {version}")
    
    try:
        import torch
        import tensorrt_llm as trtllm
        from tensorrt_llm.runtime import ModelConfig, SamplingConfig
        from tensorrt_llm._utils import torch_to_numpy
        
        # 导入构建相关模块 (TensorRT-LLM 1.0.0 API)
        try:
            from tensorrt_llm.models import GPTLMHeadModel, GPTConfig
            from tensorrt_llm.builder import Builder
            from tensorrt_llm.quantization import QuantMode
        except ImportError as e:
            logger.warning(f"导入 TensorRT-LLM 模块失败: {e}")
            logger.info("尝试使用备用方法...")
            return build_engine_fallback(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                dtype=dtype
            )
        
        # 检查检查点
        if not os.path.exists(checkpoint_path):
            logger.error(f"检查点文件不存在: {checkpoint_path}")
            return False
        
        logger.info(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取配置
        config = checkpoint.get('config', {})
        vocab_size = config.get('vocab_size', vocab_size)
        num_layers = config.get('num_layers', num_layers)
        num_heads = config.get('num_heads', num_heads)
        hidden_size = config.get('embedding_dim', hidden_size)
        
        logger.info(f"模型配置:")
        logger.info(f"  Vocab Size: {vocab_size}")
        logger.info(f"  Num Layers: {num_layers}")
        logger.info(f"  Num Heads: {num_heads}")
        logger.info(f"  Hidden Size: {hidden_size}")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # TensorRT-LLM 1.0.0 推荐使用 trtllm-build 命令行工具
        # 或者使用新的 Python API
        logger.info("使用 TensorRT-LLM 1.0.0 Python API 构建引擎...")
        
        # 构建配置
        build_config = {
            'max_batch_size': max_batch_size,
            'max_input_len': max_input_len,
            'max_output_len': max_output_len,
            'max_seq_len': max_seq_len,
            'vocab_size': vocab_size,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'hidden_size': hidden_size,
            'ffn_hidden_size': ffn_hidden_size or hidden_size * 4,
            'dtype': dtype,
            'remove_input_padding': remove_input_padding,
            'use_gpt_attention_plugin': use_gpt_attention_plugin,
            'use_gemm_plugin': use_gemm_plugin,
        }
        
        # 保存构建配置
        config_path = output_path.replace('.engine', '_build_config.json')
        import json
        with open(config_path, 'w') as f:
            json.dump(build_config, f, indent=2)
        logger.info(f"构建配置已保存: {config_path}")
        
        # 尝试使用新的构建 API
        try:
            # TensorRT-LLM 1.0.0 新 API
            from tensorrt_llm.builder import BuildConfig, Builder
            
            build_cfg = BuildConfig(
                max_batch_size=max_batch_size,
                max_input_len=max_input_len,
                max_output_len=max_output_len,
                max_seq_len=max_seq_len,
            )
            
            # 创建 GPT 配置
            gpt_config = GPTConfig(
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_heads=num_heads,
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size or hidden_size * 4,
                dtype=dtype,
            )
            
            # 创建模型
            model = GPTLMHeadModel(gpt_config)
            
            # 加载权重
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=False)
            
            # 构建引擎
            builder = Builder()
            engine = builder.build_engine(model, build_cfg)
            
            # 保存引擎
            with open(output_path, 'wb') as f:
                f.write(engine)
            
            logger.info(f"引擎构建成功: {output_path}")
            logger.info(f"引擎大小: {os.path.getsize(output_path) / (1024**2):.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"新 API 构建失败: {e}")
            logger.info("回退到命令行构建方式...")
            return build_engine_cli(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                build_config=build_config
            )
        
    except Exception as e:
        logger.error(f"构建引擎失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def build_engine_cli(
    checkpoint_path: str,
    output_path: str,
    build_config: Dict[str, Any]
) -> bool:
    """使用 trtllm-build 命令行工具构建引擎.
    
    TensorRT-LLM 1.0.0 推荐使用命令行工具进行构建.
    """
    import subprocess
    import json
    import tempfile
    
    try:
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(build_config, f, indent=2)
            config_file = f.name
        
        # 构建命令
        cmd = [
            'trtllm-build',
            '--checkpoint', checkpoint_path,
            '--output', output_path,
            '--max-batch-size', str(build_config['max_batch_size']),
            '--max-input-len', str(build_config['max_input_len']),
            '--max-output-len', str(build_config['max_output_len']),
            '--max-seq-len', str(build_config['max_seq_len']),
            '--dtype', build_config['dtype'],
        ]
        
        if build_config.get('use_gpt_attention_plugin'):
            cmd.append('--use-gpt-attention-plugin')
        
        if build_config.get('use_gemm_plugin'):
            cmd.append('--use-gemm-plugin')
        
        logger.info(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        # 清理临时文件
        os.unlink(config_file)
        
        if result.returncode == 0:
            logger.info(f"引擎构建成功: {output_path}")
            return True
        else:
            logger.error(f"构建失败:\n{result.stderr}")
            return False
            
    except FileNotFoundError:
        logger.error("trtllm-build 命令未找到，请确保 TensorRT-LLM 1.0.0 正确安装")
        return False
    except subprocess.TimeoutExpired:
        logger.error("构建超时")
        return False
    except Exception as e:
        logger.error(f"构建失败: {e}")
        return False


def build_engine_fallback(
    checkpoint_path: str,
    output_path: str,
    max_batch_size: int,
    max_seq_len: int,
    dtype: str
) -> bool:
    """备用构建方法 - 使用 ONNX + TensorRT.
    
    当 TensorRT-LLM 1.0.0 API 不可用时使用.
    """
    logger.info("使用备用构建方法 (ONNX + TensorRT)...")
    
    try:
        import tensorrt as trt
        import torch
        
        # 加载模型并导出为 ONNX
        from training.decoder.model import GenerativeDecoder
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        model = GenerativeDecoder(
            vocab_size=config.get('vocab_size', 256),
            num_quantizers=config.get('num_quantizers', 4),
            embedding_dim=config.get('embedding_dim', 256),
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            max_seq_len=config.get('max_seq_len', 512)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 导出 ONNX
        onnx_path = output_path.replace('.engine', '.onnx')
        dummy_input = torch.randint(0, 256, (1, 10, 4))
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        logger.info(f"ONNX 导出成功: {onnx_path}")
        
        # 构建 TensorRT 引擎
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(f"ONNX 解析错误: {parser.get_error(error)}")
                return False
        
        config = builder.create_builder_config()
        config.max_workspace_size = 4 * 1024 * 1024 * 1024
        
        if dtype == "float16":
            config.set_flag(trt.BuilderFlag.FP16)
        
        config.max_batch_size = max_batch_size
        
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input_ids",
            min=(1, 1, 4),
            opt=(max_batch_size // 2, max_seq_len // 2, 4),
            max=(max_batch_size, max_seq_len, 4)
        )
        config.add_optimization_profile(profile)
        
        engine = builder.build_engine(network, config)
        
        if engine is None:
            logger.error("引擎构建失败")
            return False
        
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        logger.info(f"引擎构建成功: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"备用构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_engine(engine_path: str) -> bool:
    """验证 TensorRT 引擎.
    
    Args:
        engine_path: 引擎路径
        
    Returns:
        验证是否通过
    """
    if not os.path.exists(engine_path):
        logger.error(f"引擎文件不存在: {engine_path}")
        return False
    
    try:
        import tensorrt as trt
        
        logger.info(f"验证引擎: {engine_path}")
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt.Logger())
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if engine is None:
            logger.error("引擎反序列化失败")
            return False
        
        logger.info(f"引擎验证通过:")
        logger.info(f"  绑定数量: {engine.num_bindings}")
        logger.info(f"  最大批次: {engine.max_batch_size}")
        
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            dtype = engine.get_binding_dtype(i)
            shape = engine.get_binding_shape(i)
            logger.info(f"  绑定 {i}: {name}, {dtype}, {shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"验证失败: {e}")
        return False


def main():
    """命令行入口."""
    parser = argparse.ArgumentParser(
        description='Build TensorRT-LLM 1.0.0 Engine for Decoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型路径
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoints/decoder/decoder_best.pt',
                        help='PyTorch checkpoint path')
    parser.add_argument('--output_path', type=str,
                        default='./exported/decoder/decoder.engine',
                        help='Output engine path')
    
    # 序列配置
    parser.add_argument('--max_batch_size', type=int, default=32,
                        help='Maximum batch size')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--max_input_len', type=int, default=256,
                        help='Maximum input length')
    parser.add_argument('--max_output_len', type=int, default=256,
                        help='Maximum output length')
    
    # 模型配置
    parser.add_argument('--vocab_size', type=int, default=256,
                        help='Vocabulary size')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden size')
    parser.add_argument('--ffn_hidden_size', type=int, default=1024,
                        help='FFN hidden size')
    parser.add_argument('--num_quantizers', type=int, default=4,
                        help='Number of quantizers')
    
    # 精度配置
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Data type')
    parser.add_argument('--use_weight_only', action='store_true',
                        help='Use weight-only quantization')
    parser.add_argument('--weight_only_precision', type=str, default='int8',
                        choices=['int8', 'int4'],
                        help='Weight-only precision')
    
    # 插件配置
    parser.add_argument('--use_gpt_attention_plugin', action='store_true', default=True,
                        help='Use GPT attention plugin')
    parser.add_argument('--use_gemm_plugin', action='store_true', default=True,
                        help='Use GEMM plugin')
    parser.add_argument('--remove_input_padding', action='store_true', default=True,
                        help='Remove input padding')
    
    # 其他
    parser.add_argument('--parallel_build', action='store_true',
                        help='Parallel build')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose logging')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify engine after build')

    args = parser.parse_args()

    print("=" * 70)
    print("TensorRT-LLM 1.0.0 Engine Builder")
    print("=" * 70)
    print()
    
    # 检查环境
    installed, version = check_tensorrt_llm()
    if installed:
        print(f"TensorRT-LLM 版本: {version}")
    else:
        print("警告: TensorRT-LLM 未安装，将使用备用构建方法")
    print()
    
    # 构建引擎
    success = build_gpt_decoder_engine(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_quantizers=args.num_quantizers,
        dtype=args.dtype,
        remove_input_padding=args.remove_input_padding,
        use_gpt_attention_plugin=args.use_gpt_attention_plugin,
        use_gemm_plugin=args.use_gemm_plugin,
        use_weight_only=args.use_weight_only,
        weight_only_precision=args.weight_only_precision,
        parallel_build=args.parallel_build,
        verbose=args.verbose
    )

    if not success:
        print()
        print("❌ 构建失败！")
        print()
        print("可能的解决方案:")
        print("  1. 确保 TensorRT-LLM 1.0.0 已正确安装:")
        print("     pip install tensorrt-llm==1.0.0 --extra-index-url https://pypi.nvidia.com")
        print("  2. 检查 CUDA 版本是否为 12.2:")
        print("     nvidia-smi")
        print("  3. 确保有足够的显存 (> 24GB 推荐)")
        print("  4. 查看详细错误信息:")
        print("     python build_engine.py --verbose")
        print()
        print("备选方案: 使用 PyTorch 推理服务")
        print("  ./scripts/start_trt_server.sh")
        sys.exit(1)

    # 验证引擎
    if args.verify:
        print()
        if not verify_engine(args.output_path):
            logger.warning("⚠️ 引擎验证未通过，但文件已生成")

    print()
    print("=" * 70)
    print("✅ 构建完成!")
    print(f"引擎路径: {args.output_path}")
    print("=" * 70)
    print()
    print("启动推理服务:")
    print(f"  python inference/trt_llm/server.py --model_path {args.output_path}")


if __name__ == '__main__':
    main()
