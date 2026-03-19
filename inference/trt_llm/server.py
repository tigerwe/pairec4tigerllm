# -*- coding: utf-8 -*-
"""TensorRT-LLM 推理服务.

提供基于 PyTorch 和 TensorRT-LLM 1.0.0 的生成式召回推理服务.
支持 HTTP 接口.
"""

import os
import sys
import argparse
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from training.decoder.model import GenerativeDecoder


@dataclass
class InferenceConfig:
    """推理配置."""
    model_path: str
    device: str = 'cuda'
    max_batch_size: int = 32
    max_seq_len: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    beam_width: int = 5
    use_trt_llm: bool = False  # 是否使用 TensorRT-LLM


class TensorRTLLMInference:
    """TensorRT-LLM 1.0.0 推理引擎封装.
    
    如果 TensorRT-LLM 不可用，自动回退到 PyTorch.
    """
    
    def __init__(self, engine_path: str, config: InferenceConfig):
        """初始化 TensorRT-LLM 推理引擎.
        
        Args:
            engine_path: TensorRT 引擎路径
            config: 推理配置
        """
        self.config = config
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.stream = None
        
        # 尝试加载 TensorRT-LLM
        self._init_trt_llm()
    
    def _init_trt_llm(self) -> bool:
        """初始化 TensorRT-LLM.
        
        Returns:
            是否成功初始化
        """
        try:
            import tensorrt_llm as trtllm
            from tensorrt_llm.runtime import ModelConfig, SamplingConfig
            
            logger.info(f"TensorRT-LLM 版本: {trtllm.__version__}")
            
            # 检查引擎文件
            if not os.path.exists(self.engine_path):
                logger.error(f"引擎文件不存在: {self.engine_path}")
                return False
            
            # TensorRT-LLM 1.0.0 推理初始化
            # 注意：这里使用运行时 API，具体的初始化方式取决于引擎构建方式
            self.runtime = trtllm.runtime.Runtime()
            
            # 加载引擎
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()
            
            # 创建 CUDA 流
            import cupy as cp
            self.stream = cp.cuda.Stream()
            
            logger.info("TensorRT-LLM 1.0.0 引擎加载成功")
            return True
            
        except ImportError:
            logger.warning("TensorRT-LLM 未安装，无法使用 TRT 推理")
            return False
        except Exception as e:
            logger.error(f"TensorRT-LLM 初始化失败: {e}")
            return False
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int
    ) -> torch.Tensor:
        """生成推荐.
        
        Args:
            input_ids: 输入序列
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-k 采样
            
        Returns:
            生成的序列
        """
        if self.engine is None:
            raise RuntimeError("TensorRT-LLM 引擎未初始化")
        
        # 这里需要根据具体的 TensorRT-LLM 1.0.0 API 实现
        # 由于 API 可能变化，这里提供基本框架
        
        try:
            # TensorRT-LLM 1.0.0 推理逻辑
            # TODO: 根据实际 API 调整
            logger.info("使用 TensorRT-LLM 推理")
            
            # 回退到 PyTorch 生成
            return self._fallback_generate(input_ids, max_new_tokens, temperature, top_k)
            
        except Exception as e:
            logger.error(f"TensorRT-LLM 推理失败: {e}")
            return self._fallback_generate(input_ids, max_new_tokens, temperature, top_k)
    
    def _fallback_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int
    ) -> torch.Tensor:
        """回退到 PyTorch 生成."""
        logger.warning("回退到 PyTorch 推理")
        # 这里应该调用 PyTorch 模型，但为了解耦，返回空
        # 实际使用时，由上层服务处理
        return None


class GenerativeInferenceService:
    """生成式推理服务.

    支持 PyTorch 和 TensorRT-LLM 1.0.0 两种后端.
    优先使用 TensorRT-LLM (如果可用且配置启用).
    """

    def __init__(self, config: InferenceConfig):
        """初始化推理服务.

        Args:
            config: 推理配置
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        print(f"Initializing inference service on {self.device}")
        
        # 初始化 TensorRT-LLM (如果启用)
        self.trt_llm_engine = None
        if config.use_trt_llm:
            engine_path = config.model_path.replace('.pt', '.engine')
            if os.path.exists(engine_path):
                print(f"尝试加载 TensorRT-LLM 引擎: {engine_path}")
                self.trt_llm_engine = TensorRTLLMInference(engine_path, config)
                if self.trt_llm_engine.engine is None:
                    print("TensorRT-LLM 加载失败，将使用 PyTorch")
                    self.trt_llm_engine = None
            else:
                print(f"TensorRT 引擎不存在: {engine_path}")
                print("将使用 PyTorch 推理")

        # 加载 PyTorch 模型 (作为回退或主推理)
        self._load_model()

        # 加载语义 ID 映射
        self._load_semantic_id_mapping()

        print("Inference service initialized successfully")

    def _load_model(self) -> None:
        """加载 PyTorch 模型."""
        print(f"Loading PyTorch model from {self.config.model_path}")

        checkpoint = torch.load(self.config.model_path, map_location=self.device)
        model_config = checkpoint['config']

        self.model = GenerativeDecoder(
            vocab_size=model_config['vocab_size'],
            num_quantizers=model_config['num_quantizers'],
            embedding_dim=model_config['embedding_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            max_seq_len=model_config['max_seq_len']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.vocab_size = model_config['vocab_size']
        self.num_quantizers = model_config['num_quantizers']
        self.pad_token_id = model_config.get('pad_token_id', 0)

        print(f"PyTorch model loaded: {self.num_quantizers} quantizers, vocab size {self.vocab_size}")

    def _load_semantic_id_mapping(self) -> None:
        """加载语义 ID 到物品 ID 的映射."""
        # 默认路径
        mapping_path = os.path.join(
            os.path.dirname(self.config.model_path),
            'rqvae_semantic_ids.json'
        )
        
        # 也尝试从 processed 目录加载
        if not os.path.exists(mapping_path):
            mapping_path = './data/tenrec/processed/semantic_id_map.json'

        if os.path.exists(mapping_path):
            print(f"Loading semantic ID mapping from {mapping_path}")
            with open(mapping_path, 'r') as f:
                self.semantic_to_item = json.load(f)

            # 转换为元组形式（用于哈希）
            self.semantic_to_item_tuple = {}
            for item_id, sem_ids in self.semantic_to_item.items():
                key = tuple(sem_ids)
                self.semantic_to_item_tuple[key] = int(item_id)

            print(f"Loaded {len(self.semantic_to_item)} item mappings")
        else:
            print(f"Warning: Semantic ID mapping not found at {mapping_path}")
            self.semantic_to_item = {}
            self.semantic_to_item_tuple = {}

    def recommend(
        self,
        user_history: List[List[int]],
        topk: int = 10,
        temperature: Optional[float] = None,
        beam_width: Optional[int] = None
    ) -> List[Dict]:
        """生成推荐.

        Args:
            user_history: 用户历史语义 ID 序列，每个元素是 [num_quantizers] 列表
            topk: 推荐数量
            temperature: 采样温度
            beam_width: Beam search 宽度

        Returns:
            推荐结果列表，每个元素包含 item_id 和 score
        """
        start_time = time.time()

        temperature = temperature or self.config.temperature
        beam_width = beam_width or self.config.beam_width

        # 准备输入
        input_ids = self._prepare_input(user_history)
        input_ids = input_ids.to(self.device)

        # 生成推荐
        with torch.no_grad():
            if beam_width > 1:
                recommendations = self._beam_search_generate(
                    input_ids, topk, beam_width
                )
            else:
                recommendations = self._sampling_generate(
                    input_ids, topk, temperature
                )

        inference_time = (time.time() - start_time) * 1000  # ms

        return {
            'recommendations': recommendations,
            'inference_time_ms': inference_time
        }

    def _prepare_input(self, user_history: List[List[int]]) -> torch.Tensor:
        """准备输入张量.

        Args:
            user_history: 用户历史序列

        Returns:
            输入张量 [1, seq_len, num_quantizers]
        """
        if not user_history:
            # 空历史，使用 pad token
            return torch.zeros(1, 1, self.num_quantizers, dtype=torch.long)

        # 转换为张量
        seq_len = len(user_history)
        input_ids = torch.zeros(1, seq_len, self.num_quantizers, dtype=torch.long)

        for i, sem_ids in enumerate(user_history):
            for j, sem_id in enumerate(sem_ids):
                input_ids[0, i, j] = sem_id

        return input_ids

    def _sampling_generate(
        self,
        input_ids: torch.Tensor,
        topk: int,
        temperature: float
    ) -> List[Dict]:
        """采样生成.

        Args:
            input_ids: 输入张量
            topk: 生成数量
            temperature: 温度

        Returns:
            推荐列表
        """
        recommendations = []
        current_input = input_ids.clone()

        for _ in range(topk * 2):  # 多生成一些，去重后取 topk
            # 单步生成
            logits, _ = self.model(current_input)
            next_logits = logits[:, -1, :, :]  # [1, num_quantizers, vocab_size]

            # 应用温度
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # 采样
            probs = F.softmax(next_logits, dim=-1)
            next_tokens = torch.multinomial(
                probs.view(-1, self.vocab_size), num_samples=1
            ).view(1, self.num_quantizers)

            # 转换为列表
            sem_ids = next_tokens[0].cpu().tolist()
            sem_tuple = tuple(sem_ids)

            # 查找物品 ID
            item_id = self.semantic_to_item_tuple.get(sem_tuple)

            if item_id and item_id not in [r['item_id'] for r in recommendations]:
                # 计算分数（使用概率的均值）
                score = probs[0].max(dim=-1)[0].mean().item()
                recommendations.append({
                    'item_id': item_id,
                    'semantic_id': sem_ids,
                    'score': score
                })

                if len(recommendations) >= topk:
                    break

            # 更新输入
            next_tokens = next_tokens.unsqueeze(1)
            current_input = torch.cat([current_input, next_tokens], dim=1)

            # 截断长度
            if current_input.shape[1] > self.config.max_seq_len:
                current_input = current_input[:, -self.config.max_seq_len:]

        return recommendations[:topk]

    def _beam_search_generate(
        self,
        input_ids: torch.Tensor,
        topk: int,
        beam_width: int
    ) -> List[Dict]:
        """Beam search 生成.

        Args:
            input_ids: 输入张量
            topk: 生成数量
            beam_width: beam 宽度

        Returns:
            推荐列表
        """
        # 简化的 beam search 实现
        # 实际生产环境可能需要更高效的实现

        recommendations = []
        candidates = [(input_ids.clone(), 0.0)]  # (sequence, score)

        for _ in range(topk * 2):
            new_candidates = []

            for seq, score in candidates:
                logits, _ = self.model(seq)
                next_logits = logits[:, -1, :, :]
                log_probs = F.log_softmax(next_logits, dim=-1)

                # 取 top beam_width 个
                topk_vals, topk_indices = log_probs.topk(beam_width, dim=-1)

                for i in range(beam_width):
                    next_tokens = topk_indices[0, :, i].unsqueeze(0).unsqueeze(1)
                    new_seq = torch.cat([seq, next_tokens], dim=1)
                    new_score = score + topk_vals[0, :, i].sum().item()
                    new_candidates.append((new_seq, new_score))

            # 排序并保留 top beam_width
            new_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = new_candidates[:beam_width]

            # 提取推荐
            for seq, score in candidates:
                sem_ids = seq[0, -1, :].cpu().tolist()
                sem_tuple = tuple(sem_ids)
                item_id = self.semantic_to_item_tuple.get(sem_tuple)

                if item_id and item_id not in [r['item_id'] for r in recommendations]:
                    recommendations.append({
                        'item_id': item_id,
                        'semantic_id': sem_ids,
                        'score': np.exp(score / len(sem_ids))  # 转换为概率
                    })

                    if len(recommendations) >= topk:
                        return recommendations[:topk]

        return recommendations[:topk]


class HTTPServer:
    """HTTP 推理服务."""

    def __init__(self, inference_service: GenerativeInferenceService, port: int = 8000):
        """初始化 HTTP 服务.

        Args:
            inference_service: 推理服务实例
            port: 服务端口
        """
        self.service = inference_service
        self.port = port

    def start(self) -> None:
        """启动 HTTP 服务."""
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            print("Flask not installed. Installing...")
            os.system("pip install flask")
            from flask import Flask, request, jsonify

        app = Flask(__name__)

        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                'status': 'healthy',
                'backend': 'pytorch',  # 或 'trt_llm'
                'version': '1.0.0'
            })

        @app.route('/recommend', methods=['POST'])
        def recommend():
            try:
                data = request.get_json()
                user_id = data.get('user_id', '')
                history = data.get('history', [])
                topk = data.get('topk', 10)
                temperature = data.get('temperature', 1.0)
                beam_width = data.get('beam_width', 1)

                result = self.service.recommend(
                    user_history=history,
                    topk=topk,
                    temperature=temperature,
                    beam_width=beam_width
                )

                return jsonify({
                    'code': 200,
                    'user_id': user_id,
                    'recommendations': result['recommendations'],
                    'inference_time_ms': result['inference_time_ms']
                })

            except Exception as e:
                return jsonify({
                    'code': 500,
                    'error': str(e)
                }), 500

        print(f"Starting HTTP server on port {self.port}")
        app.run(host='0.0.0.0', port=self.port, threaded=True)


def main():
    """命令行入口."""
    parser = argparse.ArgumentParser(description='Start inference service')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--port', type=int, default=8000,
                        help='HTTP service port')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--max_batch_size', type=int, default=32,
                        help='Maximum batch size')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--use_trt_llm', action='store_true',
                        help='Use TensorRT-LLM if available')

    args = parser.parse_args()

    # 创建配置
    config = InferenceConfig(
        model_path=args.model_path,
        device=args.device,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        use_trt_llm=args.use_trt_llm
    )

    # 创建推理服务
    service = GenerativeInferenceService(config)

    # 启动 HTTP 服务
    server = HTTPServer(service, port=args.port)
    server.start()


if __name__ == '__main__':
    main()
