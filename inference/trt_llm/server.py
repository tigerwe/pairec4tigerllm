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
    """TensorRT 10.x 推理引擎封装.
    
    加载通过 ONNX + TensorRT 构建的纯 TensorRT 引擎，
    为 GenerativeDecoder 提供加速推理.
    """
    
    def __init__(self, engine_path: str, config: InferenceConfig):
        """初始化 TensorRT 推理引擎.
        
        Args:
            engine_path: TensorRT 引擎路径
            config: 推理配置
        """
        self.config = config
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self._logger = logging.getLogger(__name__)
        
        self._init_engine()
    
    def _init_engine(self) -> bool:
        """初始化 TensorRT 引擎.
        
        Returns:
            是否成功初始化
        """
        try:
            import tensorrt as trt
            
            if not os.path.exists(self.engine_path):
                self._logger.error(f"引擎文件不存在: {self.engine_path}")
                return False
            
            self._logger.info(f"加载 TensorRT 引擎: {self.engine_path}")
            
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                self._logger.error("引擎反序列化失败")
                return False
            
            self.context = self.engine.create_execution_context()
            
            # 验证输入输出
            num_tensors = self.engine.num_io_tensors
            self._logger.info(f"引擎 IO Tensors: {num_tensors}")
            for i in range(num_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                dtype = self.engine.get_tensor_dtype(name)
                shape = self.engine.get_tensor_shape(name)
                self._logger.info(f"  {name}: mode={mode}, dtype={dtype}, shape={shape}")
            
            return True
            
        except Exception as e:
            self._logger.error(f"TensorRT 引擎初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """执行前向传播.
        
        Args:
            input_ids: 输入语义 ID [batch_size, seq_len, num_quantizers]
            
        Returns:
            logits: [batch_size, seq_len, num_quantizers, vocab_size]
        """
        if self.engine is None:
            raise RuntimeError("TensorRT 引擎未初始化")
        
        batch_size, seq_len, num_quantizers = input_ids.shape
        
        # 确保输入类型为 Int64 (与 ONNX 导出一致)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        
        # 设置动态输入形状
        self.context.set_input_shape("input_ids", (batch_size, seq_len, num_quantizers))
        
        # 获取输出形状并分配 GPU 内存 (转换为 tuple)
        output_shape = self.context.get_tensor_shape("logits")
        logits = torch.empty(tuple(output_shape), dtype=torch.float32, device=input_ids.device)
        
        # 绑定输入输出地址
        self.context.set_tensor_address("input_ids", input_ids.data_ptr())
        self.context.set_tensor_address("logits", logits.data_ptr())
        
        # 执行推理 (使用当前 CUDA stream)
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        
        return logits


class GenerativeInferenceService:
    """生成式推理服务.

    支持 PyTorch 和 TensorRT 两种后端.
    优先使用 TensorRT 引擎 (如果可用且配置启用).
    """

    def __init__(self, config: InferenceConfig):
        """初始化推理服务.

        Args:
            config: 推理配置
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        print(f"Initializing inference service on {self.device}")
        
        # 先从 checkpoint 读取配置信息
        checkpoint = torch.load(config.model_path, map_location='cpu')
        model_config = checkpoint['config']
        self.vocab_size = model_config['vocab_size']
        self.num_quantizers = model_config['num_quantizers']
        self.pad_token_id = model_config.get('pad_token_id', 0)
        self.max_seq_len = model_config.get('max_seq_len', 512)
        
        # 初始化 TensorRT 引擎 (如果启用)
        self.trt_llm_engine = None
        self.model = None
        
        if config.use_trt_llm:
            # 尝试多个可能的引擎路径
            possible_paths = [
                config.model_path.replace('.pt', '.engine'),
                './exported/decoder/decoder.engine',
                os.path.join(os.path.dirname(config.model_path), 'decoder.engine'),
            ]
            engine_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    engine_path = p
                    break
            
            if engine_path:
                print(f"尝试加载 TensorRT 引擎: {engine_path}")
                trt_engine = TensorRTLLMInference(engine_path, config)
                if trt_engine.engine is not None:
                    self.trt_llm_engine = trt_engine
                    print("TensorRT 引擎加载成功，将使用 TensorRT 加速推理")
                else:
                    print("TensorRT 引擎加载失败，将使用 PyTorch")
            else:
                print("TensorRT 引擎不存在，将使用 PyTorch 推理")
                print(f"搜索路径: {possible_paths}")

        # 如果 TRT 引擎未加载成功，加载 PyTorch 模型
        if self.trt_llm_engine is None:
            self._load_pytorch_model(checkpoint)

        # 加载语义 ID 映射
        self._load_semantic_id_mapping()

        print("Inference service initialized successfully")

    def _load_pytorch_model(self, checkpoint) -> None:
        """加载 PyTorch 模型."""
        print(f"Loading PyTorch model from {self.config.model_path}")

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

        print(f"PyTorch model loaded: {self.num_quantizers} quantizers, vocab size {self.vocab_size}")
    
    def _get_logits(self, input_ids: torch.Tensor):
        """获取 logits，优先使用 TensorRT 引擎.
        
        Args:
            input_ids: 输入张量
            
        Returns:
            (logits, loss) 元组，loss 始终为 None
        """
        if self.trt_llm_engine is not None:
            logits = self.trt_llm_engine.forward(input_ids)
            return logits, None
        return self.model(input_ids)

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
            logits, _ = self._get_logits(current_input)
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
                logits, _ = self._get_logits(seq)
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
            backend = 'tensorrt' if self.service.trt_llm_engine is not None else 'pytorch'
            return jsonify({
                'status': 'healthy',
                'backend': backend,
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
