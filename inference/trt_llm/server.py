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
from training.decoder.qwen3_generative_rec import Qwen3GenerativeRec

# DataSystem client (optional)
try:
    from yr.datasystem import DsClient
    _HAS_DATASYSTEM = True
except ImportError:
    DsClient = None
    _HAS_DATASYSTEM = False


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
    backbone: str = 'gpt2'     # 'gpt2' or 'qwen3'
    qwen3_model_path: str = 'Qwen/Qwen3-0.6B'
    trt_engine_dir: str = ''   # TRT-LLM engine dir (Qwen3, > PyTorch)
    datasystem_host: str = ''  # DataSystem worker host (empty = disabled)
    datasystem_port: int = 31501  # DataSystem worker port
    trt_max_kv_tokens: int = 2048  # TRT-LLM paged KV cache pressure knob
    trt_scheduler_policy: str = 'max_utilization'
    trt_max_input_len: int = 64   # TRT engine max_input_len (match trtllm-build --max_input_len)


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
        
        # 非侵入式时延打点：模型前向/语义映射累加器
        self._trace_forward_ms = 0.0
        self._trace_map_ms = 0.0

        # 初始化
        self.trt_llm_engine = None
        self.model = None
        self.kv_manager = None
        self.kv_cache_hits = 0
        self.kv_cache_misses = 0

        # TRT-LLM 结果缓存 (key: "{user_id}:{history_hash}" → dict)
        from collections import OrderedDict
        self._result_cache: OrderedDict[str, Dict] = OrderedDict()
        self._result_cache_max = 50  # LRU 容量
        self._result_ds_prefix = "pairec4tigerllm:result"  # DataSystem key 前缀 (与 KV 分离)

        # 根据 backbone 选择加载方式
        backbone = model_config.get('backbone', config.backbone)

        self._trt_backend = None  # TRT-LLM 引擎后端

        if backbone == 'qwen3':
            # 优先 TRT-LLM 引擎
            if os.path.isdir(config.trt_engine_dir):
                print(f"[TRT] Loading Qwen3 engine from {config.trt_engine_dir}")
                self._load_qwen3_trt(checkpoint)
            else:
                print("Loading Qwen3 backbone model (PyTorch)...")
                self._load_qwen3_model(checkpoint)
        else:
            # GPT2 backbone: 尝试 TRT 引擎
            if config.use_trt_llm:
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
                        print("TensorRT 引擎加载成功")
                    else:
                        print("TensorRT 引擎加载失败，回退 PyTorch")
                else:
                    print("TensorRT 引擎不存在，使用 PyTorch 推理")

            if self.trt_llm_engine is None:
                self._load_pytorch_model(checkpoint)

        # 加载语义 ID 映射
        self._load_semantic_id_mapping()

        # ── 注入物品前缀索引到模型 (约束解码用) ─
        if hasattr(self.model, '_item_prefix') and hasattr(self, '_item_prefix'):
            self.model._item_prefix = self._item_prefix
            print(f"[constrain] Item prefix trie injected into model")

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

    def _load_qwen3_model(self, checkpoint) -> None:
        """加载 Qwen3 模型."""
        print(f"Loading Qwen3 from {self.config.qwen3_model_path}")
        model_config = checkpoint['config']

        self.model = Qwen3GenerativeRec(
            model_name_or_path=model_config.get('model_name_or_path') or self.config.qwen3_model_path,
            vocab_size=model_config['vocab_size'],
            num_quantizers=model_config['num_quantizers'],
            max_seq_len=model_config.get('max_seq_len', 512),
            use_lora=False,  # checkpoint 已含 LoRA 权重
        )
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model = self.model.to(self.device).bfloat16().eval()
        self.model.merge_lora()

        # 恢复 token 映射 (checkpoint 中保存)
        if '_id_to_sem' in checkpoint:
            self.model._id_to_sem = {
                int(k): tuple(v) for k, v in checkpoint['_id_to_sem'].items()
            }
        if '_sem_to_id' in checkpoint:
            self.model._sem_to_id = {
                tuple(k): int(v) for k, v in checkpoint['_sem_to_id'].items()
            }

        self.hidden_size = model_config.get('hidden_size', 1024)
        self.num_layers = model_config.get('num_layers', 28)
        self.num_kv_heads = model_config.get('num_kv_heads', 8)
        self.head_dim = model_config.get('head_dim', 64)

        print(f"Qwen3 model loaded: layers={self.num_layers}, "
              f"kv_heads={self.num_kv_heads}, hidden={self.hidden_size}")

        # 初始化 DataSystem client + KVCacheManager (PyTorch 路径共用)
        self._init_kv_cache_manager(self.config)

    def _load_qwen3_trt(self, checkpoint) -> None:
        """加载 Qwen3 TRT-LLM 引擎 (无需 PyTorch 模型)."""
        from .trt_qwen3_backend import TRTQwen3Backend

        model_config = checkpoint['config']

        # 优先用导出的 tokenizer (含 1024 个 <s0_X> 特殊 token)
        from transformers import AutoTokenizer
        exported_tok = os.path.join(self.config.trt_engine_dir, '..', '..', 'exported', 'qwen3_rec')
        if os.path.isdir(exported_tok):
            tokenizer = AutoTokenizer.from_pretrained(exported_tok)
            print(f"[TRT] Using exported tokenizer from {exported_tok}")
        else:
            qwen3_path = model_config.get('model_name_or_path') or self.config.qwen3_model_path
            tokenizer = AutoTokenizer.from_pretrained(qwen3_path, trust_remote_code=True)
            special_tokens = []
            for i in range(model_config['num_quantizers']):
                for j in range(model_config['vocab_size']):
                    special_tokens.append(f"<s{i}_{j}>")
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"[TRT] Registered {len(special_tokens)} special tokens")

        self._trt_backend = TRTQwen3Backend(
            engine_dir=self.config.trt_engine_dir,
            tokenizer=tokenizer,
            num_quantizers=model_config['num_quantizers'],
            vocab_size=model_config['vocab_size'],
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            max_tokens_in_paged_kv_cache=self.config.trt_max_kv_tokens,
            scheduler_policy=self.config.trt_scheduler_policy,
            max_input_len=self.config.trt_max_input_len,
        )

        if '_id_to_sem' in checkpoint:
            self._trt_backend._id_to_sem = {
                int(k): tuple(v) for k, v in checkpoint['_id_to_sem'].items()
            }

        self.hidden_size = model_config.get('hidden_size', 1024)
        self.num_layers = model_config.get('num_layers', 28)
        self.num_kv_heads = model_config.get('num_kv_heads', 8)
        self.head_dim = model_config.get('head_dim', 64)

        print(f"[TRT] Qwen3 engine loaded: layers={self.num_layers}, "
              f"kv_heads={self.num_kv_heads}, hidden={self.hidden_size}")

        # 初始化 DataSystem client + KVCacheManager (TRT 路径)
        self._init_kv_cache_manager(self.config)

    def _init_datasystem_client(self, config: InferenceConfig):
        """初始化 DataSystem 客户端 (如果可用).
        
        优先从环境变量 DATASYSTEM_HOST / DATASYSTEM_PORT 读取配置，
        其次使用 config 中的值。
        
        Returns:
            DsClient 实例或 None
        """
        import os
        host = os.environ.get("DATASYSTEM_HOST", config.datasystem_host)
        port = int(os.environ.get("DATASYSTEM_PORT", str(config.datasystem_port)))
        
        if not host:
            print("[DataSystem] Disabled (no host configured)")
            return None
        
        if not _HAS_DATASYSTEM:
            print("[DataSystem] yr.datasystem not installed, skip")
            return None
        
        try:
            print(f"[DataSystem] Connecting to {host}:{port} ...")
            client = DsClient(host=host, port=port)
            client.init()
            print(f"[DataSystem] Connected OK (host={host}, port={port})")
            return client
        except Exception as e:
            print(f"[DataSystem] Connection failed: {e}")
            return None

    def _init_kv_cache_manager(self, config: InferenceConfig) -> None:
        """初始化 KVCacheManager (含 DataSystem client)."""
        ds_client = self._init_datasystem_client(config)
        try:
            from inference.kv_cache.manager import KVCacheManager
            self.kv_manager = KVCacheManager(
                num_layers=self.num_layers,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                hbm_capacity=50,
                ds_client=ds_client,
            )
            ds_status = "DataSystem" if ds_client else "HBM-only"
            print(f"[KVCacheManager] Initialized ({ds_status}) "
                  f"(layers={self.num_layers}, kv_heads={self.num_kv_heads}, "
                  f"head_dim={self.head_dim})")
        except ImportError as e:
            print(f"[KVCacheManager] Not available: {e}")

    def _get_logits(self, input_ids: torch.Tensor):
        """获取 logits，优先使用 TensorRT 引擎.
        
        Args:
            input_ids: 输入张量
            
        Returns:
            (logits, loss) 元组，loss 始终为 None
        """
        import time
        t0 = time.perf_counter()
        if self.trt_llm_engine is not None:
            logits = self.trt_llm_engine.forward(input_ids)
            result = (logits, None)
        else:
            result = self.model(input_ids)
        self._trace_forward_ms += (time.perf_counter() - t0) * 1000
        return result

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

            # ── 构建物品前缀索引 (用于约束解码) ─
            s0_set = set()
            s01_map = {}   # s0 → {s1, ...}
            s012_map = {}  # (s0,s1) → {s2, ...}
            s0123_map = {} # (s0,s1,s2) → {s3, ...}
            for item_id, sem_ids in self.semantic_to_item.items():
                s0, s1, s2, s3 = sem_ids
                s0_set.add(s0)
                s01_map.setdefault(s0, set()).add(s1)
                s012_map.setdefault((s0, s1), set()).add(s2)
                s0123_map.setdefault((s0, s1, s2), set()).add(s3)
            self._item_prefix = {
                's0_set': s0_set,
                's01_map': s01_map,
                's012_map': s012_map,
                's0123_map': s0123_map,
            }
            print(f"[prefix trie] s0={len(s0_set)}, s01={len(s01_map)}, "
                  f"s012={len(s012_map)}, s0123={len(s0123_map)}")
        else:
            print(f"Warning: Semantic ID mapping not found at {mapping_path}")
            self.semantic_to_item = {}
            self.semantic_to_item_tuple = {}
            self._item_prefix = None

    def recommend(
        self,
        user_history: List[List[int]],
        topk: int = 10,
        temperature: Optional[float] = None,
        beam_width: Optional[int] = None,
        user_id: str = "default",
    ) -> Dict:
        """生成推荐 (支持 Qwen3 KV Cache 加速)."""
        t0 = time.perf_counter()
        temperature = temperature or self.config.temperature

        # 1. 输入准备
        input_ids = self._prepare_input(user_history)
        input_ids = input_ids.to(self.device)
        history_hash = self._hash_history(user_history)

        # 2. KV Cache 查询
        kv_source = "miss"
        kv_lookup_ms = 0.0
        past_kv = None

        if self.kv_manager is not None:
            past_kv, kv_source, kv_lookup_ms = self.kv_manager.query(
                user_id, history_hash
            )
            if past_kv is not None:
                print(f"[KV Cache] hit: user={user_id}, hash={history_hash[:16]}..., "
                      f"lookup_ms={kv_lookup_ms:.1f}")

        # 3. 生成
        t_infer_start = time.perf_counter()
        recommendations = []
        final_past_kv = None

        with torch.no_grad():
            if self._trt_backend is not None:
                # ── TRT-LLM 结果缓存 (HBM + DataSystem onboard) ─
                cache_key = f"{user_id}:{history_hash}"

                # 1. 查 HBM 内存缓存
                if cache_key in self._result_cache:
                    self._result_cache.move_to_end(cache_key)
                    result = dict(self._result_cache[cache_key])
                    result['trace'] = dict(result['trace'])
                    result['trace']['kv_source'] = 'hbm_hit'
                    result['trace']['kv_lookup_ms'] = 0.0
                    total_ms = (time.perf_counter() - t0) * 1000
                    result['trace']['total_ms'] = total_ms
                    result['inference_time_ms'] = total_ms
                    self.kv_cache_hits += 1
                    print(f"[ResultCache] hbm_hit: user={user_id}, total_ms={total_ms:.0f}")
                    return result

                # 2. DataSystem onboard (内存 miss 时回读)
                onboard_result = None
                if self.kv_manager is not None and self.kv_manager.ds is not None:
                    try:
                        t_ds = time.perf_counter()
                        ds_key = f"{self._result_ds_prefix}:{cache_key}"
                        raw = self.kv_manager.ds.kv().get([ds_key], convert_to_str=False)
                        if raw and raw[0] is not None:
                            onboard_result = json.loads(
                                raw[0] if isinstance(raw[0], bytes) else raw[0]
                            )
                            ds_lookup_ms = (time.perf_counter() - t_ds) * 1000
                            print(f"[ResultCache] ds_hit: user={user_id}, "
                                  f"ds_lookup_ms={ds_lookup_ms:.0f}")
                            # 回填 HBM
                            self._result_cache[cache_key] = onboard_result
                            self._result_cache.move_to_end(cache_key)
                            kv_source = 'ds_hit'
                            kv_lookup_ms = ds_lookup_ms
                            self.kv_cache_hits += 1
                            total_ms = (time.perf_counter() - t0) * 1000
                            return {
                                **onboard_result,
                                'trace': {
                                    **onboard_result.get('trace', {}),
                                    'kv_source': 'ds_hit',
                                    'kv_lookup_ms': ds_lookup_ms,
                                    'total_ms': total_ms,
                                },
                                'inference_time_ms': total_ms,
                            }
                    except Exception as e:
                        # "Key not found" 是首次请求的正常 miss, 不打印
                        msg = str(e)
                        if "Key not found" not in msg and "not found" not in msg.lower():
                            print(f"[ResultCache] DataSystem onboard error: {e}")

                # 3. 全部 miss → TRT-LLM 引擎推理
                tokens = self._trt_backend.generate(
                    input_ids, max_new_tokens=topk * 2
                )  # [batch, max_items, 4]
                if tokens.shape[0] == 0:
                    print(f"[TRT generate] topk={topk}, output shape={tokens.shape} → empty batch, skip")
                    tokens = torch.zeros(0, self.num_quantizers, dtype=torch.long, device=self.device)
                else:
                    tokens = tokens[0]  # [max_items, 4]
                    print(f"[TRT generate] topk={topk}, output shape={tokens.shape}, "
                          f"nonzero={(tokens.sum(dim=1) != 0).sum().item()}/{tokens.shape[0]}")
            elif hasattr(self.model, '_id_to_sem'):
                # Qwen3 Prompt Mode: 原生 generate (内含 tokenizer + prompt 构造)
                result = self.model.generate(
                    input_ids,
                    max_new_tokens=topk * 2,
                    temperature=temperature,
                    use_cache=True,
                    past_key_values=past_kv,
                    return_past_kv=True,
                )
                if isinstance(result, tuple):
                    tokens, final_past_kv = result
                else:
                    tokens = result
                tokens = tokens[0]  # [max_items, 4]
                print(f"[DEBUG generate] topk={topk}, max_new={topk*2}, "
                      f"output shape={tokens.shape}, "
                      f"nonzero={(tokens.sum(dim=1) != 0).sum().item()}/{tokens.shape[0]}, "
                      f"kv_cached={final_past_kv is not None}")
            elif hasattr(self.model, 'generate'):
                # Qwen3 inputs_embeds Mode (旧)
                if past_kv is not None:
                    generated = self._decode_with_cache(
                        input_ids, topk * 2, temperature, past_kv
                    )
                    final_past_kv = generated[1] if isinstance(generated, tuple) else None
                    tokens = generated[0] if isinstance(generated, tuple) else generated
                else:
                    tokens = self.model.generate(
                        input_ids,
                        max_new_tokens=topk * 2,
                        temperature=temperature,
                        use_cache=True,
                    )[0]  # [n_tokens, 4]
            else:
                # GPT2: 原有逻辑
                if beam_width and beam_width > 1:
                    recommendations = self._beam_search_generate(
                        input_ids, topk, beam_width
                    )
                else:
                    recommendations = self._sampling_generate(
                        input_ids, topk, temperature
                    )
                # 已有 recommendations，跳过后续转换
                total_ms = (time.perf_counter() - t0) * 1000
                return {
                    'recommendations': recommendations,
                    'inference_time_ms': total_ms,
                    'trace': {'total_ms': total_ms, 'backend': 'pytorch'},
                }

        infer_ms = (time.perf_counter() - t_infer_start) * 1000

        # 4. 语义 ID → 物品 ID
        if not recommendations:
            recommendations = self._tokens_to_items(tokens, topk)

        # 5. 异步存储 KV Cache
        kv_write_ms = 0.0
        if self.kv_manager is not None and final_past_kv is not None:
            t_write = time.perf_counter()
            self.kv_manager.store(
                user_id, history_hash, final_past_kv, async_write=True
            )
            kv_write_ms = (time.perf_counter() - t_write) * 1000

        # 6. Trace
        total_ms = (time.perf_counter() - t0) * 1000
        if self._trt_backend is not None:
            backend = 'trt-qwen3'
        elif hasattr(self.model, 'hidden_size'):
            backend = 'qwen3'
        elif self.trt_llm_engine is not None:
            backend = 'tensorrt'
        else:
            backend = 'pytorch'

        if kv_source == 'hbm_hit':
            self.kv_cache_hits += 1
        elif kv_source == 'miss':
            self.kv_cache_misses += 1

        # 7. TRT-LLM 结果缓存存储
        if self._trt_backend is not None:
            cache_key = f"{user_id}:{history_hash}"
            while len(self._result_cache) >= self._result_cache_max:
                evicted_key, _ = self._result_cache.popitem(last=False)
                print(f"[ResultCache] evicted: {evicted_key}")
            self._result_cache[cache_key] = {
                'recommendations': recommendations,
                'inference_time_ms': total_ms,
                'trace': {
                    'total_ms': total_ms,
                    'infer_ms': infer_ms,
                    'kv_lookup_ms': kv_lookup_ms,
                    'kv_write_ms': kv_write_ms,
                    'kv_source': kv_source,
                    'backend': backend,
                },
            }
            # 异步写入 DataSystem (持久化)
            if self.kv_manager is not None and self.kv_manager.ds is not None:
                try:
                    payload = json.dumps(self._result_cache[cache_key]).encode()
                    ds_key = f"{self._result_ds_prefix}:{cache_key}"
                    import threading
                    threading.Thread(
                        target=lambda: self.kv_manager.ds.kv().set(
                            ds_key, payload, ttl_second=600
                        ),
                        daemon=True,
                    ).start()
                except Exception:
                    pass

        return {
            'recommendations': recommendations,
            'inference_time_ms': total_ms,
            'trace': {
                'total_ms': total_ms,
                'infer_ms': infer_ms,
                'kv_lookup_ms': kv_lookup_ms,
                'kv_write_ms': kv_write_ms,
                'kv_source': kv_source,
                'backend': backend,
            },
        }

    def _hash_history(self, user_history: List[List[int]]) -> str:
        """用户历史 → 16字符 MD5 哈希."""
        import hashlib
        raw = str(user_history).encode()
        return hashlib.md5(raw).hexdigest()[:16]

    def _tokens_to_items(
        self, tokens: torch.Tensor, topk: int
    ) -> List[Dict]:
        """语义 ID 序列 → 物品 ID 列表 (去重)."""
        recs = []
        hit = miss = 0
        for i in range(tokens.shape[0]):
            sem_ids = tokens[i].cpu().tolist()
            sem_tuple = tuple(sem_ids)
            item_id = self.semantic_to_item_tuple.get(sem_tuple)
            if item_id:
                hit += 1
                if item_id not in [r['item_id'] for r in recs]:
                    recs.append({
                        'item_id': item_id,
                        'semantic_id': sem_ids,
                        'score': 1.0,
                    })
            else:
                miss += 1
                if i < 3:  # 打印前3个miss的样例
                    print(f"[DEBUG map] miss: {sem_ids}")
            if len(recs) >= topk:
                break
        print(f"[DEBUG map] total={hit+miss}, hit={hit}, miss={miss}, "
              f"unique={len(recs)}, map_size={len(self.semantic_to_item_tuple)}")
        return recs

    def _decode_with_cache(
        self, input_ids, max_tokens, temperature, past_kv
    ):
        """使用已有 past_kv 做 Decode (跳过 Prefill).

        Returns:
            (tokens [n_tokens, 4], final_past_kv)
        """
        # 用历史最后一个 token + past_kv 做一次 forward 获得第一个 logits
        last_token = input_ids[:, -1:, :]  # [1, 1, 4]
        past_len = past_kv[0][0].size(-2)
        pos_ids = torch.tensor([[past_len]], device=self.device)

        logits, _, past_kv = self.model.forward(
            last_token,
            use_cache=True,
            past_key_values=past_kv,
            position_ids=pos_ids,
        )
        next_tokens = self.model._sample_token(
            logits[:, -1, :, :], temperature, None
        )
        generated = [next_tokens]
        current_pos = past_len + 1

        for _ in range(1, max_tokens):
            current_input = next_tokens.unsqueeze(1)
            pos_ids = torch.tensor([[current_pos]], device=self.device)
            logits, _, past_kv = self.model.forward(
                current_input,
                use_cache=True,
                past_key_values=past_kv,
                position_ids=pos_ids,
            )
            next_tokens = self.model._sample_token(
                logits[:, -1, :, :], temperature, None
            )
            generated.append(next_tokens)
            current_pos += 1

        tokens = torch.stack(generated, dim=1)[0]  # [n_tokens, 4]
        return tokens, past_kv

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
            t_map = time.perf_counter()
            item_id = self.semantic_to_item_tuple.get(sem_tuple)
            self._trace_map_ms += (time.perf_counter() - t_map) * 1000

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
                t_map = time.perf_counter()
                item_id = self.semantic_to_item_tuple.get(sem_tuple)
                self._trace_map_ms += (time.perf_counter() - t_map) * 1000

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
            if self.service._trt_backend is not None:
                backend = 'trt-qwen3'
            elif self.service.trt_llm_engine is not None:
                backend = 'tensorrt'
            else:
                backend = 'pytorch'
            ds_status = 'connected' if (
                self.service.kv_manager is not None
                and self.service.kv_manager.ds is not None
            ) else 'disabled'
            return jsonify({
                'status': 'healthy',
                'backend': backend,
                'datasystem': ds_status,
                'kv_cache_hits': self.service.kv_cache_hits,
                'kv_cache_misses': self.service.kv_cache_misses,
                'version': '1.0.0',
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
                print(f"[DEBUG req] user={user_id} history_len={len(history)} topk={topk}")

                result = self.service.recommend(
                    user_history=history,
                    topk=topk,
                    temperature=temperature,
                    beam_width=beam_width,
                    user_id=user_id,
                )

                return jsonify({
                    'code': 200,
                    'user_id': user_id,
                    'recommendations': result['recommendations'],
                    'inference_time_ms': result['inference_time_ms'],
                    'trace': result.get('trace', {}),
                })

                trace = result.get('trace', {})
                print(f"[TRACE] request_id={request.headers.get('X-Request-ID','')} "
                      f"total_ms={trace.get('total_ms',0):.1f} backend={trace.get('backend','unknown')} "
                      f"prepare_ms={trace.get('prepare_input_ms',0):.1f} forward_ms={trace.get('model_forward_ms',0):.1f} "
                      f"generate_ms={trace.get('generate_ms',0):.1f} map_ms={trace.get('map_item_ms',0):.1f} "
                      f"items={len(result['recommendations'])}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({
                    'code': 500,
                    'error': str(e),
                    'error_type': type(e).__name__,
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
    parser.add_argument('--qwen3_model_path', type=str,
                        default='Qwen/Qwen3-0.6B',
                        help='Path or HF name for Qwen3-0.6B backbone')
    parser.add_argument('--trt_engine_dir', type=str, default='',
                        help='TRT-LLM engine dir (Qwen3, takes priority over PyTorch)')
    parser.add_argument('--datasystem_host', type=str, default='',
                        help='DataSystem worker host (env: DATASYSTEM_HOST)')
    parser.add_argument('--datasystem_port', type=int, default=31501,
                        help='DataSystem worker port (env: DATASYSTEM_PORT, default: 31501)')
    parser.add_argument('--trt_max_kv_tokens', type=int,
                        default=int(os.environ.get('TRT_MAX_KV_TOKENS', '2048')),
                        help='TRT-LLM max tokens in paged KV cache '
                             '(env: TRT_MAX_KV_TOKENS, default: 2048)')
    parser.add_argument('--trt_scheduler_policy', type=str,
                        default=os.environ.get('TRT_SCHEDULER_POLICY', 'max_utilization'),
                        help='TRT-LLM scheduler policy '
                             '(env: TRT_SCHEDULER_POLICY, default: max_utilization)')
    parser.add_argument('--trt_max_input_len', type=int,
                        default=int(os.environ.get('TRT_MAX_INPUT_LEN', '64')),
                        help='TRT engine max input length '
                             '(env: TRT_MAX_INPUT_LEN, default: 64)')

    args = parser.parse_args()

    # 创建配置
    config = InferenceConfig(
        model_path=args.model_path,
        device=args.device,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        use_trt_llm=args.use_trt_llm,
        qwen3_model_path=args.qwen3_model_path,
        trt_engine_dir=args.trt_engine_dir,
        datasystem_host=args.datasystem_host,
        datasystem_port=args.datasystem_port,
        trt_max_kv_tokens=args.trt_max_kv_tokens,
        trt_scheduler_policy=args.trt_scheduler_policy,
        trt_max_input_len=args.trt_max_input_len,
    )

    # 创建推理服务
    service = GenerativeInferenceService(config)

    # 启动 HTTP 服务
    server = HTTPServer(service, port=args.port)
    server.start()


if __name__ == '__main__':
    main()
