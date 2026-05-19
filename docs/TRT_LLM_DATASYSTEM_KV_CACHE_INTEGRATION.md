# Pairec4TigerLLM × TensorRT-LLM × DataSystem KV Cache 集成方案

> 目标：在现有 `pairec4tigerllm` 推荐系统中，将推理后端升级为真正的 **TensorRT-LLM ModelRunner**，引入 **KVCacheManager** 管理 GPU HBM 中的 KV Cache，并通过 **DataSystem Worker** 实现 KV Cache 的跨请求持久化与复用，达成 `HBM ⇄ DataSystem` 传输闭环。
>
> 版本：v1.0 | 日期：2026-05-14

---

## 一、现状差距分析（Gap Analysis）

### 1.1 当前 `server.py` 的真实状态

现有代码中的 "TensorRT-LLM" 实际上是 **`tensorrt`（纯 TensorRT 10.x API）**，并非 `tensorrt_llm`（NVIDIA TensorRT-LLM 库）：

```python
# server.py 当前实现
import tensorrt as trt
runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
self.engine = runtime.deserialize_cuda_engine(engine_data)
self.context.execute_async_v3(stream.cuda_stream)
```

- ❌ 没有 `ModelRunner`、`GenerationSession`、`KVCacheManager` 等 TRT-LLM 高级抽象
- ❌ 引擎是 ONNX → 纯 TensorRT 构建的静态图，**不支持 KV Cache 传入/传出**
- ❌ 每次推理都重新计算全序列的 Attention，Prefill 与 Decode 无法分离
- ❌ 无法将 GPU HBM 中的 KV Cache 导出到外部存储

### 1.2 `build_engine.py` 的瓶颈

`build_engine.py` 尝试调用 `tensorrt_llm.models.GPTForCausalLM`，但自定义 `GenerativeDecoder` 有 **4 个独立输出头**（`num_quantizers=4`），与 TRT-LLM 预定义的标准 GPT 结构不兼容，因此实际回退到了 **ONNX + 纯 TensorRT** 方案。

### 1.3 要达到目标状态必须解决的三个问题

| 序号 | 问题 | 解决思路 |
|:----:|------|---------|
| 1 | 模型无 `past_key_values` 支持 | 改造 `GenerativeDecoder`，添加 `use_cache=True` |
| 2 | 引擎不支持 KV Cache I/O | 导出 **单步 Decode ONNX**（输入：token + past_kv；输出：logits + new_past_kv） |
| 3 | TRT-LLM API 未接入 | 使用 `tensorrt_llm.runtime.ModelRunner` 或自研轻量 `ModelRunner` 封装 TRT Decode 引擎 |

---

## 二、目标架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Request (HTTP)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  server.py (Flask HTTP Service)                                             │
│  ├─ /recommend 路由                                                         │
│  ├─ 输入：user_history (语义 ID 序列)                                       │
│  ├─ 调用 KVCacheManager.query(user_id, history_hash)                       │
│  └─ 输出：recommendations + trace (含 kv_transfer_ms)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  KVCacheManager (Python 层，管理 GPU HBM 中的活跃 KV Cache)                 │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  1. 检查 HBM 本地缓存 (LRU，按 user_id/history_hash 索引)             │  │
│  │     ├── 命中 → 直接返回 past_key_values (GPU Tensor)                  │  │
│  │     └── 未命中 → 查询 DataSystem Worker                               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  2. DataSystem 查询 (yr.datasystem SDK)                               │  │
│  │     ├── 命中 → Host → HBM 拷贝 (反序列化 + cudaMemcpyHtoD)            │  │
│  │     └── 未命中 → 标记为 "MISS"，触发 Prefill                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
              Cache Hit (DataSystem)              Cache Miss
                      │                                 │
                      ▼                                 ▼
    ┌─────────────────────────────┐       ┌─────────────────────────────┐
    │  Load KV from DataSystem    │       │  Prefill Phase              │
    │  Host Memory → GPU HBM      │       │  (PyTorch 或 TRT Full Seq)  │
    │  跳过 Attention 重复计算    │       │  计算初始 past_key_values   │
    └─────────────────────────────┘       └─────────────────────────────┘
                      │                                 │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TRT-LLM ModelRunner (单步 Decode 引擎)                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  输入：                                                               │  │
│  │    - input_ids: [batch=1, seq_len=1]  当前待生成 token                │  │
│  │    - past_key_values: [num_layers, 2, num_kv_heads, past_len, head_dim]│ │
│  │  输出：                                                               │  │
│  │    - logits: [1, 1, vocab_size]                                       │  │
│  │    - new_past_key_values: 更新后的 KV (长度 +1)                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│  内部：TensorRT Engine (decoder_step.engine) + execute_async_v3             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Sampling / Beam Search (Python)                                            │
│  循环调用 ModelRunner，直到生成 topk 个去重后的语义 ID                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  异步回写 (Background Thread)                                               │
│  ├─ 将最终 past_key_values 序列化 → Host Memory                             │
│  ├─ 压缩 (np.savez_compressed / lz4)                                        │
│  ├─ 写入 DataSystem: kv_set("pairec:kv:{user_hash}", value, ttl=600)        │
│  └─ 更新 HBM LRU 索引                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DataSystem Worker (yr.datasystem)                                          │
│  ├─ 存储介质：DRAM / SSD / RDMA (依部署而定)                                │
│  ├─ 通信：gRPC / brpc / 共享内存 (与推理 Pod 同节点时走 shm)                │
│  └─ Key 空间：pairec:kv:* / pairec:semantic_map:* / pairec:rec:*           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、核心组件设计

### 3.1 模型改造：`GenerativeDecoder` → 支持 `use_cache`

当前模型的 `forward()` 只返回 `logits`，没有 `past_key_values`。必须改造为标准的 Decoder-with-Cache 接口，才能被 TRT-LLM / ONNX / DataSystem 链路消费。

#### 3.1.1 模型结构改造点

```python
class GenerativeDecoder(nn.Module):
    def __init__(self, ..., use_cache: bool = True):
        ...
        self.use_cache = use_cache

    def forward(
        self,
        semantic_ids: torch.Tensor,           # [batch, seq_len, num_quantizers]
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor]]], Optional[torch.Tensor]]:
        """
        Returns:
            logits: [batch, seq_len, num_quantizers, vocab_size]
            past_key_values: ((k1, v1), (k2, v2), ...) for each layer
            loss: optional
        """
        use_cache = use_cache if use_cache is not None else self.use_cache
        batch_size, seq_len, _ = semantic_ids.shape

        # Embedding (与之前相同)
        x = ...  # [batch, seq_len, hidden]

        # 如果提供了 past_key_values，需要计算当前序列的 position_ids
        # past_len = past_key_values[0][0].size(-2) if past_key_values else 0
        # positions = torch.arange(past_len, past_len + seq_len, device=device)

        new_past_key_values = () if use_cache else None

        for i, block in enumerate(self.transformer_blocks):
            # 改造 TransformerBlock，让它接收和返回 (k, v)
            if use_cache and past_key_values is not None:
                past_k, past_v = past_key_values[i]
            else:
                past_k, past_v = None, None

            x, (present_k, present_v) = block(
                x, attention_mask=attention_mask,
                past_key_value=(past_k, past_v),
                use_cache=use_cache
            )
            if use_cache:
                new_past_key_values += ((present_k, present_v),)

        x = self.ln_f(x)
        logits = torch.stack([head(x) for head in self.output_heads], dim=2)
        return logits, new_past_key_values, loss
```

#### 3.1.2 `TransformerBlock` 改造

将 `CausalSelfAttention` 替换为支持 `past_key_value` 的版本：

```python
class CausalSelfAttention(nn.Module):
    def forward(self, x, attention_mask=None, past_key_value=None, use_cache=False):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape to [batch, num_heads, seq_len, head_dim]
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 拼接 past_kv
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present_key_value = (k, v) if use_cache else None

        # Attention 计算 (causal mask 需覆盖 past_len + seq_len)
        ...

        return output, present_key_value
```

> **关键收益**：Prefill 阶段一次性计算长序列的 KV，后续 Decode 阶段只需计算新 token 的 Q，与 K/V 做 Attention，复杂度从 `O(L²)` 降至 `O(L)`。

---

### 3.2 ONNX / TRT 引擎构建：分离 Prefill & Decode

#### 3.2.1 为什么必须分离？

| 阶段 | 输入 seq_len | 是否需要 past_kv | 输出 past_kv | 用途 |
|------|-------------|-----------------|-------------|------|
| **Prefill** | `L` (历史长度，如 50) | 否 | 是 | 首次请求，计算用户历史 |
| **Decode** | `1` | 是 | 是 | 逐步生成推荐 token |

- Prefill 是 **compute-bound**（矩阵大，算力密集）
- Decode 是 **memory-bound**（矩阵小，KV Cache 带宽密集）

如果只做单个全序列引擎（当前做法），每次生成都重新计算全序列 Attention，浪费巨大。

#### 3.2.2 Decode 引擎导出（核心）

Decode 引擎是 TRT-LLM ModelRunner 的直接执行对象：

```python
class DecodeStepWrapper(torch.nn.Module):
    """单步 Decode 导出包装器."""
    def __init__(self, model: GenerativeDecoder):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,        # [batch=1, seq_len=1, num_quantizers]
        past_key_values: List[torch.Tensor]  # 扁平化输入
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # 重塑 past_key_values: [num_layers, 2, batch, num_heads, past_len, head_dim]
        ...
        logits, new_past, _ = self.model(
            input_ids,
            past_key_values=restored_past_kv,
            use_cache=True
        )
        # 只返回最后一个位置的 logits
        next_logits = logits[:, -1, :, :]  # [1, num_quantizers, vocab_size]
        # 扁平化 new_past 以便 ONNX 导出
        flat_new_past = [t for layer_kv in new_past for t in layer_kv]
        return next_logits, flat_new_past
```

ONNX 导出参数：

```python
dummy_input_ids = torch.randint(0, 256, (1, 1, 4), dtype=torch.long)
dummy_past_kv = [
    torch.randn(1, 8, 50, 32, dtype=torch.float16)  # k
    for _ in range(num_layers * 2)  # k + v for each layer
]

torch.onnx.export(
    decode_wrapper,
    (dummy_input_ids, dummy_past_kv),
    "decoder_step.onnx",
    opset_version=14,
    input_names=["input_ids"] + [f"past_{i}" for i in range(num_layers * 2)],
    output_names=["logits"] + [f"present_{i}" for i in range(num_layers * 2)],
    dynamic_axes={
        "input_ids": {0: "batch"},
        "logits": {0: "batch"},
        **{f"past_{i}": {2: "past_len"} for i in range(num_layers * 2)},
        **{f"present_{i}": {2: "total_len"} for i in range(num_layers * 2)},
    }
)
```

TensorRT 构建时启用 **FP16** + **Plugin**，并用 `decoder_step.onnx` 构建 `decoder_step.engine`。

> **注意**：如果 TensorRT 对动态 `past_len` 支持不佳，可以固定 `max_past_len=512`，超出时做 KV Cache 截断（滑动窗口）。

---

### 3.3 KVCacheManager 设计

```python
# inference/kv_cache/manager.py

from typing import Dict, Tuple, Optional
import torch
import threading
import time
from collections import OrderedDict


class KVCacheManager:
    """管理 GPU HBM 中的 KV Cache，并协调 DataSystem 读写."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 512,
        hbm_capacity: int = 100,          # HBM 中最多保留多少条 KV Cache
        ds_client=None,                   # yr.datasystem.DsClient
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.ds = ds_client

        # HBM 缓存: OrderedDict for LRU
        # key: (user_id, history_hash)
        # value: { "past_kv": Tuple[Tuple[Tensor]], "seq_len": int, "timestamp": float }
        self.hbm_cache: OrderedDict[str, Dict] = OrderedDict()
        self.hbm_capacity = hbm_capacity
        self._lock = threading.RLock()

        # 序列化缓冲区（Host pinned memory，加速 HtoD/DtoH）
        self._pinned_buffer = {}

    def _make_key(self, user_id: str, history_hash: str) -> str:
        return f"{user_id}:{history_hash}"

    def _ds_key(self, local_key: str) -> str:
        return f"pairec4tigerllm:kv:{local_key}"

    def query(
        self,
        user_id: str,
        history_hash: str
    ) -> Tuple[Optional[Tuple], str, float]:
        """查询 KV Cache.

        Returns:
            (past_kv_tuple, source, lookup_ms)
            source: "hbm_hit" | "ds_hit" | "miss"
        """
        local_key = self._make_key(user_id, history_hash)
        t0 = time.perf_counter()

        with self._lock:
            # 1. 查 HBM
            if local_key in self.hbm_cache:
                self.hbm_cache.move_to_end(local_key)
                return self.hbm_cache[local_key]["past_kv"], "hbm_hit", (time.perf_counter() - t0) * 1000

        # 2. 查 DataSystem
        if self.ds is not None:
            try:
                raw = self.ds.kv().get(self._ds_key(local_key))
                if raw is not None:
                    past_kv = self._deserialize(raw)
                    # 加载到 HBM
                    self._put_hbm(local_key, past_kv)
                    return past_kv, "ds_hit", (time.perf_counter() - t0) * 1000
            except Exception as e:
                print(f"[KVCacheManager] DataSystem query failed: {e}")

        return None, "miss", (time.perf_counter() - t0) * 1000

    def store(
        self,
        user_id: str,
        history_hash: str,
        past_kv: Tuple[Tuple[torch.Tensor]],
        async_write: bool = True
    ):
        """存储 KV Cache 到 HBM 和 DataSystem."""
        local_key = self._make_key(user_id, history_hash)

        # 写入 HBM
        self._put_hbm(local_key, past_kv)

        # 异步写入 DataSystem
        if self.ds is not None:
            if async_write:
                threading.Thread(
                    target=self._write_ds,
                    args=(local_key, past_kv),
                    daemon=True
                ).start()
            else:
                self._write_ds(local_key, past_kv)

    def _put_hbm(self, local_key: str, past_kv: Tuple):
        with self._lock:
            if local_key in self.hbm_cache:
                self.hbm_cache.move_to_end(local_key)
                return
            # LRU 淘汰
            while len(self.hbm_cache) >= self.hbm_capacity:
                evicted_key, _ = self.hbm_cache.popitem(last=False)
                print(f"[KVCacheManager] HBM evicted: {evicted_key}")
            self.hbm_cache[local_key] = {
                "past_kv": past_kv,
                "timestamp": time.time(),
            }

    def _serialize(self, past_kv: Tuple[Tuple[torch.Tensor]]) -> bytes:
        """将 past_key_values 序列化为 bytes.

        格式: np.savez_compressed
        为了加速传输，可以先做 fp16 → uint8 量化（可选）
        """
        import numpy as np
        import io

        arrays = {}
        for layer_idx, (k, v) in enumerate(past_kv):
            # k, v: [batch, num_heads, seq_len, head_dim]
            arrays[f"k_{layer_idx}"] = k.cpu().numpy()
            arrays[f"v_{layer_idx}"] = v.cpu().numpy()
        arrays["meta"] = np.array([len(past_kv), past_kv[0][0].size(2), past_kv[0][0].size(3)])

        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return buf.getvalue()

    def _deserialize(self, raw: bytes) -> Tuple[Tuple[torch.Tensor]]:
        import numpy as np
        import io

        buf = io.BytesIO(raw if isinstance(raw, bytes) else raw.encode('utf-8'))
        data = np.load(buf)
        meta = data["meta"]
        num_layers, seq_len, head_dim = int(meta[0]), int(meta[1]), int(meta[2])

        past_kv = []
        for i in range(num_layers):
            k = torch.from_numpy(data[f"k_{i}"]).to(device=self.device, dtype=self.dtype)
            v = torch.from_numpy(data[f"v_{i}"]).to(device=self.device, dtype=self.dtype)
            past_kv.append((k, v))
        return tuple(past_kv)

    def _write_ds(self, local_key: str, past_kv: Tuple):
        try:
            payload = self._serialize(past_kv)
            self.ds.kv().set(self._ds_key(local_key), payload, ttl=600)
        except Exception as e:
            print(f"[KVCacheManager] DataSystem write failed: {e}")

    def estimate_size_mb(self, seq_len: int, batch_size: int = 1) -> float:
        """估算一条 KV Cache 的显存占用 (MB)."""
        element_size = 2 if self.dtype == torch.float16 else 4
        bytes_per_tensor = batch_size * self.num_kv_heads * seq_len * self.head_dim * element_size
        total = self.num_layers * 2 * bytes_per_tensor  # K + V
        return total / (1024 ** 2)
```

---

### 3.4 TRT-LLM ModelRunner（轻量封装层）

由于当前自定义模型无法直接使用 TRT-LLM 预定义的 `GPTForCausalLM`，我们采用 **"TRT-LLM 风格 API + 纯 TensorRT 引擎内核"** 的混合方案：

```python
# inference/trt_llm/model_runner.py

import tensorrt as trt
import torch
from typing import Tuple, List


class TRTLLMModelRunner:
    """TensorRT-LLM 风格的单步 Decode ModelRunner.

    封装 decoder_step.engine，提供 generate_step() 接口，
    与 KVCacheManager 配合实现增量推理。
    """

    def __init__(self, engine_path: str, num_layers: int, num_kv_heads: int, head_dim: int):
        self.logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.stream = torch.cuda.current_stream()

    @torch.no_grad()
    def generate_step(
        self,
        input_ids: torch.Tensor,                       # [batch, 1, num_quantizers]
        past_key_values: Tuple[Tuple[torch.Tensor]],   # ((k,v), ...)
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor]]]:
        """单步生成.

        Returns:
            logits: [batch, num_quantizers, vocab_size]
            new_past_key_values: 更新后的 KV
        """
        batch_size = input_ids.shape[0]
        past_len = past_key_values[0][0].size(2) if past_key_values else 0
        total_len = past_len + 1

        # 设置动态形状
        self.context.set_input_shape("input_ids", (batch_size, 1, input_ids.shape[2]))

        # 绑定 input_ids
        self.context.set_tensor_address("input_ids", input_ids.data_ptr())

        # 绑定 past_key_values (每层 k, v)
        for i, (k, v) in enumerate(past_key_values):
            # 确保形状正确并绑定
            self.context.set_input_shape(f"past_{i*2}", tuple(k.shape))
            self.context.set_input_shape(f"past_{i*2+1}", tuple(v.shape))
            self.context.set_tensor_address(f"past_{i*2}", k.data_ptr())
            self.context.set_tensor_address(f"past_{i*2+1}", v.data_ptr())

        # 预分配输出张量 (GPU HBM)
        logits = torch.empty(batch_size, 1, input_ids.shape[2], 256,
                             dtype=torch.float32, device=input_ids.device)
        new_past_kv = []

        self.context.set_tensor_address("logits", logits.data_ptr())

        for i in range(self.num_layers):
            # present_k
            pk_shape = (batch_size, self.num_kv_heads, total_len, self.head_dim)
            pk = torch.empty(pk_shape, dtype=k.dtype, device=input_ids.device)
            self.context.set_tensor_address(f"present_{i*2}", pk.data_ptr())
            # present_v
            pv_shape = (batch_size, self.num_kv_heads, total_len, self.head_dim)
            pv = torch.empty(pv_shape, dtype=v.dtype, device=input_ids.device)
            self.context.set_tensor_address(f"present_{i*2+1}", pv.data_ptr())
            new_past_kv.append((pk, pv))

        # 执行推理
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        # logits: [batch, 1, num_quantizers, vocab_size] -> [batch, num_quantizers, vocab_size]
        return logits.squeeze(1), tuple(new_past_kv)
```

> **设计说明**：`TRTLLMModelRunner` 的接口风格对齐 TRT-LLM 的 `ModelRunner.generate()`，但底层针对自定义多 quantizer 模型做了适配。未来如果迁移到标准 LLaMA/Qwen 结构，可以无缝替换为 `tensorrt_llm.runtime.ModelRunnerCpp`。

---

### 3.5 DataSystem Bridge 扩展

在 P0 已有的 `DataSystemCache` 基础上，新增 `KVCacheStorage`：

```python
# 复用 P0 的 DataSystemCache 结构，增加 KV 专用方法

class DataSystemCache:
    # ... P0 已有代码 ...

    def get_kv_cache(self, user_id: str, history_hash: str) -> Optional[bytes]:
        key = self._key(f"kv:{user_id}:{history_hash}")
        return self._safe_get(key)

    def put_kv_cache(self, user_id: str, history_hash: str, data: bytes, ttl_sec: int = 600) -> bool:
        key = self._key(f"kv:{user_id}:{history_hash}")
        return self._safe_set(key, data, ttl=ttl_sec)
```

---

## 四、推理数据流：Cache Miss vs Cache Hit

### 4.1 Cache Miss（首次请求 / 历史变更）

```
用户请求 (history=[[100,50,25,10], [101,51,26,11]])
    │
    ▼
KVCacheManager.query("user_123", "hash_abcd")
    └── 返回 (None, "miss", 0.5ms)
    │
    ▼
Prefill Phase (PyTorch / TRT Full-Seq Engine)
    输入: [1, 2, 4] 语义 ID 序列 (history)
    输出: logits + past_key_values (长度=2)
    耗时: ~15ms (TensorRT FP16)
    │
    ▼
Loop: Decode Step (调用 TRTLLMModelRunner.generate_step())
    Step 1: input=[new_token_1], past_kv=[len=2] → logits + past_kv[len=3]
    Step 2: input=[new_token_2], past_kv=[len=3] → logits + past_kv[len=4]
    ... (直到生成 topk 个去重推荐)
    每步耗时: ~3ms
    │
    ▼
生成完成 (假设生成 20 步，取 topk=10)
    │
    ▼
异步: KVCacheManager.store("user_123", "hash_abcd", final_past_kv)
    ├── HBM: 写入 LRU
    └── DataSystem: np.savez_compressed → kv.set()
    │
    ▼
返回 recommendations + trace
```

**Trace 输出示例**：
```json
{
  "total_ms": 85.0,
  "prefill_ms": 15.0,
  "decode_steps": 20,
  "decode_ms": 60.0,
  "kv_lookup_ms": 0.5,
  "kv_write_ms": 5.0,
  "kv_source": "miss",
  "backend": "tensorrt"
}
```

### 4.2 Cache Hit（DataSystem 命中）

```
用户请求 (相同 history)
    │
    ▼
KVCacheManager.query("user_123", "hash_abcd")
    └── HBM 未命中 → DataSystem 命中
    └── Host → HBM 反序列化
    └── 返回 (past_kv, "ds_hit", 8.0ms)
    │
    ▼
Skip Prefill !!
    │
    ▼
Loop: Decode Step (直接复用 loaded past_kv)
    Step 1: input=[new_token_1], past_kv=[len=50] → ...
    ...
    │
    ▼
异步更新 DataSystem (TTL 刷新)
    │
    ▼
返回 recommendations + trace
```

**Trace 输出示例**：
```json
{
  "total_ms": 45.0,
  "prefill_ms": 0.0,
  "decode_steps": 20,
  "decode_ms": 35.0,
  "kv_lookup_ms": 8.0,
  "kv_write_ms": 2.0,
  "kv_source": "ds_hit",
  "backend": "tensorrt"
}
```

**收益**：跳过 Prefill（节省 15ms），只需支付 8ms 的 DataSystem 加载延迟。对于长历史用户（seq_len=50），净收益 ~7ms；对于超长历史（如果未来扩展），收益更大。

---

## 五、HBM ⇄ DataSystem KV Cache 传输协议

### 5.1 数据格式

```
┌─────────────────────────────────────────────────────────────┐
│  Key: pairec4tigerllm:kv:{user_id}:{history_md5_16}         │
├─────────────────────────────────────────────────────────────┤
│  Value: np.savez_compressed({                               │
│           "meta": [num_layers, seq_len, head_dim],          │
│           "k_0": ndarray[float16],  # [1, 8, seq_len, 32]  │
│           "v_0": ndarray[float16],  # [1, 8, seq_len, 32]  │
│           "k_1": ndarray[float16],                         │
│           "v_1": ndarray[float16],                         │
│           ...                                              │
│         })                                                  │
│  Size: 6 layers × 2 × 1 × 8 × 50 × 32 × 2B ≈ 0.3 MB       │
│        28 layers × 2 × 1 × 8 × 50 × 64 × 2B ≈ 2.8 MB      │
├─────────────────────────────────────────────────────────────┤
│  TTL: 600s (可配置)                                         │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 传输路径优化

| 部署模式 | HBM → Host | Host → DataSystem | 延迟估计 |
|---------|-----------|-------------------|---------|
| 同节点共享内存 | `cudaMemcpyDtoH` (pinned memory) | `memcpy` 到 shm | **1~3ms** |
| 同机架 RDMA | `cudaMemcpyDtoH` | `ibv_write` | **3~5ms** |
| 跨网络 TCP | `cudaMemcpyDtoH` | gRPC/brpc | **8~15ms** |

**推荐**：推理 Pod 与 DataSystem Worker 部署在 **同一 K8s Node**，通过 **Host Path / shared memory** 通信，将 `kv_write_ms` 压到 3ms 以内。

### 5.3 量化压缩（可选 P2 优化）

如果 KV Cache 体积过大，可在序列化前做 **KV Cache Quantization**：
- FP16 → INT8：体积减半，精度损失 < 1%（参考 LLM 推理社区实践）
- 按 channel（head_dim）做 per-channel scale
- DataSystem 存储 INT8 + scale，加载后反量化回 FP16

---

## 六、`server.py` 改造要点

### 6.1 初始化流程

```python
class GenerativeInferenceService:
    def __init__(self, config: InferenceConfig):
        # 1. 加载 PyTorch 模型（用于 Prefill）
        self.pytorch_model = self._load_pytorch_model()

        # 2. 加载 TRT Decode 引擎
        self.decode_runner = TRTLLMModelRunner(
            engine_path="./exported/decoder/decoder_step.engine",
            num_layers=config.num_layers,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
        )

        # 3. 初始化 KVCacheManager
        self.kv_manager = KVCacheManager(
            num_layers=config.num_layers,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            hbm_capacity=100,
            ds_client=DataSystemCache()._client,  # 复用 P0 连接
        )

        # 4. 加载语义 ID 映射（P0 逻辑）
        self._load_semantic_id_mapping()
```

### 6.2 `recommend()` 核心逻辑

```python
def recommend(self, user_history, topk=10, temperature=1.0, beam_width=1):
    t0 = time.perf_counter()
    user_id = "anonymous"  # 实际从 request 获取
    history_hash = self._hash_history(user_history)

    # 1. KV Cache 查询
    past_kv, kv_source, kv_lookup_ms = self.kv_manager.query(user_id, history_hash)

    if past_kv is None:
        # 2a. Cache Miss: Prefill
        input_ids = self._prepare_input(user_history).to(self.device)
        with torch.no_grad():
            logits, past_kv, _ = self.pytorch_model(
                input_ids, use_cache=True
            )
        prefill_ms = ...
    else:
        prefill_ms = 0.0

    # 2b. Decode 循环（TRT-LLM ModelRunner）
    decode_ms = 0.0
    generated = []
    current_past_kv = past_kv

    for _ in range(topk * 2):
        # 准备当前 token（首次用历史最后一个，后续用上次生成的）
        if len(generated) == 0 and kv_source != "miss":
            # 如果是 Cache Hit，但还没生成任何新 token
            # 需要从历史构造 "下一个输入"
            next_input = ...
        else:
            next_input = last_generated_token

        t_step = time.perf_counter()
        logits, current_past_kv = self.decode_runner.generate_step(
            next_input, current_past_kv
        )
        decode_ms += (time.perf_counter() - t_step) * 1000

        # sampling → semantic ID → item ID（现有逻辑）
        ...

    # 3. 异步存储 KV
    t_write = time.perf_counter()
    self.kv_manager.store(user_id, history_hash, current_past_kv, async_write=True)
    kv_write_ms = (time.perf_counter() - t_write) * 1000

    total_ms = (time.perf_counter() - t0) * 1000
    return {
        'recommendations': recommendations,
        'trace': {
            'total_ms': total_ms,
            'prefill_ms': prefill_ms,
            'decode_ms': decode_ms,
            'kv_lookup_ms': kv_lookup_ms,
            'kv_write_ms': kv_write_ms,
            'kv_source': kv_source,
            'backend': 'tensorrt',
        }
    }
```

---

## 七、实施路线图

| 阶段 | 事项 | 工作量 | 依赖 |
|:----:|------|:------:|------|
| **P0.5** | 模型改造：`GenerativeDecoder` 添加 `use_cache` + `past_key_values` | 2d | 无 |
| **P0.5** | 改造 `TransformerBlock` / `CausalSelfAttention` 支持 KV Cache | 1d | 上一步 |
| **P1** | 导出 Decode Step ONNX（支持 past_kv I/O） | 1d | 模型改造完成 |
| **P1** | 构建 `decoder_step.engine`（TensorRT FP16） | 1d | ONNX 导出完成 |
| **P1** | 开发 `TRTLLMModelRunner`（单步推理封装） | 2d | 引擎就绪 |
| **P1** | 开发 `KVCacheManager`（HBM LRU + 序列化） | 2d | 无 |
| **P2** | DataSystem Bridge：扩展 `DataSystemCache` 支持 KV 读写 | 1d | P0 DataSystem 已就绪 |
| **P2** | 改造 `server.py`：Prefill-Decode 分离 + KVManager 接入 | 2d | P1 完成 |
| **P2** | 端到端联调：Cache Miss / Hit 双路径验证 | 2d | 全部完成 |
| **P3** | 性能优化：KV Cache INT8 量化、pinned memory、shm 零拷贝 | 3d | 联调通过 |

**总工期**：约 **2 周**（2 人并行可压缩到 1 周）。

---

## 八、风险与回退方案

| 风险 | 影响 | 回退方案 |
|------|------|---------|
| **ONNX 不支持 dynamic past_len** | Decode 引擎构建失败 | 固定 `max_past_len=512`，超出时截断；或回退到 PyTorch Decode |
| **TensorRT engine 对 multi-input 支持差** | 4个 past_kv 输入绑定失败 | 将 past_kv 在 Python 侧拼接为单个 tensor，引擎只接受 2 个输入（k_all, v_all） |
| **DataSystem 延迟过高** | `ds_hit` 比 `miss` 还慢 | 关闭 KV Cache 持久化，仅保留 HBM LRU（纯内存缓存） |
| **KV Cache 显存爆炸** | HBM 存不下 100 条 | 降低 `hbm_capacity`；或做 KV Cache 截断（sliding window，只保留最近 32 个位置） |
| **TRT-LLM 库版本不兼容** | `tensorrt_llm` 无法安装 | 使用纯 TensorRT 引擎 + Python 层手动管理 past_kv（即本文 `TRTLLMModelRunner` 的方案，不依赖 `tensorrt_llm` 库本身，只用 `tensorrt`） |

---

## 九、关键结论

1. **当前代码用的是纯 TensorRT，不是 TRT-LLM**。要达到目标状态，必须将模型改造为支持 `past_key_values`，并导出 **单步 Decode 引擎**。

2. **TRTLLMModelRunner** 是连接 `server.py` 与 TensorRT 引擎的桥梁，接口对齐标准 TRT-LLM 风格，但底层适配自定义多 quantizer 模型。

3. **KVCacheManager** 是 HBM 中的 LRU 缓存层，负责决定 `命中 HBM` / `回源 DataSystem` / `触发 Prefill`。

4. **DataSystem Worker** 是跨请求持久化层，Key 为 `pairec4tigerllm:kv:{user}:{hash}`，Value 为压缩后的 KV Cache 字节流。

5. **HBM ⇄ DataSystem** 的传输发生在：
   - **读**：DataSystem → Host Memory → `cudaMemcpyHtoD` → GPU HBM
   - **写**：GPU HBM → `cudaMemcpyDtoH` → Host Memory → DataSystem

6. **显存估算**：6 层模型、seq_len=50、batch=1 时，一条 KV Cache 约 **0.3 MB**；28 层 Qwen3 约 **2.8 MB**。HBM 缓存 100 条仅需 **~300MB**，完全可接受。

---

**下一步**：如需推进实施，建议先从 **模型改造（添加 `use_cache`）** 和 **Decode Step ONNX 导出** 开始，这是整个链路的基石。
