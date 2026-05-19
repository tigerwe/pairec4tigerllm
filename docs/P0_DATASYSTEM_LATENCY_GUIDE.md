# P0 落地指南：推理服务接入 DataSystem + 端到端时延分解

> 目标：在现有 `pairec4tigerllm` 仿真环境中，让 Python 推理服务通过 `yr.datasystem` SDK 接入 DataSystem，并完善端到端推荐时延分解。
>
> 前置条件：已有一个 Docker 容器成功运行 `tensorrt_llm` + `yr.datasystem`（验证过 `import tensorrt_llm` 和 `import yr.datasystem` 均正常）。

---

## 一、环境确认（5 分钟）

### 1.1 确认 DataSystem Worker 可达

在容器内执行：

```bash
export DATASYSTEM_HOST=${DATASYSTEM_HOST:-127.0.0.1}
export DATASYSTEM_PORT=${DATASYSTEM_PORT:-31501}

python -c "
import yr.datasystem
client = yr.datasystem.DsClient('$DATASYSTEM_HOST', $DATASYSTEM_PORT)
print('DsClient init OK')
print('KV methods:', [m for m in dir(client.kv()) if not m.startswith('_')])
"
```

**预期输出**：
```
DsClient init OK
KV methods: ['get', 'set', ...]  # 具体方法名以实际输出为准
```

**若失败**：检查 DataSystem Worker 和 ETCD 是否已启动，确认 `DATASYSTEM_HOST` / `DATASYSTEM_PORT` 环境变量指向正确。

---

## 二、Python 推理服务改造（`inference/trt_llm/server.py`）

### 2.1 新增 DataSystem 缓存层

在 `server.py` 顶部（`import` 区域）插入以下代码：

```python
import os
import json
import hashlib
import time
from typing import Optional


class DataSystemCache:
    """兼容 yr.datasystem DsClient(kv()/hetero()/object()) 结构的缓存层."""

    KEY_PREFIX = "pairec4tigerllm"

    def __init__(self):
        self._client = None
        self._kv = None
        self._get_fn = None
        self._set_fn = None

        try:
            import yr.datasystem as ds
        except ImportError:
            print("[DataSystem] yr.datasystem not available")
            return

        # 1. 初始化 DsClient
        try:
            host = os.getenv("DATASYSTEM_HOST", "127.0.0.1")
            port = int(os.getenv("DATASYSTEM_PORT", "31501"))
            self._client = ds.DsClient(host, port)
            print(f"[DataSystem] DsClient({host}, {port}) initialized")
        except Exception as e:
            print(f"[DataSystem] DsClient init failed: {e}")
            return

        # 2. 获取 kv 子客户端
        if hasattr(self._client, 'kv'):
            try:
                self._kv = self._client.kv()
                print("[DataSystem] KV client acquired")
            except Exception as e:
                print(f"[DataSystem] client.kv() failed: {e}")
                return
        else:
            print("[DataSystem] client has no .kv() method")
            return

        # 3. 在 kv 对象上自动探测 get/set
        kv_methods = [m for m in dir(self._kv) if not m.startswith('_')]
        print(f"[DataSystem] KV available methods: {kv_methods}")

        for name in ['get', 'Get', 'mget', 'MGet', 'kv_get', 'KVGet']:
            if hasattr(self._kv, name):
                self._get_fn = getattr(self._kv, name)
                print(f"[DataSystem] Detected READ api: {name}")
                break

        for name in ['set', 'Set', 'mset', 'MSet', 'kv_set', 'KVSet']:
            if hasattr(self._kv, name):
                self._set_fn = getattr(self._kv, name)
                print(f"[DataSystem] Detected WRITE api: {name}")
                break

        if self._get_fn is None:
            print("[DataSystem] WARNING: No get-like method found on kv client")
        if self._set_fn is None:
            print("[DataSystem] WARNING: No set-like method found on kv client")

    def _key(self, suffix: str) -> str:
        return f"{self.KEY_PREFIX}:{suffix}"

    def _safe_get(self, key: str) -> Optional[bytes]:
        if self._kv is None or self._get_fn is None:
            return None
        try:
            result = self._get_fn(key)
            if result is None:
                return None
            if isinstance(result, (bytes, str)):
                return result if isinstance(result, bytes) else result.encode('utf-8')
            if hasattr(result, 'data'):
                data = result.data
                return data if isinstance(data, bytes) else bytes(data)
            return bytes(result)
        except Exception:
            return None

    def _safe_set(self, key: str, value: bytes, ttl: int = 0) -> bool:
        if self._kv is None or self._set_fn is None:
            return False
        try:
            try:
                self._set_fn(key, value, ttl=ttl)
            except TypeError:
                self._set_fn(key, value)
            return True
        except Exception as e:
            print(f"[DataSystem] set error: {e}")
            return False

    def get_semantic_map(self) -> Optional[dict]:
        val = self._safe_get(self._key("semantic_map"))
        if val is not None:
            try:
                return json.loads(val.decode('utf-8'))
            except Exception:
                pass
        return None

    def put_semantic_map(self, mapping: dict) -> bool:
        data = json.dumps(mapping).encode('utf-8')
        return self._safe_set(self._key("semantic_map"), data)

    def get_result_cache(self, user_history: list) -> Optional[list]:
        key = self._key(f"rec:{self._hash_history(user_history)}")
        val = self._safe_get(key)
        if val is not None:
            try:
                return json.loads(val.decode('utf-8'))
            except Exception:
                pass
        return None

    def put_result_cache(self, user_history: list, recommendations: list, ttl_sec: int = 10) -> bool:
        key = self._key(f"rec:{self._hash_history(user_history)}")
        data = json.dumps(recommendations).encode('utf-8')
        return self._safe_set(key, data, ttl=ttl_sec)

    @staticmethod
    def _hash_history(user_history: list) -> str:
        s = json.dumps(user_history, sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()[:16]
```

### 2.2 改造 `GenerativeInferenceService`

#### ① `__init__` 中初始化缓存

找到 `class GenerativeInferenceService` 的 `__init__` 方法，在**加载语义 ID 映射之前**插入：

```python
    def __init__(self, config: InferenceConfig):
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

        # ===== 新增：DataSystem 缓存 =====
        self.ds_cache = DataSystemCache()
        # =================================

        # 初始化 TensorRT 引擎（原有逻辑，保持不变）
        self.trt_llm_engine = None
        self.model = None
        if config.use_trt_llm:
            # ... 原有引擎加载逻辑 ...
            pass

        if self.trt_llm_engine is None:
            self._load_pytorch_model(checkpoint)

        # 加载语义 ID 映射（内部已改造为优先 DataSystem）
        self._load_semantic_id_mapping()

        # 重置追踪器（新增 kv 字段）
        self._trace_forward_ms = 0.0
        self._trace_map_ms = 0.0
        self._trace_kv_lookup_ms = 0.0
        self._trace_kv_write_ms = 0.0

        print("Inference service initialized successfully")
```

#### ② 改造 `_load_semantic_id_mapping`

**替换**原有的 `_load_semantic_id_mapping` 方法：

```python
    def _load_semantic_id_mapping(self) -> None:
        """加载语义 ID 到物品 ID 的映射（优先 DataSystem，回退 JSON）."""
        # 1. 优先从 DataSystem 读取
        t0 = time.perf_counter()
        ds_map = self.ds_cache.get_semantic_map() if self.ds_cache else None
        t_kv = (time.perf_counter() - t0) * 1000
        self._trace_kv_lookup_ms = t_kv

        if ds_map is not None:
            print(f"[DataSystem] semantic map loaded from KV (lookup={t_kv:.1f}ms)")
            self.semantic_to_item = ds_map
            self._build_tuple_map()
            return

        # 2. 回退到本地 JSON
        mapping_path = os.path.join(
            os.path.dirname(self.config.model_path),
            'rqvae_semantic_ids.json'
        )
        if not os.path.exists(mapping_path):
            mapping_path = './data/tenrec/processed/semantic_id_map.json'

        if os.path.exists(mapping_path):
            print(f"Loading semantic ID mapping from {mapping_path}")
            with open(mapping_path, 'r') as f:
                self.semantic_to_item = json.load(f)

            # 3. 异步写入 DataSystem（供其他 Pod / 重启后复用）
            if self.ds_cache:
                import threading
                def _async_upload():
                    self.ds_cache.put_semantic_map(self.semantic_to_item)
                threading.Thread(target=_async_upload, daemon=True).start()

            self._build_tuple_map()
        else:
            print(f"Warning: Semantic ID mapping not found at {mapping_path}")
            self.semantic_to_item = {}
            self.semantic_to_item_tuple = {}

    def _build_tuple_map(self):
        """构建 tuple → item_id 的反向索引."""
        self.semantic_to_item_tuple = {}
        for item_id, sem_ids in self.semantic_to_item.items():
            key = tuple(sem_ids)
            self.semantic_to_item_tuple[key] = int(item_id)
        print(f"Loaded {len(self.semantic_to_item)} item mappings")
```

#### ③ 改造 `recommend` 方法（增加结果缓存 + 完整 trace）

**替换**原有的 `recommend` 方法：

```python
    def recommend(
        self,
        user_history: list,
        topk: int = 10,
        temperature: Optional[float] = None,
        beam_width: Optional[int] = None
    ) -> dict:
        """生成推荐（增强版：支持 DataSystem 结果缓存 + 完整 trace）."""
        t0 = time.perf_counter()

        temperature = temperature or self.config.temperature
        beam_width = beam_width or self.config.beam_width

        # ===== 新增：结果级缓存查询 =====
        t_kv_lookup_start = time.perf_counter()
        cached = self.ds_cache.get_result_cache(user_history) if self.ds_cache else None
        t_kv_lookup = (time.perf_counter() - t_kv_lookup_start) * 1000

        if cached is not None:
            total_time = (time.perf_counter() - t0) * 1000
            trace = {
                'total_ms': total_time,
                'prepare_input_ms': 0.0,
                'model_forward_ms': 0.0,
                'generate_ms': 0.0,
                'map_item_ms': 0.0,
                'kv_lookup_ms': t_kv_lookup,
                'kv_write_ms': 0.0,
                'backend': 'cache_hit',
            }
            return {
                'recommendations': cached,
                'inference_time_ms': total_time,
                'trace': trace,
            }
        # =================================

        # 阶段 1: 输入准备
        t_prepare_start = time.perf_counter()
        input_ids = self._prepare_input(user_history)
        input_ids = input_ids.to(self.device)
        t_prepare = (time.perf_counter() - t_prepare_start) * 1000

        # 阶段 2: 模型推理 + 采样生成
        t_generate_start = time.perf_counter()
        self._trace_forward_ms = 0.0
        self._trace_map_ms = 0.0

        with torch.no_grad():
            if beam_width > 1:
                recommendations = self._beam_search_generate(input_ids, topk, beam_width)
            else:
                recommendations = self._sampling_generate(input_ids, topk, temperature)

        t_generate = (time.perf_counter() - t_generate_start) * 1000
        total_time = (time.perf_counter() - t0) * 1000

        # ===== 新增：写入结果缓存 =====
        t_kv_write_start = time.perf_counter()
        if self.ds_cache:
            self.ds_cache.put_result_cache(user_history, recommendations, ttl_sec=10)
        t_kv_write = (time.perf_counter() - t_kv_write_start) * 1000
        # ===============================

        trace = {
            'total_ms': total_time,
            'prepare_input_ms': t_prepare,
            'model_forward_ms': self._trace_forward_ms,
            'generate_ms': t_generate - self._trace_forward_ms,
            'map_item_ms': self._trace_map_ms,
            'kv_lookup_ms': t_kv_lookup,
            'kv_write_ms': t_kv_write,
            'backend': 'tensorrt' if self.trt_llm_engine is not None else 'pytorch',
        }

        # 重置累加器
        self._trace_forward_ms = 0.0
        self._trace_map_ms = 0.0

        return {
            'recommendations': recommendations,
            'inference_time_ms': total_time,
            'trace': trace,
        }
```

### 2.3 增强 Flask 日志输出

找到 `HTTPServer.start` 中的 `[TRACE]` 打印行，替换为：

```python
                trace = result.get('trace', {})
                print(
                    f"[TRACE] request_id={request._trace_id} "
                    f"total_ms={trace.get('total_ms', 0):.1f} "
                    f"backend={trace.get('backend', 'unknown')} "
                    f"prepare_ms={trace.get('prepare_input_ms', 0):.1f} "
                    f"forward_ms={trace.get('model_forward_ms', 0):.1f} "
                    f"generate_ms={trace.get('generate_ms', 0):.1f} "
                    f"map_ms={trace.get('map_item_ms', 0):.1f} "
                    f"kv_lookup_ms={trace.get('kv_lookup_ms', 0):.1f} "
                    f"kv_write_ms={trace.get('kv_write_ms', 0):.1f} "
                    f"items={len(result['recommendations'])}"
                )
```

---

## 三、Go 侧时延分解增强

### 3.1 `services/recall/trtllm_client.go`

在 `TraceInfo` 结构体中新增两个字段：

```go
// TraceInfo 推理服务回传的性能追踪信息.
type TraceInfo struct {
    TotalMs        float64 `json:"total_ms"`
    PrepareInputMs float64 `json:"prepare_input_ms"`
    ModelForwardMs float64 `json:"model_forward_ms"`
    GenerateMs     float64 `json:"generate_ms"`
    MapItemMs      float64 `json:"map_item_ms"`
    KvLookupMs     float64 `json:"kv_lookup_ms"`   // 新增
    KvWriteMs      float64 `json:"kv_write_ms"`    // 新增
    Backend        string  `json:"backend"`
}
```

### 3.2 `services/recall/generative_recall.go`

在 `GetCandidateItems()` 的全链路日志输出段，增加 `tr_kv_lookup_ms`、`tr_kv_write_ms` 和排队时间估算：

```go
    totalCost := utils.CostTime(stageStart)

    // 估算排队/网络开销 = HTTP 往返 - Python 侧内部处理时间
    var queueMs int64
    if response.Trace != nil {
        queueMs = stageHTTP.Milliseconds() - int64(response.Trace.TotalMs)
        if queueMs < 0 {
            queueMs = 0
        }
    }

    if response.Trace != nil {
        log.Info(fmt.Sprintf(
            "requestId=%s\tmodule=GenerativeRecall\tname=%s\tcount=%d\tcost=%d"+
            "\tcache_ms=%d\thistory_ms=%d\tconvert_ms=%d\thttp_ms=%d\titems_ms=%d"+
            "\ttr_backend=%s\ttr_total_ms=%.0f\ttr_prepare_ms=%.0f\ttr_forward_ms=%.0f\ttr_generate_ms=%.0f\ttr_map_ms=%.0f"+
            "\ttr_kv_lookup_ms=%.0f\ttr_kv_write_ms=%.0f\tqueue_ms=%d",
            ctx.RecommendId, r.modelName, len(items), totalCost,
            stageCache.Milliseconds(), stageHistory.Milliseconds(), stageConvert.Milliseconds(),
            stageHTTP.Milliseconds(), stageItems.Milliseconds(),
            response.Trace.Backend, response.Trace.TotalMs,
            response.Trace.PrepareInputMs, response.Trace.ModelForwardMs,
            response.Trace.GenerateMs, response.Trace.MapItemMs,
            response.Trace.KvLookupMs, response.Trace.KvWriteMs,
            queueMs,
        ))
    } else {
        // 兼容旧版推理服务
        log.Info(fmt.Sprintf(
            "requestId=%s\tmodule=GenerativeRecall\tname=%s\tcount=%d\tcost=%d"+
            "\tcache_ms=%d\thistory_ms=%d\tconvert_ms=%d\thttp_ms=%d\titems_ms=%d\tinference_svc_ms=%.0f",
            ctx.RecommendId, r.modelName, len(items), totalCost,
            stageCache.Milliseconds(), stageHistory.Milliseconds(), stageConvert.Milliseconds(),
            stageHTTP.Milliseconds(), stageItems.Milliseconds(),
            response.InferenceTimeMs,
        ))
    }
```

---

## 四、编译与启动（Go 侧）

```bash
cd services
go build -o pairec-server main.go
```

---

## 五、验证步骤

### 5.1 启动推理服务

```bash
# 确保环境变量已设置
export DATASYSTEM_HOST=127.0.0.1
export DATASYSTEM_PORT=31501

python inference/trt_llm/server.py \
  --model_path ./checkpoints/decoder/decoder_best.pt \
  --port 8000 \
  --use_trt_llm
```

**观察启动日志**，应出现：
```
[DataSystem] DsClient(127.0.0.1, 31501) initialized
[DataSystem] KV client acquired
[DataSystem] KV available methods: ['get', 'set', ...]
[DataSystem] Detected READ api: get
[DataSystem] Detected WRITE api: set
```

### 5.2 首次请求（预期 Cache Miss）

```bash
curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: test-001" \
  -d '{
    "user_id": "test_user",
    "history": [[100,50,25,10], [101,51,26,11]],
    "topk": 10,
    "temperature": 1.0,
    "beam_width": 1
  }' | python -m json.tool
```

**预期 Python 侧日志**：
```
[TRACE] request_id=test-001 total_ms=25.3 backend=tensorrt ... kv_lookup_ms=2.5 kv_write_ms=1.2 items=10
```

### 5.3 重复请求（预期 Cache Hit）

用**完全相同的 body** 再发一次：

```bash
curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: test-002" \
  -d '{
    "user_id": "test_user",
    "history": [[100,50,25,10], [101,51,26,11]],
    "topk": 10,
    "temperature": 1.0,
    "beam_width": 1
  }' | python -m json.tool
```

**预期 Python 侧日志**：
```
[TRACE] request_id=test-002 total_ms=3.1 backend=cache_hit ... kv_lookup_ms=1.8 kv_write_ms=0.0 items=10
```

### 5.4 Go 侧全链路验证

启动 Go 服务后，调用推荐接口：

```bash
curl -s -X POST http://localhost:8080/api/rec/feed \
  -H "Content-Type: application/json" \
  -d '{"uid": "76295990", "size": 10}'
```

**检查 Go 日志**，应出现包含 `tr_kv_lookup_ms` / `tr_kv_write_ms` / `queue_ms` 的单行结构化日志：
```
requestId=xxx	module=GenerativeRecall	name=generative_recall	count=50	cost=45	...	tr_kv_lookup_ms=2	tr_kv_write_ms=1	queue_ms=5
```

---

## 六、端到端时延分解全景

完成上述改造后，一次推荐的完整时延分解如下，**所有字段均可从单条日志中提取**：

| 序号 | 阶段 | 字段名 | 所在服务 | 说明 |
|:----:|------|--------|---------|------|
| 1 | 本地缓存 | `cache_ms` | Go | pairec local/redis 缓存查询 |
| 2 | 特征获取 | `history_ms` | Go | Kafka/JSON 获取用户历史行为 |
| 3 | ID 转换 | `convert_ms` | Go | 物品 ID → 语义 ID |
| 4 | HTTP 往返 | `http_ms` | Go | Go → Python 网络 + 序列化 |
| 5 | 结果组装 | `items_ms` | Go | 组装 pairec Item + 写缓存 |
| 6 | 输入准备 | `tr_prepare_ms` | Python | JSON → Tensor + H2D 拷贝 |
| 7 | KV 查询 | `tr_kv_lookup_ms` | Python | DataSystem 语义映射/结果缓存查询 |
| 8 | 模型前向 | `tr_forward_ms` | Python | TensorRT `execute_async_v3` 累加 |
| 9 | 采样生成 | `tr_generate_ms` | Python | Temperature / Beam Search 循环开销 |
| 10 | 语义映射 | `tr_map_ms` | Python | 语义 ID → 物品 ID 字典查询 |
| 11 | KV 写入 | `tr_kv_write_ms` | Python | DataSystem 写回结果缓存/语义映射 |
| 12 | 排队耗时 | `queue_ms` | Go (推算) | `http_ms - tr_total_ms`，反映推理服务排队等待 |

**端到端公式**：
```
端到端时延 ≈ cache_ms + history_ms + convert_ms + http_ms + items_ms
           ≈ tr_total_ms + queue_ms + 网络协议开销
```

---

## 七、故障排查速查表

| 现象 | 排查方向 | 解决 |
|------|---------|------|
| `DsClient init failed` | `DATASYSTEM_HOST`/`DATASYSTEM_PORT` 未设置或 Worker 未启动 | 检查环境变量，确认 Worker + ETCD 健康 |
| `No get-like method found` | `yr.datasystem` 版本差异 | 把 `KV available methods` 输出贴出来，手动替换 `get`/`set` 方法名 |
| `Cache Hit 但结果为空` | DataSystem 返回的数据格式不是 bytes/str | 在 `_safe_get` 中增加 `print(type(result))` 调试 |
| `kv_write_ms 很高` | DataSystem Worker 与推理服务跨节点 / 网络延迟 | 确保 Worker 部署在推理 Pod 所在节点，走共享内存 |
| Go 侧没有 `tr_kv_*` 字段 | `server.py` 返回的 JSON 缺少字段 | 确认 `server.py` 已重启，且请求走到了新代码 |
| `semantic_map` 未写入 DataSystem | `_safe_set` 抛异常 | 检查 DataSystem Worker 是否有写入权限 / 磁盘空间 |

---

## 八、回滚方案

所有改动均为**新增代码和新增字段**，不涉及删除原有逻辑。如需回滚：

1. **Python 侧**：注释掉 `DataSystemCache` 初始化行（`self.ds_cache = DataSystemCache()`），服务立即回退到纯本地 JSON 模式。
2. **Go 侧**：`TraceInfo` 新增的 `KvLookupMs` / `KvWriteMs` 不影响旧版推理服务兼容（旧服务不返回这两个字段，Go 侧打印 `0`）。
3. **完全回滚**：用 git 恢复 `server.py`、`trtllm_client.go`、`generative_recall.go` 三个文件。

---

## 九、下一步（P1 / P2 预览）

P0 完成后，建议按以下顺序推进：

| 阶段 | 事项 | 关键动作 |
|:----:|------|---------|
| P1 | K8s 部署 + brpc 通信 | Go 侧引入 `brpc-go` HTTP Channel 替代原生 `net/http`，获得连接池、健康检查、服务发现 |
| P2 | DataSystem 引擎层集成 | 将推理后端从纯 TensorRT 引擎迁移到 TensorRT-LLM C++ 后端（`PyExecutor` + `KVCacheManager`），编译时链接 `libdatasystem.so`，实现 KV Cache 的 HBM/DRAM 分级缓存 |
