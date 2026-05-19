# P0 DataSystem 接入 + 端到端时延分解 — 最终总结

> 生成时间：2026-05-14
> 状态：Go 侧完成，Python 侧已回退

---

## 一、P0 目标回顾

在 `pairec4tigerllm` 仿真环境中：
1. 让 Python 推理服务通过 `yr.datasystem` SDK 接入 DataSystem
2. 完善端到端推荐时延分解（12 个阶段的全链路日志）

---

## 二、最终代码状态

| 文件 | 改动内容 | 状态 |
|------|---------|------|
| `services/recall/generative_recall.go` | 阶段化计时（cache/history/convert/http/items）、`queueMs` 估算、全链路日志扩展 | ✅ committed (76bb2cc) |
| `services/recall/trtllm_client.go` | TraceInfo 新增 `KvLookupMs`/`KvWriteMs`；Recommend 方法新增 `traceID` | ✅ committed (76bb2cc) |
| `inference/trt_llm/server.py` | DataSystemCache 类、语义映射/结果缓存、trace 增强 | ❌ 已回退 |

```bash
git log --oneline -6
# 76bb2cc P0: integrate DataSystem KV cache and add E2E latency decomposition
# b5d83cd feat: TensorRT 推理优化和 Kafka 特征消费增强  ← server.py 当前版本
```

---

## 三、DataSystem 对接的真相

### 3.1 三个独立的 DataSystem 客户端

| 客户端 | 语言层 | 位置 | 用途 | 状态 |
|--------|--------|------|------|------|
| `KvCacheManagerDataSystem::mkvClientPtr` | C++ | `TensorRT-LLM/cpp/.../kvCacheManager.cpp` | KV Cache **卸载** HBM→DataSystem | 源码中 |
| `KvCacheManagerDataSystemTmp::mkvClientPtr` | C++ | 同上 | KV Cache **加载** DataSystem→HBM | 源码中 |
| `DataSystemCache` (server.py) | Python | `pairec4tigerllm/.../server.py` | 语义映射 + 结果缓存 | 已回退 |

它们互不依赖，连接同一个 DataSystem Worker 但读写不同的 key。

### 3.2 TensorRT-LLM C++ 层的逻辑

**初始化**（KVCacheManager 构造函数，`kvCacheManager.cpp:1840`）：
```cpp
TLLM_LOG_INFO("[TensorRT-LLM][Datasystem] Create Datasystem class");
KvCacheManagerDataSystem& dataSystem = KvCacheManagerDataSystem::getInstance();
std::shared_ptr<datasystem::KVClient> kvClient = dataSystem.getKVClient();
datasystem::Status initRet = kvClient->Init();
// 同时初始化 KvCacheManagerDataSystemTmp
```

**运行时**（KVCacheTransferManager::copyBlock，`kvCacheTransferManager.cpp:161-209`）：
- **卸载**：`kvClient->Create(key, size, para, buffer)` → HBM→CPU copy → `kvClient->Set(buffer)`
- **加载**：`kvClient1->Get(key, buffer, 0)` → CPU→HBM copy

触发条件：GPU HBM 不足，需要驱逐/重新加载 KV Cache 块。

### 3.3 推荐系统内的 DataSystem 调用链

```
用户请求 → Go pairec (:8080)
  → GenerativeRecall.GetCandidateItems()
    → [全链路时延打点] cache_ms/history_ms/convert_ms/http_ms/items_ms
    → TRTLLMClient.Recommend() → HTTP
      → Python server.py (:8000)  ← server.py 不直接调 DataSystem
        → engine.forward()
          → KVCacheManager (C++ 引擎内)
            └── [显存不足时] KvCacheManagerDataSystem → Worker  ← DataSystem 访问
    → 返回推荐结果
```

### 3.4 一次请求访问 DataSystem 的次数

| 场景 | 次数 |
|------|------|
| 短序列、GPU 内存充裕 | **0 次** |
| 长序列多并发、显存压力大 | 动态：每个需卸载的块 2 次 (Create+Set)，每个需加载的块 1 次 (Get) |
| GPT2-Decoder（50 tokens × 32 并发） | **0 次**（KV Cache 仅 77MB，L40S 48GB 完全够用） |

---

## 四、环境调试记录

### 4.1 时间线

| 时间 | 事件 | 结论 |
|------|------|------|
| 05-12 | 代码改造完成，Go 侧 commit & push | — |
| 05-13 | aarch64 Docker 验证机尝试连接 DataSystem Worker | `RPC unavailable`（Worker 启动不足 60s，内部线程未就绪） |
| 05-13 | Worker 就绪后重试 | `segfault`（`/dev/shm` 仅 64MB + ARM 兼容性） |
| 05-14 | 重建容器 `--shm-size=2G` | 仍 segfault |
| 05-14 | 排除法测试：`.so` 加载 OK、对象构造 OK、`LD_BIND_NOW` 通过 | 确认是 C++ `Init()` 内部 ARM 兼容性 bug |
| 05-14 | 决定回退。`git checkout b5d83cd -- server.py` | 干净回退 |

### 4.2 根因

`yr.datasystem` SDK 的 aarch64 二进制在 Docker 环境中 `kv_client::Init()` 直接 segfault。非代码逻辑问题，是 C++ SDK 的 ARM 兼容性缺失。

### 4.3 回退不影响
- Go 侧的全链路时延分解（已 commit）
- TensorRT-LLM C++ 引擎层的 DataSystem 对接（在 `TensorRT-LLM/cpp/` 源码中）
- 推荐系统正常功能

---

## 五、当前架构全貌

```
┌──────────────────────────────────────────────────────────────────┐
│  pairec4tigerllm 推荐系统                                        │
│                                                                  │
│  用户 ──→ Go pairec (:8080)                                      │
│            │                                                     │
│            ├── GenerativeRecall                                  │
│            │   ├── 全链路时延打点 ✅ (76bb2cc)                   │
│            │   │   cache_ms | history_ms | convert_ms | http_ms  │
│            │   │   items_ms | queue_ms                           │
│            │   └── TRTLLMClient.Recommend()                      │
│            │                                                     │
│            └── HTTP ──→ Python server.py (:8000)                 │
│                          ├── 模型推理 (PyTorch / TensorRT)        │
│                          └── KVCacheManager (C++ 引擎内)          │
│                                └── 显存不足时                    │
│                                      └── DataSystem Worker       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 六、Go 侧全链路日志字段

一次请求输出的结构化日志包含：

| 字段 | 含义 | 所在服务 |
|------|------|---------|
| `cache_ms` | 本地/Redis 缓存查询 | Go |
| `history_ms` | Kafka/JSON 用户历史获取 | Go |
| `convert_ms` | 物品ID → 语义ID 转换 | Go |
| `http_ms` | HTTP 往返（含网络） | Go |
| `items_ms` | 结果组装 | Go |
| `inference_svc_ms` | 推理服务总耗时 | Python → Go |
| `tr_backend` | tensorrt 或 pytorch | Python → Go |
| `tr_total_ms` | 推理内部总耗时 | Python → Go |
| `tr_prepare_ms` | 输入准备 | Python → Go |
| `tr_forward_ms` | 模型前向 | Python → Go |
| `tr_generate_ms` | 采样/搜索 | Python → Go |
| `tr_map_ms` | 语义ID→物品ID映射 | Python → Go |
| `tr_kv_lookup_ms` | DataSystem 查询 | Python → Go |
| `tr_kv_write_ms` | DataSystem 写入 | Python → Go |
| `queue_ms` | 推理排队耗时（推算） | Go |

> 注：`tr_kv_lookup_ms` / `tr_kv_write_ms` 在 server.py 回退后始终为 0。

---

## 七、关键文件清单

| 文件 | 路径 |
|------|------|
| Go 全链路日志 | `services/recall/generative_recall.go` |
| TRT 客户端 Trace | `services/recall/trtllm_client.go` |
| TRT KVCacheManager (C++ DS 对接) | `TensorRT-LLM/cpp/include/tensorrt_llm/batch_manager/kvCacheManager.h` |
| KVCacheManager 实现 | `TensorRT-LLM/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` |
| KV Cache TransferManager | `TensorRT-LLM/cpp/tensorrt_llm/batch_manager/kvCacheTransferManager.cpp` |
| CMake DataSystem 链接 | `TensorRT-LLM/cpp/tensorrt_llm/batch_manager/CMakeLists.txt` |
| FindDatasystem | `TensorRT-LLM/cpp/cmake/FindDatasystem.cmake` |
| DataSystem C++ SDK | `yuanrong-datasystem/python/yr/datasystem/kv_client.py` |
| 推理服务（当前版本） | `inference/trt_llm/server.py` |
| P0 落地指南 | `docs/P0_DATASYSTEM_LATENCY_GUIDE.md` |
| 本文件 | `docs/P0_FINAL_SUMMARY.md` |

---

## 八、验证 DataSystem KV Cache 对接

### 方式 1：启动日志（推荐）

TensorRT-LLM 启动时，如果编译了 DataSystem 支持，KVCacheManager 构造函数会打印：
```
[TensorRT-LLM][Datasystem] Create Datasystem class
[TensorRT-LLM][Datasystem] Init KvCache...
```

### 方式 2：Worker 日志

DataSystem Worker 的 ZMQ 指标中包含应用请求时延分布。

### 方式 3：人为触发卸载（高成本）

在 pairec4tigerllm 的 GPT2-Decoder 场景下（KV Cache ~77MB），GPU 48GB 不会触发卸载。需要：
- 换大模型（KV Cache > 显存）
- 或改 `maxNumSequences` 强制限制 block 数
