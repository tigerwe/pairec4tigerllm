# Session 总结：P0 DataSystem 接入与端到端时延分解

> 日期：2026-05-13 ~ 2026-05-14
> 仓库：pairec4tigerllm + TensorRT-LLM + yuanrong-datasystem

---

## 一、核心发现

### 1.1 DataSystem 有三个独立的客户端，不是一套

| 客户端 | 语言层 | 位置 | 用途 |
|--------|--------|------|------|
| `KvCacheManagerDataSystem::mkvClientPtr` | C++ | `TensorRT-LLM/cpp/.../kvCacheManager.cpp` | KV Cache **卸载** |
| `KvCacheManagerDataSystemTmp::mkvClientPtr` | C++ | 同上 | KV Cache **加载** |
| `DataSystemCache` (server.py) | Python | `pairec4tigerllm/.../server.py` | 语义映射 + 结果缓存 |

它们互不依赖，各行其是。

### 1.2 三个 DataSystem"对接"的真相

| 来源 | 实际状态 |
|------|---------|
| P0 目标：Python 推理服务接入 DataSystem | ✅ 代码已写（含 DataSystemCache），后回退 |
| 背景：TensorRT-LLM 已对接 DataSystem | ✅ `/home/vivwimp/TensorRT-LLM/cpp/` 源码有 C++ 集成，但在 TRT-LLM executor 层 |
| 我们的推荐系统走到 DataSystem | ❌ 当前 `server.py` 用原始 `nvinfer1` API，绕过 KVCacheManager |

### 1.3 GPT2-Decoder 无法触发 DataSystem

- KV Cache 满载（32并发×512token）= **96 MB**
- 模型权重 = **10 MB**
- L40S = **48 GB**
- 剩余显存 = **~47 GB**
- 结论：**正常推理对 DataSystem 的访问次数为 0**

---

## 二、代码最终状态

| 文件 | 内容 | 状态 |
|------|------|:--:|
| `services/recall/generative_recall.go` | 全链路时延分解 + queueMs | ✅ committed |
| `services/recall/trtllm_client.go` | TraceInfo 扩展 + traceID 透传 | ✅ committed |
| `inference/trt_llm/server.py` | DataSystemCache | ❌ 已回退 |
| `inference/trt_llm/server.py` | 非侵入式 trace 打点（纯 time.perf_counter） | ✅ 已加入 |

Go 侧日志包含 15 个时延字段：
`cache_ms | history_ms | convert_ms | http_ms | items_ms | tr_backend | tr_total_ms | tr_prepare_ms | tr_forward_ms | tr_generate_ms | tr_map_ms | tr_kv_lookup_ms | tr_kv_write_ms | queue_ms | inference_svc_ms`

---

## 三、ARM Docker 调试记录

| 步骤 | 现象 | 根因 |
|------|------|------|
| Worker 启动 3 分钟时连接 | `RPC unavailable` | `add_node_wait_time_s=60`，线程未就绪 |
| Worker 就绪后连接 | `segfault` | `/dev/shm=64MB` 不够 SDK mmap |
| 扩容 `/dev/shm=2G` | 仍 `segfault` | `yr.datasystem` aarch64 二进制兼容性问题 |
| 排除法：`.so` 加载 OK、构造 OK、`LD_BIND_NOW` 通过 | 确认 C++ `Init()` 内部 ARM bug | SDK 自身问题，非代码逻辑问题 |

---

## 四、环境要求

运行 DataSystem 需要：
- **x86 架构**（ARM 上 SDK 不兼容）
- Docker `--shm-size=512m`（SDK 需要 `/dev/shm` 做 IPC）
- ETCD 运行（Worker 依赖）
- Worker 运行（`datasystem_worker`，端口 31501）

---

## 五、当前端到端数据流

```
用户 → Go pairec (:8080)
  → GenerativeRecall
    ┌─ cache_ms:     本地缓存查询
    ├─ history_ms:   Kafka/JSON 用户历史
    ├─ convert_ms:   itemID → semanticID (4-dim)
    ├─ http_ms:      HTTP POST :8000
    │   └→ Python server.py
    │        ┌─ _prepare_input: [][]int → torch.Tensor
    │        ├─ _get_logits:   pytorch() 或 trt_engine.execute_async_v3
    │        ├─ softmax → multinomial → 4-dim semanticID
    │        └─ semantic_to_item_tuple.get() → itemID
    ├─ items_ms:     结果组装
    └─ → 全链路日志 (15字段)
```

**DataSystem 不在这个链路里。** KVCacheManagerDataSystem 在 TRT-LLM executor 层，当前 `server.py` 用原始 `nvinfer1` API 绕过了它。

---

## 六、让 DataSystem 进入闭环的方案

### 方案：Qwen3-0.6B + TRT-LLM ModelRunner

| 改动文件 | 内容 | 行数 |
|---------|------|:--:|
| `generative_recall.go` | `convertToSemanticIDs` 加 Qwen 模式 | ~10 |
| **新增** `server_qwen.py` | ModelRunner 推理 + prompt 拼接 | ~200 |
| **新增** `build_qwen_engine.sh` | Qwen3 → TRT 引擎构建 | ~20 |

不改 `trtllm_client.go`、不改 `convertToItems`、不改全链路日志。
ModelRunner 内部自动构造 KVCacheManager → DataSystem。

### Qwen3-0.6B 推荐格式

```
输入:  "用户历史点击物品: 3035268, 248791, ...\n请推荐10个物品ID，用逗号分隔:"
输出:  "12345, 67890, 11223, ..."
解析:  → itemID 列表 → 返回给 Go
```

---

## 七、关键文件清单

| 文件 | 路径 |
|------|------|
| Go 全链路日志 | `services/recall/generative_recall.go` |
| TRT 客户端 | `services/recall/trtllm_client.go` |
| 推理服务（当前） | `inference/trt_llm/server.py` |
| TRT KVCacheManager 头文件 | `TensorRT-LLM/cpp/include/tensorrt_llm/batch_manager/kvCacheManager.h` |
| KVCacheManager 实现 | `TensorRT-LLM/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` |
| KV Cache 传输管理 | `TensorRT-LLM/cpp/tensorrt_llm/batch_manager/kvCacheTransferManager.cpp` |
| CMake DataSystem 链接 | `TensorRT-LLM/cpp/tensorrt_llm/batch_manager/CMakeLists.txt` |
| DataSystem SDK | `yuanrong-datasystem/python/yr/datasystem/` |
| P0 落地指南 | `docs/P0_DATASYSTEM_LATENCY_GUIDE.md` |
| P0 进展 | `docs/P0_PROGRESS.md` |
| P0 最终总结 | `docs/P0_FINAL_SUMMARY.md` |
| 本文件 | `docs/SESSION_SUMMARY.md` |
