# P0 DataSystem 接入 + 端到端时延分解 — 进展总结

> 生成时间：2026-05-12
> 供后续智能体继续工作参考

---

## 一、P0 目标回顾

在 `pairec4tigerllm` 仿真环境中，让 Python 推理服务通过 `yr.datasystem` SDK 接入 DataSystem，并完善端到端推荐时延分解。

---

## 二、已完成的工作

### 2.1 代码改造（3 个文件）

| 文件 | 改动内容 | 状态 |
|------|---------|------|
| `inference/trt_llm/server.py` | 新增 `DataSystemCache` 类；改造语义映射加载（优先 DataSystem，回退 JSON）；`recommend()` 增加结果级缓存查询/写入；增强 trace（`kv_lookup_ms`/`kv_write_ms`）；Flask 日志增强 | ✅ 已修改，未重新 commit |
| `services/recall/trtllm_client.go` | 新增 `TraceInfo` 结构体（含 `KvLookupMs`/`KvWriteMs`）；`Recommend` 方法新增 `traceID` 参数，HTTP Header 透传 `X-Request-ID` | ✅ 已 commit & push |
| `services/recall/generative_recall.go` | `GetCandidateItems` 阶段化计时（cache/history/convert/http/items）；新增 `queueMs` 估算；全链路日志扩展 `tr_kv_lookup_ms`/`tr_kv_write_ms`/`queue_ms` | ✅ 已 commit & push |

### 2.2 Git 提交记录

```bash
commit 76bb2cc  (origin/main)
Author: kimi
Message: P0: integrate DataSystem KV cache and add E2E latency decomposition
```

**注意**：后续又修改了 `server.py`（修复 DataSystem `Not ready` 问题），但**尚未 commit**。当前工作区 `server.py` 与 `origin/main` 不一致。

### 2.3 生成的 Patch 文件（供跨机器部署）

```
P0-datasystem-latency.patch      # 总 patch（28KB）
P0-01-server.patch               # server.py 单独 patch（15KB）
P0-02-generative_recall.patch    # generative_recall.go 单独 patch（7.2KB）
P0-03-trtllm_client.patch        # trtllm_client.go 单独 patch（2.2KB）
```

> 这些 patch 基于 commit `76bb2cc` 生成，后续 `server.py` 的修复未包含在内。

---

## 三、当前遇到的核心问题：DataSystem `set` 报 `Not ready`

### 3.1 现象

Python 推理服务启动时，`DsClient` 初始化成功、`kv()` 获取成功、`get`/`set` API 探测成功，但**实际调用 `set` 时抛错**：

```
[DataSystem] set error: code: [Not ready], msg: [
  Thread ID 281468973891968 Not ready.
  The client has not been initialized yet
  File: object_client_impl.h
]
```

### 3.2 根因分析（已确认）

1. **不是线程安全问题**：早期怀疑是 `threading.Thread` 导致 TLS 上下文丢失，已改为同步调用，问题依旧。
2. **不是 Worker 未启动**：如果 Worker 未启动，`DsClient(host, port)` 会直接连接失败。
3. **真实原因**：`yr.datasystem` 的 `DsClient` 创建后，底层 C++ client **并未完成初始化握手**。`get` 能探测到是因为 Python 层面仅做 `hasattr` 检查，但 `set` 真正发起 RPC 时才暴露底层未就绪。日志中伴随的 `AccessRecorder is not init` 也佐证了 Worker 内部模块未完全初始化。

### 3.3 已尝试的修复（当前代码状态）

当前 `server.py` 中 `DataSystemCache.__init__` 已做以下处理：

```python
# 1. 获取 kv 子客户端后，显式调用 init() 和 health_check()
if hasattr(self._kv, 'init'):
    try:
        self._kv.init()
    except Exception as e:
        print(f"[DataSystem] KV init() skipped: {e}")

if hasattr(self._kv, 'health_check'):
    try:
        self._kv.health_check()
    except Exception as e:
        print(f"[DataSystem] KV health_check() skipped: {e}")

# 2. _safe_set 中对 Not ready 静默降级
except Exception as e:
    err_msg = str(e)
    if "Not ready" in err_msg or "not been initialized" in err_msg:
        if not getattr(self, '_warned_not_ready', False):
            print(f"[DataSystem] Worker not ready for write, will retry later")
            self._warned_not_ready = True
    else:
        print(f"[DataSystem] set error: {e}")
    return False
```

### 3.4 当前结果

- 服务**可以正常启动**（不会阻塞）
- `get` 操作（结果缓存查询）**理论上可用**（待验证）
- `set` 操作（语义映射上传、结果缓存写入）**当前失败**，被静默降级

---

## 四、待验证 / 待解决问题

### 4.1 高优先级：DataSystem 写入问题

**目标**：让 `set` 操作真正成功，否则结果缓存和语义映射上传都无效。

**排查方向**：

1. **DataSystem Worker 是否真的 ready？**
   - 检查 Worker 日志，确认是否有 `AccessRecorder init done` 或类似就绪标志
   - 确认 Worker 和 ETCD 的健康状态

2. **`yr.datasystem` 版本差异**
   - 当前环境 `KV available methods` 输出：
     ```
     ['delete', 'exist', 'expire', 'generate_key', 'get', 'get_buffers',
      'get_read_only_buffers', 'health_check', 'init', 'mcreate', 'mset',
      'mset_buffer', 'msettx', 'read', 'set', 'set_value']
     ```
   - 可能需要显式等待 Worker ready，或调用其他初始化方法

3. **写权限问题**
   - 确认 DataSystem Worker 是否以 writable 模式启动
   - 检查 `DATASYSTEM_HOST`/`DATASYSTEM_PORT` 指向的实例是否有写入权限

4. **重试机制**
   - 如果 Worker 是异步初始化的，可能需要在 `DsClient` 创建后 sleep 几秒再操作

### 4.2 中优先级：端到端验证

| 验证项 | 状态 | 命令 |
|--------|------|------|
| Python 推理服务独立启动 | ✅ 可启动 | `python inference/trt_llm/server.py --model_path ... --port 8000` |
| DataSystem `get` 语义映射 | 待验证 | 观察启动日志是否 `semantic map loaded from KV` |
| DataSystem `set` 语义映射 | ❌ 当前失败 | 见 4.1 |
| 结果缓存 Cache Miss | 待验证 | `curl localhost:8000/recommend` 首次请求 |
| 结果缓存 Cache Hit | 待验证 | `curl localhost:8000/recommend` 重复请求 |
| Go 侧编译 | ✅ 通过 | `cd services && go build -o pairec-server main.go` |
| Go → Python 端到端 | 待验证 | 启动 pairec-server 后调用 `/api/rec/feed` |

### 4.3 低优先级：代码清理

- `server.py` 当前工作区修改未 commit，修复 DataSystem 问题后需重新 commit & push
- Patch 文件未更新（未包含 `server.py` 的 `Not ready` 修复）

---

## 五、关键代码位置

### 5.1 DataSystem 缓存层

```python
# inference/trt_llm/server.py
class DataSystemCache:
    KEY_PREFIX = "pairec4tigerllm"
    # __init__ 中初始化 DsClient -> kv() -> init()/health_check()
    # _safe_get / _safe_set / get_semantic_map / put_semantic_map / get_result_cache / put_result_cache
```

### 5.2 语义映射加载（优先 DataSystem）

```python
# inference/trt_llm/server.py: GenerativeInferenceService._load_semantic_id_mapping
# 1. 优先 ds_cache.get_semantic_map()
# 2. miss 后读本地 JSON
# 3. 尝试 ds_cache.put_semantic_map() 写回（当前失败）
```

### 5.3 结果缓存（recommend 方法）

```python
# inference/trt_llm/server.py: GenerativeInferenceService.recommend
# 开头：ds_cache.get_result_cache(user_history)
# 结尾：ds_cache.put_result_cache(user_history, recommendations, ttl_sec=10)
```

### 5.4 Go 侧全链路日志

```go
// services/recall/generative_recall.go: GetCandidateItems
// 阶段：cache_ms -> history_ms -> convert_ms -> http_ms -> items_ms
// queueMs = stageHTTP.Milliseconds() - response.Trace.TotalMs
// 日志字段：tr_kv_lookup_ms / tr_kv_write_ms / queue_ms
```

---

## 六、环境信息

| 组件 | 配置 |
|------|------|
| DataSystem Host | `127.0.0.1`（默认） |
| DataSystem Port | `31501`（默认） |
| 推理服务端口 | `8000` |
| pairec 服务端口 | `8080` |
| 模型路径 | `./checkpoints/decoder/decoder_best.pt` |
| 语义映射路径 | `./data/tenrec/processed/semantic_id_map.json` |
| TensorRT 引擎 | `./exported/decoder/decoder.engine`（11 MiB） |

---

## 七、下一步建议（给后续智能体）

1. **先解决 DataSystem `set` 失败问题**
   - 检查 Worker 日志确认就绪状态
   - 尝试在 `DsClient` 创建后增加 `time.sleep(3)` 等待初始化
   - 如果 Worker 确实不支持写入，考虑换一种 DataSystem 部署方式

2. **验证 Cache Miss / Cache Hit**
   - 启动服务后连续发两次相同 curl，观察 `[TRACE]` 日志
   - 首次应 `backend=pytorch`，第二次应 `backend=cache_hit`

3. **验证语义映射加载**
   - 清空 DataSystem KV 中 `pairec4tigerllm:semantic_map`
   - 重启服务，观察是否 `semantic map loaded from KV`

4. **提交最终代码**
   - `server.py` 的修复确认稳定后，重新 `git commit` 并 `git push`
   - 更新 patch 文件

5. **端到端验证**
   - 打全 3 个 patch，编译 Go 侧，启动完整链路验证
