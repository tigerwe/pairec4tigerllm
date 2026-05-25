# 交接日志

> 供 Agent 跨 session 恢复上下文。只记录关键决策和当前任务。

## 最近一次交接 (2026-05-25)

### 当前任务
DataSystem KV Cache 集成 — 核心阻断已解决，待重建引擎后完成 offload/onboard 验证。
下一步: 重建 TRT-LLM 引擎 → Go pairec 联调 (F08)

### 关键成果
- **stub 方案通过** ✅: `LD_PRELOAD=block_ds_consumer.so:libabseil_dll.so` 解决双重问题
- DataSystem 连接成功: `Init KvCache Manager DataSystem success`
- 所有代码已推送 gitcode `dev` 分支

### 问题链与解法

```
问题1: abseil 冲突 → RegisterFlag 崩溃
  → LD_PRELOAD=libabseil_dll.so

问题2: LD_PRELOAD 副作用 → PipelineRH2DQueueConsumer 线程崩溃
  → stub 函数 `ConsumerLoop` 立即返回 (C++ std::thread，pthread_create 拦不住)

问题3: 重编译 TRT-LLM 后旧引擎不兼容
  → 需重建引擎 (trtllm-build)
```

### 容器运行时环境变量 (已写入 ~/.bashrc)
```bash
export LD_PRELOAD="/workspace/pairec4tigerllm/scripts/block_ds_consumer.so:/usr/local/lib/python3.11/site-packages/yr/datasystem/lib/libabseil_dll.so.2407.0.0"
export LD_LIBRARY_PATH="/opt/openEuler/gcc-toolset-14/root/usr/lib64:/usr/local/lib/python3.11/site-packages/yr/datasystem/lib:/TensorRT-LLM/tensorrt_llm:/TensorRT-LLM/cpp/build/tensorrt_llm/thop:$(find /usr/local/lib/python3.11/site-packages/nvidia -type d -name 'lib' | tr '\n' ':')${LD_LIBRARY_PATH}"
export PYTHONPATH=/TensorRT-LLM:$PYTHONPATH
```

### 关键文件
| 文件 | 用途 |
|------|------|
| `scripts/stub_consumer.c` | stub ConsumerLoop，绕过共享内存崩溃 |
| `scripts/block_consumer.c` | 旧版 pthread_create 拦截 (保留参考) |
| `inference/trt_llm/trt_qwen3_backend.py` | 支持 scheduler_config + max_tokens_in_paged_kv_cache |
| `inference/trt_llm/server.py` | DataSystem client 初始化 + CLI |

### 待完成
1. 重建引擎: `trtllm-build` 用当前 TRT-LLM 二进制重编 `qwen3_rec` 引擎
2. 发批量请求触发 eviction，验证 offload/onboard 日志
3. Go pairec 联调 (F08)
