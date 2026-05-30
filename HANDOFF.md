# 交接日志

> 供 Agent 跨 session 恢复上下文。只记录关键决策和当前任务。

## 最近一次交接 (2026-05-30)

### 当前任务
TensorRT-LLM C++ 层 KV cache offload/onboard 已跑通并收口。任务切换到推荐链路工程化。

下一阶段优先级:
1. **F08 接通 PaiRec**
2. **F12 推荐系统各阶段时延分析**
3. **F13 K8s 部署**
4. **F14 引入 brpc**

### 本 session 解决的问题和关键结论

1. **Scheduler 真实切到 MAX_UTILIZATION**
   - `inference/trt_llm/trt_qwen3_backend.py` 现在构造 `SchedulerConfig(MAX_UTILIZATION)` 并传给 `ModelRunnerCpp.from_dir()`。
   - 前提: 远程 TensorRT-LLM 的 `tensorrt_llm/runtime/model_runner_cpp.py` 也需要支持 `scheduler_config` 参数并写入 `ExecutorConfig`。
2. **KV block reuse 不再被禁用**
   - 远程 TensorRT-LLM C++ 层已绕过 `paged_context_fmha` 对 `enableBlockReuse` 的强约束。
   - 目标日志: 不再出现 `KV cache reuse disabled`，出现 `KV cache block reuse is enabled`。
3. **DataSystem primary/TMP 连接修正**
   - 必须导出 `DATASYSTEM_HOST=127.0.0.1`、`DATASYSTEM_PORT=31501`，否则 TMP 可能默认走 `127.0.0.2`。
4. **C++ offload 写侧已验证**
   - 2048 tokens 配置下: `primaryBlocks=64 secondaryBlocks=28`。
   - 120+32 请求压测出现 `copyBlock entered=3735`、`OffLoad copy=2520`、`Create/Set Key=2520/2520`、`error=0`。
5. **C++ onboard 读侧已验证**
   - 1024 tokens 配置下: `primaryBlocks=32 secondaryBlocks=28`。
   - 180+64 串行压测出现 `copyBlock entered=6067`、`OffLoad copy=4116`、`Create/Set Key=4116/4116`、`Get/OnBoard Key=1/1`、`error=0`。
   - 这证明 TensorRT-LLM C++ 写入 DataSystem 与从 DataSystem 回读 onboard 均已真实触发。
6. **`CacheTransceiver is disabled` 不是这条链路的 blocker**
   - 当前 fork 的证据来自 `KVCacheTransferManager::copyBlock()` 内 DataSystem `Create/Set/Get` 日志。
7. **旧定位修正**
   - GCC/cuBLAS 兼容性不是主因，早期方向已废弃。
   - FMHA crash、reuse 被禁、scheduler 未生效、pool 压力不够，这几条是真 blocker。

### 本 session 关键提交

| 提交 | 内容 |
|------|------|
| `b34db88` | Python TRT backend 传入 scheduler config、block reuse 和 KV token 上限 |
| `e6d7c65` | 新增 C++ KV offload/onboard 压测脚本 |
| `156c600` | 压测脚本改为默认串行并加 fail-fast |
| `5561f5b` | 严格区分 DataSystem onboard 与 HBM reuse |
| `219a71f` | 暴露 `--trt_max_kv_tokens` / `TRT_MAX_KV_TOKENS` |
| `d795947` | 记录 C++ onboard 证据并增强脚本日志输出 |
| `75e1214` | 更新后续任务: PaiRec、时延分析、K8s、brpc |

### 关键新增/修改文件

| 文件 | 用途 |
|------|------|
| `inference/trt_llm/trt_qwen3_backend.py` | 传 `SchedulerConfig(MAX_UTILIZATION)`、`max_tokens_in_paged_kv_cache`、`kv_cache_enable_block_reuse=True` |
| `inference/trt_llm/server.py` | 新增 `--trt_max_kv_tokens` / `TRT_MAX_KV_TOKENS` 和 `--trt_scheduler_policy` |
| `scripts/test_trt_cpp_kv_offload.py` | C++ KV offload/onboard 压测和日志判定 |
| `feature_list.json` | 更新 F08/F12/F13/F14 |
| `progress.md` | 记录 C++ offload/onboard 闭环证据 |

### 远程验证命令

启动服务时建议显式设置:

```bash
export DATASYSTEM_HOST=127.0.0.1
export DATASYSTEM_PORT=31501
export TLLM_LOG_LEVEL=DEBUG
export TRT_MAX_KV_TOKENS=1024

python -m inference.trt_llm.server \
  --model_path ./checkpoints/decoder_qwen3/decoder_epoch_20.pt \
  --qwen3_model_path ./models/Qwen3-0.6B \
  --trt_engine_dir ./trt_engines/qwen3_rec_v4 \
  --port 18000 --device cuda \
  --datasystem_host 127.0.0.1 --datasystem_port 31501 \
  2>&1 | tee /tmp/server_v4.log
```

压测验证:

```bash
python scripts/test_trt_cpp_kv_offload.py \
  --log /tmp/server_v4.log \
  --request 180 \
  --repeat-requests 64 \
  --warmup-requests 0 \
  --history-len 8 \
  --concurrency 1 \
  --strict-onboard
```

预期核心证据:

```text
latest scheduler policy: MAX_UTILIZATION
block reuse enabled seen: True
reuse disabled warning seen: False
new OffLoad copy: >0
new Create/Set Key: >0/>0
new Get/OnBoard Key: >0/>0
new error-like lines: 0
```

### 运行时环境变量

```bash
source /opt/openEuler/gcc-toolset-14/enable

export LD_PRELOAD="\
/workspace/pairec4tigerllm/scripts/block_ds_consumer.so:\
/workspace/pairec4tigerllm/scripts/stub_gpu.so:\
/usr/local/lib/python3.11/site-packages/yr/datasystem/lib/libabseil_dll.so.2407.0.0"
```

### 下一步

1. 接通 PaiRec (F08): 确认 PaiRec 侧配置/代码路径，指向当前 TRT `/recommend` 服务，跑通端到端请求。
2. 推荐系统各阶段时延分析 (F12): 定义 trace 字段，拆 PaiRec、HTTP/Python、TRT generate、DataSystem/KV、item 映射耗时。
3. K8s 部署 (F13): 梳理镜像、GPU 资源、LD_PRELOAD、DataSystem 地址、模型/engine 挂载和探针。
4. 引入 brpc (F14): 在 F12 确认 HTTP/Flask 通信开销后，再决定 brpc server/client 边界。
5. 低优先级: 修复 ConsumerLoop SIGSEGV，恢复 DataSystem 异步通信线程。
