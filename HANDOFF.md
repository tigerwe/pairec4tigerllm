# 交接日志

> 供 Agent 跨 session 恢复上下文。只记录关键决策和当前任务。

## 最近一次交接 (2026-06-01)

### 当前任务
F12 推荐系统各阶段时延分析 — 第一版埋点和端到端压测脚本已完成，待远程 L40S 采集 p50/p95/p99。

### 6/1 PaiRec 端到端验证

**功能验证通过**:

| 用户 | history_len | PaiRec 响应 |
|------|-------------|-------------|
| `303` | 20 | `code=200`, `size=5`, 5 个 `generative_recall` item |
| `1201` | 1 | `code=200`, `size=5`, 5 个 `generative_recall` item |
| `130` | 1 | `code=200`, `size=5`, 5 个 `generative_recall` item |

**新发现**: 单次 TRT miss 约 `667-812ms`，Go 客户端默认超时仅 `500ms` 且最多尝试 3 次。日志显示同一用户请求会并发触发多次推理，随后由重试命中 HBM 结果缓存。

**本地修复**:
- `RecallAlgo` 支持 `timeout_ms` 和 `max_retries`
- PaiRec 配置设为 `timeout_ms=3000`、`max_retries=1`
- Go 默认值同步改为 `3s`、单次尝试
- 本地验证: `go test ./services/...` 通过

**远程复验通过**:
- Kafka 实时特征用户 `130`：`history_len=20`，只触发一次 TRT generate，`hit=10, miss=0`
- Kafka 实时特征用户 `2184`：`history_len=20`，只触发一次 TRT generate，`hit=10, miss=0`
- F08 PaiRec 对接可以标记完成

**远程启动修复**:
- 仓库原先忽略 `go.sum`，远程 `go run` 会报依赖校验缺失；现已纳入版本控制
- `scripts/start_pairec.sh` 现在会导出 `CONFIG_PATH`，确保 `main.go` 首次加载正确配置
- 移除脚本中 PaiRec 未定义的 `--port` 参数
- 远程 DNS 无法解析 `mirrors.aliyun.com`；现已将完整 `vendor/` 纳入仓库
- 启动脚本固定使用 `go run -mod=vendor`，不再依赖在线下载
- 首次 vendor 提交遗漏 128 个文件：`.gitignore` 的通用 `lib/` 规则误伤 vendor 内 ClickHouse、Apache Thrift 和 PostgreSQL 驱动目录
- 已增加 `!vendor/**` 例外并补齐文件；从 Git 索引导出临时副本后，以 `GOPROXY=off`、空 `GOMODCACHE` 和空 `GOCACHE` 编译通过，模块缓存文件数为 0
- PaiRec 启动曾 panic：`recall empty, name:generative_recall`。根因是手工注册自定义 recall 时未同步写入框架配置签名，框架二次加载时错误进入内置工厂
- 已在 vendored recall 包增加 `RegisterRecallWithConfig()` 并让 `main.go` 使用；本地短暂启动已输出 `server start`，原 panic 消失

### 6/1 探索：推理命中率优化

**背景**: PaiRec 调用推理服务 → 绝大多数用户返回 `code:299 "items size not enough"`。

**根因链**:
```
20条历史(117 tokens) > max_input_len=64 → 截断到9条
→ max_new_tokens=128 > 引擎预留32 → C++卡死不报错
→ 即使生成成功, 每轮20 tokens太短, layer 3永远为0
→ 有效四元组=0 → PaiRec拿不到item → 299
```

**已修复的5个瓶颈**:

| 提交 | 问题 | 修复 |
|------|------|------|
| `f7beeb4` | input超64 | prompt自动截断到最近9条 |
| `fb1cb98` | 语义token被非语义token隔开 | 先过滤再匹配 |
| `326e1bb` | 层序乱(2,1,0,3)→连续模式miss | 按层收集+笛卡尔积组合 |
| `5533a16` + `1544a3f` | token太少+合并池 | 8轮×32=256 token统一组合 |
| `b44dbc6` | layer 3=0(短历史用户) | 补{0}由map验证 |

### 引擎硬限制 (重要！)

```
引擎: max_seq_len=96, max_input_len=64, max_new_tokens=32

⚠️ max_new_tokens>32 → C++层buffer分配失败卡死不报错!
   Python except捕获不到, 必须代码层硬限制≤32

⚠️ prompt_len + max_new_tokens == 96 → C++层边界卡死!
   必须控制在≤95
```

### 当前推理流程

```
输入: history[20条] → 截断到[9条] → tokenize → prompt_len=62
生成: 8轮 × max_new=32 → 256 token池
解析: 按层收集 → 笛卡尔积组合 → layer缺失补{0} → semantic_id_map验证
输出: 去重后的有效item列表
```

### F12 第一版时延分解

已补齐当前 Qwen3 TRT 链路的结构化 trace：

| 层级 | 字段 |
|------|------|
| PaiRec 入口内部 | `user_feature_ms`, `recall_ms`, `filter_ms`, `general_rank_ms`, `feature_ms`, `rank_ms`, `pipeline_wait_ms`, `merge_ms`, `sort_ms` |
| GenerativeRecall | `history_ms`, `convert_ms`, `http_ms`, `items_ms`, `http_overhead_ms` |
| Python TRT 服务 | `prepare_input_ms`, `kv_lookup_ms`, `result_cache_lookup_ms`, `result_cache_ds_lookup_ms`, `result_cache_write_submit_ms` |
| TRT 后端 | `prompt_ms`, `runner_generate_ms`, `parse_combo_ms`, `output_pad_ms`, `backend_total_ms` |

新增 `scripts/benchmark_e2e_latency.py`：
- 从 PaiRec `/api/recommend` 入口发请求
- 用 PaiRec `request_id` 关联 Go 和 Python TRT 日志
- 汇总各阶段 p50/p95/p99
- 按结果缓存来源 `miss` / `hbm_hit` / `ds_hit` 分组
- percentile 使用线性插值，2 个样本的 p50 为中位数

本机验证：

```text
python -m py_compile inference/trt_llm/server.py inference/trt_llm/trt_qwen3_backend.py scripts/benchmark_e2e_latency.py
# exit 0

go test -mod=vendor ./services/...
# pass

trace parser assertions
# trace parser OK
```

### 关键修改文件

| 文件 | 改动内容 |
|------|---------|
| `inference/trt_llm/trt_qwen3_backend.py` | prompt自动截断、max_new双限制、多轮token合并池、组合模式、layer填充 |
| `inference/trt_llm/server.py` | max_new_tokens≤32、500异常traceback、防御性shape检查 |

### 远程验证命令

```bash
cd /home/workspace/zcx/pairectest/pairec4tigerllm
git pull gitcode dev

export DATASYSTEM_HOST=127.0.0.1 DATASYSTEM_PORT=31501 TRT_MAX_KV_TOKENS=1024

pkill -f "inference.trt_llm.server" || true
python -m inference.trt_llm.server \
  --model_path ./checkpoints/decoder_qwen3/decoder_epoch_20.pt \
  --qwen3_model_path ./models/Qwen3-0.6B \
  --trt_engine_dir ./trt_engines/qwen3_rec_v4 \
  --port 18000 --device cuda \
  --datasystem_host 127.0.0.1 --datasystem_port 31501 \
  2>&1 | tee /tmp/server_v4.log
```

直接打推理服务验证:

```bash
# 长历史 (9条截断)
curl -X POST http://localhost:18000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test_9","history":[[92,230,20,0],[129,73,21,0],[28,174,20,0],[174,241,21,0],[72,249,20,0],[66,221,20,0],[184,144,21,0],[224,97,22,0],[67,194,24,0],[223,106,22,0]],"topk":10}'

# 短历史 (1条)
curl -X POST http://localhost:18000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test_1","history":[[92,230,20,0]],"topk":10}'
```

期望看到 `[TRT parse] merged 8 rounds, layer_counts=[?,?,?,?]` 日志。

### 运行时环境

```bash
source /opt/openEuler/gcc-toolset-14/enable
export LD_PRELOAD="\
/workspace/pairec4tigerllm/scripts/block_ds_consumer.so:\
/workspace/pairec4tigerllm/scripts/stub_gpu.so:\
/usr/local/lib/python3.11/site-packages/yr/datasystem/lib/libabseil_dll.so.2407.0.0"
```

### 下一步

1. 远程拉取 F12 埋点并重启 TRT 服务和 PaiRec
2. 运行 `scripts/benchmark_e2e_latency.py`
3. 分别采集 `miss`、`hbm_hit`、`ds_hit` 的 p50/p95/p99
4. 根据阶段占比确定下一轮优化目标
