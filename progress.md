# 工作进度

> 最后更新: 2026-06-01 | 当前状态: F08 PaiRec 对接完成，F12 时延分解埋点已落地 | 下一步: 远程压测 miss / hbm_hit / ds_hit 的 p50/p95/p99

## 时间线

| 日期 | 进度 |
|------|------|
| **6/1** | **开始 F12 推荐系统时延分析: 补齐 PaiRec 入口、GenerativeRecall、Python TRT 服务和 TRT runner 分阶段 trace；新增端到端压测汇总脚本** |
| **6/1** | **F08 PaiRec 对接完成: 远程复验 Kafka 实时特征用户 `130`、`2184` 均单次触发 TRT 推理并返回完整映射结果** |
| **6/1** | **修复 PaiRec 自定义 recall 启动 panic: 外部注册同步写入配置签名，框架二次加载时正确跳过内置工厂** |
| **6/1** | **修复远程 DNS 不可用: 将完整 Go `vendor/` 纳入仓库，PaiRec 启动固定使用 `-mod=vendor` 离线依赖** |
| **6/1** | **修复 PaiRec 远程启动: 将 `go.sum` 纳入版本控制；启动脚本正确导出 `CONFIG_PATH` 并移除无效 `--port` 参数** |
| **6/1** | **推荐链路对接 — PaiRec 端到端返回成功；修复 Go 客户端 500ms 超时导致的重复推理** |
| **5/30** | C++ KV cache offload/onboard 闭环验证通过; 任务切换到推荐链路工程化 |
| **5/29** | C++ offload/onboard 链路突破 (FMHA crash根因/C++源码绕过/重编.so) |
| **5/28** | KV Cache + offload/onboard 闭环 (TRT/PyTorch双路径/三层缓存全通) |
| **5/27** | 推理打通 (FMHA bf16 kernel SM89非法内存访问根因) |
| **5/23** | ARM 4090D推理部署 + TRT-LLM引擎构建 + 多轮采样去重 |
| **5/22** | 训练完成 epoch 20 (loss 2.49) |
| **5/21** | DDP三卡训练启动 (epoch 11) |

## 当前状态

- **训练**: ✅ epoch 20, loss 2.49
- **TRT-LLM 引擎**: ✅ bfloat16, 1.46 GB, 限制: max_input_len=64, max_new_tokens=32, max_seq_len=96
- **C++ KV offload/onboard**: ✅ 闭环验证通过
- **推理服务**: ✅ /recommend 可用
- **PaiRec 对接**: ✅ Kafka 实时特征、生成式召回、TRT 推理和 item 映射链路已打通
- **时延分析**: 🔄 第一版 trace 埋点和压测脚本已完成，待远程 L40S 采样

## 6/1 探索：推理命中率优化 (5个bug修复)

### 问题链

PaiRec 调用推理服务 → 大部分用户返回 `code:299 "items size not enough"`。

根因是 **推理服务返回的推荐 item 太少**（1个或0个），远低于 PaiRec 期望的 size（10）。

### 修复清单

| # | 提交 | 问题 | 修复 |
|---|------|------|------|
| 1 | `f7beeb4` | 20条历史→117 tokens 超引擎 max_input_len=64 | prompt 自动截断，保留最近9条 |
| 2 | `fb1cb98` | `<s0_X><pad><s1_Y>` 因中间有非语义token被丢弃 | 先过滤有效token再匹配，跳过非语义token |
| 3 | `326e1bb` | 模型输出层序乱 (s2,s1,s0,s3) 连续递增模式找不到 | 改为按层收集+笛卡尔积组合，不要求顺序 |
| 4 | `5533a16` | 20 tokens输出太短+8轮各自为战 | 8轮token合并到一个池子统一组合 |
| 5 | `1544a3f` | max_new_tokens>32 C++层卡死不报错 | 硬限制≤32 (引擎构建时预留) |
| 6 | `b44dbc6` | layer 3永远为0 (1条历史的用户) | 缺失层用{0}填充，由semantic_id_map验证 |

### 引擎硬限制 (重要)

```
max_seq_len = 96
  ├─ max_input_len = 64   (超过报 RuntimeError)
  └─ max_new_tokens = 32  (超过 C++ 层卡死不报错!)
```

必须同时遵守两个限制。

### 核心机制

**组合模式**: 8轮采样 × 32 token = 256 token 池 → 按层收集所有有效语义token → 笛卡尔积组合 → semantic_id_map 验证

**layer 填充**: 当某层缺失时用 {0} 补位 → 组合出候选 → map 验证真假

## 6/1 验证：PaiRec 端到端功能打通

### 验证证据

| 用户 | history_len | PaiRec 响应 |
|------|-------------|-------------|
| `303` | 20 | `code=200`, `size=5`, 5 个 `generative_recall` item |
| `1201` | 1 | `code=200`, `size=5`, 5 个 `generative_recall` item |
| `130` | 1 | `code=200`, `size=5`, 5 个 `generative_recall` item |

### 新发现：Go 客户端超时导致重复推理

单次 TRT miss 约 `667-812ms`，但 Go 客户端默认超时仅 `500ms` 且最多尝试 3 次。同一个 PaiRec 请求会并发触发多次 GPU 推理，再由后续重试命中 HBM 结果缓存。

已在本地修复：`RecallAlgo` 增加 `timeout_ms=3000`、`max_retries=1`，并同步修改默认配置。待远程部署后确认单次 PaiRec 请求只触发一次 `/recommend`。

### 远程启动修复

远程 `go run` 曾因仓库未跟踪 `go.sum` 报依赖校验缺失。已将 `go.sum` 纳入版本控制，并修复 `scripts/start_pairec.sh`：导出 `CONFIG_PATH` 供 `main.go` 首次加载，移除 PaiRec 未定义的 `--port` 参数。

远程环境随后因 DNS 解析失败无法访问 `mirrors.aliyun.com`。已将完整 `vendor/` 纳入版本控制，启动脚本固定使用 `go run -mod=vendor`，不再依赖在线下载。

首次 vendor 提交仍遗漏 128 个文件：根因是 `.gitignore` 中通用 `lib/` 规则误伤 vendor 内 ClickHouse、Apache Thrift 和 PostgreSQL 驱动目录。已增加 `!vendor/**` 例外并补齐文件。验证方式：从 Git 暂存区导出临时副本，在 `GOPROXY=off`、空 `GOMODCACHE` 和空 `GOCACHE` 下执行 `go test -mod=vendor ./services/...`，全部通过且模块缓存文件数为 0。

### 自定义 recall 注册修复

PaiRec 启动时会再次执行 `recall.Load()`。此前 `main.go` 手工注册 `GenerativeRecall` 时只写入实例，没有写入框架的配置签名；二次加载时框架尝试用内置工厂重建自定义类型并 panic：`recall empty, name:generative_recall`。

已在 vendored recall 包增加 `RegisterRecallWithConfig()`，同步写入实例和配置签名，`main.go` 改用该入口。启动级验证已越过注册阶段并输出 `server start`。

## 6/1 F12：推荐系统端到端时延分解

已按当前 Qwen3 TRT 链路补齐 trace：

- PaiRec 入口内部：`user_feature_ms`、`recall_ms`、`filter_ms`、`general_rank_ms`、`feature_ms`、`rank_ms`、`pipeline_wait_ms`、`merge_ms`、`sort_ms`
- GenerativeRecall：`history_ms`、`convert_ms`、`http_ms`、`items_ms`、`http_overhead_ms`
- Python TRT 服务：`prepare_input_ms`、`kv_lookup_ms`、结果缓存查询和异步写入提交耗时
- TRT 后端：`prompt_ms`、8 轮累计 `runner_generate_ms`、`parse_combo_ms`、`output_pad_ms`
- 新增 `scripts/benchmark_e2e_latency.py`：从 PaiRec 入口发请求，用 `request_id` 关联 PaiRec 和 TRT 日志，汇总 p50/p95/p99，并按 `miss` / `hbm_hit` / `ds_hit` 分组
- 修正小样本 percentile 插值：2 个样本的 p50 使用中位数，不再错误取最小值
- 修复 PaiRec trace 采集：`glog` 默认写独立文件，`scripts/start_pairec.sh` 现在默认传入 `--alsologtostderr=true`，保留文件日志并可由 `tee /tmp/pairec.log` 捕获结构化日志

远程首次冷请求初步数据：2 个 fallback 用户均为 TRT `miss`，Python TRT 平均 `740.2ms`，其中 8 轮 `runner_generate` 平均 `680.0ms`，占比约 `92%`。首个客户端请求额外慢约 `2.7s`，结合首次输出 `Loaded fallback: 999447 users`，疑似 fallback JSON 懒加载；待 PaiRec `history_ms` 复验。

本机验证：

```text
python -m py_compile inference/trt_llm/server.py inference/trt_llm/trt_qwen3_backend.py scripts/benchmark_e2e_latency.py
# exit 0

go test -mod=vendor ./services/...
# services / config / feature / recall 均通过

python -c '<trace parser assertions>'
# trace parser OK
```

## 下一步

1. 远程拉取 F12 埋点并重启 TRT 服务和 PaiRec
2. 运行 `scripts/benchmark_e2e_latency.py`，采集端到端 p50/p95/p99
3. 分别统计 `miss`、`hbm_hit`、`ds_hit`，确认主要瓶颈占比
4. 根据报告决定是否优先优化 8 轮 runner、prompt tokenize 或结果缓存路径
