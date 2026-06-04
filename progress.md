# 工作进度

> 最后更新: 2026-06-04 | 当前状态: E2E 报告已合并 DataSystem C++ KV block Get/Set 指标 | 下一步: 重跑 TRT_NUM_SAMPLES=1 E2E，分离纯 miss 与缓存命中并同时观察 C++ Get/Set

## 时间线

| 日期 | 进度 |
|------|------|
| **6/4** | **输出 `docs/E2E_LATENCY_REPORT_2026-06-04.md`，汇总 TRT_NUM_SAMPLES=1 的 PaiRec E2E、缓存分组、当前 E2E 中 DS trace 缺失原因与下一轮充分测 DS Set/Get 的要求** |
| **6/4** | **`scripts/benchmark_e2e_latency.py` 合并 DataSystem C++ KV block 统计：同一份 E2E 报告同时输出请求级阶段耗时和 `offload.set_ms` / `onboard.get_ms` 等 per-block 指标** |
| **6/4** | **TRT_NUM_SAMPLES=1 PaiRec E2E 完成: 60 请求全成功，client p50=3.5ms、p95=90.5ms、p99=92.8ms；报告混入结果缓存命中，冷 miss 对应 p95/p99 尾部约 90ms** |
| **6/4** | **远程 runner 轮数 A/B 完成: 1/2/4/8 轮均 `full_topk=60/60`；HTTP p50 分别为 91.8/180.1/350.2/701.7ms；确认 runner 耗时近似线性缩放** |
| **6/4** | **修复 `scripts/benchmark_trt_runner_samples.py` 使用 `--stop-command 'pkill -f ...'` 时会匹配自身 `--server-cmd` 并被 `Terminated` 的问题；文档改为脚本外手动清理旧服务** |
| **6/3** | **新增 `scripts/benchmark_trt_runner_samples.py`，自动核验 `/health` 的 `trt_num_samples`、运行直连压测、解析 `runner_calls` 并输出 JSON/CSV 对比** |
| **6/2** | **TRT runner 优化准备: 定位每个 miss 请求串行执行 8 次 `runner.generate()`；增加 `TRT_NUM_SAMPLES` 参数和单轮 trace，支持 1/2/4/8 轮质量-时延 A/B** |
| **6/2** | **DataSystem onboard 专项扩样准备: 报告明确区分 host DataSystem API、host 计时同步 D2H/H2D 和总 wall-clock；新增 `--min-onboard-samples` 门槛** |
| **6/2** | **远程 replay 修正验证通过: offload 8237 次，onboard 177 次；Get p50=0.725ms、p99=1.709ms，Get+H2D p50=1.258ms、p99=2.291ms** |
| **6/2** | **远程应用 C++ trace patch 并完成首轮 DataSystem 实测: offload 4181 次，Set p99=1.093ms；onboard/Get 已观测到 1 次，需继续增加读样本** |
| **6/2** | **F12 C++ DataSystem 时延观测补齐: 新增 Create/D2H/Set、Get/H2D 结构化 trace patch；C++ 压测和 PaiRec E2E 汇总新增 p99/p9999/max** |
| **6/1** | **端到端阶段性收口: `dev` 固化为可回退基线；后续从专用分支开展 TRT-LLM C++ DataSystem 与原生 pinned DRAM 的端到端 A/B** |
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
- **时延分析**: ✅ 第一版端到端 trace 和冷请求分解已验证；✅ C++ DataSystem Set/Get 阶段性统计已完成；🔄 待扩大 onboard 样本和补充 pinned DRAM A/B

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
- 新增 `scripts/benchmark_e2e_latency.py`：从 PaiRec 入口发请求，用 `request_id` 关联 PaiRec 和 TRT 日志，汇总 p50/p95/p99/p9999/max，并按 `miss` / `hbm_hit` / `ds_hit` 分组
- 修正小样本 percentile 插值：2 个样本的 p50 使用中位数，不再错误取最小值
- 修复 PaiRec trace 采集：`glog` 默认写独立文件，`scripts/start_pairec.sh` 现在默认传入 `--alsologtostderr=true`，保留文件日志并可由 `tee /tmp/pairec.log` 捕获结构化日志

远程冷请求分解已确认：

- fallback 用户 `142`、`211` 均为 TRT `miss`
- 首个 fallback 请求：PaiRec `3392ms`，其中 `history_ms=2573ms`；对应首次加载 `999447` 用户 fallback JSON
- 后续稳态冷请求：PaiRec `687ms`，其中 TRT 服务 `685.3ms`，PaiRec 额外开销约 `2ms`
- 两次 TRT 平均 `750.0ms`，其中 8 轮 `runner_generate` 平均 `686.2ms`，占比约 `91.5%`
- TRT 次要耗时：`output_pad_ms` 平均 `27.3ms`、`prompt_ms` 平均 `10.7ms`
- 结论：fallback JSON 应在启动阶段预加载，避免首请求抖动；稳态冷请求的首要优化目标是 8 轮 TRT runner

本机验证：

```text
python -m py_compile inference/trt_llm/server.py inference/trt_llm/trt_qwen3_backend.py scripts/benchmark_e2e_latency.py
# exit 0

go test -mod=vendor ./services/...
# services / config / feature / recall 均通过

python -c '<trace parser assertions>'
# trace parser OK
```

## 6/2 F12：C++ DataSystem Get/Set 时延与尾延迟指标

已新增 `trtllm-datasystem-latency-trace.patch`，针对外部 TensorRT-LLM
`cpp/tensorrt_llm/batch_manager/kvCacheTransferManager.cpp` 增加结构化日志：

```text
[TensorRT-LLM][Datasystem][TRACE] op=offload ... create_ms=... d2h_ms=... set_ms=... total_ms=...
[TensorRT-LLM][Datasystem][TRACE] op=onboard ... get_ms=... h2d_ms=... total_ms=...
```

统计脚本同步增强：

- `scripts/test_trt_cpp_kv_offload.py`：按 warmup / pressure / replay / overall 汇总 DataSystem C++ 指标，HTTP 和 C++ 指标均输出 `avg/p50/p95/p99/p9999/max`，支持 `--json-output`
- `scripts/benchmark_e2e_latency.py`：PaiRec E2E、各阶段和 TRT 推荐结果缓存分组新增 `p9999`，缓存分组补齐 `p99/p9999/max`
- `p9999` 使用线性插值；样本量不足时仅作方向性观察，正式结论需扩大请求量

本机验证：

```text
git -C /home/vivwimp/TensorRT-LLM apply --check \
  /home/vivwimp/pairec4tigerllm/trtllm-datasystem-latency-trace.patch
# exit 0

python -m py_compile \
  scripts/benchmark_e2e_latency.py scripts/test_trt_cpp_kv_offload.py
# exit 0

python - <<'PY'
# synthetic percentile + DataSystem TRACE parser assertions
PY
# datasystem trace parser OK
```

### 远程 DataSystem 首轮实测

DataSystem runtime 已应用 trace patch 并完成首轮 pressure / replay：

```text
KV pool: primaryBlocks=32 secondaryBlocks=28
offload count=4181
  create_ms p50=0.680 p99=1.025 max=2.171
  d2h_ms    p50=0.413 p99=0.691 max=0.732
  set_ms    p50=0.759 p99=1.093 max=1.381
  total_ms  p50=1.937 p99=2.553 max=4.006
onboard count=1
  get_ms=1.562 h2d_ms=0.345 total_ms=1.961
```

结论：

- DataSystem C++ offload 写路径稳定，单个 3.5 MiB block 的 `Create + D2H + Set`
  总耗时 p50 约 `1.94ms`，p99 约 `2.55ms`
- onboard/Get 已出现 1 次，证明读路径打通；样本不足，暂不能评价 Get 分位数
- 原压测 verdict 依赖 DEBUG 级 `KV cache block reuse is enabled`、`copyBlock entered`
  和 `Set Key` 日志，在 INFO 级日志下会误报失败；脚本已改为优先使用结构化 trace 判定

### 远程 DataSystem 第二轮实测与 replay 修正

扩大 pressure 后采集到：

```text
KV pool: primaryBlocks=32 secondaryBlocks=28
offload count=10440
  create_ms p50=0.765 p99=1.117 p9999=1.783 max=2.311
  d2h_ms    p50=0.493 p99=0.700 p9999=1.475 max=2.087
  set_ms    p50=0.839 p99=1.120 p9999=1.481 max=10.906
  total_ms  p50=2.200 p99=2.796 p9999=4.198 max=12.334
onboard count=0
```

结论：

- `Set` 路径已有 `10440` 个样本，p99 仍约 `1.12ms`；出现一次
  `set_ms=10.906ms` 尾部尖峰，后续 A/B 需保留 max 和原始 JSON
- 原 replay 固定重放最早的 pressure 历史；secondary pool 只有 `28` 个 block，
  增大 pressure 反而会使这些前缀更早被回收，无法稳定触发 `Get`
- `scripts/test_trt_cpp_kv_offload.py` 新增 `--replay-source-count` 和
  `--replay-tail-offset`，默认循环最近一小组 pressure 历史；历史生成器改为
  可逆 32 位混合，避免每 `256` 个 seed 重复，可构造十万级不同请求

### 远程 DataSystem 第三轮实测：Get/onboard 样本补齐

使用近期 pressure 历史循环 replay 后，结构化 trace 判定通过：

```text
PASS: C++ KV offload and DataSystem onboard were both observed.

KV pool: primaryBlocks=32 secondaryBlocks=28
offload count=8237
  set_ms    p50=0.780 p99=1.202 p9999=10.793 max=10.848
  total_ms  p50=2.101 p99=2.760 p9999=12.028 max=12.173
onboard count=177
  get_ms    p50=0.725 p95=1.520 p99=1.709 p9999=1.960 max=1.963
  h2d_ms    p50=0.492 p95=0.607 p99=0.618 p9999=0.624 max=0.624
  total_ms  p50=1.258 p95=2.027 p99=2.291 p9999=2.471 max=2.471
```

其中 replay/onboard wave 单独贡献 `174` 次 onboard，说明新的 replay 策略可以
稳定进入 DataSystem `Get + H2D` 读路径。`Get` 的 p50/p95/p99 已具备阶段性参考
价值；读路径只有 `177` 个样本，p9999 仍接近 max，仅作为观察值。

### Host/device 指标边界与 onboard 专项扩样

已展开 TensorRT-LLM runtime 实现：

```text
create_ms  host steady_clock 包围 DataSystem Create API
set_ms     host steady_clock 包围 DataSystem Set API
get_ms     host steady_clock 包围 DataSystem Get API
d2h_ms     host steady_clock 包围同步 cudaMemcpySanitized/cudaMemcpy DeviceToHost
h2d_ms     host steady_clock 包围同步 cudaMemcpySanitized/cudaMemcpy HostToDevice
total_ms   host steady_clock 包围完整 offload/onboard 流程
```

因此：

- `create_ms/set_ms/get_ms` 是 host 侧 DataSystem API wall-clock
- `d2h_ms/h2d_ms` 是 host 观察到的同步 CPU↔GPU 传输完成耗时，包含 PCIe/DMA
  等待，但不是 CUDA event 计出的纯 device 时间
- 当前没有采集 GPU kernel 或 CUDA-event device-only 指标

`scripts/test_trt_cpp_kv_offload.py` 已增加：

- `--min-onboard-samples`：结构化 onboard trace 少于目标数量时返回失败
- 报告开头逐项打印 metric scope
- 报告结尾提示 p9999 样本门槛：`10000` 是最低尾部观测门槛，稳定结论建议
  `>=100000`

远程下一轮先采集 `>=10000` 个 onboard 样本：

```text
python scripts/test_trt_cpp_kv_offload.py \
  --log /tmp/server_v4.log \
  --warmup-requests 0 \
  --requests 360 --repeat-requests 10000 \
  --replay-source-count 4 --replay-tail-offset 1 \
  --min-onboard-samples 10000 \
  --json-output /tmp/cppkv-datasystem-onboard-10k.json
```

### 远程 DataSystem 第四轮实测：onboard 10k+

onboard 专项 replay 已完成，达到最低 p9999 尾部观测门槛：

```text
replay/onboard wave HTTP:
  count=10000 p50=694ms p95=727ms p99=749ms p9999=832ms max=1552ms

all measured phases:
offload count=166089
  create_ms p50=0.665 p99=0.990 p9999=1.637 max=10.910
  d2h_ms    p50=0.483 p99=0.675 p9999=0.836 max=1.150
  set_ms    p50=0.756 p99=1.084 p9999=1.816 max=11.055
  total_ms  p50=1.937 p99=2.582 p9999=11.877 max=12.513
onboard count=14208
  get_ms    p50=0.740 p99=1.037 p9999=1.570 max=10.801
  h2d_ms    p50=0.407 p99=0.599 p9999=0.804 max=1.021
  total_ms  p50=1.177 p99=1.546 p9999=2.121 max=11.272
```

结论：

- onboard 已有 `14208` 个样本，可观察 p9999；若需要稳定的正式 p9999
  结论，仍建议扩大到 `>=100000`
- onboard 极端 max `11.272ms` 主要来自 `Get max=10.801ms`，不是 H2D；
  `h2d_ms max=1.021ms`
- replay 请求 E2E p50 为 `694ms`，单 block onboard p50 为 `1.177ms`；
  DataSystem 单块读回不是当前推荐请求的主耗时，主耗时仍在 8 轮 TRT runner
- offload 已有 `166089` 个样本，写路径 p9999 具备更好的统计支撑

### TRT runner 耗时根因与减轮数 A/B

当前每个 TRT miss 请求会串行调用 `8` 次 `ModelRunnerCpp.generate()`，每轮最多
生成 `32` 个 token，再合并 token 池做语义四元组组合。端到端基线中：

```text
runner_generate_ms avg=675.2ms
runner calls per request=8
single runner.generate avg≈84.4ms
```

8 轮不是模型推理的硬要求，而是当前无约束解码下为提高候选覆盖率采用的召回
策略。减少轮数预计近似线性降低 runner 耗时，但可能使可映射 item 数下降。

已增加：

- 服务参数 `--trt_num_samples`，环境变量 `TRT_NUM_SAMPLES`，默认 `8`
- TRT trace：`runner_calls`、`runner_avg_ms`、`runner_max_ms`
- `scripts/benchmark_e2e_latency.py` 汇总上述字段
- `scripts/test_trt_cpp_kv_offload.py` 输出每个 phase 的 `full_topk`、item 数
  p50/p95/min/max，便于直连 TRT 做质量-时延 A/B

粗略预估，固定开销按约 `23ms` 计算：

```text
8 rounds: runner≈675ms, request≈698ms  当前基线
4 rounds: runner≈338ms, request≈361ms  待实测
2 rounds: runner≈169ms, request≈192ms  待实测
1 round:  runner≈ 84ms, request≈107ms  待实测
```

远程完整 A/B 已回传，`/health` 与服务 TRACE 均确认轮数生效：

```text
sample health kv_exit calls ok/topk http_p50 http_p99 runner_avg runner_max
1      True   2       1     60/60  91.8     150.7    85.8       177.7
2      True   2       2     60/60  180.1    247.7    85.4       178.6
4      True   2       4     60/60  350.2    423.0    84.8       178.4
8      True   2       8     60/60  701.7    795.6    83.9       178.8
```

结论：

- 对当前直连 TRT、`topk=5`、60 请求样本，1/2/4/8 轮均没有出现 top-k 返回不足
- 1 轮相对 8 轮 HTTP p50 从 `701.7ms` 降到 `91.8ms`，下降约 `87%`
- 单次 `runner.generate()` 平均稳定在 `83.9-85.8ms`，说明主耗时几乎完全随轮数线性缩放
- `kv_exit=2` 来自底层 C++ KV/DataSystem verifier 未通过；该组结果只作为 runner
  轮数 A/B，不作为 DataSystem offload/onboard 结论

远程按 `TRT_NUM_SAMPLES=1/2/4/8` 分别重启服务，再执行相同直连请求序列：

```text
python scripts/test_trt_cpp_kv_offload.py \
  --log /tmp/server_samplesN.log \
  --warmup-requests 0 --requests 60 --repeat-requests 0 \
  --topk 5 --json-output /tmp/runner-samplesN.json
```

脚本默认使用带时间戳的唯一 `user_id` 前缀，避免重复运行命中 Python 结果缓存。
优先比较 `full_topk` 比例和 HTTP p50/p99。默认值暂不下调，需根据远程 A/B
结果选择。

已补充自动编排脚本，推荐后续直接使用：

```text
python scripts/benchmark_trt_runner_samples.py \
  --samples 1,2,4,8 \
  --server-cmd 'python -m inference.trt_llm.server ...' \
  --requests 60 --repeat-requests 0 --topk 5
```

如果需要清理旧服务，先在脚本外单独执行 `pkill -f "inference.trt_llm.server" || true`。
不要把这条 `pkill -f` 放进 `--stop-command`：benchmark 进程自己的
`--server-cmd` 参数也包含 `inference.trt_llm.server`，会被误杀并显示
`Terminated`。

脚本会逐轮注入 `TRT_NUM_SAMPLES`，等待 `/health` 返回匹配的
`trt_num_samples`，运行 `scripts/test_trt_cpp_kv_offload.py`，解析服务 TRACE
中的 `runner_calls/runner_avg_ms/runner_max_ms`，最后输出
`/tmp/trt_runner_samples_ab.json` 和 `/tmp/trt_runner_samples_ab.csv`。
默认情况下，runner A/B 脚本不会因为底层 C++ KV/DataSystem verifier 返回非零而
失败；`kv_exit` 会保留在汇总中作为诊断字段。若需要严格验证 C++ KV/DataSystem，
显式加 `--fail-on-kv-verdict`。

### TRT_NUM_SAMPLES=1 PaiRec E2E 实测

从 PaiRec `:18080` 入口使用 `TRT_NUM_SAMPLES=1`、`size=5`、60 个请求完成
端到端压测：

```text
Client: avg=9.4ms p50=3.5ms p95=90.5ms p99=92.8ms max=94.6ms
PaiRec total: avg=8.4ms p50=3.0ms p95=89.0ms p99=91.6ms max=94.0ms
GenerativeRecall http_ms: avg=8.1ms p50=2.0ms p95=89.0ms p99=91.2ms
TRT trace tr_total_ms: avg=7.5ms p50=1.6ms p95=88.2ms p99=90.4ms
TRT trace tr_runner_ms: avg=5.4ms p50=0.0ms p95=80.0ms p99=81.6ms
ok=60 fail=0
```

该报告混合了推荐结果缓存命中和少量冷 miss。p50 主要代表缓存命中路径；p95/p99
代表仍需进入 TRT runner 的冷 miss，符合 1 轮 runner 约 `84ms` 的预期。由于 TRT
服务日志未按 `request_id` 关联上，`scripts/benchmark_e2e_latency.py` 已补充
item 完整率和基于 GenerativeRecall `tr_result_cache_source` 的缓存分组 fallback。

后续重跑同一个 E2E 脚本时，会追加：

```text
== DataSystem C++ KV block stages ==
  scope: per KV block, from TRT-LLM C++ [Datasystem][TRACE], host wall-clock
  offload.create_ms / offload.d2h_ms / offload.set_ms / offload.total_ms
  onboard.get_ms / onboard.h2d_ms / onboard.total_ms
```

该部分是 per-block C++ KV 传输指标，不是 per-request 请求指标；需要与上面的
请求级 E2E 阶段并排解读。

后续可实验将多轮 prompt 批量提交给 `ModelRunnerCpp.generate()`。当前本地
TensorRT-LLM API 对整批默认共用一个 sampling config，而现有逻辑依赖每轮不同
seed；批量化需要先验证候选多样性，不能直接替换串行循环。

## 下一步

`dev` 已作为端到端阶段性基线保留。后续在专用分支开展 C++ DataSystem A/B：

1. 分开采集 `TRT_NUM_SAMPLES=1` 的纯 cold/miss E2E 与缓存命中 E2E，避免一个报告里 p50/p99 语义混杂
2. 单独恢复 DataSystem C++ KV verifier：当前 runner A/B 中 `kv_exit=2`，不能作为 DataSystem 结论
3. 如需稳定的正式 onboard p9999 结论，再扩大到 `>=100000` 个 onboard 样本
4. 推理服务增加实验开关，关闭 TRT Python 结果缓存和无效的 Python KV Cache 查询，确保请求进入 C++ runner
5. TensorRT-LLM runtime 恢复 KV 配置参数化，确保两组使用相同 primary / secondary block 数
6. 基于同一份 engine 构建原生 pinned DRAM baseline，与当前 DataSystem runtime 对照
7. 后续独立优化 fallback JSON 启动预加载
