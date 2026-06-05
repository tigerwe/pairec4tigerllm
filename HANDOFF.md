# 交接日志

> 供 Agent 跨 session 恢复上下文。只记录关键决策和当前任务。

## 最近一次交接 (2026-06-02)

### 当前任务
F12 推荐系统各阶段时延分析 — `dev` 已固化为端到端阶段性基线。下一阶段在专用分支开展 TRT-LLM C++ DataSystem 与原生 pinned DRAM runtime 的端到端 A/B。

### 6/2 C++ DataSystem 时延观测补充

已新增 `trtllm-datasystem-latency-trace.patch`，不直接修改存在历史未提交变更的
`/home/vivwimp/TensorRT-LLM` 源码树。patch 为
`kvCacheTransferManager.cpp` 增加两类结构化日志：

```text
op=offload create_ms=... d2h_ms=... set_ms=... total_ms=...
op=onboard get_ms=... h2d_ms=... total_ms=...
```

`scripts/test_trt_cpp_kv_offload.py` 已按 warmup / pressure / replay / overall 汇总
DataSystem C++ trace，输出 `avg/p50/p95/p99/p9999/max` 并支持 `--json-output`。
`scripts/benchmark_e2e_latency.py` 的 PaiRec E2E 与各阶段统计也已补 `p9999`；
缓存分组补齐 `p99/p9999/max`。

本机已通过：

```text
git -C /home/vivwimp/TensorRT-LLM apply --check \
  /home/vivwimp/pairec4tigerllm/trtllm-datasystem-latency-trace.patch
python -m py_compile scripts/benchmark_e2e_latency.py scripts/test_trt_cpp_kv_offload.py
# synthetic DataSystem TRACE parser assertions: OK
```

下一步必须在隔离 TRT-LLM 源码副本应用 patch、构建 runtime，再到远程 L40S
采集真实 `Set/Get` 与 E2E 指标。小样本 `p9999` 仅供观察，不应作为正式结论。

### 6/2 DataSystem runtime 首轮实测

远程已应用 C++ trace patch 并完成首轮采集：

```text
primaryBlocks=32 secondaryBlocks=28
offload count=4181
  create_ms p50=0.680 p99=1.025 max=2.171
  d2h_ms    p50=0.413 p99=0.691 max=0.732
  set_ms    p50=0.759 p99=1.093 max=1.381
  total_ms  p50=1.937 p99=2.553 max=4.006
onboard count=1
  get_ms=1.562 h2d_ms=0.345 total_ms=1.961
```

首轮结论：

- `Set` 路径样本充足，DataSystem C++ offload 写路径稳定
- `Get` 路径已打通，但只有 1 个样本，必须增加 onboard 命中后再评价读延迟
- 当前服务输出 INFO 日志，旧版验证脚本依赖的 DEBUG 日志不可见，因此会误报
  `KV cache block reuse was not confirmed`；脚本已改为优先使用
  `[Datasystem][TRACE] op=offload/onboard` 判定

### 6/2 DataSystem runtime 第二轮实测与 replay 修正

扩大 pressure 后采集到 `offload=10440`，其中单个 3.5 MiB block：

```text
create_ms p50=0.765 p99=1.117 p9999=1.783 max=2.311
d2h_ms    p50=0.493 p99=0.700 p9999=1.475 max=2.087
set_ms    p50=0.839 p99=1.120 p9999=1.481 max=10.906
total_ms  p50=2.200 p99=2.796 p9999=4.198 max=12.334
onboard count=0
```

`Set` p99 仍稳定，但出现一次 `10.906ms` 尾部尖峰。原 replay 固定取最早的
pressure 历史，在 `secondaryBlocks=28` 时增大 pressure 反而会使前缀更早被回收。

`scripts/test_trt_cpp_kv_offload.py` 已调整：

- `--replay-source-count`：循环重放多少个 pressure 历史，默认 `4`
- `--replay-tail-offset`：从 pressure 尾部跳过多少个最新历史，默认 `0`
- 历史生成器使用可逆 32 位混合，避免旧实现每 `256` 个 seed 重复，可构造
  十万级不同请求

远程下一轮先执行：

```text
 python scripts/test_trt_cpp_kv_offload.py \
  --log /tmp/server_v4.log \
  --requests 360 --repeat-requests 128 \
  --replay-source-count 4 --replay-tail-offset 1 \
  --strict-onboard --json-output /tmp/cppkv-datasystem.json
```

### 6/2 DataSystem runtime 第三轮实测：Get/onboard 样本补齐

新的 replay 策略已在远程验证通过：

```text
PASS: C++ KV offload and DataSystem onboard were both observed.

offload count=8237
  set_ms    p50=0.780 p99=1.202 p9999=10.793 max=10.848
  total_ms  p50=2.101 p99=2.760 p9999=12.028 max=12.173
onboard count=177
  get_ms    p50=0.725 p95=1.520 p99=1.709 p9999=1.960 max=1.963
  h2d_ms    p50=0.492 p95=0.607 p99=0.618 p9999=0.624 max=0.624
  total_ms  p50=1.258 p95=2.027 p99=2.291 p9999=2.471 max=2.471
```

replay/onboard wave 单独贡献 `174` 次 onboard。C++ DataSystem `Create/D2H/Set`
和 `Get/H2D` 两条路径的结构化耗时均已采集。读路径只有 `177` 个样本，p9999
接近 max，仅作观察。

### 6/2 Host/device 指标边界与 onboard 专项扩样

已确认 DataSystem trace 中所有指标均由 host `steady_clock` 采集：

```text
create_ms/set_ms/get_ms  host 侧 DataSystem API wall-clock
d2h_ms/h2d_ms            host 计时的同步 cudaMemcpy 完成耗时
total_ms                 host 侧完整 offload/onboard wall-clock
```

底层 `BufferManager::offLoadCopy/onBoardCopy` 调用
`cudaMemcpySanitized`，最终执行同步 `cudaMemcpy`。因此 `d2h_ms/h2d_ms` 包含
CPU↔GPU 传输完成等待，但不是 CUDA event 计出的纯 device 时间；当前没有采集
GPU kernel 或 device-only 指标。

`scripts/test_trt_cpp_kv_offload.py` 已增加：

- `--min-onboard-samples`：结构化 onboard trace 未达到目标时返回失败
- 报告逐字段输出 metric scope
- p9999 样本提示：`10000` 是最低尾部观测门槛，稳定结论建议 `>=100000`

下一轮先采集 `>=10000` 个 onboard：

```text
python scripts/test_trt_cpp_kv_offload.py \
  --log /tmp/server_v4.log \
  --warmup-requests 0 \
  --requests 360 --repeat-requests 10000 \
  --replay-source-count 4 --replay-tail-offset 1 \
  --min-onboard-samples 10000 \
  --json-output /tmp/cppkv-datasystem-onboard-10k.json
```

### 6/2 DataSystem runtime 第四轮实测：onboard 10k+

onboard 专项 replay 已完成：

```text
replay HTTP: count=10000 p50=694ms p95=727ms p99=749ms p9999=832ms max=1552ms

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

- onboard 已达到 `14208` 个样本，可观察 p9999；稳定结论仍建议 `>=100000`
- onboard 极端 max 来自 `Get max=10.801ms`，不是 H2D
- 单 block onboard p50 `1.177ms`，相对 replay E2E p50 `694ms` 较小；主耗时
  仍在 8 轮 TRT runner

### 6/2 TRT runner 耗时根因与减轮数 A/B

当前每个 TRT miss 请求串行执行 `8` 次 `ModelRunnerCpp.generate()`，每轮最多
生成 `32` 个 token。基线 `runner_ms=675.2ms`，即单轮约 `84.4ms`。8 轮是
提高无约束解码候选覆盖率的策略，不是引擎硬限制。

已增加：

- `--trt_num_samples` / `TRT_NUM_SAMPLES`，默认 `8`
- trace：`runner_calls`、`runner_avg_ms`、`runner_max_ms`
- `scripts/test_trt_cpp_kv_offload.py` 输出 `full_topk` 和 item 数分位值
- `scripts/benchmark_trt_runner_samples.py` 自动编排 `1/2/4/8` 轮 A/B，核验
  `/health` 的 `trt_num_samples`，解析 `runner_calls`，输出 JSON/CSV 汇总

远程按 `TRT_NUM_SAMPLES=1/2/4/8` 分别重启 TRT 服务。示例：

```text
export TRT_NUM_SAMPLES=4
pkill -f "inference.trt_llm.server" || true
python -m inference.trt_llm.server \
  --model_path ./checkpoints/decoder_qwen3/decoder_epoch_20.pt \
  --qwen3_model_path ./models/Qwen3-0.6B \
  --trt_engine_dir ./trt_engines/qwen3_rec_v4 \
  --port 18000 --device cuda \
  --datasystem_host 127.0.0.1 --datasystem_port 31501 \
  2>&1 | tee /tmp/server_samples4.log
```

另一个终端执行：

```text
python scripts/test_trt_cpp_kv_offload.py \
  --log /tmp/server_samples4.log \
  --warmup-requests 0 --requests 60 --repeat-requests 0 \
  --topk 5 --json-output /tmp/runner-samples4.json
```

比较各轮数的 HTTP p50/p99 和 `full_topk` 比例。脚本默认使用带时间戳的唯一
`user_id` 前缀，可避免重复运行命中 Python 结果缓存。默认轮数暂不修改。

完整 runner A/B 已回传，`/health` 与 `runner_calls` 均确认轮数生效：

```text
sample health kv_exit calls ok/topk http_p50 http_p99 runner_avg runner_max
1      True   2       1     60/60  91.8     150.7    85.8       177.7
2      True   2       2     60/60  180.1    247.7    85.4       178.6
4      True   2       4     60/60  350.2    423.0    84.8       178.4
8      True   2       8     60/60  701.7    795.6    83.9       178.8
```

结论：当前直连 TRT、`topk=5`、60 请求样本下，1 轮已经 `full_topk=60/60`。
1 轮相对 8 轮 HTTP p50 从 `701.7ms` 降到 `91.8ms`，可作为下一轮 PaiRec
E2E 验证候选。`kv_exit=2` 来自 C++ KV/DataSystem verifier 未通过，不能作为
DataSystem offload/onboard 结论。

也可以直接用自动编排脚本：

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

输出：

```text
/tmp/trt_runner_samples_ab.json
/tmp/trt_runner_samples_ab.csv
```

默认情况下，runner A/B 脚本不会因为底层 C++ KV/DataSystem verifier 返回非零而
失败；`kv_exit` 会保留在汇总中作为诊断字段。若需要严格验证 C++ KV/DataSystem，
显式加 `--fail-on-kv-verdict`。

### 6/4 TRT_NUM_SAMPLES=1 PaiRec E2E

从 PaiRec `:18080` 入口完成 `TRT_NUM_SAMPLES=1` 端到端压测，60 请求全成功：

```text
Client: avg=9.4ms p50=3.5ms p95=90.5ms p99=92.8ms max=94.6ms
PaiRec total: avg=8.4ms p50=3.0ms p95=89.0ms p99=91.6ms max=94.0ms
GenerativeRecall http_ms: avg=8.1ms p50=2.0ms p95=89.0ms p99=91.2ms
TRT trace tr_total_ms: avg=7.5ms p50=1.6ms p95=88.2ms p99=90.4ms
TRT trace tr_runner_ms: avg=5.4ms p50=0.0ms p95=80.0ms p99=81.6ms
ok=60 fail=0
```

注意：这组报告混入推荐结果缓存命中，p50 主要是缓存路径；p95/p99 是少量冷 miss
进入 1 轮 TRT runner 的路径。`scripts/benchmark_e2e_latency.py` 已补充 item 完整率
和基于 GenerativeRecall `tr_result_cache_source` 的缓存分组 fallback，后续重跑可更清楚
区分 `miss/hbm_hit/ds_hit`。

`scripts/benchmark_e2e_latency.py` 也已合并 DataSystem C++ KV block 统计。重跑 E2E
时同一份报告会追加：

```text
== DataSystem C++ KV block stages ==
  offload.create_ms / offload.d2h_ms / offload.set_ms / offload.total_ms
  onboard.get_ms / onboard.h2d_ms / onboard.total_ms
```

该部分来自 TRT-LLM C++ `[Datasystem][TRACE]`，是 per-block host wall-clock，不是
per-request 指标。

首轮减轮数远程结果已回传。根据请求 p50 接近预测值，推断为 `TRT_NUM_SAMPLES=4`；
仍需通过 `/health` 的 `trt_num_samples` 或服务 TRACE 的 `runner_calls=4` 确认：

```text
pressure requests=60 concurrency=1
HTTP: avg=360ms p50=354ms p95=375ms p99=430ms max=495ms
items: full_topk=60/60 min=5 p50=5 p95=5 max=5
DataSystem offload blocks=508 total p50=2.079ms p99=3.041ms max=3.413ms
```

相对 8 轮请求约 `698ms` 基线，4 轮请求 p50 约下降 `49%`。该轮样本只用于轮数
A/B，其 p9999 不作为正式尾延迟结论。

批量提交多个相同 prompt 是后续可实验方向，但当前 `ModelRunnerCpp.generate()`
对整批默认共用 sampling config，现有逻辑依赖每轮不同 seed；需要先验证候选
多样性再替换串行循环。

### 6/1 阶段性收口

当前端到端链路已经阶段性完善：

```text
PaiRec :18080
  → GenerativeRecall
  → Python TRT 服务 :18000
  → TensorRT-LLM ModelRunnerCpp
  → semantic_id 映射
  → generative_recall items
```

已验证：
- PaiRec API 可返回 `code=200` 和完整推荐结果
- Kafka 实时特征用户单次触发一次 TRT 推理，不再因 500ms 超时重复请求
- PaiRec、GenerativeRecall、Python TRT 服务和 TRT runner 的第一版结构化 trace 可通过 `request_id` 关联
- 冷请求分解已确认：稳态 PaiRec 约 `687ms`，TRT 约 `685.3ms`，8 轮 `runner_generate` 占 TRT 内部约 `91.5%`

尚未完成：
- Python 结果缓存会掩盖 TRT-LLM C++ KV 路径，需要实验开关彻底绕过
- TensorRT-LLM C++ DataSystem 与原生 pinned DRAM runtime 尚未做严格 A/B

下一阶段原则：
- 不重建 `rank0.engine`，两组加载同一 engine
- 主对比为原生 pinned DRAM baseline 与 DataSystem runtime
- 两组使用同一 TRT-LLM 源码基线、相同 FMHA 兼容修复、相同 KV block 配置和请求序列
- 从 PaiRec `:18080` 执行 `preload + warmup + pressure + replay`，pressure 和 replay 分开统计

外部 TensorRT-LLM 源码树 `/home/vivwimp/TensorRT-LLM` 当前存在历史未提交修改。后续应先使用独立 worktree 或容器副本构建两套 runtime，不要覆盖现有跑通环境。

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
- 汇总各阶段 p50/p95/p99/p9999/max
- 按结果缓存来源 `miss` / `hbm_hit` / `ds_hit` 分组
- percentile 使用线性插值，2 个样本的 p50 为中位数
- 可通过 `--uid-file` 构造大量不同 UID 的 pressure 流量
- 可通过 `--repeat-requests` + `--replay-source-count` 循环近期 pressure UID，端到端观察 C++ DataSystem onboard/Get
- 可按 `[pressure]` 和 `[replay/onboard]` 分别输出 C++ `[Datasystem][TRACE]` 的 `offload.set_ms` / `onboard.get_ms`
- PaiRec 的结构化日志使用 `glog`；启动脚本默认传入 `--alsologtostderr=true`，保留文件日志并确保 `tee /tmp/pairec.log` 能捕获 trace

远程冷请求分解已确认：
- fallback 用户 `142`、`211` 均为 TRT `miss`
- 首个 fallback 请求：PaiRec `3392ms`，其中 `history_ms=2573ms`；对应首次加载 `999447` 用户 fallback JSON
- 后续稳态冷请求：PaiRec `687ms`，其中 TRT 服务 `685.3ms`，PaiRec 额外开销约 `2ms`
- 两次 TRT 平均 `750.0ms`
- `runner_generate_ms` 平均 `686.2ms`，约占 TRT 内部 `91.5%`
- `output_pad_ms` 平均 `27.3ms`，`prompt_ms` 平均 `10.7ms`
- 下一步：启动阶段预加载 fallback JSON，并使用更多新用户采集稳态 `miss` p50/p95/p99；随后补 `hbm_hit` 和 `ds_hit`

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

默认期望看到 `[TRT parse] merged 8 rounds, layer_counts=[?,?,?,?]` 日志；
设置 `TRT_NUM_SAMPLES=N` 做 A/B 时应显示 `merged N rounds`。

### 运行时环境

```bash
source /opt/openEuler/gcc-toolset-14/enable
export LD_PRELOAD="\
/workspace/pairec4tigerllm/scripts/block_ds_consumer.so:\
/workspace/pairec4tigerllm/scripts/stub_gpu.so:\
/usr/local/lib/python3.11/site-packages/yr/datasystem/lib/libabseil_dll.so.2407.0.0"
```

### 下一步

1. F13 K8s 最小闭环配置已成型：`k8s/deployment-inference.yaml`、`k8s/deployment-pairec.yaml`、`k8s/configmap.yaml`、`k8s/service.yaml`、`docker/entrypoint-inference.sh`、`docker/entrypoint-pairec.sh` 已对齐当前 TRT/DataSystem/PaiRec 链路；执行步骤见 `k8s/README.md`
2. 推理 Pod：端口 `18000`，单 GPU，`Recreate`，挂载 `models/`、`checkpoints/`、`trt_engines/`、`data/`，传 `TRT_ENGINE_DIR`、`DATASYSTEM_HOST/PORT`、`TRT_NUM_SAMPLES`、`TRT_RESULT_CACHE_ENABLED`、`LD_PRELOAD`
3. PaiRec Pod：端口 `18080`，ConfigMap 注入真实 `pairec_config.json`，服务发现地址为 `http://inference:18000`，探针用框架实际存在的 `/ping`；Dockerfile 已改为 `go build -mod=vendor`
4. 本机验证已完成：entrypoint shell 语法 OK，K8s YAML 与内嵌 PaiRec JSON 解析 OK，`git diff --check -- docker k8s` OK，`go build -mod=vendor -ldflags "-s -w" -o /tmp/pairec-server-k8s ./services/main.go` OK
5. 远程 K8s 下一步：基于已验证 TensorRT-LLM/DataSystem runtime 准备 `registry.example.com/pairec-inference:trt-datasystem` 镜像，构建 `pairec-server:k8s`，准备 PVC 目录后按 `k8s/README.md` 顺序 `kubectl apply`，验证 inference `/health` 与 PaiRec `/api/recommend`
6. F12 后续仍保留：DataSystem vs pinned DRAM A/B；remote H2D 暂列遗留，需后续改 C++ ConnectOptions 和 onboard 路径；如需稳定发布 p9999，再扩到 `>=100000` onboard events
