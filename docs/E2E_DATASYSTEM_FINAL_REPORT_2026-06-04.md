# PaiRec 端到端 DataSystem Set/Get 时延报告

> 日期：2026-06-04  
> 分支：`cpp-datasystem-ab-latency`  
> 目标：在 PaiRec 端到端请求中关闭 Python TRT 推荐结果缓存，构造 pressure/replay 流量，同时观测请求级阶段耗时和 TensorRT-LLM C++ KV block DataSystem `Set/Get` 耗时。

## 1. 实验配置

| 参数 | 值 |
|---|---:|
| PaiRec endpoint | `http://127.0.0.1:18080/api/recommend` |
| TRT service | `http://127.0.0.1:18000` |
| `TRT_NUM_SAMPLES` | `1` |
| `TRT_RESULT_CACHE_ENABLED` | `0` |
| `TRT_MAX_KV_TOKENS` | `1024` |
| concurrency | `1` |
| pressure requests | `2000` |
| replay/onboard requests | `10000` |
| replay source count | `16` |
| expected item size | `3` |
| successful responses | `11853/12000` |
| success rate | `98.78%` |

说明：

- `TRT_RESULT_CACHE_ENABLED=0` 已生效，TRT result cache group 显示 `disabled count=11804`。
- `size=3` 用于降低 `TRT_NUM_SAMPLES=1` 下候选不足导致的 `code=299`，成功响应均返回完整 `3` 个 item。
- DataSystem C++ 指标来自 TensorRT-LLM C++ `[Datasystem][TRACE]`，是 per KV block 级别的 host wall-clock，不是 per request 指标。
- `Create/Set/Get` 是 host DataSystem API wall-clock；`D2H/H2D` 是 host 计时的同步 cudaMemcpy 完成耗时；当前没有 CUDA event device-only 指标。

## 2. 总体结论

这轮测试已经在端到端链路中同时观测到 DataSystem C++ KV offload 和 onboard：

| C++ 路径 | 样本数 | 核心指标 | p50 | p95 | p99 | p9999 | max |
|---|---:|---|---:|---:|---:|---:|---:|
| offload | 29330 | `set_ms` | `0.746ms` | `0.910ms` | `1.015ms` | `1.689ms` | `10.769ms` |
| offload | 29330 | `total_ms` | `1.870ms` | `2.256ms` | `2.388ms` | `3.445ms` | `11.648ms` |
| onboard | 13421 | `get_ms` | `0.623ms` | `0.747ms` | `0.818ms` | `1.094ms` | `10.691ms` |
| onboard | 13421 | `total_ms` | `0.992ms` | `1.178ms` | `1.293ms` | `2.116ms` | `11.078ms` |

请求级端到端耗时仍主要由单轮 TRT runner 决定：

| 指标 | count | p50 | p95 | p99 | p9999 | max |
|---|---:|---:|---:|---:|---:|---:|
| `client_e2e_ms` | 11853 | `92.8ms` | `96.6ms` | `103.2ms` | `119.9ms` | `125.2ms` |
| `tr_total_ms` | 11853 | `90.7ms` | `94.5ms` | `101.0ms` | `114.6ms` | `114.7ms` |
| `tr_runner_ms` | 11853 | `84.9ms` | `88.8ms` | `95.4ms` | `107.6ms` | `109.1ms` |
| `tr_kv_lookup_ms` | 11853 | `1.3ms` | `1.6ms` | `1.7ms` | `2.4ms` | `2.4ms` |

结论：

- 端到端 cold path p50 约 `93ms`，p99 约 `103ms`。
- TRT runner p50 约 `85ms`，仍是主要瓶颈。
- DataSystem 单 block `Set` p50 约 `0.75ms`，p99 约 `1.02ms`。
- DataSystem 单 block `Get` p50 约 `0.62ms`，p99 约 `0.82ms`。
- `Set/Get` 都存在约 `10ms` 级别 max 尾部尖峰，p9999 和 max 需要在报告中保留。

## 3. 请求级阶段耗时

### 3.1 Client / PaiRec

| metric | count | avg | p50 | p95 | p99 | p9999 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `client_e2e_ms` | 11853 | `93.3` | `92.8` | `96.6` | `103.2` | `119.9` | `125.2` |
| `pairec.total_ms` | 11853 | `92.3` | `92.0` | `96.0` | `102.0` | `115.8` | `116.0` |
| `pairec.recall_ms` | 11853 | `92.2` | `92.0` | `96.0` | `102.0` | `115.8` | `116.0` |

PaiRec 内部耗时几乎全部在 `recall_ms`，其它阶段在当前毫秒级统计下为 `0.0ms`。

### 3.2 GenerativeRecall / TRT

| metric | count | avg | p50 | p95 | p99 | p9999 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tr_total_ms` | 11853 | `91.2` | `90.7` | `94.5` | `101.0` | `114.6` | `114.7` |
| `tr_infer_ms` | 11853 | `89.4` | `88.8` | `92.6` | `99.0` | `111.9` | `112.7` |
| `tr_runner_ms` | 11853 | `85.9` | `84.9` | `88.8` | `95.4` | `107.6` | `109.1` |
| `tr_prompt_ms` | 11853 | `2.5` | `2.7` | `2.9` | `3.0` | `5.4` | `5.6` |
| `tr_pad_ms` | 11853 | `0.7` | `0.6` | `1.7` | `2.1` | `3.0` | `3.3` |
| `tr_map_ms` | 11853 | `0.1` | `0.1` | `0.2` | `0.2` | `0.8` | `0.8` |
| `tr_kv_lookup_ms` | 11853 | `1.3` | `1.3` | `1.6` | `1.7` | `2.4` | `2.4` |

TRT service 侧 correlated trace 数为 `11804`，与 GenerativeRecall 成功请求数略有差异；未关联的请求被归入 `unknown`，不影响已关联请求的阶段统计。

### 3.3 TRT Result Cache

| cache group | count | total p50 | total p95 | total p99 | total p9999 | max |
|---|---:|---:|---:|---:|---:|---:|
| `disabled` | 11804 | `90.7ms` | `94.5ms` | `101.0ms` | `114.6ms` | `114.7ms` |
| `unknown` | 49 | `0.0ms` | `0.0ms` | `0.0ms` | `0.0ms` | `0.0ms` |

这证明 Python TRT 推荐结果缓存已关闭，当前 E2E 统计代表 cold runner path，而不是 Python `hbm_hit`。

## 4. DataSystem C++ KV Block 耗时

### 4.1 All Measured Phases

| metric | count | avg | p50 | p95 | p99 | p9999 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `offload.create_ms` | 29330 | `0.607` | `0.605` | `0.758` | `0.833` | `1.578` | `2.758` |
| `offload.d2h_ms` | 29330 | `0.488` | `0.468` | `0.727` | `0.751` | `1.173` | `1.259` |
| `offload.set_ms` | 29330 | `0.757` | `0.746` | `0.910` | `1.015` | `1.689` | `10.769` |
| `offload.total_ms` | 29330 | `1.887` | `1.870` | `2.256` | `2.388` | `3.445` | `11.648` |
| `onboard.get_ms` | 13421 | `0.632` | `0.623` | `0.747` | `0.818` | `1.094` | `10.691` |
| `onboard.h2d_ms` | 13421 | `0.343` | `0.336` | `0.434` | `0.515` | `1.382` | `1.416` |
| `onboard.total_ms` | 13421 | `1.009` | `0.992` | `1.178` | `1.293` | `2.116` | `11.078` |

### 4.2 Pressure Phase

| metric | count | avg | p50 | p95 | p99 | p9999 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `offload.create_ms` | 5996 | `0.661` | `0.661` | `0.787` | `0.869` | `1.413` | `1.574` |
| `offload.d2h_ms` | 5996 | `0.607` | `0.692` | `0.744` | `0.765` | `1.234` | `1.259` |
| `offload.set_ms` | 5996 | `0.779` | `0.767` | `0.915` | `1.018` | `1.792` | `1.811` |
| `offload.total_ms` | 5996 | `2.082` | `2.089` | `2.341` | `2.499` | `3.338` | `3.390` |
| `onboard.get_ms` | 87 | `0.731` | `0.712` | `0.866` | `0.909` | `0.972` | `0.973` |
| `onboard.total_ms` | 87 | `1.242` | `1.244` | `1.416` | `1.484` | `1.552` | `1.553` |

### 4.3 Replay/Onboard Phase

| metric | count | avg | p50 | p95 | p99 | p9999 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `offload.create_ms` | 23334 | `0.593` | `0.591` | `0.741` | `0.817` | `1.524` | `2.758` |
| `offload.d2h_ms` | 23334 | `0.458` | `0.464` | `0.547` | `0.739` | `1.107` | `1.164` |
| `offload.set_ms` | 23334 | `0.751` | `0.739` | `0.908` | `1.013` | `1.434` | `10.769` |
| `offload.total_ms` | 23334 | `1.837` | `1.829` | `2.163` | `2.345` | `3.689` | `11.648` |
| `onboard.get_ms` | 13334 | `0.631` | `0.623` | `0.745` | `0.816` | `1.095` | `10.691` |
| `onboard.h2d_ms` | 13334 | `0.342` | `0.336` | `0.420` | `0.510` | `1.382` | `1.416` |
| `onboard.total_ms` | 13334 | `1.008` | `0.992` | `1.171` | `1.282` | `2.116` | `11.078` |

replay/onboard 阶段贡献了绝大多数 Get 样本：`13334/13421`。因此当前 Get 结论主要来自 replay/onboard 设计，而不是 pressure 阶段偶发 onboard。

## 5. 失败请求说明

本轮总请求 `12000`，成功 `11853`，失败 `147`：

| phase | total | ok | fail | success rate |
|---|---:|---:|---:|---:|
| pressure | 2000 | 1976 | 24 | `98.80%` |
| replay/onboard | 10000 | 9877 | 123 | `98.77%` |
| all | 12000 | 11853 | 147 | `98.78%` |

失败均为：

```text
code=299 error=items size not enough
```

这表示 `TRT_NUM_SAMPLES=1` 下少量 UID 生成的有效推荐 item 不足 `size=3`，不是 DataSystem 或服务错误。成功响应的 `full_size=11853/11853`。

## 6. 样本量与 p9999 可信度

| 路径 | count | p9999 可信度 |
|---|---:|---|
| offload | 29330 | 达到最低观察门槛，可作为阶段性 p9999 |
| onboard | 13421 | 达到最低观察门槛，可作为阶段性 p9999 |

注意：p9999 对样本量非常敏感。当前 `onboard=13421` 可以给最低限度参考，但如果要稳定发布 p9999 结论，建议继续扩到 `>=100000` onboard events。

## 7. 结论

本轮端到端压测完成了目标：

- 在 PaiRec E2E 中同时观测到 C++ DataSystem `Set` 和 `Get`。
- 关闭 Python TRT 推荐结果缓存后，端到端 cold path p50 约 `93ms`，p99 约 `103ms`。
- 单轮 TRT runner p50 约 `85ms`，仍是当前主耗时。
- DataSystem C++ 单 block `Set` p50 约 `0.75ms`，p99 约 `1.02ms`。
- DataSystem C++ 单 block `Get` p50 约 `0.62ms`，p99 约 `0.82ms`。
- `Set/Get` max 均出现 `10ms+` 尾部尖峰，正式汇报应保留 `p9999/max`。

下一步若要做系统级优化，优先级仍应放在 TRT runner；若要做 DataSystem 存储路径 A/B，则应继续用同一 pressure/replay 脚本对比 DataSystem 与 pinned DRAM，并保持 `TRT_RESULT_CACHE_ENABLED=0`。

## 8. 远端 DataSystem Get 到本地测试

> 补充日期：2026-06-05
> 远端 DataSystem：`141.61.91.188:18581`
> 测试目标：将 DataSystem worker 放到远端，测量 TensorRT-LLM C++ KV onboard 时从远端 DataSystem `Get` 到本机 host buffer，再 H2D 到本机 GPU 的耗时。

### 8.1 测试配置与路径

远端启动日志确认两个 C++ DataSystem client 都连接到远端：

```text
[TensorRT-LLM][Datasystem] Init KvCache Manager DataSystem. host = 141.61.91.188, ip = 18581.
[TensorRT-LLM][Datasystem] Init KvCache TMP Manager DataSystem. host = 141.61.91.188, ip = 18581.
```

当前代码路径仍是：

```text
远端 DataSystem Get -> 本机 host buffer -> 本机 GPU H2D
```

不是 remote H2D。原因是当前 C++ 配置仍为：

```cpp
conn_opts.enableCrossNodeConnection = false;
conn_opts.enableRemoteH2D = false;
```

并且 onboard 路径仍显式执行：

```cpp
kvClient1->Get(key, buffer, 0);
mOnboardManager.onBoardCopy(*dstPtr, buffer->MutableData(), buffer->GetSize());
```

因此 `onboard.get_ms` 表示远端 DataSystem Get 到本机 host buffer 的耗时，`onboard.h2d_ms` 表示之后本机 host 到本机 GPU 的普通 H2D。

### 8.2 远端 DS 测试结论

本轮远端 DS 为中等样本规模：

| phase | total | ok | fail | success rate |
|---|---:|---:|---:|---:|
| pressure | 300 | 295 | 5 | `98.33%` |
| replay/onboard | 1000 | 995 | 5 | `99.50%` |
| all | 1300 | 1290 | 10 | `99.23%` |

请求级耗时显著上升：

| metric | count | p50 | p95 | p99 | p9999 | max |
|---|---:|---:|---:|---:|---:|---:|
| `client_e2e_ms` | 1290 | `1050.3ms` | `1687.2ms` | `1689.2ms` | `1690.9ms` | `1691.0ms` |
| `tr_total_ms` | 1290 | `1048.0ms` | `1684.7ms` | `1686.6ms` | `1688.6ms` | `1688.6ms` |
| `tr_runner_ms` | 1290 | `1034.2ms` | `1670.6ms` | `1671.9ms` | `1672.8ms` | `1672.8ms` |
| `tr_kv_lookup_ms` | 1290 | `9.7ms` | `10.3ms` | `10.5ms` | `10.6ms` | `10.6ms` |

远端 DataSystem C++ KV block 耗时：

| metric | count | avg | p50 | p95 | p99 | p9999 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `offload.set_ms` | 3202 | `314.491` | `314.149` | `315.288` | `315.610` | `325.289` | `328.397` |
| `offload.total_ms` | 3202 | `316.721` | `316.451` | `317.597` | `317.915` | `327.545` | `330.647` |
| `onboard.get_ms` | 1346 | `315.930` | `315.572` | `316.862` | `317.038` | `321.050` | `321.650` |
| `onboard.h2d_ms` | 1346 | `0.405` | `0.405` | `0.421` | `0.429` | `0.487` | `0.494` |
| `onboard.total_ms` | 1346 | `316.370` | `316.010` | `317.307` | `317.480` | `321.488` | `322.087` |

关键结论：

- 远端 DataSystem `Get` 到本机 host buffer 的 p50 为 `315.572ms`，p99 为 `317.038ms`。
- `onboard.h2d_ms` p50 仅 `0.405ms`，与本地 H2D 同量级，说明远端开销主要集中在 `onboard.get_ms`，不是本机 H2D。
- 请求级 p50 从本地 DS 的约 `92.8ms` 上升到远端 DS 的约 `1050.3ms`，主要原因是 runner 过程中触发大量远端 KV block `Set/Get`。
- 当前 `onboard=1346` 足够判断 p50/p95/p99 趋势；如需正式发布远端 DS p9999，建议扩到 `>=10000` onboard events，稳定 p9999 则建议 `>=100000` onboard events。

### 8.3 本地 DS vs 远端 DS 对比

| 指标 | 本地 DS p50 | 远端 DS p50 | 放大倍数 |
|---|---:|---:|---:|
| `client_e2e_ms` | `92.8ms` | `1050.3ms` | `~11.3x` |
| `tr_runner_ms` | `84.9ms` | `1034.2ms` | `~12.2x` |
| `offload.set_ms` | `0.746ms` | `314.149ms` | `~421x` |
| `onboard.get_ms` | `0.623ms` | `315.572ms` | `~507x` |
| `onboard.h2d_ms` | `0.336ms` | `0.405ms` | `~1.2x` |
| `onboard.total_ms` | `0.992ms` | `316.010ms` | `~319x` |

远端 DS 的 `Set/Get` 都稳定在 `315ms` 左右，这表明远端访问/网络/远端 DS 路径是主要新增开销。

### 8.4 遗留事项

1. remote H2D 暂未测试。

当前测到的是：

```text
远端 DS Get -> 本机 host buffer -> 本机 GPU H2D
```

如需测试 remote H2D，需要改造 TensorRT-LLM C++ DataSystem 接入：

```cpp
conn_opts.enableCrossNodeConnection = true;
conn_opts.enableRemoteH2D = true;
```

同时 onboard 路径不能继续只走 `Get -> buffer->MutableData() -> onBoardCopy`，需要确认并使用 DataSystem 直接写 GPU device pointer 的 API，新增类似 `remote_h2d_ms` 的 trace。

2. PaiRec timeout 需要随远端 DS 场景调大。

远端 DS 下请求级 p95/p99 已到 `1.68s` 左右。如果 PaiRec TRT HTTP client timeout 仍为 `1000ms`，会出现：

```text
context deadline exceeded (Client.Timeout exceeded while awaiting headers)
```

远端 DS 测试建议将 `timeout_ms` 调到 `5000ms`，并保持 `max_retries=1`，避免重试放大 GPU/DS 压力。

## 9. 指标口径详细说明

本节解释报告中所有主要 `*_ms` 指标的来源和含义。需要先区分三类统计粒度：

1. **请求级**：`client_e2e_ms`、PaiRec stages、GenerativeRecall stages、TRT service stages，单位是每个推荐请求。
2. **服务内阶段级**：`tr_*` 和 TRT service stages 来自 inference 服务返回的 `trace`，部分字段是同一底层字段在不同日志里的别名。
3. **KV block 级**：`offload.*_ms` 和 `onboard.*_ms` 来自 TensorRT-LLM C++ `[Datasystem][TRACE]`，单位是每个 KV block，不是每个请求。

所有指标均为 host wall-clock 计时：Go 侧使用 `time.Now()` / `time.Since()`，Python 侧使用 `time.perf_counter()`，C++ DataSystem trace 使用 host 计时。当前没有 CUDA event 的 device-only 指标。

### 9.1 Client 指标

| 指标 | 粒度 | 来源 | 含义 | 解读 |
|---|---|---|---|---|
| `client_e2e_ms` | request | `scripts/benchmark_e2e_latency.py` | 压测客户端从发起 `POST /api/recommend` 到收到 PaiRec 响应的完整 HTTP 往返耗时。 | 最接近用户侧体感延迟，包含客户端到 PaiRec、PaiRec 内部、PaiRec 调 inference、inference 内部和响应解析。 |

本地 DS 报告中 `client_e2e_ms p50=92.8ms`，而 PaiRec 服务端 `total_ms p50=92.0ms`，两者只差约 `0.8ms`，说明压测客户端到 PaiRec 的额外开销很小。

### 9.2 PaiRec recommend stages

这些字段来自 PaiRec Go 服务的 `RecommendTrace` 日志，描述 `/api/recommend` 在 PaiRec 内部的阶段耗时。

| 指标 | 粒度 | 含义 | 本报告中的判断 |
|---|---|---|---|
| `total_ms` | request | PaiRec 处理一次 `/api/recommend` 的服务端总耗时，不包含压测客户端侧网络开销。 | 本地 DS p50=`92.0ms`，几乎等于 `recall_ms`。 |
| `user_feature_ms` | request | 获取用户特征的耗时，可能来自实时特征、fallback 文件或上下文。 | 本轮为 `0.0ms`，不是瓶颈。 |
| `recall_ms` | request | PaiRec 召回阶段总耗时，包含 GenerativeRecall 调 inference。 | 几乎等于 `total_ms`，说明本轮 pipeline 的主要工作就是生成式召回。 |
| `filter_ms` | request | 过滤阶段耗时，例如过滤已看、黑名单或业务规则。 | 本轮为 `0.0ms`。 |
| `general_rank_ms` | request | 粗排/通用排序阶段耗时。 | 本轮为 `0.0ms`，没有明显排序负载。 |
| `feature_ms` | request | 排序或模型需要的特征加载耗时。 | 本轮为 `0.0ms`。 |
| `rank_ms` | request | 精排阶段耗时。 | 本轮为 `0.0ms`，当前主要测召回链路。 |
| `pipeline_wait_ms` | request | pipeline 并发等待耗时，例如等待多路召回或 rank stage。 | 本轮为 `0.0ms`。 |
| `merge_ms` | request | 多路召回结果合并耗时。 | 本轮为 `0.0ms`，主要只有一路 `generative_recall`。 |
| `sort_ms` | request | 最终排序耗时。 | 本轮为 `0.0ms`。 |

因此，本地 DS 下 PaiRec 侧几乎没有额外瓶颈：`total_ms ≈ recall_ms ≈ GenerativeRecall http_ms`。

### 9.3 GenerativeRecall stages

这些字段来自 `services/recall/generative_recall.go`，描述 PaiRec 自定义召回内部从取历史、转换、调用 inference 到转换返回 item 的耗时。

| 指标 | 粒度 | 来源/计算方式 | 含义 | 解读 |
|---|---|---|---|---|
| `cost` | request | `utils.CostTime(stageStart)` | GenerativeRecall 总耗时，单位 ms。 | 基本等于 PaiRec `recall_ms`。 |
| `cache_ms` | request | Go recall cache 查询计时 | GenerativeRecall 自己的 Go 侧缓存查询耗时。 | 本轮为 `0.0ms`，没有靠 Go recall cache 命中。 |
| `history_ms` | request | `getUserHistory()` 计时 | 获取用户历史行为耗时。 | 本轮为 `0.0ms`，历史获取不是瓶颈。 |
| `convert_ms` | request | `convertToSemanticIDs()` 计时 | 将用户历史 item 转成 semantic id 历史的耗时。 | 本轮为 `0.0ms`，转换成本很低。 |
| `http_ms` | request | `r.client.Recommend()` 外围计时 | GenerativeRecall 调 inference `/recommend` 的 HTTP 总耗时。 | 本地 DS p50=`92.0ms`，主要由 inference 服务耗时贡献。 |
| `items_ms` | request | `convertToItems()` 计时 | 将 inference 返回的 recommendations 转成 PaiRec item 的耗时。 | 本轮为 `0.0ms`。 |
| `http_overhead_ms` | request | `http_ms - response.Trace.TotalMs`，负数归零 | HTTP/JSON/Flask 路由/网络/排队等 inference 自报耗时之外的开销。 | 本地 DS p50 约 `1ms`，服务间 HTTP 开销很小。 |

`tr_*` 字段是 inference 服务 response 中 `trace` 字段的透传，GenerativeRecall 只是加了 `tr_` 前缀：

| GenerativeRecall 字段 | inference trace 字段 | 含义 |
|---|---|---|
| `tr_total_ms` | `total_ms` | inference 服务处理 `/recommend` 的总耗时。 |
| `tr_prepare_ms` | `prepare_input_ms` | 输入准备耗时。 |
| `tr_infer_ms` | `infer_ms` | 推理分支总耗时。 |
| `tr_prompt_ms` | `prompt_ms` | TRT prompt 构造、tokenizer 编码和历史截断耗时。 |
| `tr_runner_ms` | `runner_generate_ms` | TensorRT-LLM `ModelRunnerCpp.generate()` 累计耗时。 |
| `tr_parse_ms` | `parse_combo_ms` | TRT 输出 token 解析、semantic id 组合和去重耗时。 |
| `tr_pad_ms` | `output_pad_ms` | 将结果 padding 成统一 tensor 的耗时。 |
| `tr_map_ms` | `map_item_ms` | semantic id 映射回 item id 的耗时。 |
| `tr_kv_lookup_ms` | `kv_lookup_ms` | Python 层 KVCacheManager 查询耗时。 |
| `tr_kv_write_ms` | `kv_write_ms` | Python 层 KV cache 异步写入提交耗时。 |
| `tr_result_cache_lookup_ms` | `result_cache_lookup_ms` | Python 侧推荐结果 HBM LRU cache 查询耗时。 |
| `tr_result_cache_ds_lookup_ms` | `result_cache_ds_lookup_ms` | Python 侧推荐结果缓存从 DataSystem 查询耗时。 |
| `tr_result_cache_write_submit_ms` | `result_cache_write_submit_ms` | Python 侧推荐结果缓存异步写入提交耗时。 |

注意：`tr_kv_lookup_ms`、`tr_kv_write_ms` 和 `tr_result_cache_*` 都是 Python inference 服务层指标，不是 TensorRT-LLM C++ KV block 的 `offload/onboard` 指标。

### 9.4 TRT service stages

这些字段来自 inference 服务日志 `[TRACE]`，与上面的 `tr_*` 基本是同一批数据，只是没有 `tr_` 前缀。

| 指标 | 粒度 | 含义 | 解读 |
|---|---|---|---|
| `total_ms` | request | inference 服务处理 `/recommend` 的总耗时。 | 等价于 `tr_total_ms`。 |
| `prepare_ms` | request | 输入准备耗时，对应 `prepare_input_ms`。 | 包括构造模型输入、转 tensor、移动到 GPU、计算 history hash。 |
| `kv_lookup_ms` | request | Python 层 KVCacheManager 查询耗时。 | 不是 C++ KV block 的 Set/Get。 |
| `result_cache_lookup_ms` | request | Python 推荐结果 HBM LRU cache 查询耗时。 | 本报告关闭 result cache，因此为 `0.0ms`。 |
| `result_cache_ds_lookup_ms` | request | Python 推荐结果缓存从 DataSystem 查询耗时。 | 本报告关闭 result cache，因此为 `0.0ms`。 |
| `prompt_ms` | request | TRT backend prompt 构造、tokenizer 编码和历史截断耗时。 | 本地 DS p50 约 `2.7ms`。 |
| `runner_ms` | request | TensorRT-LLM `runner.generate()` 累计耗时。 | 本地 DS p50 约 `84.9ms`，是主瓶颈。 |
| `runner_calls` | request | 本请求内调用 `runner.generate()` 的次数。 | 本报告 `TRT_NUM_SAMPLES=1`，因此为 `1`。 |
| `runner_avg_ms` | request | `runner_ms / runner_calls`。 | 当前等于 `runner_ms`。 |
| `runner_max_ms` | request | 多次 `runner.generate()` 中最慢一次耗时。 | 当前 `runner_calls=1`，也等于 `runner_ms`。 |
| `parse_ms` | request | 输出 token 解析、semantic id 组合和去重耗时。 | 本轮接近 `0.0ms`。 |
| `map_ms` | request | semantic id 映射回 item id 的耗时。 | 本轮约 `0.1ms`。 |

本地 DS 下，`runner_ms p50=84.9ms`，`total_ms p50=90.7ms`，runner 约占 inference 总耗时的 `94%`。这就是报告判断“TRT runner 仍是主要瓶颈”的依据。

### 9.5 inference trace 内部字段补充

以下字段没有全部单独出现在 TRT service table 中，但会出现在 GenerativeRecall 的 `tr_*` 或 response trace 里：

| 字段 | 粒度 | 含义 |
|---|---|---|
| `prepare_input_ms` | request | inference 服务输入准备耗时，对应 `tr_prepare_ms` / `prepare_ms`。 |
| `infer_ms` | request | 推理分支总耗时，对应 `tr_infer_ms`。TRT path 下主要覆盖 TRT backend generate。 |
| `generate_ms` | request | 生成总耗时。TRT path 下通常等于 `backend_total_ms`。 |
| `model_forward_ms` | request | PyTorch forward path 的模型前向耗时。TRT path 下通常为 `0.0ms`。 |
| `backend_total_ms` | request | TRT backend `generate()` 端到端耗时，包含 prompt、runner、parse、pad。 |
| `map_item_ms` | request | semantic id 到 item id 的映射耗时。 |
| `kv_write_ms` | request | Python 层 KV cache store 的异步提交耗时；TRT C++ KV path 下通常为 `0.0ms`。 |

### 9.6 DataSystem C++ KV block stages

这些字段来自 TensorRT-LLM C++ `[Datasystem][TRACE]`。它们是 per KV block 指标，不是 per request 指标，不能直接与 `client_e2e_ms` 相加。一个请求可能触发多个 KV block 的 offload/onboard。

| 指标 | 粒度 | 含义 | 本地 DS 结论 |
|---|---|---|---|
| `offload.create_ms` | KV block | offload 时创建/准备 DataSystem object 或 buffer 的耗时。 | p50=`0.605ms`。 |
| `offload.d2h_ms` | KV block | GPU device KV block 拷贝到 host buffer 的耗时，即 GPU -> host。 | p50=`0.468ms`。 |
| `offload.set_ms` | KV block | 调用 DataSystem `Set` 将 host buffer 写入 DataSystem 的耗时。 | p50=`0.746ms`，p99=`1.015ms`。 |
| `offload.total_ms` | KV block | 一次 KV block offload 的总耗时，包含 create、D2H、Set 及少量 C++ 路径开销。 | p50=`1.870ms`。 |
| `onboard.get_ms` | KV block | 从 DataSystem `Get` 取回 KV block 到 host buffer 的耗时。 | p50=`0.623ms`，p99=`0.818ms`。 |
| `onboard.h2d_ms` | KV block | host buffer 拷贝回 GPU KV cache block 的耗时，即 host -> GPU。 | p50=`0.336ms`。 |
| `onboard.total_ms` | KV block | 一次 KV block onboard 的总耗时，包含 Get、H2D 及少量 C++ 路径开销。 | p50=`0.992ms`。 |

本地 DS 下，per request 近似触发量为：

```text
offload blocks/request ≈ 29330 / 11853 ≈ 2.47
onboard blocks/request ≈ 13421 / 11853 ≈ 1.13
```

这意味着本地 DS 的 KV 传输成本通常是每请求几毫秒量级，远小于 `tr_runner_ms p50=84.9ms`，因此本地 DS 不是端到端主瓶颈。

远端 DS 下，关键字段变化是：

```text
offload.set_ms p50 = 314.149ms
onboard.get_ms p50 = 315.572ms
onboard.h2d_ms p50 = 0.405ms
```

这说明远端慢点集中在 DataSystem `Set/Get` 到本机 host buffer 的路径，而不是本机 H2D。当前远端路径仍是：

```text
远端 DataSystem Get -> 本机 host buffer -> 本机 GPU H2D
```

还不是 remote H2D。

### 9.7 cache group 指标

| 指标/分组 | 含义 |
|---|---|
| `disabled` | Python TRT 推荐结果缓存已关闭，当前请求走 cold runner path。 |
| `hbm_hit` | Python 侧推荐结果 HBM LRU cache 命中。该场景会绕过 TRT runner。 |
| `ds_hit` | Python 侧推荐结果缓存从 DataSystem 命中。注意这不是 C++ KV block onboard。 |
| `unknown` | request id 未能和 TRT trace 关联，通常用于保底归类，不作为性能结论主体。 |

本报告中 `TRT_RESULT_CACHE_ENABLED=0`，因此主要分组为：

```text
disabled count=11804
```

这证明当前 E2E 统计代表 cold runner path，而不是 Python 推荐结果缓存命中后的低延迟路径。

## 附录 A. 原始 Benchmark 输出

以下为远程 `scripts/benchmark_e2e_latency.py` 输出原始摘录，保留 pressure/replay、请求级阶段和 C++ DataSystem block 指标，便于复核上文汇总表。

```text
== pressure wave: 2000 requests, concurrency=1 ==
  ok=1976 fail=24 elapsed=189.0s
  client_ms avg=94.5 p50=94.4 p95=96.3 p99=97.6 p9999=118.0 max=120.5
  items full_size=1976/1976 expected=3 min=3 p50=3.0 p95=3.0 max=3
  failed[25] uid=26 code=299 error=items size not enough
  failed[148] uid=149 code=299 error=items size not enough
  failed[169] uid=170 code=299 error=items size not enough
  failed[237] uid=238 code=299 error=items size not enough
  failed[241] uid=242 code=299 error=items size not enough

  replay_uids=986,987,988,989,990,991,992,993,994,995,996,997,998,999,1000,1001

== replay/onboard wave: 10000 requests, concurrency=1 ==
  ok=9877 fail=123 elapsed=930.5s
  client_ms avg=93.0 p50=92.2 p95=96.7 p99=104.3 p9999=117.6 max=125.2
  items full_size=9877/9877 expected=3 min=3 p50=3.0 p95=3.0 max=3
  failed[131] uid=989 code=299 error=items size not enough
  failed[155] uid=997 code=299 error=items size not enough
  failed[267] uid=997 code=299 error=items size not enough
  failed[324] uid=990 code=299 error=items size not enough
  failed[390] uid=992 code=299 error=items size not enough

== Client (all measured phases) ==
  metric                              count      avg      p50      p95      p99    p9999      max
  client_e2e_ms                       11853     93.3     92.8     96.6    103.2    119.9    125.2

== Response items ==
  full_size=11853/11853 expected=3 min=3 p50=3.0 p95=3.0 max=3

== PaiRec recommend stages ==
  metric                              count      avg      p50      p95      p99    p9999      max
  total_ms                            11853     92.3     92.0     96.0    102.0    115.8    116.0
  user_feature_ms                     11853      0.0      0.0      0.0      0.0      0.0      0.0
  recall_ms                           11853     92.2     92.0     96.0    102.0    115.8    116.0
  filter_ms                           11853      0.0      0.0      0.0      0.0      0.0      0.0
  general_rank_ms                     11853      0.0      0.0      0.0      0.0      0.0      0.0
  feature_ms                          11853      0.0      0.0      0.0      0.0      0.0      0.0
  rank_ms                             11853      0.0      0.0      0.0      0.0      0.0      0.0
  pipeline_wait_ms                    11853      0.0      0.0      0.0      0.0      0.0      0.0
  merge_ms                            11853      0.0      0.0      0.0      0.0      0.0      0.0
  sort_ms                             11853      0.0      0.0      0.0      0.0      0.0      0.0

== GenerativeRecall stages ==
  metric                              count      avg      p50      p95      p99    p9999      max
  cost                                11853     92.2     92.0     95.0    102.0    115.8    116.0
  cache_ms                            11853      0.0      0.0      0.0      0.0      0.0      0.0
  history_ms                          11853      0.0      0.0      0.0      0.0      0.0      0.0
  convert_ms                          11853      0.0      0.0      0.0      0.0      0.0      0.0
  http_ms                             11853     92.1     92.0     95.0    102.0    115.8    116.0
  items_ms                            11853      0.0      0.0      0.0      0.0      0.0      0.0
  tr_total_ms                         11853     91.2     90.7     94.5    101.0    114.6    114.7
  tr_prepare_ms                       11853      0.4      0.4      0.5      0.5      1.2      9.1
  tr_infer_ms                         11853     89.4     88.8     92.6     99.0    111.9    112.7
  tr_prompt_ms                        11853      2.5      2.7      2.9      3.0      5.4      5.6
  tr_runner_ms                        11853     85.9     84.9     88.8     95.4    107.6    109.1
  tr_parse_ms                         11853      0.0      0.0      0.1      0.2      0.2      0.2
  tr_pad_ms                           11853      0.7      0.6      1.7      2.1      3.0      3.3
  tr_map_ms                           11853      0.1      0.1      0.2      0.2      0.8      0.8
  tr_kv_lookup_ms                     11853      1.3      1.3      1.6      1.7      2.4      2.4
  tr_kv_write_ms                      11853      0.0      0.0      0.0      0.0      0.0      0.0
  tr_result_cache_lookup_ms           11853      0.0      0.0      0.0      0.0      0.0      0.0
  tr_result_cache_ds_lookup_ms        11853      0.0      0.0      0.0      0.0      0.0      0.0
  tr_result_cache_write_submit_ms     11853      0.0      0.0      0.0      0.0      0.0      0.0
  http_overhead_ms                    11853      1.3      1.0      2.0      2.0      3.8      4.0

== TRT service stages ==
  metric                              count      avg      p50      p95      p99    p9999      max
  total_ms                            11804     91.2     90.7     94.5    101.0    114.6    114.7
  prepare_ms                          11804      0.4      0.4      0.5      0.5      1.2      9.1
  kv_lookup_ms                        11804      1.3      1.3      1.6      1.7      2.4      2.4
  result_cache_lookup_ms              11804      0.0      0.0      0.0      0.0      0.0      0.0
  result_cache_ds_lookup_ms           11804      0.0      0.0      0.0      0.0      0.0      0.0
  prompt_ms                           11804      2.5      2.7      2.9      3.0      5.4      5.6
  runner_ms                           11804     85.9     84.9     88.8     95.5    107.6    109.1
  runner_calls                        11804      1.0      1.0      1.0      1.0      1.0      1.0
  runner_avg_ms                       11804     85.9     84.9     88.8     95.5    107.6    109.1
  runner_max_ms                       11804     85.9     84.9     88.8     95.5    107.6    109.1
  parse_ms                            11804      0.0      0.0      0.1      0.2      0.2      0.2
  map_ms                              11804      0.1      0.1      0.2      0.2      0.8      0.8

== TRT result cache groups (trt_log) ==
  disabled     count=11804 total_p50=    90.7ms total_p95=    94.5ms total_p99=   101.0ms total_p9999=   114.6ms total_max=   114.7ms
  unknown      count=  49 total_p50=     0.0ms total_p95=     0.0ms total_p99=     0.0ms total_p9999=     0.0ms total_max=     0.0ms

== DataSystem C++ KV block stages (all measured phases) ==
  scope: per KV block, from TRT-LLM C++ [Datasystem][TRACE], host wall-clock
  metric                              count      avg      p50      p95      p99    p9999      max
  offload: events=29330
  offload.create_ms                   29330    0.607    0.605    0.758    0.833    1.578    2.758
  offload.d2h_ms                      29330    0.488    0.468    0.727    0.751    1.173    1.259
  offload.set_ms                      29330    0.757    0.746    0.910    1.015    1.689   10.769
  offload.total_ms                    29330    1.887    1.870    2.256    2.388    3.445   11.648
  onboard: events=13421
  onboard.get_ms                      13421    0.632    0.623    0.747    0.818    1.094   10.691
  onboard.h2d_ms                      13421    0.343    0.336    0.434    0.515    1.382    1.416
  onboard.total_ms                    13421    1.009    0.992    1.178    1.293    2.116   11.078

== DataSystem C++ KV block stages [pressure] ==
  scope: per KV block, from TRT-LLM C++ [Datasystem][TRACE], host wall-clock
  metric                              count      avg      p50      p95      p99    p9999      max
  offload: events=5996
  offload.create_ms                    5996    0.661    0.661    0.787    0.869    1.413    1.574
  offload.d2h_ms                       5996    0.607    0.692    0.744    0.765    1.234    1.259
  offload.set_ms                       5996    0.779    0.767    0.915    1.018    1.792    1.811
  offload.total_ms                     5996    2.082    2.089    2.341    2.499    3.338    3.390
  onboard: events=87
  onboard.get_ms                         87    0.731    0.712    0.866    0.909    0.972    0.973
  onboard.h2d_ms                         87    0.471    0.509    0.530    0.531    0.534    0.534
  onboard.total_ms                       87    1.242    1.244    1.416    1.484    1.552    1.553

== DataSystem C++ KV block stages [replay/onboard] ==
  scope: per KV block, from TRT-LLM C++ [Datasystem][TRACE], host wall-clock
  metric                              count      avg      p50      p95      p99    p9999      max
  offload: events=23334
  offload.create_ms                   23334    0.593    0.591    0.741    0.817    1.524    2.758
  offload.d2h_ms                      23334    0.458    0.464    0.547    0.739    1.107    1.164
  offload.set_ms                      23334    0.751    0.739    0.908    1.013    1.434   10.769
  offload.total_ms                    23334    1.837    1.829    2.163    2.345    3.689   11.648
  onboard: events=13334
  onboard.get_ms                      13334    0.631    0.623    0.745    0.816    1.095   10.691
  onboard.h2d_ms                      13334    0.342    0.336    0.420    0.510    1.382    1.416
  onboard.total_ms                    13334    1.008    0.992    1.171    1.282    2.116   11.078
```

## 附录 B. 远端 DataSystem Benchmark 原始输出

以下为 2026-06-05 远端 DataSystem worker 测试的原始输出摘录，DataSystem host 为 `141.61.91.188:18581`。

```text
== pressure wave: 300 requests, concurrency=1 ==
  ok=295 fail=5 elapsed=310.1s
  client_ms avg=1033.2 p50=1048.2 p95=1153.6 p99=1369.1 p9999=1370.8 max=1370.8
  items full_size=295/295 expected=3 min=3 p50=3.0 p95=3.0 max=3
  failed[84] uid=85 code=299 error=items size not enough
  failed[171] uid=172 code=299 error=items size not enough
  failed[172] uid=173 code=299 error=items size not enough
  failed[219] uid=220 code=299 error=items size not enough
  failed[247] uid=248 code=299 error=items size not enough

  replay_uids=285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300

== replay/onboard wave: 1000 requests, concurrency=1 ==
  ok=995 fail=5 elapsed=1258.4s
  client_ms avg=1258.8 p50=1050.6 p95=1687.7 p99=1689.3 p9999=1690.9 max=1691.0
  items full_size=995/995 expected=3 min=3 p50=3.0 p95=3.0 max=3
  failed[231] uid=292 code=299 error=items size not enough
  failed[318] uid=299 code=299 error=items size not enough
  failed[340] uid=289 code=299 error=items size not enough
  failed[411] uid=296 code=299 error=items size not enough
  failed[825] uid=294 code=299 error=items size not enough

== Client (all measured phases) ==
  metric                              count      avg      p50      p95      p99    p9999      max
  client_e2e_ms                        1290   1207.2   1050.3   1687.2   1689.2   1690.9   1691.0

== PaiRec recommend stages ==
  metric                              count      avg      p50      p95      p99    p9999      max
  total_ms                             1290   1206.1   1049.0   1686.0   1688.0   1690.0   1690.0
  recall_ms                            1290   1206.0   1049.0   1686.0   1688.0   1690.0   1690.0

== GenerativeRecall stages ==
  metric                              count      avg      p50      p95      p99    p9999      max
  cost                                 1290   1206.0   1049.0   1686.0   1688.0   1690.0   1690.0
  http_ms                              1290   1205.8   1049.0   1686.0   1688.0   1690.0   1690.0
  tr_total_ms                          1290   1204.8   1048.0   1684.7   1686.6   1688.6   1688.6
  tr_prepare_ms                        1290      0.4      0.4      0.5      0.6      8.2      9.3
  tr_infer_ms                          1290   1196.6   1038.1   1674.7   1676.2   1677.9   1677.9
  tr_prompt_ms                         1290      2.7      2.9      3.1      3.3      3.9      3.9
  tr_runner_ms                         1290   1192.9   1034.2   1670.6   1671.9   1672.8   1672.8
  tr_parse_ms                          1290      0.1      0.1      0.1      0.2      0.6      0.6
  tr_pad_ms                            1290      0.8      0.7      1.8      2.3      2.8      2.8
  tr_map_ms                            1290      0.1      0.1      0.2      0.2      0.3      0.3
  tr_kv_lookup_ms                      1290      7.6      9.7     10.3     10.5     10.6     10.6
  http_overhead_ms                     1290      1.5      2.0      2.0      3.0      3.9      4.0

== TRT service stages ==
  metric                              count      avg      p50      p95      p99    p9999      max
  total_ms                             1220   1203.0   1048.0   1684.6   1686.6   1688.6   1688.6
  prepare_ms                           1220      0.4      0.4      0.5      0.6      8.3      9.3
  kv_lookup_ms                         1220      7.6      9.7     10.3     10.5     10.6     10.6
  prompt_ms                            1220      2.7      2.9      3.1      3.3      3.6      3.6
  runner_ms                            1220   1191.1   1034.2   1670.5   1671.9   1672.8   1672.8
  runner_calls                         1220      1.0      1.0      1.0      1.0      1.0      1.0
  runner_avg_ms                        1220   1191.1   1034.2   1670.5   1671.9   1672.8   1672.8
  runner_max_ms                        1220   1191.1   1034.2   1670.5   1671.9   1672.8   1672.8
  parse_ms                             1220      0.1      0.1      0.1      0.2      0.6      0.6
  map_ms                               1220      0.1      0.1      0.2      0.2      0.3      0.3

== TRT result cache groups (trt_log) ==
  disabled     count=1220 total_p50=  1048.0ms total_p95=  1684.6ms total_p99=  1686.6ms total_p9999=  1688.6ms total_max=  1688.6ms
  unknown      count=  70 total_p50=     0.0ms total_p95=     0.0ms total_p99=     0.0ms total_p9999=     0.0ms total_max=     0.0ms

== DataSystem C++ KV block stages (all measured phases) ==
  scope: per KV block, from TRT-LLM C++ [Datasystem][TRACE], host wall-clock
  metric                              count      avg      p50      p95      p99    p9999      max
  offload: events=3202
  offload.create_ms                    3202    1.669    1.662    1.790    1.826    2.265    2.286
  offload.d2h_ms                       3202    0.524    0.516    0.643    0.736    0.786    0.787
  offload.set_ms                       3202  314.491  314.149  315.288  315.610  325.289  328.397
  offload.total_ms                     3202  316.721  316.451  317.597  317.915  327.545  330.647
  onboard: events=1346
  onboard.get_ms                       1346  315.930  315.572  316.862  317.038  321.050  321.650
  onboard.h2d_ms                       1346    0.405    0.405    0.421    0.429    0.487    0.494
  onboard.total_ms                     1346  316.370  316.010  317.307  317.480  321.488  322.087

== DataSystem C++ KV block stages [pressure] ==
  scope: per KV block, from TRT-LLM C++ [Datasystem][TRACE], host wall-clock
  metric                              count      avg      p50      p95      p99    p9999      max
  offload: events=871
  offload.create_ms                     871    1.696    1.745    1.795    1.823    2.211    2.220
  offload.d2h_ms                        871    0.555    0.545    0.730    0.759    0.787    0.787
  offload.set_ms                        871  314.497  314.138  315.294  315.644  327.552  328.397
  offload.total_ms                      871  316.785  316.481  317.668  318.017  329.804  330.647
  onboard: events=15
  onboard.get_ms                         15  316.029  315.604  316.877  316.899  316.905  316.905
  onboard.h2d_ms                         15    0.404    0.407    0.423    0.423    0.423    0.423
  onboard.total_ms                       15  316.470  316.044  317.336  317.350  317.354  317.354

== DataSystem C++ KV block stages [replay/onboard] ==
  scope: per KV block, from TRT-LLM C++ [Datasystem][TRACE], host wall-clock
  metric                              count      avg      p50      p95      p99    p9999      max
  offload: events=2331
  offload.create_ms                    2331    1.659    1.633    1.787    1.829    2.210    2.286
  offload.d2h_ms                       2331    0.513    0.509    0.615    0.637    0.657    0.658
  offload.set_ms                       2331  314.489  314.153  315.281  315.601  318.162  318.255
  offload.total_ms                     2331  316.697  316.415  317.554  317.835  320.396  320.472
  onboard: events=1331
  onboard.get_ms                       1331  315.929  315.572  316.861  317.041  321.057  321.650
  onboard.h2d_ms                       1331    0.405    0.405    0.421    0.429    0.487    0.494
  onboard.total_ms                     1331  316.368  316.009  317.304  317.483  321.495  322.087
```
