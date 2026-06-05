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
