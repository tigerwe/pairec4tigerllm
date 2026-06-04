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
