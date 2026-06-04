# PaiRec 端到端各阶段时延测试报告

> 整理日期：2026-06-04  
> 分支：`cpp-datasystem-ab-latency`  
> 目标：汇总 `TRT_NUM_SAMPLES=1` 后的 PaiRec 端到端时延、缓存命中/冷 miss 分组，以及 DataSystem C++ Get/Set 在当前 E2E 报告中的观测状态。

## 1. 实验配置

| 参数 | 值 |
|---|---|
| 推荐入口 | `http://127.0.0.1:18080/api/recommend` |
| TRT 服务 | `http://127.0.0.1:18000` |
| TRT 采样轮数 | `TRT_NUM_SAMPLES=1` |
| 请求 UID | `130,2184,7494,142,211,303,1201` |
| 请求数 | `60` |
| Warmup | `3` |
| 并发 | `1` |
| 返回数量 | `size=5` |
| 场景 | `home_feed` |
| 成功率 | `60/60` |
| 返回完整率 | `full_size=60/60` |

运行命令：

```bash
python scripts/benchmark_e2e_latency.py \
  --url http://127.0.0.1:18080/api/recommend \
  --uids 130,2184,7494,142,211,303,1201 \
  --warmup 3 \
  --requests 60 \
  --concurrency 1 \
  --size 5 \
  --timeout 10 \
  --pairec-log /tmp/pairec_e2e_samples1.log \
  --trt-log /tmp/server_e2e_samples1.log \
  --json-output /tmp/e2e_samples1_combined.json
```

## 2. 核心结论

本轮 E2E 报告的 p50 很低，原因是大多数请求命中了 Python TRT 推荐结果 HBM 缓存：

```text
hbm_hit = 58
miss    = 2
```

因此本轮报告需要拆成两条路径理解：

| 路径 | 结论 |
|---|---|
| 缓存命中路径 | Client E2E p50 `3.4ms`，TRT total p50 `1.6ms` |
| 冷 miss 路径 | TRT total 约 `88.8ms`，runner 约 `82-84ms` |

`TRT_NUM_SAMPLES=1` 后，冷 miss 已从 8 轮基线约 `694ms` 降到约 `89ms`。这与直连 TRT runner A/B 的结果一致。

本轮 E2E 日志切片没有出现 TensorRT-LLM C++ `[Datasystem][TRACE]`，因此没有在本轮 E2E 中观测到 C++ KV block 的 DataSystem `set_ms/get_ms`。这不是脚本解析问题，而是当前请求大多被结果缓存拦截，只有 2 个 miss，不足以触发 C++ KV offload/onboard。

## 3. 端到端分层时延

### 3.1 Client

| 指标 | count | avg | p50 | p95 | p99 | p9999 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `client_e2e_ms` | 60 | `6.3ms` | `3.4ms` | `4.3ms` | `90.8ms` | `91.2ms` | `91.2ms` |

### 3.2 返回完整率

| 指标 | 值 |
|---|---:|
| `full_size` | `60/60` |
| expected size | `5` |
| min items | `5` |
| p50 items | `5.0` |
| p95 items | `5.0` |
| max items | `5` |

### 3.3 PaiRec recommend stages

| 阶段 | count | avg | p50 | p95 | p99 | p9999 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `total_ms` | 60 | `5.3` | `2.0` | `3.0` | `90.0` | `90.0` | `90.0` |
| `recall_ms` | 60 | `5.1` | `2.0` | `3.0` | `89.4` | `90.0` | `90.0` |
| `user_feature_ms` | 60 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| `filter_ms` | 60 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| `general_rank_ms` | 60 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| `feature_ms` | 60 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| `rank_ms` | 60 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| `pipeline_wait_ms` | 60 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| `merge_ms` | 60 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| `sort_ms` | 60 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |

PaiRec 内部耗时几乎全部在 `recall_ms`。其它阶段在当前毫秒级日志精度下低于 `1ms`。

### 3.4 GenerativeRecall stages

| 阶段 | count | avg | p50 | p95 | p99 | p9999 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cost` | 60 | `5.0` | `2.0` | `3.0` | `89.4` | `90.0` | `90.0` |
| `http_ms` | 60 | `5.0` | `2.0` | `3.0` | `89.4` | `90.0` | `90.0` |
| `tr_total_ms` | 60 | `4.5` | `1.6` | `2.1` | `88.7` | `89.1` | `89.1` |
| `tr_prepare_ms` | 60 | `0.3` | `0.4` | `0.4` | `0.5` | `0.5` | `0.5` |
| `tr_infer_ms` | 60 | `2.9` | `0.0` | `0.0` | `85.4` | `85.9` | `85.9` |
| `tr_prompt_ms` | 60 | `0.0` | `0.0` | `0.0` | `0.4` | `0.4` | `0.4` |
| `tr_runner_ms` | 60 | `2.8` | `0.0` | `0.0` | `82.7` | `83.6` | `83.6` |
| `tr_parse_ms` | 60 | `0.0` | `0.0` | `0.0` | `0.1` | `0.1` | `0.1` |
| `tr_pad_ms` | 60 | `0.0` | `0.0` | `0.0` | `0.8` | `1.2` | `1.2` |
| `tr_map_ms` | 60 | `0.0` | `0.0` | `0.0` | `0.1` | `0.1` | `0.1` |
| `tr_kv_lookup_ms` | 60 | `1.2` | `1.2` | `1.6` | `1.7` | `1.7` | `1.7` |
| `tr_result_cache_ds_lookup_ms` | 60 | `0.0` | `0.0` | `0.0` | `1.1` | `1.2` | `1.2` |
| `tr_result_cache_write_submit_ms` | 60 | `0.1` | `0.0` | `0.0` | `1.5` | `1.7` | `1.7` |
| `http_overhead_ms` | 60 | `1.1` | `1.0` | `1.0` | `2.0` | `2.0` | `2.0` |

`tr_runner_ms` 的 p50 为 `0.0ms`，不是 runner 没有耗时，而是 58/60 请求命中了推荐结果缓存，没有进入 runner。p99 `82.7ms` 对应少量冷 miss。

## 4. 缓存分组

由于 TRT 服务日志未按 `request_id` 关联上，本报告使用 GenerativeRecall 回传的 `tr_result_cache_source` 做缓存分组。

| 结果缓存来源 | count | total p50 | total p95 | total p99 | total p9999 | max |
|---|---:|---:|---:|---:|---:|---:|
| `hbm_hit` | 58 | `1.6ms` | `2.1ms` | `2.1ms` | `2.1ms` | `2.1ms` |
| `miss` | 2 | `88.8ms` | `89.1ms` | `89.1ms` | `89.1ms` | `89.1ms` |

这说明当前 E2E p50 基本代表缓存命中路径，p99 才能看到冷 miss 的 runner 代价。

## 5. DataSystem 指标

### 5.1 本轮 E2E 中的 DataSystem 指标

本轮 E2E 报告中的 Python DataSystem 相关字段：

| 字段 | avg | p50 | p99 | 含义 |
|---|---:|---:|---:|---|
| `tr_kv_lookup_ms` | `1.2ms` | `1.2ms` | `1.7ms` | Python `KVCacheManager.query()` 开销 |
| `tr_result_cache_ds_lookup_ms` | `0.0ms` | `0.0ms` | `1.1ms` | Python 推荐结果缓存 DataSystem 查询 |
| `tr_result_cache_write_submit_ms` | `0.1ms` | `0.0ms` | `1.5ms` | Python 推荐结果缓存异步写入提交 |

这些不是 C++ KV block 的 DataSystem `Get/Set`。

本轮 E2E 的 TRT 日志切片中没有：

```text
[TensorRT-LLM][Datasystem][TRACE]
```

因此本轮没有观测到：

```text
offload.set_ms
onboard.get_ms
```

### 5.2 C++ DataSystem 专项压测结果

此前专项压测已证明 C++ DataSystem Get/Set 可观测，且样本量更充足：

| 路径 | 样本数 | 指标 | p50 | p99 | p9999 | max |
|---|---:|---|---:|---:|---:|---:|
| offload | 166089 | `set_ms` | `0.756ms` | `1.084ms` | `1.816ms` | `11.055ms` |
| offload | 166089 | `total_ms` | `1.937ms` | `2.582ms` | `11.877ms` | `12.513ms` |
| onboard | 14208 | `get_ms` | `0.740ms` | `1.037ms` | `1.570ms` | `10.801ms` |
| onboard | 14208 | `total_ms` | `1.177ms` | `1.546ms` | `2.121ms` | `11.272ms` |

这部分是 per KV block 指标，不是 per request 指标。`d2h_ms/h2d_ms` 是 host 计时的同步 `cudaMemcpy` wall-clock，当前没有 CUDA event device-only 时间。

## 6. 与 8 轮基线对比

| 场景 | Client p50 | TRT total p50 | runner p50/avg | 说明 |
|---|---:|---:|---:|---|
| 8 轮冷 miss 基线 | `695.9ms` | `693.0ms` | `675.2ms avg` | 10 个请求均 miss |
| 1 轮直连 TRT A/B | `91.8ms` | - | `85.8ms avg` | 60 个请求，`full_topk=60/60` |
| 1 轮 PaiRec E2E 缓存混合 | `3.4ms` | `1.6ms` | p99 `82.7ms` | 58 个 hbm_hit，2 个 miss |

`TRT_NUM_SAMPLES=1` 已将冷 miss 的 runner 开销降到约 `84ms`。当前 PaiRec E2E 的 p50 已被缓存命中路径主导。

## 7. 为什么本轮没有测到 E2E 内的 C++ Set/Get

要看到 C++ DataSystem Set/Get，需要请求进入 TRT-LLM runner 并触发 KV block eviction/reuse：

```text
pressure 阶段：大量 miss → C++ offload → DataSystem Set
replay 阶段：重放近期 prefix → C++ onboard → DataSystem Get
```

本轮只有 2 个 miss，且大多数请求命中 Python 推荐结果 HBM cache，因此：

- runner 调用次数太少；
- KV block 压力不足；
- 没有触发 C++ offload/onboard；
- E2E 报告中自然没有 C++ `set_ms/get_ms`。

## 8. 下一轮充分测 DS Set/Get 的要求

要在端到端报告里充分看到 C++ DataSystem Set/Get，建议下一轮满足：

1. 关闭 Python TRT 推荐结果缓存，避免 replay 被 HBM 结果缓存拦截。
2. 使用大量不同 UID 或不同 history，保证 pressure 阶段进入 runner。
3. pressure 请求数至少数百，推荐 `>=1000`，用于触发 offload/Set。
4. replay 请求循环近期 pressure UID，推荐 `>=10000`，用于积累 onboard/Get。
5. TRT 服务日志必须包含 `[Datasystem][TRACE]`。
6. 报告同时输出请求级 E2E 和 per-block C++ DataSystem 指标。

当前仓库中的 `scripts/benchmark_e2e_latency.py` 已能解析并输出 C++ DataSystem block 指标；缺口在于实验流量需要绕开结果缓存并制造足够 KV 压力。

## 9. 报告结论

当前可确认：

- `TRT_NUM_SAMPLES=1` 在直连 TRT 与 PaiRec E2E 中均有效。
- 当前 PaiRec E2E 混合缓存路径下，客户端 p50 已到 `3.4ms`。
- 冷 miss 路径约 `89ms`，主要由单轮 runner `~83ms` 构成。
- 推荐返回完整率为 `60/60`。
- 本轮 E2E 没有观测到 C++ DataSystem `set_ms/get_ms`。
- C++ DataSystem `set_ms/get_ms` 已在专项压测中观测到，但仍需下一轮构造端到端 pressure/replay，才能把它们和 E2E 请求阶段放进同一份充分报告。
