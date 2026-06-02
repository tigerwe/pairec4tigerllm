# PaiRec 端到端稳态冷请求时延基线

> 整理日期：2026-06-02  
> 实验类型：PaiRec 入口端到端 benchmark，TRT 推荐结果缓存全部 miss  
> 用途：作为后续 TRT-LLM C++ DataSystem 与原生 pinned DRAM A/B 的参考基线

## 1. 实验配置

| 参数 | 值 |
|---|---|
| 推荐入口 | `http://localhost:18080/api/recommend` |
| 用户 | `300,301,302,303,304,305,306,307,308,309` |
| 请求数 | `10` |
| Warmup | `0` |
| 并发数 | `1` |
| 返回数量 | `5` |
| Scene | `home_feed` |
| 成功率 | `10 / 10` |
| 总耗时 | `7.0s` |
| TRT 推荐结果缓存分组 | `miss = 10` |

这组请求属于稳态的 TRT 推荐结果缓存 miss 基线：10 个请求均进入 TRT-LLM runner，没有被 Python 推荐结果缓存提前返回。

需要注意：

- `warmup=0` 表示 benchmark 脚本没有主动预热，但报告中 `history_ms=0`，未观察到首次加载用户特征文件的秒级抖动。
- 仅凭这份报告无法确认用户特征来自 Kafka 实时缓存还是已加载的 fallback 文件。
- `miss=10` 指 Python TRT 推荐结果缓存 miss，不等价于 TRT-LLM C++ KV Block 的 DataSystem miss。
- 当前报告没有 C++ offload / onboard 日志，因此不能单独用于证明 DataSystem 对 C++ KV Cache 的收益。

## 2. 核心结论

客户端平均端到端时延为 `701.7ms`，p95 为 `728.3ms`。

```text
客户端 E2E                     701.7ms
  └─ PaiRec                    700.0ms
      └─ GenerativeRecall      699.7ms
          └─ HTTP 调用 TRT     699.6ms
              └─ TRT 服务      698.4ms
                  └─ runner    675.2ms
```

TRT 服务占客户端端到端平均时延约 `99.5%`。其中 `runner_ms=675.2ms`，占 TRT 服务平均时延约 `96.7%`，占客户端端到端平均时延约 `96.2%`。

当前主要瓶颈不是 PaiRec、HTTP、特征读取、语义映射或 Python 后处理，而是 TRT-LLM C++ runner 内的 8 轮生成：

```text
单轮 runner.generate() 平均耗时约 675.2 / 8 = 84.4ms
```

## 3. 端到端分层分析

### 3.1 客户端与 PaiRec

| 阶段 | 平均耗时 | p50 | p95 | p99 | 最大值 | 说明 |
|---|---:|---:|---:|---:|---:|---|
| Client E2E | `701.7ms` | `695.9ms` | `728.3ms` | `739.0ms` | `741.7ms` | 客户端感知的完整请求耗时 |
| PaiRec `total_ms` | `700.0ms` | `694.0ms` | `725.4ms` | `735.5ms` | `738.0ms` | PaiRec 服务内部完整耗时 |
| 差值 | `1.7ms` | - | - | - | - | 客户端到 PaiRec 的网络、序列化和测量边界开销 |

PaiRec 自身额外开销很小。`recall_ms=699.7ms`，约占 PaiRec 平均耗时的 `99.96%`。当前请求几乎全部耗在生成式召回。

### 3.2 PaiRec 内部阶段

| 阶段 | 平均耗时 | 说明 |
|---|---:|---|
| `recall_ms` | `699.7ms` | 生成式召回，主要耗时 |
| `user_feature_ms` | `<1ms` | PaiRec 用户特征阶段，日志精度下显示为 `0.0ms` |
| `filter_ms` | `<1ms` | 过滤阶段 |
| `general_rank_ms` | `<1ms` | 通用排序阶段 |
| `feature_ms` | `<1ms` | 排序特征阶段 |
| `rank_ms` | `<1ms` | 排序阶段 |
| `pipeline_wait_ms` | `<1ms` | Pipeline 等待 |
| `merge_ms` | `<1ms` | 合并阶段 |
| `sort_ms` | `<1ms` | 最终排序阶段 |

日志中的 `0.0ms` 不应解释为绝对没有执行，而是该阶段低于当前毫秒级统计精度。

### 3.3 GenerativeRecall 阶段

| 阶段 | 平均耗时 | p95 | 说明 |
|---|---:|---:|---|
| `cost` | `699.7ms` | `725.4ms` | GenerativeRecall 总耗时 |
| `http_ms` | `699.6ms` | `725.4ms` | Go 调用 TRT 服务的 HTTP 往返 |
| `http_overhead_ms` | `1.9ms` | `2.5ms` | HTTP 往返与 TRT 服务内部耗时之间的差异 |
| `cache_ms` | `<1ms` | `<1ms` | PaiRec recall cache 查询 |
| `history_ms` | `<1ms` | `<1ms` | 获取用户历史 |
| `convert_ms` | `<1ms` | `<1ms` | Item ID 转 semantic ID |
| `items_ms` | `<1ms` | `<1ms` | TRT 返回结果转 PaiRec Item |

`http_ms` 几乎覆盖全部 GenerativeRecall 时延，说明 Go 层优化空间有限。

### 3.4 TRT 服务阶段

| 阶段 | 平均耗时 | p50 | p95 | p99 | 最大值 | 占 TRT 总耗时 |
|---|---:|---:|---:|---:|---:|---:|
| TRT `total_ms` | `698.4ms` | `693.0ms` | `723.6ms` | `733.4ms` | `735.9ms` | `100.0%` |
| `runner_ms` | `675.2ms` | `672.1ms` | `687.5ms` | `694.3ms` | `696.0ms` | `96.7%` |
| `pad_ms` | `12.9ms` | `11.1ms` | `20.9ms` | `21.4ms` | `21.5ms` | `1.8%` |
| `prompt_ms` | `2.8ms` | `3.0ms` | `3.1ms` | `3.1ms` | `3.1ms` | `0.4%` |
| `kv_lookup_ms` | `1.9ms` | `1.8ms` | `2.5ms` | `2.7ms` | `2.8ms` | `0.3%` |
| `result_cache_write_submit_ms` | `1.6ms` | `1.5ms` | `1.9ms` | `1.9ms` | `1.9ms` | `0.2%` |
| `result_cache_ds_lookup_ms` | `1.4ms` | `1.4ms` | `1.7ms` | `1.7ms` | `1.7ms` | `0.2%` |
| `prepare_ms` | `1.4ms` | `0.5ms` | `5.5ms` | `8.5ms` | `9.2ms` | `0.2%` |
| `parse_ms` | `0.6ms` | `0.5ms` | `1.0ms` | `1.2ms` | `1.2ms` | `<0.1%` |
| `map_ms` | `0.1ms` | `0.1ms` | `0.2ms` | `0.3ms` | `0.3ms` | `<0.1%` |

上述 TRT 子阶段大致覆盖完整 `total_ms`。`tr_infer_ms=693.4ms` 是嵌套指标，已经包含结果缓存检查、prompt 构造、runner、解析和 padding，不应与这些子阶段再次相加。

外围阶段即使全部消除，也只能减少约二十几毫秒。首要优化对象仍然是 `runner_ms`。

## 4. DataSystem 指标边界

当前报告中可见的 DataSystem 相关字段属于 Python 服务层：

| 字段 | 平均耗时 | 当前含义 |
|---|---:|---|
| `tr_kv_lookup_ms` | `1.9ms` | Python `KVCacheManager.query()` |
| `tr_result_cache_ds_lookup_ms` | `1.4ms` | Python 推荐结果缓存从 DataSystem 查询 |
| `tr_result_cache_write_submit_ms` | `1.6ms` | Python 推荐结果缓存异步写入任务提交 |

其中：

- 10 个请求的 Python 推荐结果缓存均为 `miss`。
- `tr_result_cache_ds_lookup_ms` 表示 miss 前仍查询了 DataSystem。
- 异步写入只统计任务提交耗时，不等于 DataSystem 实际写入耗时。
- TRT-LLM C++ KV Block DataSystem 路径发生在 runner 内部，当前没有拆分到这些字段中。

后续 C++ DataSystem A/B 需要增加：

```text
C++ offload_count / onboard_count
DataSystem Create / Set / Get 耗时
D2H / H2D copy 耗时
pressure 阶段与 replay 阶段的独立 p50 / p95 / p99
```

## 5. 尾延迟观察

| 指标 | p50 | p95 | p99 | 最大值 | p95 - p50 |
|---|---:|---:|---:|---:|---:|
| Client E2E | `695.9ms` | `728.3ms` | `739.0ms` | `741.7ms` | `32.4ms` |
| PaiRec total | `694.0ms` | `725.4ms` | `735.5ms` | `738.0ms` | `31.4ms` |
| TRT total | `693.0ms` | `723.6ms` | `733.4ms` | `735.9ms` | `30.6ms` |
| TRT runner | `672.1ms` | `687.5ms` | `694.3ms` | `696.0ms` | `15.4ms` |

TRT runner 本身波动相对有限。TRT 总耗时的额外尾部波动主要来自 runner 外围阶段，尤其是：

- `prepare_ms`：最大 `9.2ms`
- `pad_ms`：最大 `21.5ms`

本轮只有 10 个请求，p95 和 p99 接近最大值。该结果适合用作方向性基线，正式 A/B 应增加样本量并重复多轮。

## 6. 后续 A/B 的参考基线

本轮可记录为：

```text
稳态 Python 结果缓存 miss
成功率               = 100%
Client E2E p50       = 695.9ms
Client E2E p95       = 728.3ms
PaiRec total p50     = 694.0ms
TRT total p50        = 693.0ms
TRT runner p50       = 672.1ms
TRT runner avg       = 675.2ms
TRT runner / TRT     = 96.7%
TRT runner / E2E     = 96.2%
```

正式比较 DataSystem 的积极影响时，应使用同一 engine、同一请求序列和相同 KV Block 配置，分别运行原生 pinned DRAM runtime 与 DataSystem runtime，并将 `pressure` 和 `replay` 阶段分开统计。

## 7. 原始 Benchmark 输出

```text
PaiRec end-to-end latency benchmark
  url=http://localhost:18080/api/recommend
  uids=300,301,302,303,304,305,306,307,308,309 requests=10 warmup=0
  concurrency=1 size=5 scene=home_feed

== Benchmark ==
  ok=10 fail=0 elapsed=7.0s

== Client ==
  metric                              count      avg      p50      p95      p99      max
  client_e2e_ms                          10    701.7    695.9    728.3    739.0    741.7

== PaiRec recommend stages ==
  metric                              count      avg      p50      p95      p99      max
  total_ms                               10    700.0    694.0    725.4    735.5    738.0
  user_feature_ms                        10      0.0      0.0      0.0      0.0      0.0
  recall_ms                              10    699.7    694.0    725.4    735.5    738.0
  filter_ms                               10      0.0      0.0      0.0      0.0      0.0
  general_rank_ms                        10      0.0      0.0      0.0      0.0      0.0
  feature_ms                             10      0.0      0.0      0.0      0.0      0.0
  rank_ms                                10      0.0      0.0      0.0      0.0      0.0
  pipeline_wait_ms                       10      0.0      0.0      0.0      0.0      0.0
  merge_ms                               10      0.0      0.0      0.0      0.0      0.0
  sort_ms                                10      0.0      0.0      0.0      0.0      0.0

== GenerativeRecall stages ==
  metric                              count      avg      p50      p95      p99      max
  cost                                   10    699.7    694.0    725.4    735.5    738.0
  cache_ms                               10      0.0      0.0      0.0      0.0      0.0
  history_ms                             10      0.0      0.0      0.0      0.0      0.0
  convert_ms                             10      0.0      0.0      0.0      0.0      0.0
  http_ms                                10    699.6    694.0    725.4    735.5    738.0
  items_ms                               10      0.0      0.0      0.0      0.0      0.0
  tr_total_ms                            10    698.4    693.0    723.6    733.4    735.9
  tr_prepare_ms                          10      1.4      0.5      5.5      8.5      9.2
  tr_infer_ms                            10    693.4    689.1    713.7    720.1    721.7
  tr_prompt_ms                           10      2.8      3.0      3.1      3.1      3.1
  tr_runner_ms                           10    675.2    672.1    687.5    694.3    696.0
  tr_parse_ms                            10      0.6      0.5      1.0      1.2      1.2
  tr_pad_ms                              10     12.9     11.1     20.9     21.4     21.5
  tr_map_ms                              10      0.1      0.1      0.2      0.3      0.3
  tr_kv_lookup_ms                        10      1.9      1.8      2.5      2.7      2.8
  tr_kv_write_ms                         10      0.0      0.0      0.0      0.0      0.0
  tr_result_cache_lookup_ms              10      0.0      0.0      0.0      0.0      0.0
  tr_result_cache_ds_lookup_ms           10      1.4      1.4      1.7      1.7      1.7
  tr_result_cache_write_submit_ms        10      1.6      1.5      1.9      1.9      1.9
  http_overhead_ms                       10      1.9      2.0      2.5      2.9      3.0

== TRT service stages ==
  metric                              count      avg      p50      p95      p99      max
  total_ms                               10    698.4    693.0    723.6    733.4    735.9
  prepare_ms                             10      1.4      0.5      5.5      8.5      9.2
  kv_lookup_ms                           10      1.9      1.8      2.5      2.7      2.8
  result_cache_lookup_ms                 10      0.0      0.0      0.0      0.0      0.0
  result_cache_ds_lookup_ms              10      1.4      1.4      1.7      1.7      1.7
  prompt_ms                              10      2.8      3.0      3.1      3.1      3.1
  runner_ms                              10    675.2    672.1    687.5    694.3    696.0
  parse_ms                               10      0.6      0.5      1.0      1.2      1.2
  map_ms                                 10      0.1      0.1      0.2      0.3      0.3

== TRT result cache groups ==
  miss         count=  10 total_p50=   693.0ms total_p95=   723.6ms
```
