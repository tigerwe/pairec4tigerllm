# KVC Burst Wrapper 第一阶段设计

## 目标

在不改变真实推荐请求 KVC 调度的前提下，复刻 DataSystem `dsbench` 的一次性 Get burst：

```text
1 路业务 Get + C-1 路合成 Get = 配置并发 C
```

第一阶段只实现独立 Wrapper 和模拟业务 Get。TensorRT-LLM 接入留在第二阶段。

## 已确定的语义

- 最大并发为 100，首测档位为 `1/10/100`。
- 每条压力 lane 使用独立、长驻且开启 exclusive connection 的 `KVClient`。
- 每条压力 lane 使用独立 Key；Key 在启动阶段 Set 并通过 Get 验证。
- 每轮在计时窗口外随机打乱 Key 与 lane 的映射，并记录 seed。
- `batch_num=1`：每条 lane 每轮只执行一次同步单 Key Get。
- 业务 lane 与压力 lane 通过跨进程共享 futex Barrier 统一放行。
- Barrier 默认超时 10ms。超时后业务 Get 仍执行，但样本无效。
- 业务 Get 完成后即可返回；压力尾部由 Wrapper 异步汇总。
- 压力失败不改变业务结果，但实验样本无效。

## 第一阶段进程

```text
kvc_burst_wrapper
  - C-1 个 DataSystem 客户端
  - C-1 个常驻压力线程
  - 压力 Key 预填充和验证
  - 随机 lane/Key 映射
  - 压力结果异步聚合

kvc_burst_business_probe
  - 1 个独占 DataSystem 客户端
  - 1 个模拟业务 Key
  - 作为 Barrier 的第 C 个参与者
  - 执行一次真实 DataSystem Get
```

两个进程通过文件映射的 `SharedControl` 通信。协议使用固定宽度整数和 GCC process-shared
atomic builtins；futex 必须使用 `FUTEX_WAIT/FUTEX_WAKE`，不能使用进程私有 futex。

## 有效性门禁

每轮要求：

```text
clients_connected       = C-1
keys_verified           = C-1
pressure_workers_armed  = C-1
barrier_participants    = C
business_get_success    = true
pressure_get_success    = C-1
pressure_get_errors     = 0
max_active_all_gets     >= ceil(C * 0.95)
business_overlap_gets   >= ceil((C-1) * 0.95)
barrier_timeouts        = 0
```

`max_active_all_gets` 和 `business_overlap_gets` 根据同一主机的 `CLOCK_MONOTONIC_RAW`
时间区间事后计算，不使用“线程已经创建”替代真实并发证据。时延只报告，不设置硬目标。

## 构建

在具备 DataSystem C++ SDK 的 ARM 容器或主机执行：

```bash
DATASYSTEM_ROOT=/path/to/yr/datasystem \
INSTALL_DIR=/opt/pairec-kvc-burst \
  bash scripts/build_kvc_burst_wrapper.sh
```

也可以显式指定：

```bash
DATASYSTEM_INCLUDE_DIR=/path/to/include \
DATASYSTEM_LIBRARY=/path/to/libdatasystem.so \
  bash scripts/build_kvc_burst_wrapper.sh
```

## 独立验证

默认对象为 1792KiB，DataSystem 使用严格25G endpoint：

```bash
DS_ENDPOINT=192.168.100.12:18482 \
CONCURRENCY_LEVELS='1 10 100' \
REPEATS=3 \
OBJECT_SIZE=$((1792 * 1024)) \
  bash scripts/benchmark_kvc_burst_wrapper.sh
```

17.5MiB档建议只测 `1/10`：

```bash
DS_ENDPOINT=192.168.100.12:18482 \
CONCURRENCY_LEVELS='1 10' \
REPEATS=3 \
OBJECT_SIZE=$((17920 * 1024)) \
  bash scripts/benchmark_kvc_burst_wrapper.sh
```

正式分布采样可在 smoke 通过后提高 `REPEATS`，但不应先设置延迟目标。

## 第二阶段边界

TensorRT-LLM 接入时，只将 `kvc_burst_business_probe` 中以下逻辑替换为真实请求首次
DataSystem Get hook：

1. 按 `request_id` 领取一个 generation。
2. 业务线程作为 Barrier 的第 C 个参与者。
3. Barrier 放行后执行原有真实 Get。
4. 写入业务 Get 起止时间并立即继续原调用链。

真实请求原有 `3 Set + 2 Get` 不做合并、重排或强制并行；同一 Recommend 只由第一个
真实 Get 触发一轮 burst，后续 Get 只记录自然重叠情况。
