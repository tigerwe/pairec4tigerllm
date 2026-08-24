# KVC Burst Coordinator V6 设计

## 目标

在不修改 DataSystem SDK 的前提下，验证同一个生产 `KVClient` 上的持续 Get 压力是否会拖慢
一次推荐请求中的两次真实业务 Get，并定界压力停止后仍在飞的 Get 是否继续影响 `addToken`：

```text
第一次真实业务 Get
  -> 启动 C-1 路持续压力并确认至少 95% 已进入 Get
  -> 执行真实 Get #1
  -> 压力持续运行
  -> 执行真实 Get #2
  -> 非阻塞 stop，禁止提交下一轮压力 Get
  -> 观测剩余在飞 Get 与 addToken 的时间交集
```

V6由TensorRT-LLM进程内`KvcOperationProxy`代理真实DataSystem Get。压力 worker 在进程内长驻，
按值持有 prime 阶段从真实 single-key onboard 路径注册的生产 `KVClient`；sidecar只负责压力 key、
跨进程控制和结果聚合。真实 KV 数据始终通过原生 `KVClient` 直连 DataSystem，共享内存只承载
控制和统计字段。

## 已确定的语义

- 当前 in-process 验证形状固定为 `c32 = 31 pressure lanes + 1 business lane`。
- 31条压力 lane 复用同一个生产 `KVClient`，每次执行同步单 Key Get并循环提交。
- 压力 key 数为1..31，lane按`lane % pressure_key_count`映射；key在计时窗口外Set并Get验证。
- 第一次业务 Get 在启动压力后等待至少30条压力 Get真实在飞；该协调耗时单独记录。
- 真实 Get #1/#2 的计时均不包含协调等待。
- 第二次成功业务 Get 返回后只写stop generation并唤醒worker，不join、不等待在飞Get排空。
- stop之后已进入SDK的压力Get允许自然完成，但worker不得提交下一轮。
- sidecar等待业务lifecycle与31条worker均完成后再聚合，确保addToken和压力尾流观测完整。
- 压力失败不改变业务结果，但实验样本无效。

## 进程边界

```text
kvc_burst_coordinator sidecar
  - 压力 Key 预填充、刷新和验证
  - arm/disarm 与结果聚合
  - 不创建压力 KVClient，不传输压力 payload

brpc-inference
  - 从真实 single-key onboard 路径注册生产 KVClient
  - 31 个进程级常驻压力线程
  - 第一次业务Get启动压力并等待在飞门禁
  - 第二次业务Get完成后非阻塞stop
  - 记录addToken和request lifecycle边界

legacy sidecar-exclusive 模式
  - C-1 个独立 DataSystem 客户端
  - C-1 个常驻压力线程
  - 压力 Key 预填充和验证
  - 随机 lane/Key 映射
  - 压力结果异步聚合

```

两个容器通过tmpfs文件映射的 `SharedControl` V6 通信。协议使用固定宽度整数和 GCC process-shared
atomic builtins；futex 必须使用 `FUTEX_WAIT/FUTEX_WAKE`，不能使用进程私有 futex。

## 有效性门禁

每轮要求：

```text
pressure_client_registered              = 1
keys_verified                           = 31
business_get_count                      = 2
business_get_success_count              = 2
pressure_active_at_business_get_1       >= 30
pressure_active_at_second_get_start     >= 30
pressure_active_at_stop                 >= 30
pressure_get_errors                     = 0
pressure_completions_after_stop         > 0
pressure_tail_after_stop_ms             > 0
add_token_observation_count             = native add_token_count
business_lifecycle_done                 = true
native DataSystem attribution           = 2 Get + 3 Set
```

业务Get、stop、压力尾流、addToken和request lifecycle均使用同一进程的`CLOCK_MONOTONIC_RAW`。
结果同时报告stop时在飞数、stop前后完成数、stop到首个addToken间隔、addToken窗口以及两者交集，
不使用“线程已经创建”替代真实并发证据。

`client_e2e_ms`和`runner_ms`保留真实墙钟；`client_e2e_adjusted_ms`和`runner_adjusted_ms`只减去
第一次Get之前人为引入的pressure establishment等待，用于分析而非生产SLO。

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

默认对象严格匹配当前真实KV block的`3670016`字节，DataSystem使用25G endpoint：

```bash
DS_ENDPOINT=192.168.100.12:18482 \
CONCURRENCY_LEVELS='1 10 100' \
REPEATS=3 \
OBJECT_SIZE=3670016 \
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

## 真实Get边界语义

1. 第一次真实single-key Get按`request_id`领取generation。
2. 唤醒31个已注册worker，等待至少30个`pressure_active_gets`，再开始Get #1计时。
3. Get #1返回只记录结果，压力继续循环。
4. 同request的第二次真实Get记录Get #2独立起止时间。
5. Get #2成功返回后记录stop快照并发出非阻塞stop；业务线程立即继续。
6. 每个pressure worker完成当前SDK Get后看到stop，不再提交下一轮并退出。
7. addToken start/end和removeSequence completion只观测，不等待压力排空。
8. sidecar在业务lifecycle和压力worker都完成后按`request_id + generation`输出完整结果。

真实请求原有`3 Set + 2 Get`不做合并、重排或强制并行。若两次Get中任一次失败、缺失或出现
额外业务Get，该轮按`kBusinessGetCountMismatch`或业务失败处理，不把部分样本冒充有效结果。

Proxy热路径默认不打印完成或busy日志，只写共享控制块；唯一完成事件由sidecar异步输出。
`KVC_BURST_VERBOSE=1`仅用于故障诊断，不能用于正式时延采样。
