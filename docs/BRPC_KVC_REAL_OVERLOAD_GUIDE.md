# BRPC/KVC 真实过载实验指南

## 目标与边界

目标时延分布按均值或 p50 验收：

| 阶段 | 目标 |
|---|---:|
| E2E | 320-360ms |
| BRPC | 40-50ms |
| KVC | 200-210ms |
| 服务端其他阶段 | 80-90ms |

实验只允许通过真实 BRPC/TCP 和 DataSystem Set/Get 竞争制造过载。禁止 sleep、网络延迟注入、代理等待和服务端测试延迟。

前台请求必须满足 `3 Set + 2 Get`。不满足该缓存形态的样本不能进入统计。

## 1. 部署独立 BRPC 压力目标

在 master 的 PaiRec 仓库执行：

```bash
bash scripts/k8s_apply_brpc_pressure_target_188.sh
```

默认部署关系：

```text
正式推荐: 192.168.100.11:18100
压力目标: 192.168.100.11:18101
```

压力目标使用真实 BRPC Health、protobuf 和 TCP，但不初始化 TRT-LLM。这样 BRPC 压力主要竞争 25G 网卡和网络栈，不直接占用推荐服务的 TensorRT-LLM Executor。

## 2. 构建持续模式 dsbench

在拥有 DataSystem 源码和构建环境的 188 节点执行：

```bash
cd /home/zcx/workspace/pairec4tigerllm

DATASYSTEM_DIR=/home/zcx/yuanrong-datasystem-v081 \
  bash scripts/apply_datasystem_dsbench_sustained_patch.sh

cmake --build /home/zcx/yuanrong-datasystem-v081/build-sustained \
  --target dsbench_cpp -j"$(nproc)"

find /home/zcx/yuanrong-datasystem-v081/build-sustained \
  -type f -name dsbench_cpp -perm -111
```

如果远端使用的构建目录不是 `build`，替换为现有 DataSystem CMake build 目录。验证新二进制：

```bash
DSBENCH=/path/to/rebuilt/dsbench_cpp
DSBENCH_HELP=$("$DSBENCH" kv --help 2>&1 || true)
grep -E 'duration_seconds|ready_file|prepared_file|start_file|stats_file' <<<"$DSBENCH_HELP"
ldd "$DSBENCH" | grep 'not found' && exit 1 || true
```

持续模式会复用已初始化的 KVClient 和固定 key 集合，在指定时长内持续执行真实请求，不会不断重启 dsbench，也不会无限增加驻留对象。当前补丁还增加了两阶段门控和真实负载指标：

- `prepared_file`：客户端、固定 key 和预填充对象已经准备完成，但还没有发测量 RPC；
- `start_file`：benchmark 完成 inference 重置和 prime 后统一放行；
- `ready_file`：所有 worker 已经离开门控；
- `stats_file`：分别记录 GET/SET 的 calls、errors、QPS、Gbps 和 `max_inflight`。

如果新二进制依赖构建目录中的动态库，使用仓库脚本生成固定运行环境的 wrapper，避免依赖当前 Shell 中临时设置的 `LD_LIBRARY_PATH`：

```bash
cd /home/zcx/workspace/pairec4tigerllm

DATASYSTEM_DIR=/home/zcx/yuanrong-datasystem-v081 \
  bash scripts/create_datasystem_dsbench_wrapper.sh
```

脚本默认生成 `/home/zcx/bin/dsbench-v081-sustained`，并自动检查动态库、DataSystem `0.8.1` 版本以及全部持续压测参数。后续将该 wrapper 作为 `KVC_DSBENCH_CPP`，不要直接传裸二进制。

## 3. 校准 `3 Set + 2 Get`

校准会对每个候选值重启 inference Pod、执行固定 prime，再发一次 replay。候选值首次命中后，还会重启并连续确认 3 轮；只有 3 轮都保持 `3 Set + 2 Get` 才会被选中。默认尝试 `PRIME_REQUESTS=192..200`：

```bash
bash scripts/calibrate_brpc_kvc_cache_shape.sh \
  | tee /tmp/calibrate-brpc-kvc-cache-shape.log
```

成功后会生成：

```text
/tmp/brpc-kvc-cache-shape/<run_id>/selected.env
```

正式实验前加载该文件：

```bash
source /tmp/brpc-kvc-cache-shape/<run_id>/selected.env
```

其中包含经过验证的 `PRIME_REQUESTS`、`STRICT_COUNTS=1` 和每轮 inference 重置开关。

## 4. 按阶段调压

先跑同缓存形态 baseline：

```bash
CASES=baseline REPEATS=10 \
  bash scripts/benchmark_brpc_kvc_pressure_matrix.sh
```

再单独调 BRPC。100KB 压力发送到独立的 `18101`，前台推荐仍访问 `18100`：

```bash
CASES=brpc-c10,brpc-c100,brpc-c1000 REPEATS=10 \
  BRPC_LOAD_ENDPOINT=192.168.100.11:18101 \
  bash scripts/benchmark_brpc_kvc_pressure_matrix.sh
```

再单独调 KVC。两档均为 `Set:Get=3:2` 的 mixed 压力：

```bash
CASES=kvc-17.5m-c10,kvc-1.75m-c100 REPEATS=10 \
  KVC_LOAD_HOST=root@141.61.91.188 \
  KVC_DSBENCH_CPP="$DSBENCH" \
  bash scripts/benchmark_brpc_kvc_pressure_matrix.sh
```

对应并发：

| 档位 | Set | Get | 总并发 | 常驻对象总量 |
|---|---:|---:|---:|---:|
| 17.5MiB | 6 | 4 | 10 | 175MiB |
| 1.75MiB | 60 | 40 | 100 | 175MiB |

每个 worker 固定操作一个 key，`batch_num=1`，因此一次循环对应一次普通 DataSystem RPC。脚本不再把“worker 已启动”当成并发证据；只有 GET/SET 两侧都满足 `calls > 0`、`errors = 0`、QPS 非零且 `max_inflight` 达到配置值，样本才有效。

每轮顺序固定为：

```text
dsbench 预填充并等待 start_file
-> 重启 inference
-> prime 前台缓存形态
-> 启动 BRPC 压力
-> 写入 start_file 放行 KVC 压力
-> 等待压力稳定
-> replay
```

背景对象在 inference 重置和 prime 之前创建，避免 dsbench 的预填充把刚构造好的前台 KV 淘汰，从而提高 `3 Set + 2 Get` 的稳定性。

最后运行组合压力：

```bash
CASES=combined-c100-kvc100 REPEATS=30 \
  BRPC_LOAD_ENDPOINT=192.168.100.11:18101 \
  KVC_LOAD_HOST=root@141.61.91.188 \
  KVC_DSBENCH_CPP="$DSBENCH" \
  bash scripts/benchmark_brpc_kvc_pressure_matrix.sh
```

combined 模式会同时验收 BRPC 的真实 `max_active_calls`、dsbench GET/SET 各自的真实 `max_inflight` 和前台 `3 Set + 2 Get`，任一条件不满足都会将样本判为无效。GET 与 SET 来自两个进程，报告分别展示两侧峰值，不把两个峰值相加伪装成同一时刻的总在途 RPC。

## 5. 验收目标分布

```bash
python3 scripts/evaluate_brpc_kvc_latency_target.py \
  /tmp/brpc-kvc-pressure-matrix/<run_id>/combined-c100-kvc100/result.json
```

结果必须同时满足：

- 所有请求成功；
- 所有统计样本均为 `3 Set + 2 Get`；
- E2E、BRPC、KVC 和服务端其他阶段进入目标区间；
- inference Pod 没有重启或崩溃。

真实过载会产生波动，因此先用 10 轮调参，最终用至少 30 轮验收，并同时检查 p95 和最大值。
