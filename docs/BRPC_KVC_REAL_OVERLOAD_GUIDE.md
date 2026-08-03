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
压力目标1: 192.168.100.11:18101
压力目标2: 192.168.100.11:18102
```

两个压力目标进程位于同一个 Pod 和同一个 CPU cgroup，使用真实 BRPC Health、
protobuf 和 TCP，但不初始化 TRT-LLM。这样可以绕过单个 BRPC 进程的吞吐拐点，
同时保持总 CPU 配额不变；压力仍主要竞争25G网卡和网络栈，不直接占用推荐服务的
TensorRT-LLM Executor。
压力 Pod 固定申请 8 CPU、上限 16 CPU。8 CPU 实测在约 21.4k QPS、
17.5Gbps 应用层载荷下仍有 `138/888=15.5%` 的 CFS period 发生限流，
因此继续提高上限以避免压力服务先于 25G 链路成为瓶颈。apply 脚本会输出资源配置、
生效的 cgroup quota 和累计 `cpu.stat`；正式 benchmark 还会按轮记录前后差值，
汇总 `brpc_gbps` 与 `cpu_thr_pct`。

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

### 恢复严格 25G Worker

严格复现 `124.072ms` 基线时，inference 和背景压力必须共同使用
`192.168.100.12:18482`，不能将背景压力单独切到管理网 Worker
`141.61.91.189:18481`。如果 18482 没有监听，在 master 执行：

```bash
APPLY=0 bash scripts/restore_strict_25g_link.sh
```

默认只检查 master/worker1 的 `enp41s0f1`、carrier、速率、固定地址和双向
ping，不修改网络。如果分类为 `STRICT_25G_LINK_NOT_CONFIGURED`，且两端均确认
存在 `enp41s0f1`，再显式恢复原实验地址：

```bash
APPLY=1 bash scripts/restore_strict_25g_link.sh
```

必须看到 `STRICT_25G_LINK_RESTORED` 或 `STRICT_25G_LINK_READY`。该脚本只将
接口置为 UP 并恢复 `192.168.100.12/24`、`192.168.100.11/24`，不会修改 MTU、
默认路由、etcd 或 Kubernetes。之后部署 Worker：

```bash
KVC_LOAD_HOST=root@141.61.91.188 \
KVC_DSBENCH_CPP=/home/zcx/bin/dsbench-v081-sustained \
  bash scripts/deploy_datasystem_25g_master.sh \
  | tee /tmp/deploy-datasystem-25g-master.log
```

该脚本部署独立 `datasystem-25g-master`，固定绑定 master 的 25G 地址
`192.168.100.12:18482`，使用共享 etcd 但独立 cluster
`pairec-25g` 和独立 6Gi `/dev/shm`，不会修改现有 18481 pool。部署门禁包括：

- master 确实持有 `192.168.100.12`；
- etcd `141.61.91.189:12379` 可达且 18482 没有未知监听；
- Pod rollout/ready、188 到 18482 TCP 和 1KB DataSystem Set/Delete 通过；
- inference 的 `DATASYSTEM_HOST/PORT` 严格等于 `192.168.100.12/18482`。

最终必须输出 `DATASYSTEM_25G_MASTER_DEPLOYMENT_OK`，之后才能继续生命周期或
压力矩阵。删除该独立 Worker 的回滚命令为：

```bash
kubectl -n pairec delete deployment datasystem-25g-master
```

### Worker RPC 可用性诊断

如果 dsbench 在 `prefill` 或 `WarmUp` 阶段报告 `RPC_RECV_TIMEOUT`、
`RPC unavailable`，说明尚未进入 key 生命周期测试。先在 master 执行：

```bash
KVC_LOAD_HOST=root@141.61.91.188 \
KVC_DS_ENDPOINT=192.168.100.12:18482 \
KVC_DSBENCH_CPP=/home/zcx/bin/dsbench-v081-sustained \
  bash scripts/diagnose_datasystem_worker_rpc.sh \
  | tee /tmp/diagnose-datasystem-worker-rpc.log
```

脚本检查 188 到 Worker 的 TCP、残留压力进程和 PID 文件、master 监听、
DataSystem Pod/日志，并执行一次 1KB Set/Delete smoke。它不会重启 Worker、
inference 或批量终止进程。最终分类：

- `ENDPOINT_TCP_UNREACHABLE_FROM_LOAD_HOST`：188 无法建立到 Worker 的 TCP；
- `REMOTE_DSBENCH_UNAVAILABLE`：远端 0.8.1 wrapper 缺失或不可执行；
- `DATASYSTEM_RPC_UNAVAILABLE`：TCP 正常，但最小 Set RPC 失败；
- `DATASYSTEM_RPC_SMOKE_OK`：Worker RPC 已恢复，可继续 Get 生命周期诊断。

完整证据位于输出目录的 `diagnosis.txt`、`remote-state.log`、
`master-state.log`、`kubernetes-worker-logs.log` 和 `rpc-set-smoke.log`。

### 持续 Get 生命周期诊断

如果 dsbench 已输出 `prepared_workers=10`，放行后却报
`Cannot get objects from worker`，先不要重跑压力矩阵。在 master 的当前仓库执行：

```bash
KVC_LOAD_HOST=root@141.61.91.188 \
KVC_REMOTE_REPO=/home/zcx/workspace/pairec4tigerllm \
KVC_DSBENCH_CPP=/home/zcx/bin/dsbench-v081-sustained \
PRIME_REQUESTS=195 \
  bash scripts/diagnose_datasystem_sustained_get_lifecycle.sh \
  | tee /tmp/diagnose-datasystem-sustained-get.log
```

脚本会完成以下检查：

1. 在 188 预填充 4 个 Get key 和 6 个 Set key，并等待 10 个 worker 进入 prepared；
2. inference reset 前串行读取同一批 4 个 Get key；
3. 重启 inference 并执行前台 prime；
4. prime 后再次读取同一批 key；
5. 两次读取均成功后才写入 `start_file`，放行 4 Get + 6 Set，并收集 ready、stats 和远端日志。

`diagnosis.txt` 的 `failure_stage` 用于区分：

- `before_reset`：dsbench 预填充或对象生命周期本身有问题；
- `after_prime`：对象在 inference reset 或 prime 期间消失；
- `PASS`：key 生命周期正常，问题位于持续子进程放行或后续压力阶段。

脚本退出时会停止远端压力进程并清理本轮 key。

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

### 优先测量同实例 burst p99

需要测量突发请求对单次正式推荐 BRPC 通信时延的影响时，使用 burst Wrapper，
不要用持续 Health 压力的平均时延代替正式请求。每个 burst 固定包含：

```text
1 路正式 Recommend + N-1 路 100KB Health
```

所有 lane 先进入客户端屏障，再同时向同一个 `18100` 推理实例发起短连接请求。
Wrapper 单独报告：

- `armed_workers`：已到达统一释放屏障的 lane 数；
- `max_active_workers`：释放后真实调用尚未返回的最大 lane 数；
- `start_skew_us`：最早与最晚 lane 进入调用函数的时间差；
- `business_client_wall_ms`、`business_inference_ms`；
- `business_runner_generate_ms`：TRT runner.generate 耗时；
- `business_brpc_delta_ms = client_wall_ms - inference_ms`。

先用少量样本验证实验链路：

```bash
ENDPOINT=192.168.100.11:18100 \
BURST_CONCURRENCY_LEVELS='10 100' \
REPEATS=3 \
PRESSURE_PAYLOAD_BYTES=102400 \
  bash scripts/benchmark_go_brpc_burst_wrapper.sh
```

该命令会自动增加并发1的无压力 Recommend 基线，结果标记为 `PASS_SMOKE`，不能作为
正式p99。正式测试每档至少执行1000轮：

```bash
ENDPOINT=192.168.100.11:18100 \
BURST_CONCURRENCY_LEVELS='10 100 1000' \
REPEATS=1000 \
PRESSURE_PAYLOAD_BYTES=102400 \
  bash scripts/benchmark_go_brpc_burst_wrapper.sh
```

每个burst只产生一个正式Recommend样本。报告分别给出 client wall、服务端
inference、`runner_generate` 和 BRPC差值的p50/p95/p99/p999/max，并给出它们相对
并发1基线的p99变化。脚本不要求并发100或任何档位命中30-40ms，也不会按时延选择
并发档位。基线与压力档位按轮次交错执行，降低温度和缓存随时间漂移造成的偏差。
失败请求单独计数，不会被静默排除后仍将实验判为有效。

所有请求都发往同一个推理实例时，Health虽然不执行TRT/KVC，仍会竞争BRPC worker、
CPU、内存分配和网络队列，因此可能抬高正式推理时延。若 `runner_generate` p99也明显
上升，说明模型执行受到干扰；若主要是BRPC差值上升，则更偏向连接、排队和协议开销。
只有Recommend和Health全部成功、`max_active_workers >= N*95%`、Pod身份和重启数不变
且没有崩溃标记时，该档位才有效。

### 持续压力矩阵

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

单进程 `c1000` 若比 `c100` 吞吐更低，不再继续增加单进程并发。改用双进程档位，
总并发200会在两个端点间均分，每个压力进程保持100并发：

```bash
CASES=brpc-c100x2 REPEATS=3 \
  BRPC_LOAD_ENDPOINT=192.168.100.11:18101 \
  BRPC_LOAD_ENDPOINT_2=192.168.100.11:18102 \
  bash scripts/benchmark_brpc_kvc_pressure_matrix.sh
```

若独立压力端点已经达到单机最强有效吞吐但前台BRPC仍无明显变化，可以单独运行
shared-service档位。该档位把100KB Health压力发送到真实推理服务 `18100`，并把
CPU限流统计切换到inference Pod；它衡量的是BRPC服务队列竞争，不再是纯粹的
独立端点网络竞争，结果必须单独标记：

```bash
CASES=brpc-shared-c10 REPEATS=3 \
  bash scripts/benchmark_brpc_kvc_pressure_matrix.sh
```

只有c10保持严格 `3 Set + 2 Get`、restart/crash为0且server-other没有失控后，
才继续 `brpc-shared-c100`。

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
