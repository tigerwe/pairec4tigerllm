# F14 严格 25G BRPC/KVC 压力测试收工记录

日期：2026-08-02

分支：`brpc-cross-node-189-ds188`

前台链路：`master/189 PaiRec -> 192.168.100.11:18100 worker1 TRT-LLM -> 192.168.100.12:18482 master DataSystem`

缓存形态：`PRIME_REQUESTS=195`，每个有效前台请求严格 `3 offload Set + 2 onboard Get`

## 判定口径

- 报告中的 `PASS` 只表示请求成功、调用数严格为 `3+2`、压力并发达到配置、且无 Pod 重启或 crash marker。
- `PASS` 不表示同时达到原定时延区间。
- BRPC 是 `Go probe RPC wall-clock - C++ server method latency` 的估算值，包含客户端编解码、brpc framing、TCP/CNI/kube-proxy，以及 method 计时边界之外的服务端开销。
- KVC 是同一前台请求中 3 次 offload 和 2 次 onboard 的结构化日志耗时之和。
- Gbps 是应用层 payload 吞吐，不是网卡物理计数器线速。

## 基线

| 样本 | E2E avg | E2E p95 | server avg | BRPC avg | KVC avg | offload avg | onboard avg | server-other avg | outer avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 同环境 baseline，10/10 有效 | 143.355 | 148.858 | 128.300 | 2.600 | 36.662 | 21.910 | 14.752 | 91.638 | 12.455 |

历史原始 25G 单请求基线为 E2E `124.072ms`、server `118ms`、BRPC `1ms`、KVC `29.835ms`、server-other `88.165ms`。本轮正式比较统一使用上表同环境 10 轮基线，不与历史单样本直接计算回退比例。

## 压力结果总表

| 压力案例 | 有效样本 | E2E avg/p95 | BRPC avg | KVC avg/p95/max | server-other avg | 压力证据 | 结论 |
|---|---:|---:|---:|---:|---:|---|---|
| KVC 17.5MiB，4 Get + 6 Set | 10/10 | 204.072 / 347.103 | 1.900 | 90.791 / 206.359 / 219.082 | 100.509 | Get约97-111 QPS，Set约112-119 QPS，inflight=4/6 | 最接近原目标；KVC p95进入200-210ms，但均值未进入 |
| KVC 1.75MiB，40 Get + 60 Set | 10/10 | 171.692 / 180.439 | 2.000 | 62.870 / 75.443 / 77.858 | 95.130 | Get约697-813 QPS，Set约976-1051 QPS，inflight=40/60 | 高并发小对象比低并发大对象影响更弱 |
| 独立 BRPC c10 | 10/10 | 148.529 / 157.136 | 1.500 | 40.169 / 47.382 / 48.686 | 94.831 | 约13.6-14.5k QPS | BRPC通信差值基本不变 |
| 独立 BRPC c100 | 10/10 | 146.432 / 158.666 | 2.200 | 40.414 / 53.330 / 58.583 | 未单列 | 独立 pressure target | BRPC通信差值基本不变 |
| 独立 BRPC c1000 | 10/10 | 162.244 / 191.665 | 4.500 | 53.270 / 77.558 / 80.438 | 93.730 | 独立 pressure target | 并发增加带来波动，但未形成40-50ms稳定BRPC均值 |
| KVC c100 + BRPC c100 | 10/10 | 162.194 / 169.628 | 2.000 | 51.236 / 60.268 / 61.959 | 97.664 | BRPC约6.1-6.6k QPS；KVC inflight=40/60 | 组合压力未线性叠加 |
| BRPC 512KiB c100，旧2 CPU | 3/3 | 157.056 / 167.375 | 3.667 | 41.607 / 45.324 / 45.651 | 98.727 | 约1.8k QPS | CPU配额限制压力源，不能作为网络饱和证据 |
| BRPC 100KiB c100，8 CPU | 3/3 | 167.048 / 169.190 | 2.333 | 58.099 / 60.560 / 60.691 | 93.901 | 约21.4k QPS/17.5Gbps；累计限流明显 | 提升压力吞吐，但前台BRPC仍约2ms |
| BRPC 100KiB c100，16 CPU | 3/3 | 166.556 / 175.157 | 2.333 | 57.525 / 68.151 / 69.873 | 93.808 | 24.68k QPS/20.22Gbps；限流周期2.486% | 当前最强且相对干净的network-only压力档 |
| BRPC 100KiB c1000，16 CPU | 3/3 | 157.491 / 175.465 | 10.000 | 44.513 / 53.375 / 54.248 | 94.821 | 19.14k QPS/15.68Gbps；限流6.648% | 过高并发降低吞吐，BRPC为7/3/20ms，波动大 |
| 双进程 BRPC c100x2，共享16 CPU cgroup | 3/3 | 142.630 / 152.937 | 1.333 | 38.223 / 49.544 / 51.458 | 92.110 | 19.47k QPS/15.95Gbps；限流11.295% | 拆进程但不增加CPU预算不能叠加吞吐 |
| shared-service c10，直接压18100 | 3/3 | 170.281 / 181.930 | 1.333 | 35.192 / 40.051 / 40.737 | 121.474 | 15.88k QPS/13.01Gbps；推理Pod限流周期97.253% | 主要制造推理Pod CPU竞争，不是BRPC通信压力；停止shared c100 |

## 最接近原目标的结果

原目标是均值 E2E `320-360ms`、BRPC `40-50ms`、KVC `200-210ms`、server-other `80-90ms`。没有任何一组有效实验同时进入四个区间。

最接近的是 `kvc-17.5m-c10`：

- E2E p95 `347.103ms`，进入原 E2E 区间。
- KVC p95 `206.359ms`，进入原 KVC 区间。
- BRPC均值仅 `1.900ms`，没有进入40-50ms。
- server-other均值 `100.509ms`，高于80-90ms。
- 个别轮次：round 10 为 E2E `330.829ms`、KVC `190.809ms`；round 3 为 E2E `360.417ms`、KVC `219.082ms`。这说明接近目标依赖排队尾部，不能解释为稳定均值命中。

## 实事求是的结论

1. 严格 `3 Set + 2 Get` 的缓存形态已经可重复构造，结构性门禁有效。
2. 真实 KVC 大对象低并发压力能够显著抬高尾延迟，且 KVC p95 已达到原200-210ms目标。
3. 当前独立 BRPC 压力的最强有效档约20.22Gbps，但前台 BRPC通信差值仍约2.33ms；没有证据支持40-50ms BRPC均值目标可以通过当前拓扑的纯网络压力稳定构造。
4. 增加并发到1000、在同一cgroup拆双进程，都会因压力源CPU/调度开销降低实际吞吐。
5. 直接压真实推理18100会让4 CPU推理Pod接近持续限流，主要抬高server-other，不是所需的BRPC通信时延。
6. 原目标把均值区间、尾延迟和共享资源竞争混在一起。更可执行的验收方式是固定负载下使用上界：成功率100%、严格3+2、restart/crash为0、E2E p95不超过360ms、KVC p95不超过210ms、BRPC p95不超过50ms，并单独报告server-other。

## 复现关键档位

```bash
source /tmp/brpc-kvc-cache-shape/20260801-114647/selected.env

RUN_ID="strict-25g-kvc17m-c10-$(date +%Y%m%d-%H%M%S)"
CASES=kvc-17.5m-c10 \
REPEATS=10 \
RUN_ID="$RUN_ID" \
OUT_ROOT="/tmp/brpc-kvc-pressure-matrix/$RUN_ID" \
BRPC_ENDPOINT=192.168.100.11:18100 \
KVC_DS_ENDPOINT=192.168.100.12:18482 \
KVC_LOAD_HOST=root@141.61.91.188 \
KVC_DSBENCH_CPP=/home/zcx/bin/dsbench-v081-sustained \
KVC_LOAD_READY_TIMEOUT_SECONDS=300 \
MATRIX_STRICT_COUNTS=1 \
bash scripts/benchmark_brpc_kvc_pressure_matrix.sh | tee "/tmp/${RUN_ID}.log"
```

主要结果目录：

- baseline: `/tmp/brpc-kvc-pressure-matrix/strict-25g-baseline-20260801-141404/baseline/result.json`
- strong KVC: `/tmp/brpc-kvc-pressure-matrix/strict-25g-kvc17m-c10-20260801-144136/kvc-17.5m-c10/result.json`
- mild KVC: `/tmp/brpc-kvc-pressure-matrix/strict-25g-kvc1750k-c100-20260801-155152/kvc-1.75m-c100/result.json`
- BRPC c10/c100/c1000: `/tmp/brpc-kvc-pressure-matrix/strict-25g-brpc-20260801-170130/`
- combined: `/tmp/brpc-kvc-pressure-matrix/strict-25g-combined-c100-kvc100-20260801-180750/combined-c100-kvc100/result.json`
- 16 CPU c100: `/tmp/brpc-kvc-pressure-matrix/brpc-100k-c100-16cpu-20260802-125751/brpc-c100/result.json`
- 16 CPU c1000: `/tmp/brpc-kvc-pressure-matrix/brpc-100k-c1000-16cpu-20260802-143238/brpc-c1000/result.json`
- dual process: `/tmp/brpc-kvc-pressure-matrix/brpc-100k-c100x2-16cpu-20260802-144939/brpc-c100x2/result.json`
- shared service: `/tmp/brpc-kvc-pressure-matrix/brpc-shared-100k-c10-20260802-150140/brpc-shared-c10/result.json`

## 收工状态

本轮停止继续加压，不运行 `shared-c100`。下一步开始前，先确认是否接受基于固定负载和p95上界的新验收口径；如果仍要求四项均值同时落入窄区间，需要先重新论证目标的业务依据和可控变量，不能用sleep或人工时延注入凑数。
