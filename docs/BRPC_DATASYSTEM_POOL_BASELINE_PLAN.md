# brpc + DataSystem Pool Baseline Plan

## 目标

先基于当前已打通链路摸一版可复现基线：

```text
PaiRec Go
  -> brpc/TCP
  -> C++ brpc_inference_server
  -> TensorRT-LLM C++ Executor
  -> DataSystem KV
```

DataSystem pool 进程先按 worker1 + master 两节点保留：

```text
datasystem-pool-etcd     141.61.91.188:12379
datasystem-pool-worker   worker1 141.61.91.188:18481
datasystem-pool-worker   master  141.61.91.189:18481
```

当前阶段的重点是拿到 E2E、brpc/TCP 和 KV/DataSystem 的 p99/p9999 基线，并统计一次推荐请求里 brpc 调用次数、KV/DataSystem 访问次数。

## 当前限制

这版基线不宣称已经完成 DataSystem ServiceDiscovery 路由。

原因是当前 runtime 镜像 `docker.io/library/zcx-pairec-image:v1.1` 内 DataSystem SDK 缺少：

```text
ServiceAffinityPolicy
yr.datasystem.service_discovery
```

所以除非 `inference-brpc-trtllm` 日志明确出现 `with ServiceDiscovery`，否则本阶段报告要按“DataSystem pool 已启动，但 TRT-LLM 可能仍走固定 endpoint”记录。

完整 worker pool 路由需要后续基于 `/home/vivwimp/workspace/yuanrong-datasystem` 重建 SDK/runtime 后再做。

## 基线范围

先测三层：

1. E2E：`PaiRec /api/recommend` 客户端时延。
2. brpc/TCP：PaiRec Go 到 C++ `brpc_inference_server` 的单次 `Recommend` RPC 时延。
3. KV/DataSystem：TensorRT-LLM C++ KV block `Set/Get` 次数和单次耗时。

目标统计：

```text
p50 / p95 / p99 / p9999 / max
request_count / success_count / error_count
brpc_calls_per_request
kv_set_count_per_request
kv_get_count_per_request
datasystem_worker endpoint 或 fixed endpoint 标记
```

## 验证前置

确认 pool 进程：

```bash
kubectl -n pairec get pods -l app.kubernetes.io/part-of=datasystem-pool -o wide
```

确认 brpc TRT-LLM 服务：

```bash
kubectl -n pairec get pods -l app=inference-brpc-trtllm -o wide
kubectl -n pairec logs deploy/inference-brpc-trtllm -c brpc-inference --tail=120
```

确认 PaiRec 已走 brpc 配置：

```bash
kubectl -n pairec get deploy pairec -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'
kubectl -n pairec exec deploy/pairec -- sh -c 'grep -n "brpc_endpoint" /app/configs/pairec_config.json'
```

做一次功能 smoke：

```bash
kubectl -n pairec exec deploy/pairec -- \
  wget -q -O - \
  --header='Content-Type: application/json' \
  --post-data='{"scene_id":"home_feed","uid":"6312","size":10}' \
  http://127.0.0.1:18080/api/recommend

kubectl -n pairec logs deploy/inference-brpc-trtllm \
  -c brpc-inference --since=2m | grep "method=Recommend" || true
```

## 摸测步骤

第一步先做单请求确认：

```bash
TARGET=deployment/inference-brpc-trtllm \
CONTAINER=brpc-inference \
SERVER=10.96.15.101:18100 \
REQUESTS=1 \
TOPK=10 \
bash scripts/test_brpc_native_inference_smoke.sh
```

第二步再做 E2E 小并发，固定并发 10、payload 约 100KB 的配置另开报告记录。若压测脚本需要补字段，优先复用 `scripts/benchmark_e2e_latency.py`，不要手工拼散乱日志。

第三步收集 DataSystem/KV 指标：

- 从 `brpc_inference_server` 日志统计 `method=Recommend` 次数和 `latency_ms`。
- 从 TensorRT-LLM DataSystem trace 统计 `op=offload`、`op=onboard`、`set_ms`、`get_ms`。
- 如果当前镜像没有 ServiceDiscovery 日志，则把 worker 选择标记为 `fixed_endpoint`。

## 输出要求

基线报告至少包含：

```text
环境:
  cluster nodes: master + worker1
  inference image:
  pairec image:
  datasystem image:
  brpc endpoint:
  datasystem mode: fixed_endpoint | service_discovery

E2E:
  concurrency:
  requests:
  success:
  p99:
  p9999:

brpc/TCP:
  calls_per_request:
  p99:
  p9999:

KV/DataSystem:
  set_count_per_request:
  get_count_per_request:
  set_ms_p99:
  set_ms_p9999:
  get_ms_p99:
  get_ms_p9999:
```

如果样本量不足以支撑 p9999，需要在报告中注明“方向性观察”，不能当稳定 SLA 结论。
