# DataSystem Worker Pooling

## 目标

当前线上验证链路已经跑通：

```text
PaiRec Go -> brpc/TCP -> C++ brpc_inference_server -> TensorRT-LLM C++ -> DataSystem KV
```

但 DataSystem 仍是固定 endpoint：

```text
DATASYSTEM_HOST=141.61.91.188
DATASYSTEM_PORT=18481
```

这次改造先把 DataSystem 从单 worker 改成共享 ETCD 下的 worker pool：

```text
datasystem-pool-etcd
  -> 141.61.91.188:12379

datasystem-pool-worker DaemonSet
  -> <node-host-ip>:18481

TRT-LLM C++ KVClient
  -> DATASYSTEM_ETCD_ADDRESS
  -> ServiceDiscovery
  -> same-node preferred worker
```

这里还不是 RH2D。当前 KV block 仍走 host buffer staging 的 `Create/Set/Get` 路径；RH2D 需要后续单独接 `MSetD2H/MGetH2D` 或设备 blob。

## 当前状态与阻塞点

截至 2026-06-25，远端 K8s 已经能把 pool 进程拉起来：

```text
datasystem-pool-etcd     worker1 141.61.91.188
datasystem-pool-worker   worker1 141.61.91.188
datasystem-pool-worker   master  141.61.91.189
```

这说明 hostNetwork 端口、DaemonSet 形态、master/worker1 双节点进程部署已经基本可用。

当前阻塞点在运行镜像 SDK 版本，不在 K8s 调度：

```text
image: docker.io/library/zcx-pairec-image:v1.1
yr.datasystem.__file__: /usr/local/lib/python3.11/site-packages/yr/datasystem/__init__.py
missing: ServiceAffinityPolicy
missing: yr.datasystem.service_discovery
available submodules: cli, ds_client, ds_tensor_client, hetero_client, kv_client, object_client, stream_client, util
```

因此：

- `scripts/test_datasystem_pool_smoke.sh` 依赖的 Python ServiceDiscovery API 在当前镜像里不可用。
- TensorRT-LLM C++ 侧的 ServiceDiscovery patch 即使写好，也必须等 runtime 内 DataSystem SDK/C++ headers/libs 更新后才能真实验证。
- 当前 `inference-brpc-trtllm` 仍应按固定 endpoint 风险看待，除非日志明确出现 `with ServiceDiscovery`。

后续恢复完整池化接入时，优先用 `/home/vivwimp/workspace/yuanrong-datasystem` 重建 DataSystem SDK/runtime，再重建 `zcx-pairec-trtllm-brpc-sdk:v1` 与 `pairec-brpc-inference:k8s-arm64-trtllm-v1`。

当前决策：完整 ServiceDiscovery 接入暂存为阻塞任务；基线摸测先复用 worker1+master pool 已启动的环境，但报告中要明确该阶段不等价于“TRT-LLM 已按 pool 路由 KV”。

## 跨节点固定 endpoint 验证

如果目标是先稳定验证：

```text
188 PaiRec -> brpc/TCP -> 189 inference -> 188 DataSystem worker
```

不要走 ServiceDiscovery 选点。当前 189 上也有 `datasystem-pool-worker`，如果启用
`DATASYSTEM_ETCD_ADDRESS` 且 runtime patch 生效，`PREFERRED_SAME_NODE` 会优先选择 189
本地 worker，不保证走 188。

本仓库提供独立 manifest：

- `k8s/deployment-inference-brpc-trtllm-cross-node-189-ds188.yaml`
- `scripts/k8s_apply_inference_brpc_trtllm_cross_node_189_ds188.sh`

该 manifest 保持 Deployment/Service 名称仍为 `inference-brpc-trtllm`，所以 PaiRec 的
`brpc_endpoint` 不需要变化；区别是：

- `nodeName: master`，把 inference Pod 固定到 189。
- `DATASYSTEM_HOST=141.61.91.188`、`DATASYSTEM_PORT=18481`，固定访问 188 worker。
- 不设置 `DATASYSTEM_ETCD_ADDRESS`，避免 ServiceDiscovery 覆盖固定 endpoint。

执行：

```bash
bash scripts/k8s_apply_inference_brpc_trtllm_cross_node_189_ds188.sh
```

验证：

```bash
kubectl -n pairec get pods -l app=inference-brpc-trtllm -o wide
kubectl -n pairec logs deploy/inference-brpc-trtllm -c brpc-inference --tail=160 \
  | grep -E 'with host endpoint|with ServiceDiscovery|Init KvCache'
```

期望：

```text
NODE=master
with host endpoint. host=141.61.91.188 port=18481
```

回滚到默认 inference 部署：

```bash
bash scripts/k8s_apply_inference_brpc_trtllm.sh
```

## 新增内容

- `k8s/deployment-datasystem-pool-hostnetwork.yaml`
  - 一个共享 ETCD Deployment。
  - 一个 DataSystem worker DaemonSet，默认在 master/worker1 都起 worker。
  - worker 使用 `HOST_IP` 注册 host id，便于同节点亲和选择。

- `scripts/k8s_apply_datasystem_pool.sh`
  - 删除旧 `datasystem` 单 worker Deployment。
  - 应用 pool manifest 并等待 ETCD/worker rollout。

- `scripts/test_datasystem_pool_smoke.sh`
  - 在 worker Pod 内使用 Python ServiceDiscovery。
  - 校验至少能发现 `EXPECT_MIN_WORKERS` 个 worker。
  - 随机选择 worker 后做一次 KV `set/get`。

- `trtllm-datasystem-service-discovery.patch`
  - 修改 TensorRT-LLM `kvCacheManager.cpp`。
  - 如果设置 `DATASYSTEM_ETCD_ADDRESS`，则用 `ConnectOptions.serviceDiscovery` 初始化 `KVClient`。
  - 如果未设置，则回退原来的 `DATASYSTEM_HOST/PORT` 固定 endpoint。

- `scripts/apply_trtllm_datasystem_service_discovery_patch.sh`
  - 给外部 TensorRT-LLM 源码树应用上述 patch。

## 部署顺序

先确保 master 和 worker1 的 K8s containerd 都有 DataSystem runtime 镜像：

```bash
sudo ctr -n k8s.io images ls | grep zcx-pairec-image
```

如果某个节点没有，需要先把本地镜像导入该节点的 `k8s.io` namespace；否则 DaemonSet 会尝试从外部仓库拉 `docker.io/library/zcx-pairec-image:v1.1`。

应用 DataSystem pool：

```bash
bash scripts/k8s_apply_datasystem_pool.sh
kubectl -n pairec get pods -l app.kubernetes.io/part-of=datasystem-pool -o wide
```

验证 ServiceDiscovery 能看到多个 worker：

```bash
EXPECT_MIN_WORKERS=2 bash scripts/test_datasystem_pool_smoke.sh
```

注意：当前 `zcx-pairec-image:v1.1` 会在该步骤失败，因为镜像内 DataSystem Python SDK 缺少 `ServiceAffinityPolicy` 和 `yr.datasystem.service_discovery`。该命令用于更新 SDK/runtime 后复验。

如果 smoke 没有输出或卡住，先跑分段诊断：

```bash
EXPECT_MIN_WORKERS=2 bash scripts/debug_datasystem_pool_smoke.sh
```

诊断脚本会依次检查 Pod/exec 通道、Python `yr.datasystem` import、ETCD 端口连通、ServiceDiscovery worker 选择和 KV `set/get`。最后一条已打印的阶段就是当前卡点。

## 让 TRT-LLM 真正使用池化

只 apply K8s 清单还不够。`brpc_inference_server` 里调用的是 TensorRT-LLM C++ 动态库，必须把 patch 编进当前 runtime 里的 TensorRT-LLM。

在源码树上先检查/应用 patch：

```bash
git -C /home/vivwimp/TensorRT-LLM apply --check \
  /home/vivwimp/pairec4tigerllm/trtllm-datasystem-service-discovery.patch

TRTLLM_DIR=/home/vivwimp/TensorRT-LLM \
  bash scripts/apply_trtllm_datasystem_service_discovery_patch.sh

rg "DATASYSTEM_ETCD_ADDRESS|makeDataSystemConnectOptions" \
  /home/vivwimp/TensorRT-LLM/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp
```

远端 runtime 里需要在对应的 TensorRT-LLM 源码目录应用同一 patch，并重新构建包含 `libtensorrt_llm.so` 的 base image。之后再重建 brpc TRT-LLM inference 镜像：

```bash
BASE_IMAGE=docker.io/library/zcx-pairec-trtllm-brpc-sdk:v1 \
  bash scripts/build_brpc_trtllm_inference_image.sh

bash scripts/k8s_apply_inference_brpc_trtllm.sh
```

如果只重建 `pairec-brpc-inference:k8s-arm64-trtllm-v1`，但 base image 里的 TensorRT-LLM C++ 库没有更新，进程仍然不会使用 ServiceDiscovery。

## 运行时配置

`k8s/deployment-inference-brpc-trtllm.yaml` 已增加：

```text
HOST_IP=(status.hostIP)
DATASYSTEM_ETCD_ADDRESS=141.61.91.188:12379
DATASYSTEM_CLUSTER_NAME=pairec
DATASYSTEM_HOST_ID_ENV_NAME=HOST_IP
DATASYSTEM_AFFINITY_POLICY=PREFERRED_SAME_NODE
DATASYSTEM_ENABLE_CROSS_NODE_CONNECTION=true
```

`PREFERRED_SAME_NODE` 的含义是：本节点有 worker 就优先用本节点 worker；没有时退到其他 ready worker。要强制只用本节点 worker，可改成 `REQUIRED_SAME_NODE`，但缺本地 worker 时推理进程会初始化失败。

## 验证点

TRT-LLM patch 生效后，`inference-brpc-trtllm` 日志应出现类似：

```text
[TensorRT-LLM][Datasystem] Init KvCache primary DataSystem with ServiceDiscovery. etcd=141.61.91.188:12379 cluster=pairec host_id_env=HOST_IP affinity=PREFERRED_SAME_NODE cross_node=1
```

然后再跑 brpc smoke：

```bash
TARGET=deployment/inference-brpc-trtllm \
CONTAINER=brpc-inference \
SERVER=10.96.15.101:18100 \
REQUESTS=1 \
TOPK=5 \
bash scripts/test_brpc_native_inference_smoke.sh
```

如果日志仍显示 `with host endpoint`，说明 `DATASYSTEM_ETCD_ADDRESS` 没有进入容器，或运行的 TensorRT-LLM C++ 库还不是 patched 版本。

## 回滚

最快回滚方式是取消 `DATASYSTEM_ETCD_ADDRESS`，让 C++ patch 自动回退到原来的固定 worker：

```text
DATASYSTEM_HOST=141.61.91.188
DATASYSTEM_PORT=18481
```

DataSystem 自身也可以切回旧单 worker：

```bash
bash scripts/k8s_apply_datasystem_hostnetwork.sh
```
