# 工作进度

> 最后更新: 2026-06-26 | 当前状态: F14 PaiRec Go brpc client 端到端验证通过；DataSystem pool 已在 worker1 与 master 上达到 Pod Running，但完整 ServiceDiscovery 接入暂存为阻塞任务：当前 `zcx-pairec-image:v1.1` 内的 `yr.datasystem` SDK 缺少 `ServiceAffinityPolicy` 与 `yr.datasystem.service_discovery`；brpc + DataSystem pool 基线摸测已固化为并发脚本 `scripts/benchmark_brpc_datasystem_pool_baseline.sh` 和单请求分解脚本 `scripts/trace_single_brpc_datasystem_request.sh`。单请求脚本已跑出当前 PaiRec Go brpc 链路口径：一次 PaiRec 请求触发 1 次 brpc `Recommend`，C++ brpc server 约 107ms，client E2E 约 114-118ms，KVC/DataSystem 本轮观测为 2 次 offload、0 次 onboard；指标口径已修正：`brpc-inference latency_ms` 是 C++ inference 服务端推荐总耗时，不是 PaiRec 到 inference 的纯 brpc/TCP 通信耗时。跨节点验证当前切到 `brpc-cross-node-189-ds188` 分支的反向形态：`189 PaiRec -> brpc/TCP -> 188 inference -> 189 DataSystem`；远端 apply 后 Pod placement 已符合预期，inference 日志确认 `Rank 0 is using GPU 0` 且 DataSystem 固定到 `141.61.91.189:18481`。当前 `189 -> ClusterIP -> 188 inference` 的 Go brpc 探针已证明可用，PaiRec fallback 数据同步脚本也已补齐。为单独拆出 brpc 通信和 KVC 访问，新增 `scripts/benchmark_go_brpc_probe_kvc_latency.sh`：基于 Go brpc probe 直接测 `Go client -> brpc/TCP -> C++ inference`，输出 `go_probe_brpc_rpc_ms`、`server_method_ms`、估算 `brpc_comm_est_ms=client-server`，并从 inference `[Datasystem][TRACE]` 汇总 `offload.set/d2h/total` 与 `onboard.get/h2d/total` 的 p99/p9999。

## 时间线

| 日期 | 进度 |
|------|------|
| **6/26** | **新增探针式 brpc/KVC 跨节点时延脚本：`scripts/probe_go_brpc_client.go` 支持 `--requests` 在单进程内连续发送请求，避免 `go run` 编译开销混入单次延迟；`scripts/test_go_brpc_client_probe.sh` 同步支持 `REQUESTS`。新增 `scripts/benchmark_go_brpc_probe_kvc_latency.sh`：自动读取 `inference-brpc-trtllm` ClusterIP，启动 inference 日志采集，运行 Go brpc probe，并汇总 probe stdout 与 brpc inference/DataSystem 日志。输出口径：`go_probe_brpc_rpc_ms` 是 Go brpc client 调用 wall-clock，`server_method_ms` 是 C++ brpc server method latency，`brpc_comm_est_ms` 是两者差值上界，包含 Go encode/decode、brpc framing、TCP/CNI/kube-proxy 和 server method 外排队；KVC 口径来自 TRT-LLM C++ `[Datasystem][TRACE]`，按 per-block 汇总 offload/onboard 的 p50/p95/p99/p9999/max。本地 `gofmt`、`bash -n` 与 `go build -mod=vendor ./scripts/probe_go_brpc_client.go` 通过。** |
| **6/26** | **跨节点 PaiRec 请求未进入 inference 的根因定位完成：Go brpc client 探针成功，inference 日志出现 `[brpc-inference] method=Recommend user=go_brpc_probe code=200 items=8 latency_ms=89 backend=trtllm_cpp`，证明 `189 -> ClusterIP -> 188 inference` 的 Go brpc/TCP 通道正常。PaiRec `/tmp/recall_debug.log` 显示 `getUserHistory error: user 6312 not found in fallback`，PaiRec 日志显示 master/189 上 fallback 只加载了 2 个用户。结论：当前 `code=299` 不是 brpc/RPC/DataSystem 问题，而是 PaiRec 固定到 master 后 hostPath `/home/zcx/workspace/pairec4tigerllm/data/user_features.json` 与 worker1 不一致。新增 `scripts/sync_pairec_fallback_features_to_master.sh`，默认从 `141.61.91.188:/home/zcx/workspace/pairec4tigerllm/data/user_features.json` 同步到 master 同路径，先备份目标文件，再重启 PaiRec，因为 fallback 数据在进程内缓存。** |
| **6/26** | **补充 Go brpc client 独立探针：新增 `scripts/probe_go_brpc_client.go` 和 `scripts/test_go_brpc_client_probe.sh`。诊断已推进到新阶段：PaiRec ConfigMap 已改为 `brpc_endpoint=10.96.15.101:18100`，PaiRec Pod 内 `nc` 显示 TCP open，inference Pod 内 C++ brpc smoke 对 `127.0.0.1:18100` 返回 Recommend OK；但 PaiRec `/api/recommend` 后 inference 仍只有 smoke 的 `method=Recommend`，没有 user=`6312`。该探针复用 `services/recall` 中同一套 Go 手写 baidu_std brpc client，直接从 master host 访问 inference Service ClusterIP，区分“Go brpc client 协议/响应解析问题”和“PaiRec 召回流程没有走到 client 调用”。** |
| **6/26** | **跨节点路由诊断结果明确：`scripts/diagnose_pairec_brpc_route.sh` 输出显示 inference Pod 内 direct brpc smoke 成功，`brpc_inference_server` 返回 Health OK 且 Recommend user=`brpc_smoke_1` code=200/items=5/latency_ms=222；但 PaiRec `/api/recommend` 返回 `code=299` 后，inference 近 5 分钟日志只有 smoke 的 `method=Recommend`，没有 user=`6312` 的调用。PaiRec Pod 内 `/etc/resolv.conf` nameserver 为 `10.96.0.20`，`nslookup` 对该 DNS 超时并出现 `Message too large`，说明当前断点是 PaiRec 到 inference Service DNS 解析失败，而不是 inference/TRT-LLM/DataSystem 服务端故障。已修复诊断脚本对转义 `RecallAlgo` 的 `brpc_endpoint` 解析，并新增 `scripts/k8s_patch_pairec_brpc_endpoint_to_cluster_ip.sh` 自动将 `brpc_endpoint` 改为 inference Service ClusterIP，重启 PaiRec 以绕过 CoreDNS。** |
| **6/26** | **新增跨节点 PaiRec -> brpc inference 路由顺序诊断脚本：`scripts/diagnose_pairec_brpc_route.sh`。脚本会依次打印 Pod 拓扑、DataSystem pool、Service/Endpoints、Deployment nodeName、inference 的 `DATASYSTEM_*` env 与 args、PaiRec 配置内 `brpc_endpoint`，再从 PaiRec Pod 内检查 `/etc/resolv.conf`、Service DNS 与 TCP 连通性；可选执行 `scripts/test_brpc_native_inference_smoke.sh` 直连 inference Pod 的 `127.0.0.1:18100`，再发送一次 PaiRec `/api/recommend` 请求，并收集 inference `method=Recommend`、KVC/DataSystem trace、PaiRec stdout 和 `/tmp/recall_debug.log`。本地 `bash -n` 通过，目标是定位当前 `189 PaiRec -> 188 inference -> 189 DataSystem` 形态下 PaiRec 返回 `code=299` 且 inference 无 `method=Recommend` 的断点。** |
| **6/26** | **反向跨节点拓扑进入 inference 启动参数兼容问题：当前 `189 PaiRec -> 188 inference -> 189 DataSystem` 已调度成型，`pairec` 在 master/189 Running，`inference-brpc-trtllm` 在 worker1/188 CrashLoop；日志显示镜像内 `/opt/pairec-brpc/bin/brpc_inference_server` 不认识 `--trt_max_batch_size=1`，Usage 中也没有 `--trt_max_num_tokens`。这是远端镜像内 brpc 二进制版本与清单参数不一致导致。已从 `k8s/deployment-inference-brpc-trtllm.yaml`、`k8s/deployment-inference-brpc-trtllm-cross-node-189-ds188.yaml`、`k8s/deployment-inference-brpc-trtllm-cross-node-188-ds189.yaml` 中移除 `--trt_max_batch_size` 与 `--trt_max_num_tokens`，先兼容当前已导入镜像；若后续需要容量参数，应重建并导入支持这些 flags 的 brpc inference 镜像。** |
| **6/26** | **因 189 GPU 不再作为 inference 资源，新增反向跨节点验证方案：目标拓扑改为 `189 PaiRec -> brpc/TCP -> 188 inference -> 189 DataSystem`。新增 `k8s/deployment-inference-brpc-trtllm-cross-node-188-ds189.yaml`：保持 `inference-brpc-trtllm` Service/Deployment 名称，`nodeName=worker1` 固定推理到 188，`DATASYSTEM_HOST=141.61.91.189`/`DATASYSTEM_PORT=18481` 固定访问 189 DataSystem，并继续省略 `DATASYSTEM_ETCD_ADDRESS` 避免 ServiceDiscovery。新增 `k8s/deployment-pairec-brpc-master.yaml`：PaiRec CPU 服务 `nodeName=master` 固定到 189，使用 `pairec-server:k8s-arm64-brpc-v1` 与 `PAIREC_TRACE_STDOUT=1`。新增 `scripts/k8s_cleanup_master_gpu_attempt.sh` 用于 scale 到 0、删除失败 inference Pod、移除 master 的 `pairec/gpu` label 并清理 master 上的 nvidia-device-plugin Pod；新增 `scripts/k8s_apply_189_pairec_188_inference_189_ds.sh` 一键 apply 反向拓扑并输出验证命令。验证：两个脚本 `bash -n` 通过，两个新增 YAML parse 均为 2 docs，`git diff --check` 通过。** |
| **6/26** | **master/189 GPU 分配继续诊断：`inference-brpc-trtllm` scale 到 0 后，master 的 `Allocated resources` 显示 `nvidia.com/gpu 0/0`，说明 GPU 没被业务 Pod 占用；但 `kube-system/nvidia-device-plugin-daemonset-644p2` 在 master 上变为 `CrashLoopBackOff`，worker1 上同 DaemonSet 正常 Running。此前 `OutOfnvidia.com/gpu` 不是业务 Pod 抢占导致，而是 master 的 NVIDIA device plugin 不健康。用户曾用 `kubectl -n pairec describe pod nvidia-device-plugin...` 查询，因 namespace 错误返回 NotFound；下一步应在 `kube-system` 查看该 Pod 的 current/previous logs 与 describe，定位 device plugin crash 原因。** |
| **6/26** | **跨节点 inference apply 后发现远端 live Deployment 残留开发二进制 overlay：Pod 事件显示 `MountVolume.SetUp failed for volume "brpc-dev-bin": hostPath type check failed: /home/zcx/pairec-brpc-dev/bin/brpc_inference_server is not a file`。本地 `k8s/deployment-inference-brpc-trtllm-cross-node-189-ds188.yaml` 并无 `brpc-dev-bin`，说明这是早期为避免重传 150G 镜像而在远端 live Deployment 上留下的临时 hostPath 覆盖。跨节点稳定验证应移除该 `volumeMount`/`volume`，使用镜像内 `/opt/pairec-brpc/bin/brpc_inference_server`；也可临时在 master 上补文件，但不建议作为稳定方案。** |
| **6/26** | **master/189 GPU 接入完成：`nvidia-device-plugin-daemonset` 在 master 与 worker1 上均为 `1/1 Running`，`kubectl describe node master` 的 Capacity 与 Allocatable 均出现 `nvidia.com/gpu: 1`。此前的 master Calico、containerd `nvidia` runtime handler、device plugin 镜像问题均已越过。下一步可以正式 apply `scripts/k8s_apply_inference_brpc_trtllm_cross_node_189_ds188.sh`，验证 `inference-brpc-trtllm` 调度到 master/189，并从日志确认 TensorRT-LLM 使用 GPU 且 DataSystem 连接固定到 `141.61.91.188:18481`。** |
| **6/26** | **新增 master/189 NVIDIA device plugin 镜像同步脚本：`scripts/sync_nvidia_device_plugin_image_to_master.sh`。脚本默认在 master 本机执行，自动读取 `kube-system/nvidia-device-plugin-daemonset` 当前镜像（如 `nvcr.io/nvidia/k8s-device-plugin:v0.19.2`），通过 ssh 在 worker1/188 用 `ctr -n k8s.io images export` 导出该镜像，复制并导入 master 本机 `k8s.io` containerd namespace，然后只重建调度到 master 的 nvidia-device-plugin Pod，并打印 device plugin Pod 与 master `nvidia.com/gpu` allocatable 验证结果。验证：`bash -n scripts/sync_nvidia_device_plugin_image_to_master.sh` 与 `git diff --check` 通过。** |
| **6/26** | **master/189 Calico 已恢复：`calico-kube-controllers`、master/worker1 上的 `calico-node` 均为 `1/1 Running`，说明 master 的 CNI 网络创建问题已越过。新的阻塞是 master 上 `nvidia-device-plugin` 拉取 `nvcr.io/nvidia/k8s-device-plugin:v0.19.2` 失败，报 `x509: certificate signed by unknown authority`，状态 `ImagePullBackOff`；worker1 上已有同镜像运行，因此下一步应从 worker1 的 containerd 导出该 nvidia device plugin 镜像并导入 master，再重建 master 上的 nvidia-device-plugin Pod。** |
| **6/26** | **新增 master/189 Calico 镜像同步脚本：`scripts/sync_calico_images_to_master.sh`。脚本默认在 master 本机执行，从当前 kube-system 的 `calico-node` DaemonSet 和 `calico-kube-controllers` Deployment 自动提取镜像列表，通过 ssh 在 worker1/188 用 `ctr -n k8s.io images export` 导出这些镜像，复制到 master 并导入本机 `k8s.io` containerd namespace，随后只重建调度到 master 的 Calico Pod。该脚本不直接修改 `/etc/cni/net.d`，优先解决 master 上 `calico-node Init:ImagePullBackOff`。验证：`bash -n scripts/sync_calico_images_to_master.sh` 与 `git diff --check` 通过。** |
| **6/26** | **master/189 Calico CNI 根因继续定位：`calico-kube-controllers` 被调度到 master 后同样因 Calico CNI `error getting ClusterInformation: connection is unauthorized` 无法创建 sandbox；`kubectl -n kube-system get pods` 显示 master 上 `calico-node-47v5z` 已 33 天处于 `Init:ImagePullBackOff`，worker1 上 calico-node 仍在运行。这说明 master 上新 Pod 失败不是 nvidia-device-plugin 特例，而是 master 节点 Calico CNI/节点网络组件长期未正常初始化。下一步应先让 master 的 calico-node 正常起来，优先同步 worker1 已有 Calico 镜像到 master containerd 并重建 master calico-node；若 calico-node Running 后仍 Unauthorized，再检查/同步 `/etc/cni/net.d` 下 Calico CNI kubeconfig。** |
| **6/26** | **master/189 NVIDIA runtime 配置后进入下一阻塞：`nvidia-device-plugin-daemonset` 已同时出现在 worker1 与 master，其中 master Pod 能被调度但卡 `ContainerCreating`；事件已从此前的 `no runtime for "nvidia" is configured` 变为 Calico CNI `plugin type="calico" failed (add): error getting ClusterInformation: connection is unauthorized: Unauthorized`。这说明 containerd `nvidia` runtime handler 已不再是当前报错点，新的阻塞是 master 节点 Calico CNI 本地 kubeconfig/token/RBAC 授权异常，导致新 Pod sandbox 网络创建失败。下一步优先重启/重建 master 上 calico-node 以刷新 `/etc/cni/net.d`，若仍失败再对比 worker1 与 master 的 Calico CNI kubeconfig 并谨慎同步。** |
| **6/26** | **新增 master/189 NVIDIA runtime 一键同步脚本：`scripts/sync_nvidia_runtime_to_master.sh`。脚本默认在 master/189 本机执行，从 worker1/188 通过 ssh 读取 `/etc/containerd/conf.d/99-nvidia.toml`，安装到本机同路径，备份 `/etc/containerd/config.toml` 后只补齐 `imports` 中的 `99-nvidia.toml`，不覆盖整份 containerd 配置；随后可自动重启 containerd/kubelet、重建 nvidia device plugin Pod、给 master 打 `pairec/gpu=true` 标签，并打印 `containerd config dump`、`crictl info` 与 `kubectl describe node master` 的验证结果。验证：`bash -n scripts/sync_nvidia_runtime_to_master.sh` 与 `git diff --check` 通过。** |
| **6/26** | **追溯 GPU 节点差异来源：项目历史在 6/9 只记录 `worker1 已注册 nvidia.com/gpu=1` 并固化 inference hostPath manifest，K8s README 也只把 `nvidia.com/gpu` 与 `runtimeClassName=nvidia` 写成部署前置条件；仓库内没有安装/同步 NVIDIA device plugin、containerd `99-nvidia.toml` 或 master GPU runtime 的脚本。结合现场 `containerd config dump` 对比，结论是当时只在 worker1/188 做过节点本地 NVIDIA runtime 配置，master/189 虽有 GPU 和 `/usr/bin/nvidia-container-runtime`，但没有导入 `/etc/containerd/conf.d/99-nvidia.toml` 到 containerd CRI runtime handler，因此 K8s 无法在 master 上使用 `RuntimeClass nvidia`。** |
| **6/26** | **master/189 containerd NVIDIA runtime 差异已定位：worker1/188 的 `containerd config dump` 显示 `imports = ["/etc/containerd/config.toml", "/etc/containerd/conf.d/99-nvidia.toml"]`，并在 CRI runtimes 中存在 `nvidia` handler，`BinaryName=/usr/bin/nvidia-container-runtime`；master/189 的 `containerd config dump` 与 `crictl info` 中均无 `nvidia`。两台机器都有 `/usr/bin/nvidia-container-runtime`，因此修复方向是把 worker1 的 `/etc/containerd/conf.d/99-nvidia.toml` 与必要的 imports 配置同步到 master，并重启 containerd/kubelet，再让 `nvidia-device-plugin-daemonset` 在 master 变为 Running。** |
| **6/26** | **master/189 GPU 接入继续推进：给 master 补齐调度条件后，`nvidia-device-plugin-daemonset` 已能被调度到 master，但 Pod 卡在 `ContainerCreating`，事件为 `failed to get sandbox runtime: no runtime for "nvidia" is configured`。用户检查到 worker1/188 与 master/189 都存在 `/usr/bin/nvidia-container-runtime`，但直接 grep `/etc/containerd/config.toml /etc/containerd/*.toml` 未发现 `nvidia` 配置；因此当前差异不在二进制是否存在，而在 containerd 实际加载的 CRI runtime handler 配置。下一步应对比两台机器的 `containerd config dump`、`crictl info` 与 `systemctl cat containerd`，找出 worker1 上 `nvidia` handler 的来源并同步到 master，重启 containerd/kubelet 后确认 master allocatable 出现 `nvidia.com/gpu`。** |
| **6/26** | **跨节点拓扑前置状态更新：用户确认 master/189 物理有 GPU，且 master/189 上的 `datasystem-pool-worker` 已从 Unknown 恢复为 `1/1 Running`，说明 pause/sandbox 侧问题暂时越过；但 `kubectl describe node master | grep nvidia.com/gpu` 仍无输出，而 worker1/188 有 `nvidia.com/gpu: 1`。`nvidia-device-plugin-daemonset` 当前只在 worker1 上有 Pod，master 上没有 device-plugin Pod，因此 189 仍不能承载请求 GPU 的 `inference-brpc-trtllm`。当前剩余关键动作是让 NVIDIA device plugin 在 master 上运行并使 node allocatable 出现 `nvidia.com/gpu`，随后再 apply 跨节点 inference。** |
| **6/26** | **远端跨节点拓扑前置检查发现尚未满足：`kubectl get nodes` 显示 master/189 与 worker1/188 均为 Ready，但 `kubectl describe node master` 的 Allocatable 未出现 `nvidia.com/gpu`，说明 189 当前还没有把 GPU 资源暴露给 K8s；同时 `datasystem-pool-worker` 在 master/189 上为 Unknown，事件显示 pause sandbox 镜像 `docker.io/library/pause-aarch64:3.8` 拉取超时以及 `cpuset.mems` 缺失导致 sandbox 反复重建。当前实际拓扑仍是 188 PaiRec -> 188 inference -> 188 DataSystem；要切到 188 PaiRec -> 189 inference -> 188 DataSystem，需先修 189 的 pause/sandbox 与 NVIDIA device plugin/GPU allocatable，再导入 `pairec-brpc-inference:k8s-arm64-trtllm-v1` 到 189 containerd，最后 apply `scripts/k8s_apply_inference_brpc_trtllm_cross_node_189_ds188.sh`。** |
| **6/26** | **修正基线指标口径：`brpc@TCP*n` 不应直接使用 C++ `brpc-inference` 日志里的 `latency_ms`。该字段覆盖 brpc server 收到请求后执行 Recommend 的服务端总耗时，包含 TensorRT-LLM generate、semantic map、KVC offload/onboard 等后端处理；真正的 PaiRec -> inference brpc/TCP 通信/RPC耗时需要在 PaiRec Go brpc client 调用前后打点，或在 brpc 框架/client 侧采集 request send/response receive wall-clock。后续基线报告需拆成：E2E client、PaiRec Go 内部、brpc RPC wall-clock、inference server total、KVC Get/Set、other residual。** |
| **6/25** | **新增稳定跨节点固定 endpoint 验证分支：已切到 `brpc-cross-node-189-ds188`，新增 `k8s/deployment-inference-brpc-trtllm-cross-node-189-ds188.yaml` 与 `scripts/k8s_apply_inference_brpc_trtllm_cross_node_189_ds188.sh`。该 manifest 保持 `inference-brpc-trtllm` Deployment/Service 名称不变，PaiRec `brpc_endpoint` 不用改；通过 `nodeName: master` 将 inference 固定到 189，通过 `DATASYSTEM_HOST=141.61.91.188`、`DATASYSTEM_PORT=18481` 固定访问 188 DataSystem，并故意不设置 `DATASYSTEM_ETCD_ADDRESS`，避免 ServiceDiscovery 在 189 上优先选本地 worker。`docs/DATASYSTEM_WORKER_POOLING.md` 已补验证和回滚说明。** |
| **6/25** | **补 PaiRec Go stdout 观测增强：`services/recall/generative_recall.go` 新增 `PAIREC_TRACE_STDOUT=1` 门控的 `[PAIREC_TRACE]` 输出，默认关闭，不改变推荐链路行为；开启后在 cache hit、history error/empty、convert error、inference error 和成功返回时向 stdout 输出 `requestId/request_id/module=GenerativeRecall/from/protocol/cache_ms/history_ms/convert_ms/rpc_ms/brpc_ms/http_ms/items_ms/tr_*` 等字段，便于 `kubectl logs --tail=0 -f deploy/pairec` 稳定采集单请求内部阶段。`scripts/trace_single_brpc_datasystem_request.sh` 已同步展示 `from/protocol/rpc_ms/brpc_ms/inference_svc_ms` 字段。本机验证：`gofmt`、`bash -n scripts/trace_single_brpc_datasystem_request.sh`、`go test -mod=vendor ./services/...`、`go build -mod=vendor ./...`、`git diff --check` 均通过。下一步远端重建 PaiRec 小镜像并开启 `PAIREC_TRACE_STDOUT=1` 复跑单请求 trace。** |
| **6/25** | **手工交接状态更新：当前推荐链路是 `HTTP /api/recommend -> PaiRec Go -> Go baidu_std brpc/TCP client -> C++ brpc_inference_server -> TensorRT-LLM C++ -> KVC/DataSystem`。`brpc_recommend_client` 只是 C++ smoke/probe client，不是当前 PaiRec Go 改造口径。最近两次 `scripts/trace_single_brpc_datasystem_request.sh` 结果分别为：E2E `114.158ms/117.509ms`，brpc calls `1/1`，brpc server `107ms/107ms`，KVC `offload=2/onboard=0`，offload total 分别约 `14.479+8.010ms` 和 `16.816+7.427ms`。PaiRec Go 内部阶段日志仍未从 `kubectl logs` 或 Pod `/tmp` glog 文件中抓到；下一步应给 `services/recall/generative_recall.go` 增加可开关的 stdout trace，再重建 `pairec-server:k8s-arm64-brpc-v1` 复测。** |
| **6/25** | **补充单请求链路分解脚本：新增 `scripts/trace_single_brpc_datasystem_request.sh`，默认只发一次 `USER_ID=6312 SIZE=1` 的 PaiRec `/api/recommend` 请求，自动选择 port-forward 空闲端口，并以 `kubectl logs --tail=0 -f` 只采本次请求后的 PaiRec/brpc TRT-LLM 日志；请求完成后按 response `request_id` 额外在 PaiRec Pod `/tmp` glog 文件中检索 `GenerativeRecall/RecommendTrace` 行。输出包含 client E2E latency、response code/size、brpc `Recommend` 调用次数与 server `latency_ms`、PaiRec `cache/history/convert/http/items` 及 `tr_*` 阶段耗时（若日志可抓到）、KVC/DataSystem `offload/onboard` 次数，以及每次 `Create/D2H/Set`、`Get/H2D`、`total_ms` 明细；输出目录为 `/tmp/pairec_single_request_trace/<run_id>`。该脚本用于先解释一次请求内部发生了什么，再做并发基线。** |
| **6/25** | **固化 brpc + DataSystem pool 基线一键脚本：新增 `scripts/benchmark_brpc_datasystem_pool_baseline.sh`，默认检查 `pairec/inference-brpc-trtllm/datasystem-pool` Pod 与 Service，验证 PaiRec `brpc_endpoint` 配置，跑一次 PaiRec `/api/recommend` 功能 smoke，再用 `scripts/test_brpc_native_inference_smoke.sh` 做 C++ brpc/TCP 串行 smoke；随后启动本地 port-forward 与分轮日志采集，默认跑 `size=1, concurrency=10` 的系统 E2E 基线，并可选跑 `size=10, concurrency=10` 的召回质量基线。每轮会输出 benchmark 原始 stdout、JSON、PaiRec/brpc 日志，并自动统计 brpc Recommend 调用数、code 分布、item 分布、brpc server latency p50/p95/p99/p9999/max、DataSystem offload/onboard 事件数、offload/onboard per brpc，以及 `set_ms/get_ms/total_ms` p99/p9999。脚本已补充空闲本地端口自动选择：默认从 18080 开始，若被旧 port-forward 占用则自动换后续端口并同步更新 benchmark URL；日志采集改为 `kubectl logs --tail=0 -f`，避免历史 brpc/KV 日志混入本轮 summary；默认 `E2E_WARMUP=0`，让 summary 的 brpc/KV 调用数与 pressure/replay 请求数对齐，需预热时可显式设置。`docs/BRPC_DATASYSTEM_POOL_BASELINE_PLAN.md` 已改为优先使用该脚本。** |
| **6/25** | **DataSystem worker pool 任务暂存并转入基线摸测：远端 `datasystem-pool-etcd` 与两个 `datasystem-pool-worker` 已经在 worker1/master 上 Running，说明 hostNetwork、DaemonSet 和共享 ETCD 部署形态已通过 Pod 级验证；但 `debug_datasystem_pool_smoke.sh` 在 Python import 阶段确认当前 `docker.io/library/zcx-pairec-image:v1.1` 里的 DataSystem SDK 没有 `ServiceAffinityPolicy`，也没有 `yr.datasystem.service_discovery` 子模块，可用子模块仅包括 `cli/ds_client/ds_tensor_client/hetero_client/kv_client/object_client/stream_client/util`。结论：这不是 K8s 调度问题，而是运行镜像 SDK 版本不支持当前 ServiceDiscovery smoke/API；完整池化接入需后续基于 `/home/vivwimp/workspace/yuanrong-datasystem` 重建 DataSystem SDK/runtime，并重新构建 TRT-LLM/brpc runtime。当前决策：池化完整接入暂列阻塞，先在 worker1+master DataSystem pool 已启动的环境下摸 E2E/brpc/KV 基线，并在报告中标注 TRT-LLM C++ 可能仍走固定 endpoint 的限制。** |
| **6/25** | **补充并修复 DataSystem pool smoke 卡顿诊断脚本：远端 `datasystem-pool-etcd` 与两个 `datasystem-pool-worker` 已进入 Running，但 `scripts/test_datasystem_pool_smoke.sh` 执行无输出，后续诊断确认根因是 `kubectl exec ... python -` 缺少 `-i`，导致 heredoc 没有传入容器，Python 实际没有执行脚本内容。修复后进一步定位到当前 worker 镜像的 `yr.datasystem` 顶层没有导出 `ServiceAffinityPolicy`；脚本已改为优先从 `yr.datasystem` 导入，失败时 fallback 到 `yr.datasystem.service_discovery` 子模块，并在子模块也缺失时打印 `yr.datasystem.__file__`、可用 ServiceDiscovery 符号与子模块列表，便于判断是导出问题还是 SDK 版本过旧。验证：`bash -n scripts/test_datasystem_pool_smoke.sh scripts/debug_datasystem_pool_smoke.sh` 与 `git diff --check` 通过。** |
| **6/24** | **DataSystem worker 池化第一版本地落地：新增 `k8s/deployment-datasystem-pool-hostnetwork.yaml`，以 `datasystem-pool-etcd` 提供共享 ETCD `141.61.91.188:12379`，以 `datasystem-pool-worker` DaemonSet 在各节点启动 `<HOST_IP>:18481` worker，并通过 `HOST_IP` 写入 `host_id_env_name` 支持同节点亲和；新增 `scripts/k8s_apply_datasystem_pool.sh` 和 `scripts/test_datasystem_pool_smoke.sh`，用于部署 pool、验证 ServiceDiscovery 至少发现多个 worker 并做一次 KV `set/get`。新增 `trtllm-datasystem-service-discovery.patch` 和 `scripts/apply_trtllm_datasystem_service_discovery_patch.sh`，将 TensorRT-LLM `KvCacheManagerDataSystem/Tmp` 的 `KVClient` 初始化改为：存在 `DATASYSTEM_ETCD_ADDRESS` 时使用 `ConnectOptions.serviceDiscovery`，否则回退原 `DATASYSTEM_HOST/PORT`；同步给 `k8s/deployment-inference-brpc-trtllm.yaml` 增加 `DATASYSTEM_ETCD_ADDRESS`、`DATASYSTEM_CLUSTER_NAME`、`DATASYSTEM_HOST_ID_ENV_NAME`、`DATASYSTEM_AFFINITY_POLICY=PREFERRED_SAME_NODE`、`DATASYSTEM_ENABLE_CROSS_NODE_CONNECTION=true` 等运行时参数。新增 `docs/DATASYSTEM_WORKER_POOLING.md` 记录部署/重建/验证/回滚流程。验证：`bash -n` 三个新增脚本通过；两个 K8s YAML parse 输出 `yaml ok`；`git -C /home/vivwimp/TensorRT-LLM apply --check trtllm-datasystem-service-discovery.patch` 通过；`git diff --check` 通过。远程遗留：需要实际 apply pool、确认两个节点均有本地 DataSystem runtime 镜像，并重建包含该 TensorRT-LLM patch 的 runtime/base image，否则运行中 brpc TRT-LLM 仍不会真正使用 worker pool。** |
| **6/24** | **调研最新 `/home/vivwimp/workspace/yuanrong-datasystem` 及 RH2D：最新仓 `ConnectOptions` 已包含 `serviceDiscovery`、`enableRemoteH2D`、`fastTransportMemSize`、`deviceId`，C++ `ObjectClientImpl::Init()` 在传入 `serviceDiscovery` 时会通过 `ServiceDiscovery::SelectWorker()` 从 ETCD ready worker 中选点，并支持 `PREFERRED_SAME_NODE`/`REQUIRED_SAME_NODE`/`RANDOM` 亲和策略；Python `ServiceDiscovery`/`KVClient`/`DsClient` 也暴露相同能力。上游 Helm DaemonSet 已有 `host_id_env_name`、`etcd_address`、`cluster_name`、`enable_worker_worker_batch_get` 等参数，说明共享后端和 worker 池化应优先通过同一 ETCD + ServiceDiscovery 接入。RH2D 方面，最佳实践明确该路径是 Ascend NPU + CANN + RoCE 的 remote host-to-device 传输，worker 端需 `--remote_h2d_device_ids`，client 端需 `enableRemoteH2D=true`，API 侧以 `HeteroClient::MSetD2H/MGetH2D` 或 `KVClient::MGetH2D` 的设备指针/DeviceBlobList 为核心；当前 NVIDIA CUDA/TensorRT-LLM 部署和现有 patch 的 `Create/Set/Get` host buffer staging 不能直接启用 RH2D。建议阶段顺序：先把当前 TRT-LLM `KVClient` 初始化改造成可选 ServiceDiscovery 并验证跨 worker Set/Get，再在 Ascend/NPU 环境单独做 RH2D standalone benchmark，最后再改 KV block transfer 逻辑对接设备 blob，保留 host staging fallback。** |
| **6/24** | **从 `/home/vivwimp/TensorRT-LLM` 侧细化未使用 DataSystem worker 池化能力的根因：当前 patch 只在 `kvCacheManager.h` 引入 `datasystem/kv_client.h`，没有引入 `ServiceDiscovery`/`RouterClient`；`KvCacheManagerDataSystem` 与 `KvCacheManagerDataSystemTmp` 两个单例都只从 `DATASYSTEM_HOST`/`DATASYSTEM_PORT` 构造 `datasystem::KVClient`，没有读取 ETCD 地址、clusterName 或 affinity 参数，并显式设置 `enableCrossNodeConnection=false`、`enableRemoteH2D=false`；`kvCacheTransferManager.cpp` 的 offload/onboard 只通过这两个单例执行 `Create/Set/Get`，因此所有 KV block 都落到同一个固定 worker endpoint。结论：当前是为验证 C++ KV offload/onboard 做的最小 host/port 接入，不是 DataSystem 官方多 worker service discovery 接入。** |
| **6/24** | **调研 yuanrong-datasystem 共享后端能力：上游 README 明确 DataSystem 依赖 ETCD 做节点发现/健康检测/扩缩容，每节点部署 worker 并注册 ETCD，worker-worker 当前支持 TCP 传输；Helm chart 默认 `replicaNum=2`，并有 `enableWorkerWorkerBatchGet`、`enableRedirect`、`enableDataReplication` 等集群/远端取数相关配置。源码层 `ConnectOptions` 已包含 `serviceDiscovery`，`ObjectClientImpl::Init()` 可通过 `ServiceDiscovery::SelectWorker()` 从 ETCD 选择 worker，`KVClient` 内部复用 `ObjectClientImpl`；同时 worker 侧存在 `BatchGetObjectRemote` / `GetObjectRemote` 远端取数实现。当前项目差距是 TensorRT-LLM patch 只从 `DATASYSTEM_HOST/PORT` 构造 `KVClient`，且 K8s `inference-brpc-trtllm` 固定指向 worker1 `141.61.91.188:18481`，`deployment-datasystem-hostnetwork.yaml` 也是单 worker + isolated etcd，因此当前运行时尚未真正启用共享后端。下一步应先用两个 worker 注册同一 ETCD 做跨 worker Set/Get 验证，再决定是否改 TensorRT-LLM DataSystem 初始化参数** |
| **6/23** | **F14 PaiRec Go brpc client 远端 K8s 端到端验证通过：用户重新构建 `pairec-server:k8s-arm64-brpc-v1`、导入 worker1 并重启 PaiRec 后，`kubectl -n pairec exec deploy/pairec -- wget ... /api/recommend` 返回 `{"code":200,"msg":"success","size":10,...}`，10 个 item 均来自 `retrieve_id=generative_recall`；同时 `kubectl -n pairec logs deploy/inference-brpc-trtllm -c brpc-inference --since=2m | grep method=Recommend` 输出 `[brpc-inference] method=Recommend user=6312 code=200 items=10 latency_ms=228 backend=trtllm_cpp`。这证明最终链路已打通：PaiRec Go `GenerativeRecall` -> Go baidu_std brpc/TCP client -> `inference-brpc-trtllm:18100`/ClusterIP -> C++ `brpc_inference_server` -> TensorRT-LLM C++ backend。遗留：CoreDNS 当前 `0/1 ContainerCreating`，Pod 内 DNS 解析 `10.96.0.20` 超时，因此本次用 `10.96.15.101:18100` ClusterIP 绕过；后续需修 CoreDNS 并恢复 Service DNS 配置** |
| **6/23** | **修正 PaiRec Go baidu_std brpc `RpcMeta` 字段号：远端切到新镜像和 ClusterIP 后，`/tmp/recall_debug.log` 显示 brpc 请求已不再 DNS 超时，但返回 `service error: ;`，且 C++ brpc inference 未打印 `method=Recommend`，说明 Go client 能收到 brpc frame 但服务端没有正确识别请求业务 meta。对照 Apache brpc `baidu_rpc_meta.proto` 后确认 Go 手写 meta 字段号错误；已改为 `request=1,response=2,compress_type=3,correlation_id=4,attachment_size=5,content_type=10,checksum_type=11`，并补齐 `CONTENT_TYPE_PROTO` 与 `CHECKSUM_NONE`。验证：`go test -mod=vendor ./services/...`、`go build -mod=vendor ./...`、`git diff --check` 通过。下一步远端 pull 后重新构建 `pairec-server:k8s-arm64-brpc-v1` 并复测 `/api/recommend`** |
| **6/23** | **补充 PaiRec 小镜像完全绕过 Docker Hub builder 镜像的构建路径：用户远端构建继续失败于 `FROM golang:1.24-alpine`，错误为 BuildKit layer `failed size validation`，判断为 Docker registry mirror/cache 返回异常 layer，不是 Go 代码问题。新增 `docker/Dockerfile.pairec.binary` 与 `scripts/build_pairec_binary_image.sh`：先在宿主机执行 `GOPROXY=off GOSUMDB=off CGO_ENABLED=0 go build -mod=vendor` 生成静态 `/app/pairec-server`，再以本地已有 `docker.io/library/pairec-server:k8s-arm64-static` 为 runtime base 只覆盖二进制和配置，避免拉取 `golang:1.24-alpine` 和 `alpine:3.19`。验证：`bash -n scripts/build_pairec_binary_image.sh` 与 `git diff --check` 通过。远端下一步优先执行 `bash scripts/build_pairec_binary_image.sh docker.io/library/pairec-server:k8s-arm64-brpc-v1`** |
| **6/23** | **修复 PaiRec 小镜像构建依赖外网 Alpine apk 的问题：用户远端构建 `docker/Dockerfile.pairec` 时在 runtime stage `apk add --no-cache ca-certificates curl bash` 因 Alpine package index 拉取失败而中断。由于 `pairec-server` 是 CGO disabled 静态 Go 二进制，当前 K8s hostpath 部署也直接执行 `/app/pairec-server`，runtime 不需要联网安装 `bash/curl`；已删除 Dockerfile runtime stage 的 `apk add`，将健康检查从 `curl` 改为 Alpine busybox 自带 `wget`，并将 `docker/entrypoint-pairec.sh` 从 bash 改为 POSIX sh。验证：`sh -n docker/entrypoint-pairec.sh` 与 `git diff --check` 通过。下一步远端 `git pull` 后重新执行 `docker build -f docker/Dockerfile.pairec -t docker.io/library/pairec-server:k8s-arm64-brpc-v1 .`** |
| **6/23** | **F14 PaiRec Go brpc client 第一版落地：在 `services/recall` 新增手写 protobuf 消息和 baidu_std brpc/TCP frame client，不引入在线 Go SDK 依赖，覆盖 `RecommendService.Health/Recommend`；`TRTLLMClient` 新增 `protocol=brpc` 分支，brpc 成功时直接返回，失败时按 `brpc_fallback_to_http` 回退原 HTTP `/recommend`；`RecallAlgo` 支持 `protocol`、`brpc_endpoint`、`brpc_service_name`、`brpc_fallback_to_http`。新增 `configs/pairec_config.brpc.json`、`k8s/configmap-brpc.yaml` 和 `scripts/k8s_apply_pairec_brpc_hostpath.sh`，默认 HTTP 配置保持不变。验证：`go test -mod=vendor ./services/...` 通过，`go build -mod=vendor ./...` 通过，`bash -n scripts/k8s_apply_pairec_brpc_hostpath.sh`、`python -m json.tool configs/pairec_config.brpc.json`、`k8s/configmap-brpc.yaml` YAML parse 和 embedded RecallAlgo JSON parse 均通过。下一步需远端重建/替换 PaiRec server 镜像并跑 `/api/recommend` 端到端 brpc 验证** |
| **6/22** | **F14 真实 C++ TRT-LLM native brpc smoke 通过：用户对 `deployment/inference-brpc-trtllm` 执行 `scripts/test_brpc_native_inference_smoke.sh`，Health 返回 `health ok latency_ms=1 code=200 status=healthy`；Recommend 返回 `recommend ok index=1 latency_ms=224 code=200 user_id=brpc_smoke_1 items=5 inference_ms=223`，`summary ok=1 total=1 total_ms=224`。这条链路为 `brpc_recommend_client -> brpc_inference_server:18100 -> TensorRT-LLM C++ Executor -> semantic_id_map`，不经过 Flask HTTP，已证明 F14 真实 C++ brpc 推理服务可用。下一步进入 PaiRec Go 侧 brpc/TCP client 接入，并保留 HTTP fallback 方便灰度/回滚** |
| **6/22** | **F14 真实 C++ TRT-LLM brpc 后端远程启动成功：用户在 `inference-brpc-trtllm` Pod 日志中确认 `trtllm plugin preflight ok namespace=tensorrt_llm legacy_creators=36 v3_creators=3`，creator 列表包含 `GPTAttention/Gemm/GemmSwiglu` 等核心插件；TensorRT-LLM engine 反序列化成功，`Engine load time 313 ms`，容量参数生效为 `maxNumSequences=1/maxBatchSize=1/maxNumTokens=96`；KV cache 显示 `primaryBlocks=8 secondaryBlocks=28`，DataSystem C++ KV 初始化成功：`Create Datasystem class`、`Init KvCache Manager DataSystem success`；`brpc_inference_server` 完成初始化并监听 `0.0.0.0:18100`。这说明前序插件缺失、OOMKilled、tokenizer/config 路径问题均已越过；下一步是执行 brpc Health/Recommend smoke 验证真实推理响应** |
| **6/22** | **补强 TensorRT-LLM plugin registry 同类问题预检：在 `brpc_inference_server` 中新增 `--trt_plugin_preflight_only=1` 模式，只执行 `initTrtLlmPlugins()` 并列出 `libnvinfer_plugin_tensorrt_llm.so` 注册出的 plugin creators，不加载 engine、不启动 brpc server；正常 `trtllm_cpp` 启动路径也会先执行该预检并输出 `legacy_creators/v3_creators/creators` 日志。预检会 fail-fast 检查 Qwen engine 常见核心 creator：`Gemm`、`GPTAttention`、`GemmSwiglu`，避免只修 `Gemmtensorrt_llm` 后又等到 engine 反序列化阶段才发现另一个 creator 未注册。本机验证：brpc build 脚本 `bash -n`、`git diff --check`、TRT-LLM CMake 配置阶段均通过；真实 ARM 编译和 GPU Pod 预检仍需远程执行** |
| **6/22** | **F14 `backend=trtllm_cpp` 远程验证推进到 TensorRT-LLM plugin registry 阶段：worker1 K8s 日志显示 `maxNumSequences=1`、`maxBatchSize=1`、`maxNumTokens=96`，说明前序 Executor 容量参数已生效，OOMKilled 不再是当前主因；最新 CrashLoopBackOff 为 `Cannot find plugin: Gemmtensorrt_llm, version: 1, namespace:tensorrt_llm`，即 C++ brpc 进程在创建 TensorRT-LLM Executor 前没有显式注册 TensorRT-LLM 自定义插件。已修 `brpc_inference_server.cpp`：引入 `tensorrt_llm/plugins/api/tllmPlugin.h` 并在构造 Executor 前调用 `initTrtLlmPlugins()`；同步修 CMake/Docker/build 脚本，新增 `TRTLLM_PLUGIN_LIBRARY`，默认链接 `/TensorRT-LLM/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so` 并加入 install RPATH。本机验证：`bash -n` 三个 brpc build 脚本通过，`git diff --check` 通过，CMake TRT-LLM 配置阶段能识别新增 plugin lib 参数；远程下一步无需重新传 150G tar，优先用 dev binary overlay 覆盖 worker1 上 `/home/zcx/pairec-brpc-dev/bin/brpc_inference_server` 后 rollout restart** |
| **6/18** | **F14 `backend=trtllm_cpp` 第一版代码已落地，待远程 ARM/GPU runtime 验证：`cpp/brpc_gateway/brpc_inference_server.cpp` 新增 TensorRT-LLM C++ Executor 后端，启动时加载 `/app/trt_engines/qwen3_rec_v4`、`/app/exported/qwen3_rec/pairec_cpp_tokenizer.txt` 和 `semantic_id_map.json`；请求侧按当前 Python `TRTQwen3Backend` 的 prompt 结构构造 token ids，保留 `max_input_len=64`、`max_new_tokens=32`、`num_samples/top_k/temperature` 参数，生成后按 layer 收集 semantic special tokens、缺失层补 0、笛卡尔积组合并映射 item；trace 保留 `prompt_ms`、`runner_generate_ms`、`runner_calls`、`parse_combo_ms`、`map_item_ms`、`backend_total_ms`。新增 `PAIREC_ENABLE_TRTLLM_CPP` CMake 开关、TRT-LLM brpc 镜像构建脚本、`scripts/export_cpp_trt_tokenizer_config.py`、`k8s/deployment-inference-brpc-trtllm.yaml` 和 apply 脚本。本机验证：`python -m py_compile scripts/export_cpp_trt_tokenizer_config.py`、新增脚本 `bash -n`、`k8s/deployment-inference-brpc-trtllm.yaml` YAML parse、`protoc --cpp_out`、`git diff --check`、CMake TRT OFF 配置和 TRT ON 配置分支均通过；尚未在远程 ARM TensorRT-LLM runtime 内完成真实编译、镜像导入和 K8s smoke** |
| **6/18** | **F14 native C++ brpc inference K8s smoke 通过：`scripts/test_brpc_native_inference_smoke.sh` 输出 `health ok latency_ms=1 code=200 status=healthy`，`Recommend` 输出 `recommend ok index=1 latency_ms=1 code=200 user_id=brpc_smoke_1 items=5 inference_ms=0`。这条链路是 `brpc_recommend_client -> inference-brpc-native:18100 -> brpc_inference_server -> semantic_map backend`，不再经过 Flask HTTP 转发；当前验证的是协议、镜像、Service/Deployment 和 C++ brpc server 可用性，不代表真实模型推理时延。下一步进入 `backend=trtllm_cpp`，把 TensorRT-LLM C++ runner、tokenizer/prompt、semantic id 解析和 DataSystem/KV 运行时接入该 server** |
| **6/18** | **补强 brpc 镜像离线构建：master 本地已有 `zcx-pairec-image:v1.1` 与 `zcx-pairec-brpc-sdk:v1`，但使用 `docker.io/library/zcx-*` 作为 `BASE_IMAGE` 会触发 daocloud 元数据解析并因私有镜像不在白名单返回 403。已将 brpc build 脚本默认 base image 改成本地 tag；新增 `.dockerignore` 排除 `datasystem/uds` Unix socket，避免 legacy builder 打包 build context 时输出 `archive/tar: sockets not supported`；`docker/Dockerfile.brpc.gateway` 在 build/final stage 清空继承的 `LD_PRELOAD`，避免 DataSystem 预加载路径污染 brpc 镜像构建和 ldd 检查** |
| **6/18** | **统一镜像 tar 保存路径：后续 build/ship 脚本默认把导出的 `.tar` 或 `.tar.gz` 放到 `/home/zcx`，不再使用 `/tmp` 作为镜像中转目录；`ship_brpc_gateway_to_worker.sh` 与 `ship_brpc_inference_to_worker.sh` 会先创建本地 tar 目录，scp 到 worker 时也默认放到 `/home/zcx`，再由 worker 执行 `ctr -n k8s.io images import /home/zcx/*.tar`。已同步 `k8s/README.md` 示例和相关构建脚本输出** |
| **6/18** | **F14 调整方向：已用 `git revert` 生成提交撤销 PaiRec 侧 `brpc_http_proxy` sidecar 方案，因为该形态仍是 `HTTP -> brpc -> HTTP`，对时延优化意义不大；新增 `cpp/brpc_gateway/brpc_inference_server.cpp`，直接实现 `RecommendService`，不调用 Flask HTTP。当前 `backend=semantic_map` 会加载 `/app/data/tenrec/processed/semantic_id_map.json` 并返回确定性 mapped items，用于证明 native C++ brpc service、镜像、K8s 和客户端协议链路；`backend=trtllm_cpp` 当前 fail-fast，作为后续真实 TensorRT-LLM C++ runner 接入口。新增 `k8s/deployment-inference-brpc-native.yaml`、`scripts/build_brpc_inference_image.sh`、`scripts/ship_brpc_inference_to_worker.sh`、`scripts/k8s_apply_inference_brpc_native.sh`、`scripts/test_brpc_native_inference_smoke.sh`。下一步先远程构建/导入 `pairec-brpc-inference:k8s-arm64-v1` 并做 native smoke，再迁移 Python `TRTQwen3Backend` 的 prompt/tokenizer/runner.generate/parse/map trace 到 C++ backend** |
| **6/18** | **F14 brpc phase-1 K8s smoke 验证通过：基于 `docker.io/library/pairec-brpc-gateway:k8s-arm64-v1` sidecar，`scripts/test_brpc_gateway_smoke.sh` 输出 `health ok latency_ms=2 code=200 status=healthy`；`Recommend` 输出 `recommend ok index=1 latency_ms=267 code=200 user_id=brpc_smoke_1 items=5 inference_ms=264.52`，返回 5 个推荐 item，trace 显示 `backend=trt-qwen3`、`runner_calls=1`、`runner_generate_ms=212.35ms`、`total_ms=264.52ms`、`result_cache_source=disabled`。这证明 native brpc/`baidu_std` over TCP -> brpc-gateway sidecar -> localhost Flask `/recommend` -> TRT-LLM 的 phase-1 闭环已跑通；该结果暂不代表最终性能收益，因为 gateway 内部仍转发 HTTP，下一步需要同批 HTTP vs brpc 延迟和结果等价对比** |
| **6/16** | **补强 F14 brpc gateway 镜像构建防线：`scripts/build_brpc_gateway_image.sh` 在 docker build 后自动进入最终镜像，对 `/opt/pairec-brpc/bin/brpc_gateway` 与 `/opt/pairec-brpc/bin/brpc_recommend_client` 执行 `ldd`，若出现 `not found` 立即失败，避免再次出现编译成功但 K8s sidecar 运行时缺 `libbrpc.so`/Abseil/protobuf 等动态库导致 CrashLoopBackOff 的问题；`docs/F14_BRPC_GATEWAY_DESIGN.md` 已同步说明该校验。下一步重新构建 gateway 镜像时应先看到 ldd 全部解析成功，再执行 worker1 导入和 K8s smoke** |
| **6/16** | **F14 brpc 环境搭建流程已固化：新增 `scripts/build_brpc_sdk_image.sh`，基于 `zcx-pairec-image:v1.1` 构建 `docker.io/library/zcx-pairec-brpc-sdk:v1`，在专用 SDK base image 中安装编译依赖并构建/安装 Apache brpc headers/libs；新增 `scripts/ship_brpc_gateway_to_worker.sh`，固化 master 构建 gateway 镜像后的 `docker save`、`scp` 到 worker1、`ctr -n k8s.io images import` 或 `docker load` 导入流程；更新 `docs/F14_BRPC_GATEWAY_DESIGN.md` 为当前 master 构建、worker1 导入的实际 K8s 镜像流转方式。新增脚本本机 `bash -n` 通过，`feature_list.json` JSON parse 和 `git diff --check` 通过。当前远程 master 应使用本地 base tag：`BASE_IMAGE=zcx-pairec-image:v1.1 bash scripts/build_brpc_sdk_image.sh docker.io/library/zcx-pairec-brpc-sdk:v1`、`BASE_IMAGE=zcx-pairec-brpc-sdk:v1 bash scripts/build_brpc_gateway_image.sh`、`WORKER=root@141.61.91.188 bash scripts/ship_brpc_gateway_to_worker.sh`，再 apply/smoke** |
| **6/15** | **F14 brpc phase-1 PoC 设计与实现骨架完成：新增 `proto/recommend.proto` 定义 `RecommendService.Recommend/Health`，字段对齐当前 `/recommend` JSON 并保留 `raw_json` 便于对比；新增 `cpp/brpc_gateway/recommend_gateway.cpp`，作为 native brpc `baidu_std`/TCP server 监听 `18100`，内部通过 brpc HTTP channel 转发到 `127.0.0.1:18000/recommend`；新增 `cpp/brpc_gateway/recommend_client.cpp` 作为 C++ smoke client；新增 `docker/Dockerfile.brpc.gateway`、`k8s/deployment-inference-brpc-image.yaml` 和构建/部署/验证脚本。当前本机验证：`protoc --proto_path=proto --cpp_out=/tmp/pairec-brpc-proto-check proto/recommend.proto` 通过；`bash -n scripts/build_brpc_gateway_image.sh scripts/k8s_apply_inference_brpc_gateway.sh scripts/test_brpc_gateway_smoke.sh` 通过；`python -c "import yaml; list(yaml.safe_load_all(open('k8s/deployment-inference-brpc-image.yaml'))); print('yaml ok')"` 输出 `yaml ok`；`cmake -S cpp/brpc_gateway -B /tmp/pairec-brpc-cmake-check` 找到 Protobuf 3.12.4 后按预期失败于 `brpc SDK was not found`，说明本机缺 brpc headers/libs，需远程或专用 builder 镜像完成编译。已补 `scripts/build_brpc_sdk_image.sh` 固化 brpc SDK base image 构建，并补 `scripts/ship_brpc_gateway_to_worker.sh` 固化 master 构建、tar 导出、scp 到 worker1、containerd/docker 导入流程。F14 仍为 pending，下一步是远程构建 gateway 镜像并在 inference Pod 内做 `Health`/`Recommend` smoke，再输出 HTTP vs brpc 延迟对比** |
| **6/12** | **补充 E2E DataSystem 时延报告指标口径说明：在 `docs/E2E_DATASYSTEM_FINAL_REPORT_2026-06-04.md` 新增“指标口径详细说明”，逐项解释 `client_e2e_ms`、PaiRec `total/user_feature/recall/filter/rank/merge/sort`、GenerativeRecall `cache/history/convert/http/items/http_overhead`、inference `tr_*`/TRT service stages，以及 C++ DataSystem per KV block `offload.create/d2h/set/total`、`onboard.get/h2d/total`。报告明确区分 request 级、服务内阶段级和 KV block 级指标，并标注 `tr_*` 与 inference trace 字段的别名关系，避免将 Python 结果缓存/DataSystem 指标与 TensorRT-LLM C++ KV block Set/Get 混淆** |
| **6/12** | **K8s C++ KV DataSystem offload/onboard 功能闭环验证通过：基于正式 `pairec-inference:k8s-arm64-ds-runtime-v1` 镜像运行 pressure/replay 压测，日志判定显示 KV pool `primaryBlocks=32 secondaryBlocks=28`、scheduler `MAX_UTILIZATION`、`reuse disabled warning seen=False`，新增结构化 DataSystem trace `offload/onboard=1313/3`，`Get Key=3`，说明 C++ 层已真实触发 GPU KV block offload 到 DataSystem 并从 DataSystem onboard 回 HBM。当前样本量仍不足以给 p9999 结论：offload count=1313、onboard count=3，低于脚本建议 minimum=10000/recommended>=100000；下一步只需扩大 replay/onboard 压测样本用于分位数报告，不再需要改 runtime 镜像或 engine** |
| **6/12** | **正式 K8s inference DS runtime 镜像启动验证通过：基于 `scripts/build_datasystem_runtime_base_image.sh` 固化的 patched TensorRT-LLM runtime 重建 `docker.io/library/pairec-inference:k8s-arm64-ds-runtime-v1` 后，inference Pod 启动日志显示 `[TRTQwen3Backend] Scheduler policy: max_utilization, max_kv_tokens=1024, host_cache_size=104857600`、KV pool `primaryBlocks=32 secondaryBlocks=28`、`[TensorRT-LLM][Datasystem] Create Datasystem class`、`Init KvCache Manager DataSystem. host = 141.61.91.188, ip = 18481` 与 `Init KvCache Manager DataSystem success`。Python DataSystem 按预期保持 `PYTHON_DATASYSTEM_ENABLED=0`，当前只验证 C++ KV path。下一步运行 `scripts/test_trt_cpp_kv_offload.py` 的 pressure/replay 压测，确认结构化 `[Datasystem][TRACE] op=offload/onboard` 出现并汇总 C++ Set/Get latency** |
| **6/12** | **确认 K8s C++ DataSystem 差异来自 runtime 镜像而非 engine：用户在历史跑通容器 `3d25ebe028d6` 内用同一 `qwen3_rec_v4` 启动服务，日志出现 `[TensorRT-LLM][Datasystem] Create Datasystem class`、`Init KvCache Manager DataSystem success`，且 `Blocks per window size` 显示 `primaryBlocks=32 secondaryBlocks=28`，说明该容器内 TensorRT-LLM C++ patch 与 `block_ds_consumer.so/stub_gpu.so/libabseil` 预加载链路有效。进一步确认 `tensorrt-llm` 是 editable project，实际位置为 `/TensorRT-LLM`，而该路径是宿主机 `/home/zcx/TensorRT-LLM` 的 bind mount，因此不能直接 `docker commit` 旧容器。已新增 `scripts/build_datasystem_runtime_base_image.sh`，从旧容器复制 `/TensorRT-LLM` 到新的 `docker.io/library/zcx-pairec-ds-runtime:v1` base image；正式 runtime Dockerfile 继续基于该 base 构建，并在镜像内编译 `/opt/pairec/lib/block_ds_consumer.so` 与 `/opt/pairec/lib/stub_gpu.so`；entrypoint 默认 `PYTHONPATH` 同时包含 `/home/TensorRT-LLM`、`/TensorRT-LLM` 与 cutlass python 路径，启动时清理旧 `LD_PRELOAD`，按顺序加载 block/stub/NVML/abseil；K8s manifest 默认 tag 为 `docker.io/library/pairec-inference:k8s-arm64-ds-runtime-v1`。下一步重建/导入新镜像后复验 C++ offload/onboard TRACE** |
| **6/12** | **修正 K8s C++ DataSystem KV 失败判断：用户在历史跑通容器中检查 `trt_engines/qwen3_rec_v4/config.json`，同样显示 `context_fmha=False`、`use_paged_context_fmha=False`、`kv_cache_type=PAGED`，说明不能把当前失败简单归因于 `qwen3_rec_v4` engine 配置。已将 `k8s/deployment-inference-image.yaml` 恢复为默认 `qwen3_rec_v4`，新增 `scripts/k8s_check_trtllm_datasystem_runtime.sh`，用于在当前 inference Pod 内同时打印 engine config、`ModelRunnerCpp.from_dir` 签名，并扫描 TensorRT-LLM `.so` 是否包含 `Create Datasystem class`、`Init KvCache Manager DataSystem`、`op=offload/onboard` 等 C++ DataSystem patch 字符串。新的判断顺序：先确认 runtime 镜像内 C++ DataSystem patch 是否存在并被加载；若不存在，优先重建/替换 TensorRT-LLM runtime；若存在但 upstream block reuse warning 仍阻断路径，再使用 `qwen3_rec_v4_paged_fmha` engine build job 作为后备方案** |
| **6/12** | **远程 verifier 显示 `reuse disabled warning seen=True`、offload/onboard/TRACE 均为 0；进一步检查当前 K8s 挂载的 `/app/trt_engines/qwen3_rec_v4/config.json`，确认 `build_config/plugin_config/context_fmha=False`、`use_paged_context_fmha=False`。该配置会触发 upstream TensorRT-LLM 关闭 `kv_cache_enable_block_reuse` 的 warning，但由于历史容器同样配置曾跑通 C++ DataSystem offload/onboard，当前不能仅凭 engine 配置下结论。为后备验证新增 `k8s/job-build-trt-engine-paged-fmha.yaml` 与 `scripts/k8s_build_trt_engine_paged_fmha.sh`，可在 worker1/GPU 上构建独立 `trt_engines/qwen3_rec_v4_paged_fmha`，构建参数包含 `--kv_cache_type paged --remove_input_padding enable --context_fmha enable --use_paged_context_fmha enable`；旧 `qwen3_rec_v4` 保留为默认 HTTP/TRT 基线 engine。当前本地 `bash -n` 与 YAML parse 通过；下一步优先诊断 inference 镜像内 TRT-LLM C++ DataSystem patch/动态库是否存在** |
| **6/11** | **为 C++ DataSystem KV 验证版 inference 镜像切换独立 tag：默认镜像从 `docker.io/library/pairec-inference:k8s-arm64-runtime` 改为 `docker.io/library/pairec-inference:k8s-arm64-ds-kv-v1`，当前导出 tar 统一保存在 `/home/zcx` 下；`k8s/deployment-inference-image.yaml`、`scripts/build_inference_runtime_image.sh` 与 `k8s/README.md` 已同步，避免覆盖已验证 HTTP/TRT 基线镜像，也避免 kubelet/containerd 因 `IfNotPresent` 使用旧 tag 缓存** |
| **6/11** | **修复 DataSystem K8s Pod 固定 worker 二进制路径错误：远程首次启动 `datasystem` Pod 报 `datasystem_worker not found or not executable: /usr/local/lib64/python3.11/site-packages/yr/datasystem/datasystem_worker`；已将 manifest 改为启动时自动探测 `datasystem_worker`（先 `command -v`，再搜索 `/usr/local`、`/usr`、`/opt`、`/root` 下 DataSystem 路径），找不到时打印 `dscli` 与候选 DataSystem 文件，避免继续猜固定路径。下一步远程重新 apply，若仍找不到 worker 二进制，则根据候选日志决定改用镜像内实际路径或改为 `dscli start` 包装方式** |
| **6/11** | **新增 K8s DataSystem hostNetwork 部署清单：`k8s/deployment-datasystem-hostnetwork.yaml` 固定调度到 worker1，使用 `docker.io/library/zcx-pairec-image:v1.1` 作为 runtime，挂载 worker1 已解压的 `/home/zcx/workspace/pairec4tigerllm/etcd-v3.5.10-linux-arm64`，在容器内以前台进程启动独立 etcd `141.61.91.188:12379/12380` 与 `datasystem_worker --worker_address=141.61.91.188:18481 --shared_memory_size_mb=5120`；为 DataSystem worker 挂载 6Gi memory `emptyDir` 到 `/dev/shm`，避免 K8s 默认 64MiB shm 导致 worker 启动失败；新增 `scripts/k8s_apply_datasystem_hostnetwork.sh`，本地 YAML parse 与 bash 语法检查通过。当前 18481/12379/12380 在 master/worker1 均未占用，下一步远程 apply DataSystem，再重建/apply inference 镜像验证 C++ offload/onboard TRACE** |
| **6/11** | **C++ DataSystem KV trace 验证链路已在 K8s inference runtime 配置侧补齐，待远程重建镜像复验：新增 `TRT_KV_CACHE_HOST_CACHE_SIZE` 传参，构建期 patch `ModelRunnerCpp.from_dir()` 可把 `host_cache_size/onboard_blocks` 转发进 `KvCacheConfig`，用于将 `secondaryBlocks=0` 改为可 offload 的二级池；新增 `PYTHON_DATASYSTEM_ENABLED=0`，允许容器继续向 TensorRT-LLM C++ 暴露非空 `DATASYSTEM_HOST/PORT`，但不启用 Python `yr.datasystem`；`k8s/deployment-inference-image.yaml` 已临时关闭 Python 结果缓存并按当前 `dscli start -w --worker_address ${HOST_IP}:18481` 设置 `DATASYSTEM_HOST=141.61.91.188`、`DATASYSTEM_PORT=18481`、`TRT_KV_CACHE_HOST_CACHE_SIZE=104857600`，下一步远程目标是日志出现 `secondaryBlocks>0`、`[TensorRT-LLM][Datasystem] Init KvCache Manager DataSystem` 和 `[Datasystem][TRACE] op=offload/onboard`** |
| **6/11** | **正式 inference runtime 镜像远程复验通过：重建并导入 `docker.io/library/pairec-inference:k8s-arm64-runtime` 后，`k8s/deployment-inference-image.yaml` 成功启动 TRT 服务；`GET /health` 返回 `status=healthy backend=trt-qwen3 trt_num_samples=1 datasystem=disabled`；直连 `POST /recommend` 返回 `code=200` 和 5 个 recommendations，trace 显示 `runner_calls=1`、`runner_generate_ms≈187ms`、`total_ms≈232ms`；随后 PaiRec 端到端 `GET /ping -> success`，`POST /api/recommend` uid=6312,size=10 返回 `code=200` 和 10 个 `generative_recall` item。正式 inference 镜像已替代启动时 patch/hostPath 代码依赖，当前仅模型/数据仍为 hostPath** |
| **6/11** | **正式 inference runtime 镜像首次远程验证发现镜像缺 `training.rqvae`：`server.py` 导入 `training.decoder.model` 时触发 `training/__init__.py` 导入 `RQVAE`，旧 Dockerfile 只复制 `training/decoder` 导致 `ModuleNotFoundError: No module named 'training.rqvae'`；已修 `docker/Dockerfile.inference.runtime` 为复制完整 `training/` 包，待远程重新构建镜像并复验** |
| **6/10** | **正式 inference runtime 镜像方案已新增：`docker/Dockerfile.inference.runtime` 基于已验证 `zcx-pairec-image:v1.1` runtime，复制 `/app` 业务代码并在构建期 patch TensorRT-LLM `ModelRunnerCpp.from_dir()` 支持 `scheduler_config`；新增 `docker/entrypoint-inference-runtime.sh` 统一设置 gcc-toolset-14 `LD_LIBRARY_PATH`、有效 NVML 和 DataSystem abseil `LD_PRELOAD`；新增 `k8s/deployment-inference-image.yaml` 与 `scripts/build_inference_runtime_image.sh`、`scripts/k8s_apply_inference_image.sh`。本机静态校验 `py_compile`、`bash -n`、YAML parse 通过；待远程 ARM 构建、导入 worker1 并复验 HTTP/TRT 基线** |
| **6/10** | **F13 K8s PaiRec 端到端闭环验证通过：基于 `go build -mod=vendor` 生成 arm64 静态 PaiRec 二进制并打包为 `docker.io/library/pairec-server:k8s-arm64-static`；修复 Alpine 动态链接导致的 `exec /app/pairec-server: no such file or directory`；修复 PaiRec `../data/...` 相对路径，通过 hostPath 将 worker1 `/home/zcx/workspace/pairec4tigerllm/data` 挂到 `/data`，`workingDir=/app`；首次 fallback JSON 触发 1Gi OOM 后将 PaiRec memory limit 提到 8Gi；远程验证 `GET /ping -> success`，`POST /api/recommend` uid=6312,size=10 返回 `code=200` 和 10 个 `generative_recall` item；新增 `k8s/deployment-pairec-hostpath.yaml` 与 `scripts/k8s_apply_pairec_hostpath.sh`** |
| **6/9** | **F13 K8s inference 单服务闭环跑通并固化 hostPath manifest：worker1 已注册 `nvidia.com/gpu=1`；`zcx-pairec-image:v1.1` 作为 TRT/DataSystem runtime，hostPath 挂 `/home/zcx/workspace/pairec4tigerllm`；修正 `LD_LIBRARY_PATH` 使用 gcc-toolset-14、`LD_PRELOAD` 使用有效 NVML + DataSystem abseil，并在启动时临时 patch `ModelRunnerCpp.from_dir()` 转发 `scheduler_config`；远程验证 `/health` 返回 `status=healthy backend=trt-qwen3 trt_num_samples=1`，`/recommend` 返回 `code=200` 和 5 个 item；新增 `k8s/deployment-inference-hostpath.yaml` 与 `scripts/k8s_apply_inference_hostpath.sh`。遗留：Python DataSystem 为 disabled，C++ `CacheTransceiver` disabled，PaiRec Go 镜像/Deployment 待接入** |
| **6/5** | **F13 K8s 最小闭环部署配置已成型：修正 inference `:18000`、PaiRec `:18080`、PaiRec `/ping` 探针、`http://inference:18000` 服务发现、TRT engine/Qwen3/checkpoint/DataSystem/LD_PRELOAD 环境变量、GPU 单副本 `Recreate` 策略；新增 `k8s/README.md` 执行步骤；PaiRec Dockerfile 改为 `-mod=vendor` 离线构建并本机验证 `go build -mod=vendor` 通过** |
| **6/5** | **远端 DataSystem Get 到本地测试完成并补充到 `docs/E2E_DATASYSTEM_FINAL_REPORT_2026-06-04.md`：DS host=`141.61.91.188:18581`，client p50=1050.3ms、p99=1689.2ms；C++ onboard=1346，Get p50=315.572ms、p99=317.038ms；H2D p50=0.405ms，确认当前路径是远端 DS Get 到本机 host 后再本机 H2D，remote H2D 暂未测试** |
| **6/4** | **`docs/E2E_DATASYSTEM_FINAL_REPORT_2026-06-04.md` 补充原始 benchmark stdout 附录，保留 pressure/replay、请求级阶段和 C++ DataSystem block 指标的原始输出，便于复核摘要表** |
| **6/4** | **最终 E2E DataSystem Set/Get 压测完成并生成 `docs/E2E_DATASYSTEM_FINAL_REPORT_2026-06-04.md`：12000 请求成功 11853，client p50=92.8ms、p99=103.2ms；C++ offload=29330，Set p50=0.746ms、p99=1.015ms、p9999=1.689ms；C++ onboard=13421，Get p50=0.623ms、p99=0.818ms、p9999=1.094ms** |
| **6/4** | **新增 `TRT_RESULT_CACHE_ENABLED=0`，可关闭 Python TRT 推荐结果缓存；`scripts/benchmark_e2e_latency.py` 新增 `--uid-file`、pressure/replay 两段流量和按阶段 DataSystem C++ 指标，支持在端到端报告里同时观察 request stages 与 C++ `offload.set_ms` / `onboard.get_ms`** |
| **6/4** | **输出 `docs/E2E_LATENCY_REPORT_2026-06-04.md`，汇总 TRT_NUM_SAMPLES=1 的 PaiRec E2E、缓存分组、当前 E2E 中 DS trace 缺失原因与下一轮充分测 DS Set/Get 的要求** |
| **6/4** | **`scripts/benchmark_e2e_latency.py` 合并 DataSystem C++ KV block 统计：同一份 E2E 报告同时输出请求级阶段耗时和 `offload.set_ms` / `onboard.get_ms` 等 per-block 指标** |
| **6/4** | **TRT_NUM_SAMPLES=1 PaiRec E2E 完成: 60 请求全成功，client p50=3.5ms、p95=90.5ms、p99=92.8ms；报告混入结果缓存命中，冷 miss 对应 p95/p99 尾部约 90ms** |
| **6/4** | **远程 runner 轮数 A/B 完成: 1/2/4/8 轮均 `full_topk=60/60`；HTTP p50 分别为 91.8/180.1/350.2/701.7ms；确认 runner 耗时近似线性缩放** |
| **6/4** | **修复 `scripts/benchmark_trt_runner_samples.py` 使用 `--stop-command 'pkill -f ...'` 时会匹配自身 `--server-cmd` 并被 `Terminated` 的问题；文档改为脚本外手动清理旧服务** |
| **6/3** | **新增 `scripts/benchmark_trt_runner_samples.py`，自动核验 `/health` 的 `trt_num_samples`、运行直连压测、解析 `runner_calls` 并输出 JSON/CSV 对比** |
| **6/2** | **TRT runner 优化准备: 定位每个 miss 请求串行执行 8 次 `runner.generate()`；增加 `TRT_NUM_SAMPLES` 参数和单轮 trace，支持 1/2/4/8 轮质量-时延 A/B** |
| **6/2** | **DataSystem onboard 专项扩样准备: 报告明确区分 host DataSystem API、host 计时同步 D2H/H2D 和总 wall-clock；新增 `--min-onboard-samples` 门槛** |
| **6/2** | **远程 replay 修正验证通过: offload 8237 次，onboard 177 次；Get p50=0.725ms、p99=1.709ms，Get+H2D p50=1.258ms、p99=2.291ms** |
| **6/2** | **远程应用 C++ trace patch 并完成首轮 DataSystem 实测: offload 4181 次，Set p99=1.093ms；onboard/Get 已观测到 1 次，需继续增加读样本** |
| **6/2** | **F12 C++ DataSystem 时延观测补齐: 新增 Create/D2H/Set、Get/H2D 结构化 trace patch；C++ 压测和 PaiRec E2E 汇总新增 p99/p9999/max** |
| **6/1** | **端到端阶段性收口: `dev` 固化为可回退基线；后续从专用分支开展 TRT-LLM C++ DataSystem 与原生 pinned DRAM 的端到端 A/B** |
| **6/1** | **开始 F12 推荐系统时延分析: 补齐 PaiRec 入口、GenerativeRecall、Python TRT 服务和 TRT runner 分阶段 trace；新增端到端压测汇总脚本** |
| **6/1** | **F08 PaiRec 对接完成: 远程复验 Kafka 实时特征用户 `130`、`2184` 均单次触发 TRT 推理并返回完整映射结果** |
| **6/1** | **修复 PaiRec 自定义 recall 启动 panic: 外部注册同步写入配置签名，框架二次加载时正确跳过内置工厂** |
| **6/1** | **修复远程 DNS 不可用: 将完整 Go `vendor/` 纳入仓库，PaiRec 启动固定使用 `-mod=vendor` 离线依赖** |
| **6/1** | **修复 PaiRec 远程启动: 将 `go.sum` 纳入版本控制；启动脚本正确导出 `CONFIG_PATH` 并移除无效 `--port` 参数** |
| **6/1** | **推荐链路对接 — PaiRec 端到端返回成功；修复 Go 客户端 500ms 超时导致的重复推理** |
| **5/30** | C++ KV cache offload/onboard 闭环验证通过; 任务切换到推荐链路工程化 |
| **5/29** | C++ offload/onboard 链路突破 (FMHA crash根因/C++源码绕过/重编.so) |
| **5/28** | KV Cache + offload/onboard 闭环 (TRT/PyTorch双路径/三层缓存全通) |
| **5/27** | 推理打通 (FMHA bf16 kernel SM89非法内存访问根因) |
| **5/23** | ARM 4090D推理部署 + TRT-LLM引擎构建 + 多轮采样去重 |
| **5/22** | 训练完成 epoch 20 (loss 2.49) |
| **5/21** | DDP三卡训练启动 (epoch 11) |

## 当前状态

- **训练**: ✅ epoch 20, loss 2.49
- **TRT-LLM 引擎**: ✅ bfloat16, 1.46 GB, 限制: max_input_len=64, max_new_tokens=32, max_seq_len=96
- **C++ KV offload/onboard**: ✅ 闭环验证通过
- **推理服务**: ✅ /recommend 可用
- **PaiRec 对接**: ✅ Kafka 实时特征、生成式召回、TRT 推理和 item 映射链路已打通
- **时延分析**: ✅ 第一版端到端 trace 和冷请求分解已验证；✅ C++ DataSystem Set/Get 阶段性统计已完成；✅ 关闭结果缓存后的 E2E pressure/replay 最终报告已生成；✅ 远端 DS Get 到本地中等样本已完成；🔄 remote H2D 与 pinned DRAM A/B 待补充
- **K8s 部署**: ✅ ARM worker1 HTTP/TRT 基线闭环已通过：正式 inference runtime 镜像 `/health` + `/recommend` 和 PaiRec `/ping` + `/api/recommend` 均已验证；🔄 仍需将模型/数据 hostPath 替换为 PVC，并单独修 DataSystem Python/C++ KV 路径
- **brpc 改造**: 🔄 F14 已回退 PaiRec proxy 方案，转向 C++ brpc inference service；`semantic_map` native smoke 已在 K8s 通过；`trtllm_cpp` 后端第一版已实现并通过本机静态/CMake 配置验证，待远程 ARM TRT-LLM runtime 构建、部署和真实 brpc smoke

## 6/1 探索：推理命中率优化 (5个bug修复)

### 问题链

PaiRec 调用推理服务 → 大部分用户返回 `code:299 "items size not enough"`。

根因是 **推理服务返回的推荐 item 太少**（1个或0个），远低于 PaiRec 期望的 size（10）。

### 修复清单

| # | 提交 | 问题 | 修复 |
|---|------|------|------|
| 1 | `f7beeb4` | 20条历史→117 tokens 超引擎 max_input_len=64 | prompt 自动截断，保留最近9条 |
| 2 | `fb1cb98` | `<s0_X><pad><s1_Y>` 因中间有非语义token被丢弃 | 先过滤有效token再匹配，跳过非语义token |
| 3 | `326e1bb` | 模型输出层序乱 (s2,s1,s0,s3) 连续递增模式找不到 | 改为按层收集+笛卡尔积组合，不要求顺序 |
| 4 | `5533a16` | 20 tokens输出太短+8轮各自为战 | 8轮token合并到一个池子统一组合 |
| 5 | `1544a3f` | max_new_tokens>32 C++层卡死不报错 | 硬限制≤32 (引擎构建时预留) |
| 6 | `b44dbc6` | layer 3永远为0 (1条历史的用户) | 缺失层用{0}填充，由semantic_id_map验证 |

### 引擎硬限制 (重要)

```
max_seq_len = 96
  ├─ max_input_len = 64   (超过报 RuntimeError)
  └─ max_new_tokens = 32  (超过 C++ 层卡死不报错!)
```

必须同时遵守两个限制。

### 核心机制

**组合模式**: 8轮采样 × 32 token = 256 token 池 → 按层收集所有有效语义token → 笛卡尔积组合 → semantic_id_map 验证

**layer 填充**: 当某层缺失时用 {0} 补位 → 组合出候选 → map 验证真假

## 6/1 验证：PaiRec 端到端功能打通

### 验证证据

| 用户 | history_len | PaiRec 响应 |
|------|-------------|-------------|
| `303` | 20 | `code=200`, `size=5`, 5 个 `generative_recall` item |
| `1201` | 1 | `code=200`, `size=5`, 5 个 `generative_recall` item |
| `130` | 1 | `code=200`, `size=5`, 5 个 `generative_recall` item |

### 新发现：Go 客户端超时导致重复推理

单次 TRT miss 约 `667-812ms`，但 Go 客户端默认超时仅 `500ms` 且最多尝试 3 次。同一个 PaiRec 请求会并发触发多次 GPU 推理，再由后续重试命中 HBM 结果缓存。

已在本地修复：`RecallAlgo` 增加 `timeout_ms=3000`、`max_retries=1`，并同步修改默认配置。待远程部署后确认单次 PaiRec 请求只触发一次 `/recommend`。

### 远程启动修复

远程 `go run` 曾因仓库未跟踪 `go.sum` 报依赖校验缺失。已将 `go.sum` 纳入版本控制，并修复 `scripts/start_pairec.sh`：导出 `CONFIG_PATH` 供 `main.go` 首次加载，移除 PaiRec 未定义的 `--port` 参数。

远程环境随后因 DNS 解析失败无法访问 `mirrors.aliyun.com`。已将完整 `vendor/` 纳入版本控制，启动脚本固定使用 `go run -mod=vendor`，不再依赖在线下载。

首次 vendor 提交仍遗漏 128 个文件：根因是 `.gitignore` 中通用 `lib/` 规则误伤 vendor 内 ClickHouse、Apache Thrift 和 PostgreSQL 驱动目录。已增加 `!vendor/**` 例外并补齐文件。验证方式：从 Git 暂存区导出临时副本，在 `GOPROXY=off`、空 `GOMODCACHE` 和空 `GOCACHE` 下执行 `go test -mod=vendor ./services/...`，全部通过且模块缓存文件数为 0。

### 自定义 recall 注册修复

PaiRec 启动时会再次执行 `recall.Load()`。此前 `main.go` 手工注册 `GenerativeRecall` 时只写入实例，没有写入框架的配置签名；二次加载时框架尝试用内置工厂重建自定义类型并 panic：`recall empty, name:generative_recall`。

已在 vendored recall 包增加 `RegisterRecallWithConfig()`，同步写入实例和配置签名，`main.go` 改用该入口。启动级验证已越过注册阶段并输出 `server start`。

## 6/1 F12：推荐系统端到端时延分解

已按当前 Qwen3 TRT 链路补齐 trace：

- PaiRec 入口内部：`user_feature_ms`、`recall_ms`、`filter_ms`、`general_rank_ms`、`feature_ms`、`rank_ms`、`pipeline_wait_ms`、`merge_ms`、`sort_ms`
- GenerativeRecall：`history_ms`、`convert_ms`、`http_ms`、`items_ms`、`http_overhead_ms`
- Python TRT 服务：`prepare_input_ms`、`kv_lookup_ms`、结果缓存查询和异步写入提交耗时
- TRT 后端：`prompt_ms`、8 轮累计 `runner_generate_ms`、`parse_combo_ms`、`output_pad_ms`
- 新增 `scripts/benchmark_e2e_latency.py`：从 PaiRec 入口发请求，用 `request_id` 关联 PaiRec 和 TRT 日志，汇总 p50/p95/p99/p9999/max，并按 `miss` / `hbm_hit` / `ds_hit` 分组
- 修正小样本 percentile 插值：2 个样本的 p50 使用中位数，不再错误取最小值
- 修复 PaiRec trace 采集：`glog` 默认写独立文件，`scripts/start_pairec.sh` 现在默认传入 `--alsologtostderr=true`，保留文件日志并可由 `tee /tmp/pairec.log` 捕获结构化日志

远程冷请求分解已确认：

- fallback 用户 `142`、`211` 均为 TRT `miss`
- 首个 fallback 请求：PaiRec `3392ms`，其中 `history_ms=2573ms`；对应首次加载 `999447` 用户 fallback JSON
- 后续稳态冷请求：PaiRec `687ms`，其中 TRT 服务 `685.3ms`，PaiRec 额外开销约 `2ms`
- 两次 TRT 平均 `750.0ms`，其中 8 轮 `runner_generate` 平均 `686.2ms`，占比约 `91.5%`
- TRT 次要耗时：`output_pad_ms` 平均 `27.3ms`、`prompt_ms` 平均 `10.7ms`
- 结论：fallback JSON 应在启动阶段预加载，避免首请求抖动；稳态冷请求的首要优化目标是 8 轮 TRT runner

本机验证：

```text
python -m py_compile inference/trt_llm/server.py inference/trt_llm/trt_qwen3_backend.py scripts/benchmark_e2e_latency.py
# exit 0

go test -mod=vendor ./services/...
# services / config / feature / recall 均通过

python -c '<trace parser assertions>'
# trace parser OK
```

## 6/2 F12：C++ DataSystem Get/Set 时延与尾延迟指标

已新增 `trtllm-datasystem-latency-trace.patch`，针对外部 TensorRT-LLM
`cpp/tensorrt_llm/batch_manager/kvCacheTransferManager.cpp` 增加结构化日志：

```text
[TensorRT-LLM][Datasystem][TRACE] op=offload ... create_ms=... d2h_ms=... set_ms=... total_ms=...
[TensorRT-LLM][Datasystem][TRACE] op=onboard ... get_ms=... h2d_ms=... total_ms=...
```

统计脚本同步增强：

- `scripts/test_trt_cpp_kv_offload.py`：按 warmup / pressure / replay / overall 汇总 DataSystem C++ 指标，HTTP 和 C++ 指标均输出 `avg/p50/p95/p99/p9999/max`，支持 `--json-output`
- `scripts/benchmark_e2e_latency.py`：PaiRec E2E、各阶段和 TRT 推荐结果缓存分组新增 `p9999`，缓存分组补齐 `p99/p9999/max`
- `p9999` 使用线性插值；样本量不足时仅作方向性观察，正式结论需扩大请求量

本机验证：

```text
git -C /home/vivwimp/TensorRT-LLM apply --check \
  /home/vivwimp/pairec4tigerllm/trtllm-datasystem-latency-trace.patch
# exit 0

python -m py_compile \
  scripts/benchmark_e2e_latency.py scripts/test_trt_cpp_kv_offload.py
# exit 0

python - <<'PY'
# synthetic percentile + DataSystem TRACE parser assertions
PY
# datasystem trace parser OK
```

### 远程 DataSystem 首轮实测

DataSystem runtime 已应用 trace patch 并完成首轮 pressure / replay：

```text
KV pool: primaryBlocks=32 secondaryBlocks=28
offload count=4181
  create_ms p50=0.680 p99=1.025 max=2.171
  d2h_ms    p50=0.413 p99=0.691 max=0.732
  set_ms    p50=0.759 p99=1.093 max=1.381
  total_ms  p50=1.937 p99=2.553 max=4.006
onboard count=1
  get_ms=1.562 h2d_ms=0.345 total_ms=1.961
```

结论：

- DataSystem C++ offload 写路径稳定，单个 3.5 MiB block 的 `Create + D2H + Set`
  总耗时 p50 约 `1.94ms`，p99 约 `2.55ms`
- onboard/Get 已出现 1 次，证明读路径打通；样本不足，暂不能评价 Get 分位数
- 原压测 verdict 依赖 DEBUG 级 `KV cache block reuse is enabled`、`copyBlock entered`
  和 `Set Key` 日志，在 INFO 级日志下会误报失败；脚本已改为优先使用结构化 trace 判定

### 远程 DataSystem 第二轮实测与 replay 修正

扩大 pressure 后采集到：

```text
KV pool: primaryBlocks=32 secondaryBlocks=28
offload count=10440
  create_ms p50=0.765 p99=1.117 p9999=1.783 max=2.311
  d2h_ms    p50=0.493 p99=0.700 p9999=1.475 max=2.087
  set_ms    p50=0.839 p99=1.120 p9999=1.481 max=10.906
  total_ms  p50=2.200 p99=2.796 p9999=4.198 max=12.334
onboard count=0
```

结论：

- `Set` 路径已有 `10440` 个样本，p99 仍约 `1.12ms`；出现一次
  `set_ms=10.906ms` 尾部尖峰，后续 A/B 需保留 max 和原始 JSON
- 原 replay 固定重放最早的 pressure 历史；secondary pool 只有 `28` 个 block，
  增大 pressure 反而会使这些前缀更早被回收，无法稳定触发 `Get`
- `scripts/test_trt_cpp_kv_offload.py` 新增 `--replay-source-count` 和
  `--replay-tail-offset`，默认循环最近一小组 pressure 历史；历史生成器改为
  可逆 32 位混合，避免每 `256` 个 seed 重复，可构造十万级不同请求

### 远程 DataSystem 第三轮实测：Get/onboard 样本补齐

使用近期 pressure 历史循环 replay 后，结构化 trace 判定通过：

```text
PASS: C++ KV offload and DataSystem onboard were both observed.

KV pool: primaryBlocks=32 secondaryBlocks=28
offload count=8237
  set_ms    p50=0.780 p99=1.202 p9999=10.793 max=10.848
  total_ms  p50=2.101 p99=2.760 p9999=12.028 max=12.173
onboard count=177
  get_ms    p50=0.725 p95=1.520 p99=1.709 p9999=1.960 max=1.963
  h2d_ms    p50=0.492 p95=0.607 p99=0.618 p9999=0.624 max=0.624
  total_ms  p50=1.258 p95=2.027 p99=2.291 p9999=2.471 max=2.471
```

其中 replay/onboard wave 单独贡献 `174` 次 onboard，说明新的 replay 策略可以
稳定进入 DataSystem `Get + H2D` 读路径。`Get` 的 p50/p95/p99 已具备阶段性参考
价值；读路径只有 `177` 个样本，p9999 仍接近 max，仅作为观察值。

### Host/device 指标边界与 onboard 专项扩样

已展开 TensorRT-LLM runtime 实现：

```text
create_ms  host steady_clock 包围 DataSystem Create API
set_ms     host steady_clock 包围 DataSystem Set API
get_ms     host steady_clock 包围 DataSystem Get API
d2h_ms     host steady_clock 包围同步 cudaMemcpySanitized/cudaMemcpy DeviceToHost
h2d_ms     host steady_clock 包围同步 cudaMemcpySanitized/cudaMemcpy HostToDevice
total_ms   host steady_clock 包围完整 offload/onboard 流程
```

因此：

- `create_ms/set_ms/get_ms` 是 host 侧 DataSystem API wall-clock
- `d2h_ms/h2d_ms` 是 host 观察到的同步 CPU↔GPU 传输完成耗时，包含 PCIe/DMA
  等待，但不是 CUDA event 计出的纯 device 时间
- 当前没有采集 GPU kernel 或 CUDA-event device-only 指标

`scripts/test_trt_cpp_kv_offload.py` 已增加：

- `--min-onboard-samples`：结构化 onboard trace 少于目标数量时返回失败
- 报告开头逐项打印 metric scope
- 报告结尾提示 p9999 样本门槛：`10000` 是最低尾部观测门槛，稳定结论建议
  `>=100000`

远程下一轮先采集 `>=10000` 个 onboard 样本：

```text
python scripts/test_trt_cpp_kv_offload.py \
  --log /tmp/server_v4.log \
  --warmup-requests 0 \
  --requests 360 --repeat-requests 10000 \
  --replay-source-count 4 --replay-tail-offset 1 \
  --min-onboard-samples 10000 \
  --json-output /tmp/cppkv-datasystem-onboard-10k.json
```

### 远程 DataSystem 第四轮实测：onboard 10k+

onboard 专项 replay 已完成，达到最低 p9999 尾部观测门槛：

```text
replay/onboard wave HTTP:
  count=10000 p50=694ms p95=727ms p99=749ms p9999=832ms max=1552ms

all measured phases:
offload count=166089
  create_ms p50=0.665 p99=0.990 p9999=1.637 max=10.910
  d2h_ms    p50=0.483 p99=0.675 p9999=0.836 max=1.150
  set_ms    p50=0.756 p99=1.084 p9999=1.816 max=11.055
  total_ms  p50=1.937 p99=2.582 p9999=11.877 max=12.513
onboard count=14208
  get_ms    p50=0.740 p99=1.037 p9999=1.570 max=10.801
  h2d_ms    p50=0.407 p99=0.599 p9999=0.804 max=1.021
  total_ms  p50=1.177 p99=1.546 p9999=2.121 max=11.272
```

结论：

- onboard 已有 `14208` 个样本，可观察 p9999；若需要稳定的正式 p9999
  结论，仍建议扩大到 `>=100000`
- onboard 极端 max `11.272ms` 主要来自 `Get max=10.801ms`，不是 H2D；
  `h2d_ms max=1.021ms`
- replay 请求 E2E p50 为 `694ms`，单 block onboard p50 为 `1.177ms`；
  DataSystem 单块读回不是当前推荐请求的主耗时，主耗时仍在 8 轮 TRT runner
- offload 已有 `166089` 个样本，写路径 p9999 具备更好的统计支撑

### TRT runner 耗时根因与减轮数 A/B

当前每个 TRT miss 请求会串行调用 `8` 次 `ModelRunnerCpp.generate()`，每轮最多
生成 `32` 个 token，再合并 token 池做语义四元组组合。端到端基线中：

```text
runner_generate_ms avg=675.2ms
runner calls per request=8
single runner.generate avg≈84.4ms
```

8 轮不是模型推理的硬要求，而是当前无约束解码下为提高候选覆盖率采用的召回
策略。减少轮数预计近似线性降低 runner 耗时，但可能使可映射 item 数下降。

已增加：

- 服务参数 `--trt_num_samples`，环境变量 `TRT_NUM_SAMPLES`，默认 `8`
- TRT trace：`runner_calls`、`runner_avg_ms`、`runner_max_ms`
- `scripts/benchmark_e2e_latency.py` 汇总上述字段
- `scripts/test_trt_cpp_kv_offload.py` 输出每个 phase 的 `full_topk`、item 数
  p50/p95/min/max，便于直连 TRT 做质量-时延 A/B

粗略预估，固定开销按约 `23ms` 计算：

```text
8 rounds: runner≈675ms, request≈698ms  当前基线
4 rounds: runner≈338ms, request≈361ms  待实测
2 rounds: runner≈169ms, request≈192ms  待实测
1 round:  runner≈ 84ms, request≈107ms  待实测
```

远程完整 A/B 已回传，`/health` 与服务 TRACE 均确认轮数生效：

```text
sample health kv_exit calls ok/topk http_p50 http_p99 runner_avg runner_max
1      True   2       1     60/60  91.8     150.7    85.8       177.7
2      True   2       2     60/60  180.1    247.7    85.4       178.6
4      True   2       4     60/60  350.2    423.0    84.8       178.4
8      True   2       8     60/60  701.7    795.6    83.9       178.8
```

结论：

- 对当前直连 TRT、`topk=5`、60 请求样本，1/2/4/8 轮均没有出现 top-k 返回不足
- 1 轮相对 8 轮 HTTP p50 从 `701.7ms` 降到 `91.8ms`，下降约 `87%`
- 单次 `runner.generate()` 平均稳定在 `83.9-85.8ms`，说明主耗时几乎完全随轮数线性缩放
- `kv_exit=2` 来自底层 C++ KV/DataSystem verifier 未通过；该组结果只作为 runner
  轮数 A/B，不作为 DataSystem offload/onboard 结论

远程按 `TRT_NUM_SAMPLES=1/2/4/8` 分别重启服务，再执行相同直连请求序列：

```text
python scripts/test_trt_cpp_kv_offload.py \
  --log /tmp/server_samplesN.log \
  --warmup-requests 0 --requests 60 --repeat-requests 0 \
  --topk 5 --json-output /tmp/runner-samplesN.json
```

脚本默认使用带时间戳的唯一 `user_id` 前缀，避免重复运行命中 Python 结果缓存。
优先比较 `full_topk` 比例和 HTTP p50/p99。默认值暂不下调，需根据远程 A/B
结果选择。

已补充自动编排脚本，推荐后续直接使用：

```text
python scripts/benchmark_trt_runner_samples.py \
  --samples 1,2,4,8 \
  --server-cmd 'python -m inference.trt_llm.server ...' \
  --requests 60 --repeat-requests 0 --topk 5
```

如果需要清理旧服务，先在脚本外单独执行 `pkill -f "inference.trt_llm.server" || true`。
不要把这条 `pkill -f` 放进 `--stop-command`：benchmark 进程自己的
`--server-cmd` 参数也包含 `inference.trt_llm.server`，会被误杀并显示
`Terminated`。

脚本会逐轮注入 `TRT_NUM_SAMPLES`，等待 `/health` 返回匹配的
`trt_num_samples`，运行 `scripts/test_trt_cpp_kv_offload.py`，解析服务 TRACE
中的 `runner_calls/runner_avg_ms/runner_max_ms`，最后输出
`/tmp/trt_runner_samples_ab.json` 和 `/tmp/trt_runner_samples_ab.csv`。
默认情况下，runner A/B 脚本不会因为底层 C++ KV/DataSystem verifier 返回非零而
失败；`kv_exit` 会保留在汇总中作为诊断字段。若需要严格验证 C++ KV/DataSystem，
显式加 `--fail-on-kv-verdict`。

### TRT_NUM_SAMPLES=1 PaiRec E2E 实测

从 PaiRec `:18080` 入口使用 `TRT_NUM_SAMPLES=1`、`size=5`、60 个请求完成
端到端压测：

```text
Client: avg=9.4ms p50=3.5ms p95=90.5ms p99=92.8ms max=94.6ms
PaiRec total: avg=8.4ms p50=3.0ms p95=89.0ms p99=91.6ms max=94.0ms
GenerativeRecall http_ms: avg=8.1ms p50=2.0ms p95=89.0ms p99=91.2ms
TRT trace tr_total_ms: avg=7.5ms p50=1.6ms p95=88.2ms p99=90.4ms
TRT trace tr_runner_ms: avg=5.4ms p50=0.0ms p95=80.0ms p99=81.6ms
ok=60 fail=0
```

该报告混合了推荐结果缓存命中和少量冷 miss。p50 主要代表缓存命中路径；p95/p99
代表仍需进入 TRT runner 的冷 miss，符合 1 轮 runner 约 `84ms` 的预期。由于 TRT
服务日志未按 `request_id` 关联上，`scripts/benchmark_e2e_latency.py` 已补充
item 完整率和基于 GenerativeRecall `tr_result_cache_source` 的缓存分组 fallback。

后续重跑同一个 E2E 脚本时，会追加：

```text
== DataSystem C++ KV block stages ==
  scope: per KV block, from TRT-LLM C++ [Datasystem][TRACE], host wall-clock
  offload.create_ms / offload.d2h_ms / offload.set_ms / offload.total_ms
  onboard.get_ms / onboard.h2d_ms / onboard.total_ms
```

该部分是 per-block C++ KV 传输指标，不是 per-request 请求指标；需要与上面的
请求级 E2E 阶段并排解读。

后续可实验将多轮 prompt 批量提交给 `ModelRunnerCpp.generate()`。当前本地
TensorRT-LLM API 对整批默认共用一个 sampling config，而现有逻辑依赖每轮不同
seed；批量化需要先验证候选多样性，不能直接替换串行循环。

## 下一步

`dev` 已作为端到端阶段性基线保留。后续在专用分支开展 C++ DataSystem A/B：

1. 分开采集 `TRT_NUM_SAMPLES=1` 的纯 cold/miss E2E 与缓存命中 E2E，避免一个报告里 p50/p99 语义混杂
2. 单独恢复 DataSystem C++ KV verifier：当前 runner A/B 中 `kv_exit=2`，不能作为 DataSystem 结论
3. 如需稳定的正式 onboard p9999 结论，再扩大到 `>=100000` 个 onboard 样本
4. 推理服务增加实验开关，关闭 TRT Python 结果缓存和无效的 Python KV Cache 查询，确保请求进入 C++ runner
5. TensorRT-LLM runtime 恢复 KV 配置参数化，确保两组使用相同 primary / secondary block 数
6. 基于同一份 engine 构建原生 pinned DRAM baseline，与当前 DataSystem runtime 对照
7. 后续独立优化 fallback JSON 启动预加载
