# PaiRec4TigerLLM K8s Minimal Deployment

目标是先跑通最小闭环：

```text
PaiRec :18080 -> inference :18000 -> TensorRT-LLM -> DataSystem
```

## 前置条件

1. K8s 集群已安装 NVIDIA device plugin，节点能识别 `nvidia.com/gpu`。
2. 已构建并推送两个镜像。

推理镜像必须基于当前已验证的 TensorRT-LLM/DataSystem runtime，至少包含：

- TensorRT-LLM 1.0.0 Python 包与 C++ 动态库
- 已应用 DataSystem trace patch 的 TRT-LLM runtime
- `yr.datasystem` 及其动态库
- `libabseil_dll.so.2407.0.0`

PaiRec 镜像可用 vendored Go 依赖离线构建：

```bash
docker build -f docker/Dockerfile.pairec -t registry.example.com/pairec-server:k8s .
```

推理镜像在已验证 runtime 基础上构建或重新打标为：

```text
registry.example.com/pairec-inference:trt-datasystem
```

3. `model-pvc` 中准备以下目录：

```text
models/Qwen3-0.6B/
checkpoints/decoder_qwen3/decoder_epoch_20.pt
trt_engines/qwen3_rec_v4/rank0.engine
```

4. `data-pvc` 中准备：

```text
tenrec/processed/semantic_id_map.json
```

5. DataSystem worker 已启动，按环境修改 `k8s/configmap.yaml`：

```yaml
datasystem_host: "127.0.0.1"
datasystem_port: "31501"
```

如果 DataSystem 在集群外部，`datasystem_host` 应改为远端可访问 IP 或 DNS。

## 部署顺序

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/deployment-inference.yaml
```

### 已验证 hostPath 推理服务方案

2026-06-09 在 ARM `worker1` 上已验证一个临时 hostPath 推理部署：

- 节点已有 `nvidia.com/gpu: 1`，并使用 `runtimeClassName: nvidia`。
- runtime 镜像为 `docker.io/library/zcx-pairec-image:v1.1`。
- 代码、Qwen3 模型、checkpoint、TRT engine 和 semantic map 均来自
  `worker1:/home/zcx/workspace/pairec4tigerllm`。
- 启动时临时 patch
  `/home/TensorRT-LLM/tensorrt_llm/runtime/model_runner_cpp.py`，使
  `ModelRunnerCpp.from_dir()` 支持 `scheduler_config`。
- Python DataSystem 当前为 `disabled`，C++ `CacheTransceiver` 仍未启用；该
  manifest 只用于先跑通 HTTP/TRT 推荐闭环。

应用方式：

```bash
bash scripts/k8s_apply_inference_hostpath.sh
kubectl -n pairec port-forward svc/inference 18002:18000
curl http://127.0.0.1:18002/health
curl -X POST http://127.0.0.1:18002/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"test","history":[[169,41,0,0],[20,53,0,0],[80,201,0,0]],"topk":5}'
```

已验证响应：

```text
GET /health -> status=healthy, backend=trt-qwen3, trt_num_samples=1
POST /recommend -> code=200, recommendations 非空
```

### 正式 inference runtime 镜像方案

hostPath 推理方案验证通过后，优先使用
`docker/Dockerfile.inference.runtime` 构建正式 inference 镜像。该镜像基于已验证
的 `docker.io/library/zcx-pairec-image:v1.1` runtime，内置：

- `/app/inference` 和必要的 `/app/training` 代码。
- 构建期 TensorRT-LLM `ModelRunnerCpp.from_dir()` `scheduler_config` patch。
- 推理 entrypoint，统一设置 `LD_LIBRARY_PATH`、有效 NVML 与 DataSystem abseil
  `LD_PRELOAD`。

模型、checkpoint、TRT engine 和数据仍由挂载提供，当前 manifest 继续使用
worker1 hostPath；后续再替换为 PVC。

在 master 构建并导入 worker1：

```bash
cd /home/zcx/workspace/pairec4tigerllm

# 如果 docker daemon 里没有基础镜像，先 load 已验证 runtime tar。
# docker load -i /path/to/zcx-pairec-image.tar

bash scripts/build_inference_runtime_image.sh \
  docker.io/library/pairec-inference:k8s-arm64-ds-kv-v1
docker save docker.io/library/pairec-inference:k8s-arm64-ds-kv-v1 \
  -o /tmp/pairec-inference-k8s-arm64-ds-kv-v1.tar
scp /tmp/pairec-inference-k8s-arm64-ds-kv-v1.tar root@141.61.91.188:/tmp/
ssh root@141.61.91.188 \
  'ctr -n k8s.io images import /tmp/pairec-inference-k8s-arm64-ds-kv-v1.tar'
```

应用并验证：

```bash
bash scripts/k8s_apply_inference_image.sh
kubectl -n pairec port-forward svc/inference 18002:18000
curl http://127.0.0.1:18002/health
curl -X POST http://127.0.0.1:18002/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"test","history":[[169,41,0,0],[20,53,0,0],[80,201,0,0]],"topk":5}'
```

该方案默认保持 Python DataSystem disabled，用于稳定复现已验证 HTTP/TRT 基线。
DataSystem Python client 与 C++ `CacheTransceiver` 后续作为独立变量打开。

### C++ DataSystem KV 验证 engine

`qwen3_rec_v4` 是当前 HTTP/TRT 基线 engine。K8s DataSystem KV 验证需要
TensorRT-LLM block reuse；如果 engine config 中
`plugin_config.context_fmha=false` 或
`plugin_config.use_paged_context_fmha=false`，运行时会输出：

```text
KV cache reuse disabled because model was not built with paged context FMHA support
```

此时压测不会进入 C++ offload/onboard 路径。先构建独立的验证 engine：

```bash
bash scripts/k8s_build_trt_engine_paged_fmha.sh
```

该 Job 固定调度到 worker1，使用 `pairec-inference:k8s-arm64-ds-kv-v1`
runtime，在 hostPath repo 下生成：

```text
trt_engines/qwen3_rec_v4_paged_fmha/
```

然后应用 inference DS 验证版 manifest：

```bash
bash scripts/k8s_apply_inference_image.sh
```

该 manifest 默认设置：

```text
TRT_ENGINE_DIR=/app/trt_engines/qwen3_rec_v4_paged_fmha
PYTHON_DATASYSTEM_ENABLED=0
DATASYSTEM_HOST=141.61.91.188
DATASYSTEM_PORT=18481
TRT_KV_CACHE_HOST_CACHE_SIZE=104857600
TRT_RESULT_CACHE_ENABLED=0
```

验证目标是启动日志不再出现 `KV cache reuse disabled`，并且压测日志出现：

```text
[TensorRT-LLM][Datasystem][TRACE] op=offload
[TensorRT-LLM][Datasystem][TRACE] op=onboard
```

### 已验证 hostPath PaiRec 方案

2026-06-10 在 ARM `worker1` 上已验证 PaiRec 通过 K8s Service 调用
`svc/inference:18000`：

- PaiRec 镜像为 `docker.io/library/pairec-server:k8s-arm64-static`，由
  `go build -mod=vendor` 生成的静态 arm64 二进制打包。
- `worker1:/home/zcx/workspace/pairec4tigerllm/data` 挂载到容器 `/data`。
- 容器 `workingDir=/app`，代码中 `../data/...` 相对路径解析为 `/data/...`。
- 首次请求会加载 fallback 用户特征 JSON，PaiRec 内存限制已提升到 `8Gi`，避免
  `OOMKilled`。

构建并导入 PaiRec 镜像示例：

```bash
cd /home/zcx/workspace/pairec4tigerllm
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 \
  go build -mod=vendor -ldflags="-s -w" -o ./pairec-server ./services/main.go
docker build -f /tmp/Dockerfile.pairec.binary \
  -t docker.io/library/pairec-server:k8s-arm64-static .
docker save docker.io/library/pairec-server:k8s-arm64-static \
  -o /tmp/pairec-server-k8s-arm64-static.tar
scp /tmp/pairec-server-k8s-arm64-static.tar root@141.61.91.188:/tmp/
ssh root@141.61.91.188 \
  'ctr -n k8s.io images import /tmp/pairec-server-k8s-arm64-static.tar'
```

应用方式：

```bash
bash scripts/k8s_apply_pairec_hostpath.sh
kubectl -n pairec port-forward svc/pairec 18081:18080
curl http://127.0.0.1:18081/ping
curl -X POST http://127.0.0.1:18081/api/recommend \
  -H 'Content-Type: application/json' \
  -d '{"uid":"6312","size":10,"scene_id":"home_feed"}'
```

已验证响应：

```text
GET /ping -> success
POST /api/recommend -> code=200, size=10, 10 个 generative_recall item
```

先确认推理服务：

```bash
kubectl -n pairec get pods -l app=inference
kubectl -n pairec logs deploy/inference -f
kubectl -n pairec port-forward svc/inference 18000:18000
curl http://127.0.0.1:18000/health
```

再部署 PaiRec：

```bash
kubectl apply -f k8s/deployment-pairec.yaml
kubectl -n pairec get pods -l app=pairec
kubectl -n pairec logs deploy/pairec -f
kubectl -n pairec port-forward svc/pairec 18080:18080
curl http://127.0.0.1:18080/ping
```

端到端请求示例：

```bash
curl -X POST http://127.0.0.1:18080/api/recommend \
  -H 'Content-Type: application/json' \
  -d '{"scene_id":"home_feed","uid":"130","size":5}'
```

## 验收标准

- inference Pod Ready。
- `GET /health` 返回 `backend=trt-qwen3`。
- `GET /health` 中 DataSystem 不应为 `disabled`。
- PaiRec `GET /ping` 返回 `success`。
- PaiRec `/api/recommend` 返回 `code=200`，且 item 数满足 `size`。
- inference 日志中 `trt_num_samples` 与 ConfigMap 一致。

## 当前限制

- 当前 manifests 是最小闭环，不启用 HPA。
- inference Deployment 使用单副本 `Recreate`，避免 GPU 资源滚动更新时重叠占用。
- `LD_PRELOAD` 路径假设镜像内 DataSystem abseil 库位于：

```text
/usr/local/lib/python3.11/site-packages/yr/datasystem/lib/libabseil_dll.so.2407.0.0
```

如果镜像 Python 版本或 DataSystem 安装路径不同，需要同步修改
`k8s/deployment-inference.yaml`。
