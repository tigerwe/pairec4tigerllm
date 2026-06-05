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
