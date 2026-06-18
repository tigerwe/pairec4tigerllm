# F14 brpc Native Inference Design

> Date: 2026-06-18

## Goal

Introduce native brpc into the inference path without adding HTTP-brpc-HTTP
protocol conversion layers.

The 2026-06-18 `brpc_http_proxy` sidecar experiment was reverted because it
kept HTTP on both ends:

```text
PaiRec -> HTTP proxy -> brpc -> HTTP gateway -> Flask
```

That shape is useful as deployment scaffolding only, not as a latency-oriented
brpc integration. The active direction is now:

```text
PaiRec
  -> brpc/TCP
  -> C++ brpc inference service
  -> TensorRT-LLM C++ backend
```

## Validated Baseline

The previous inference-side gateway remains a useful baseline:

- protobuf IDL for the current `/recommend` contract
- native brpc service over TCP port `18100`
- K8s sidecar deployment beside the existing Python Flask inference container
- smoke client that can call `Health` and `Recommend`

Validated output on 2026-06-18:

```text
health ok latency_ms=2 code=200 status=healthy
recommend ok index=1 latency_ms=267 code=200 items=5 inference_ms=264.52
```

That gateway still forwards to Flask over local HTTP, so it is not the final
brpc inference service.

## Native C++ Service Track

`brpc_inference_server` directly implements `RecommendService` and does not call
Flask HTTP. It has backend modes:

| Backend | Purpose |
|---------|---------|
| `semantic_map` | Protocol/K8s smoke backend. Loads `semantic_id_map.json` and returns deterministic mapped items. It proves the native brpc service path without HTTP forwarding. |
| `trtllm_cpp` | Production target. Links TensorRT-LLM C++ `Executor`, builds the same recommendation prompt token sequence, runs generation, parses semantic special tokens, maps them back to items, and returns over brpc/TCP. |

The C++ backend is guarded by `PAIREC_ENABLE_TRTLLM_CPP`. Normal brpc SDK
images can still build the smoke server with the default `OFF` value. The real
TRT-LLM image must set `ENABLE_TRTLLM_CPP=ON` and provide TensorRT-LLM C++
headers, `libtensorrt_llm.so`, and CUDA headers.

The first C++ tokenizer path intentionally does not reimplement HuggingFace BPE
logic. Instead, `scripts/export_cpp_trt_tokenizer_config.py` exports the fixed
prompt fragment token ids and all semantic special-token ids from the same
exported tokenizer used by the TRT engine:

```bash
python scripts/export_cpp_trt_tokenizer_config.py \
  --tokenizer-dir ./exported/qwen3_rec \
  --output ./exported/qwen3_rec/pairec_cpp_tokenizer.txt
```

The service then consumes the generated file through
`--trt_tokenizer_config_path=/app/exported/qwen3_rec/pairec_cpp_tokenizer.txt`.

## Files

| Path | Purpose |
|------|---------|
| `proto/recommend.proto` | brpc/protobuf IDL, aligned with current JSON request and response |
| `cpp/brpc_gateway/recommend_gateway.cpp` | native brpc server, forwards to Flask HTTP locally |
| `cpp/brpc_gateway/brpc_inference_server.cpp` | native C++ brpc inference service, no HTTP forwarding |
| `cpp/brpc_gateway/recommend_client.cpp` | native brpc smoke client |
| `cpp/brpc_gateway/CMakeLists.txt` | CMake build for gateway, native inference server, and client |
| `docker/Dockerfile.brpc.gateway` | brpc binary image build, expects brpc SDK in the base image |
| `k8s/deployment-inference-brpc-image.yaml` | optional inference deployment with brpc sidecar |
| `k8s/deployment-inference-brpc-native.yaml` | native C++ brpc inference deployment |
| `k8s/deployment-inference-brpc-trtllm.yaml` | native brpc deployment using `backend=trtllm_cpp` and GPU |
| `scripts/build_brpc_gateway_image.sh` | build helper |
| `scripts/build_brpc_inference_image.sh` | native brpc inference image build helper |
| `scripts/build_brpc_trtllm_inference_image.sh` | native brpc TRT-LLM C++ image build helper |
| `scripts/ship_brpc_inference_to_worker.sh` | master-to-worker image export/import helper |
| `scripts/k8s_apply_inference_brpc_gateway.sh` | apply helper |
| `scripts/k8s_apply_inference_brpc_native.sh` | native C++ service apply helper |
| `scripts/k8s_apply_inference_brpc_trtllm.sh` | native C++ TRT-LLM service apply helper |
| `scripts/test_brpc_gateway_smoke.sh` | in-cluster smoke test helper |
| `scripts/test_brpc_native_inference_smoke.sh` | native C++ service smoke test helper |

## Build

The local workstation currently does not have brpc headers or libraries. Build
on the ARM master node using a dedicated SDK image:

```bash
BASE_IMAGE=zcx-pairec-image:v1.1 \
  bash scripts/build_brpc_sdk_image.sh \
    docker.io/library/zcx-pairec-brpc-sdk:v1
```

Then build the gateway image from that SDK image:

```bash
BASE_IMAGE=zcx-pairec-brpc-sdk:v1 \
  bash scripts/build_brpc_gateway_image.sh
```

The gateway build script runs `ldd` inside the final image for both
`brpc_gateway`, `brpc_inference_server`, and `brpc_recommend_client`. The build
fails immediately if any runtime `.so` is missing, before the image is exported
to worker1.

Build the native brpc inference image with a separate tag to avoid reusing the
already-imported gateway image:

```bash
BASE_IMAGE=zcx-pairec-brpc-sdk:v1 \
  bash scripts/build_brpc_inference_image.sh
```

For the real C++ TensorRT-LLM backend, first build a combined SDK image on top
of the already validated TRT-LLM/DataSystem runtime image:

```bash
BASE_IMAGE=docker.io/library/pairec-inference:k8s-arm64-ds-runtime-v1 \
  bash scripts/build_brpc_sdk_image.sh \
    docker.io/library/zcx-pairec-trtllm-brpc-sdk:v1
```

Then build the TRT-LLM enabled brpc inference image:

```bash
BASE_IMAGE=zcx-pairec-trtllm-brpc-sdk:v1 \
  bash scripts/build_brpc_trtllm_inference_image.sh
```

If TensorRT-LLM is mounted or built under a non-default path, pass explicit
locations:

```bash
BASE_IMAGE=zcx-pairec-trtllm-brpc-sdk:v1 \
TRTLLM_INCLUDE_DIR=/TensorRT-LLM/cpp/include \
TRTLLM_LIBRARY=/TensorRT-LLM/cpp/build/tensorrt_llm/libtensorrt_llm.so \
TRTLLM_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  bash scripts/build_brpc_trtllm_inference_image.sh
```

The image build still runs `ldd` on the final `brpc_inference_server`. Missing
TensorRT-LLM, brpc, protobuf, Abseil, CUDA, or TensorRT `.so` dependencies
should fail during image build rather than later in K8s.

If GitHub is not reachable from the master node, set `BRPC_REPO` to an internal
mirror before running `scripts/build_brpc_sdk_image.sh`.

## Ship to worker1

The current K8s flow builds images on master, then copies the tar to worker1 and
imports it into containerd:

```bash
WORKER=root@141.61.91.188 \
  bash scripts/ship_brpc_gateway_to_worker.sh
```

For native brpc inference:

```bash
WORKER=root@141.61.91.188 \
  bash scripts/ship_brpc_inference_to_worker.sh
```

## Deploy Gateway Baseline

```bash
bash scripts/k8s_apply_inference_brpc_gateway.sh
```

The Service exposes:

```text
inference:18000  HTTP Flask baseline
inference:18100  native brpc gateway
```

## Deploy Native C++ Service

```bash
bash scripts/k8s_apply_inference_brpc_native.sh
```

This deploys `inference-brpc-native:18100` with `backend=semantic_map`.

## Deploy Native C++ TRT-LLM Service

Generate the tokenizer config on the master/worker workspace before starting
the deployment:

```bash
python scripts/export_cpp_trt_tokenizer_config.py \
  --tokenizer-dir ./exported/qwen3_rec \
  --output ./exported/qwen3_rec/pairec_cpp_tokenizer.txt
```

Then apply the TRT-LLM brpc service:

```bash
bash scripts/k8s_apply_inference_brpc_trtllm.sh
```

This deploys `inference-brpc-trtllm:18100` with `backend=trtllm_cpp`, requests
one GPU, mounts `/app/trt_engines`, `/app/exported`, and `/app/data`, and keeps
the DataSystem/KV runtime env aligned with the current HTTP/TRT baseline.

## Smoke Test Gateway

```bash
bash scripts/test_brpc_gateway_smoke.sh
```

Expected output:

```text
health ok latency_ms=... code=200 status=healthy
recommend ok index=1 latency_ms=... code=200 items=...
```

## Smoke Test Native C++ Service

```bash
bash scripts/test_brpc_native_inference_smoke.sh
```

Expected output:

```text
health ok latency_ms=... code=200 status=healthy
recommend ok index=1 latency_ms=... code=200 items=...
```

The response backend should be `cpp-semantic-map-fallback` in this first native
smoke stage. That is not model inference; it is a no-HTTP brpc service proof.

For the TRT-LLM C++ service, reuse the same client against the new deployment:

```bash
TARGET=deployment/inference-brpc-trtllm \
CONTAINER=brpc-inference \
  bash scripts/test_brpc_native_inference_smoke.sh
```

The response backend should be `trtllm_cpp`, and the trace should include
`prompt_ms`, `runner_generate_ms`, `runner_calls`, `parse_combo_ms`, and
`map_item_ms`.

## Validation Criteria

Gateway baseline is successful when:

- `Health` over brpc returns success
- `Recommend` over brpc returns `code=200`
- recommendations are non-empty
- `raw_json` matches the HTTP `/recommend` response shape

Native C++ service smoke is successful when:

- `brpc_inference_server` starts without Flask/Python.
- `Health` over brpc returns `backend=cpp-semantic-map-fallback`.
- `Recommend` over brpc returns `code=200` and non-empty recommendations.
- No HTTP server or HTTP gateway is involved in the native smoke path.

## Remaining Work

The first `trtllm_cpp` implementation is now present, but still needs remote ARM
runtime validation:

- Build the combined TRT-LLM/brpc SDK image on master.
- Generate `pairec_cpp_tokenizer.txt` from the exported tokenizer.
- Build/import `pairec-brpc-inference:k8s-arm64-trtllm-v1` to worker1.
- Deploy `inference-brpc-trtllm` and run smoke.
- Compare returned items and trace fields against the current Python
  `TRTQwen3Backend` for the same request before wiring PaiRec Go to brpc.
