# F14 brpc Gateway Phase 1 Design

> Date: 2026-06-15

## Goal

Build a minimal native brpc-over-TCP proof of concept between the recommendation
stack and the inference service without changing the validated HTTP/TRT runtime.

Phase 1 validates:

- protobuf IDL for the current `/recommend` contract
- native brpc service over TCP port `18100`
- K8s sidecar deployment beside the existing Python Flask inference container
- smoke client that can call `Health` and `Recommend`
- latency and result comparison against the existing HTTP endpoint

It does not replace PaiRec's production path yet.

## Current Baseline

```text
PaiRec Go
  -> HTTP/JSON
  -> inference:18000 /recommend
  -> Python Flask
  -> TensorRT-LLM ModelRunnerCpp
```

The current baseline stays intact.

## Phase 1 Architecture

```text
C++ brpc smoke client
  -> baidu_std/protobuf over TCP :18100
  -> brpc-gateway sidecar
  -> HTTP/JSON localhost:18000 /recommend
  -> Python Flask + TRT-LLM
```

The sidecar is intentionally a gateway. This avoids moving tokenizer, prompt,
semantic-id parsing, item mapping, TRT runtime, and DataSystem initialization
into C++ before we have evidence that the RPC transport work is valuable.

## Files

| Path | Purpose |
|------|---------|
| `proto/recommend.proto` | brpc/protobuf IDL, aligned with current JSON request and response |
| `cpp/brpc_gateway/recommend_gateway.cpp` | native brpc server, forwards to Flask HTTP locally |
| `cpp/brpc_gateway/recommend_client.cpp` | native brpc smoke client |
| `cpp/brpc_gateway/CMakeLists.txt` | CMake build for gateway and client |
| `docker/Dockerfile.brpc.gateway` | gateway image build, expects brpc SDK in the base image |
| `k8s/deployment-inference-brpc-image.yaml` | optional inference deployment with brpc sidecar |
| `scripts/build_brpc_gateway_image.sh` | build helper |
| `scripts/k8s_apply_inference_brpc_gateway.sh` | apply helper |
| `scripts/test_brpc_gateway_smoke.sh` | in-cluster smoke test helper |

## Build

The local workstation currently does not have brpc headers or libraries. Build
inside an ARM runtime/builder image that has Apache brpc, protobuf, and cmake:

```bash
BASE_IMAGE=docker.io/library/zcx-pairec-image:v1.1 \
  bash scripts/build_brpc_gateway_image.sh

docker save docker.io/library/pairec-brpc-gateway:k8s-arm64-v1 \
  -o /tmp/pairec-brpc-gateway-k8s-arm64-v1.tar
```

If the base image does not contain brpc, either install brpc into a dedicated
builder image or pass `BRPC_INCLUDE_DIR` / `BRPC_LIBRARY` during CMake configure.

## Deploy

```bash
bash scripts/k8s_apply_inference_brpc_gateway.sh
```

The Service exposes:

```text
inference:18000  HTTP Flask baseline
inference:18100  native brpc gateway
```

## Smoke Test

```bash
bash scripts/test_brpc_gateway_smoke.sh
```

Expected output:

```text
health ok latency_ms=... code=200 status=healthy
recommend ok index=1 latency_ms=... code=200 items=...
```

## Validation Criteria

Phase 1 is successful when:

- `Health` over brpc returns success
- `Recommend` over brpc returns `code=200`
- recommendations are non-empty
- `raw_json` matches the HTTP `/recommend` response shape
- brpc client latency, gateway forward latency, and inference `trace.total_ms`
  can be compared with the existing HTTP path

## Next Step

After phase 1 passes, add a PaiRec-side transport switch:

```text
transport = http | brpc_proxy
```

The lowest-risk production-shaped path is:

```text
PaiRec Go
  -> localhost brpc-client-proxy
  -> native brpc/TCP
  -> inference brpc-gateway
```

Direct Go-to-native-brpc should be deferred unless we decide to accept cgo or
maintain a Go-compatible native brpc client.
