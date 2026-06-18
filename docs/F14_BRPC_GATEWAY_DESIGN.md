# F14 brpc Gateway / PaiRec Proxy Design

> Date: 2026-06-18

## Goal

Build a minimal native brpc-over-TCP proof of concept between the recommendation
stack and the inference service without changing the validated HTTP/TRT runtime.

Phase 1 validates:

- protobuf IDL for the current `/recommend` contract
- native brpc service over TCP port `18100`
- K8s sidecar deployment beside the existing Python Flask inference container
- smoke client that can call `Health` and `Recommend`
- PaiRec-side local HTTP to brpc proxy, so the recommendation service can use
  brpc on the inter-service hop without adding a Go brpc SDK

It does not move tokenizer, prompt construction, semantic-id parsing, item
mapping, TRT runtime, or DataSystem initialization into C++.

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

## PaiRec Integration Architecture

```text
PaiRec Go
  -> HTTP/JSON localhost:18090 /recommend
  -> brpc-http-proxy sidecar
  -> baidu_std/protobuf over TCP inference:18100
  -> brpc-gateway sidecar
  -> HTTP/JSON localhost:18000 /recommend
  -> Python Flask + TRT-LLM
```

The Go service still uses its existing HTTP client. The brpc protocol is
introduced between the PaiRec Pod and the inference Pod. This keeps the
business code stable and avoids cgo or an unverified Go native brpc client.

`RecommendRequest.raw_json` carries the original PaiRec `/recommend` JSON body
through brpc. The inference-side gateway forwards `raw_json` directly when it is
present, so the proxy does not need to re-parse business fields.

## Files

| Path | Purpose |
|------|---------|
| `proto/recommend.proto` | brpc/protobuf IDL; `RecommendRequest.raw_json` preserves the current JSON request |
| `cpp/brpc_gateway/recommend_gateway.cpp` | native brpc server, forwards to Flask HTTP locally |
| `cpp/brpc_gateway/brpc_http_proxy.cpp` | PaiRec-side HTTP `/recommend` proxy, forwards to inference over native brpc |
| `cpp/brpc_gateway/recommend_client.cpp` | native brpc smoke client |
| `cpp/brpc_gateway/CMakeLists.txt` | CMake build for gateway, proxy, and client |
| `docker/Dockerfile.brpc.gateway` | gateway image build, expects brpc SDK in the base image |
| `k8s/deployment-inference-brpc-image.yaml` | optional inference deployment with brpc sidecar |
| `k8s/configmap-pairec-brpc.yaml` | PaiRec config that points `server_url` to `127.0.0.1:18090` |
| `k8s/deployment-pairec-brpc-hostpath.yaml` | PaiRec deployment with local brpc proxy sidecar |
| `scripts/build_brpc_gateway_image.sh` | build helper |
| `scripts/k8s_apply_inference_brpc_gateway.sh` | apply helper |
| `scripts/k8s_apply_pairec_brpc_hostpath.sh` | apply PaiRec brpc-proxy helper |
| `scripts/test_brpc_gateway_smoke.sh` | in-cluster smoke test helper |
| `scripts/test_pairec_brpc_e2e.sh` | PaiRec `/api/recommend` through brpc proxy validation |

## Build

The local workstation currently does not have brpc headers or libraries. Build
on the ARM master node using a dedicated SDK image:

```bash
BASE_IMAGE=docker.io/library/zcx-pairec-image:v1.1 \
  bash scripts/build_brpc_sdk_image.sh \
    docker.io/library/zcx-pairec-brpc-sdk:v1
```

Then build the gateway image from that SDK image:

```bash
BASE_IMAGE=docker.io/library/zcx-pairec-brpc-sdk:v1 \
  bash scripts/build_brpc_gateway_image.sh
```

The current PaiRec-proxy build uses
`docker.io/library/pairec-brpc-gateway:k8s-arm64-v2`. Rebuild and re-import this
tag before applying the brpc PaiRec manifest; the older `v1` image does not
contain `brpc_http_proxy`.

The gateway build script runs `ldd` inside the final image for `brpc_gateway`,
`brpc_recommend_client`, and `brpc_http_proxy`. The build fails immediately if
any runtime `.so` is missing, before the image is exported to worker1.

If GitHub is not reachable from the master node, set `BRPC_REPO` to an internal
mirror before running `scripts/build_brpc_sdk_image.sh`.

## Ship to worker1

The current K8s flow builds images on master, then copies the tar to worker1 and
imports it into containerd:

```bash
WORKER=root@141.61.91.188 \
  bash scripts/ship_brpc_gateway_to_worker.sh
```

## Deploy

Deploy or update the inference side first:

```bash
bash scripts/k8s_apply_inference_brpc_gateway.sh
```

The Service exposes:

```text
inference:18000  HTTP Flask baseline
inference:18100  native brpc gateway
```

Then deploy PaiRec with the local HTTP to brpc proxy sidecar:

```bash
bash scripts/k8s_apply_pairec_brpc_hostpath.sh
```

This apply replaces the `pairec-config` ConfigMap with the brpc-proxy version,
where `RecallAlgo.server_url` is `http://127.0.0.1:18090`.

## Smoke Test

Inference brpc gateway smoke:

```bash
bash scripts/test_brpc_gateway_smoke.sh
```

Expected output:

```text
health ok latency_ms=... code=200 status=healthy
recommend ok index=1 latency_ms=... code=200 items=...
```

PaiRec recommendation-system brpc path smoke:

```bash
bash scripts/test_pairec_brpc_e2e.sh
```

Expected output includes:

```text
brpc proxy health
pairec health
pairec recommend through brpc proxy
pairec brpc e2e ok uid=6312 size=10 scene=home_feed
```

## Validation Criteria

Inference-side phase 1 is successful when:

- `Health` over brpc returns success
- `Recommend` over brpc returns `code=200`
- recommendations are non-empty
- `raw_json` matches the HTTP `/recommend` response shape

PaiRec integration is successful when:

- `brpc_http_proxy` `/health` returns success from inside the PaiRec Pod
- PaiRec `/ping` returns success
- PaiRec `/api/recommend` returns `code=200` and non-empty items while its
  `server_url` points to `127.0.0.1:18090`
- inference `brpc-gateway` logs show the corresponding `Recommend` request

## Next Step

After the sidecar path is stable, add a small transport switch in config or
deployment overlay:

```text
transport = http | brpc_proxy
```

The HTTP deployment should remain available as rollback. Direct Go-to-native
brpc should stay deferred unless we decide to accept cgo or maintain a
Go-compatible native brpc client.
