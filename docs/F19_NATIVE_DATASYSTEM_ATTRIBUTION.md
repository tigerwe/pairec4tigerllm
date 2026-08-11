# F19 Native DataSystem Request Attribution

## Contract

PaiRec keeps its current response path. It does not wait for DataSystem Set completion.
The native TensorRT-LLM process emits one later completion event for each PaiRec UUID:

```json
{"event":"datasystem_request_complete","request_id":"<uuid>","get_count":2,"get_us":12000,"set_count":3,"set_us":19000,"get_failed_count":0,"set_failed_count":0,"pending_count":0,"unknown_count":0,"attribution_complete":true}
```

The correlation path is exact:

```text
PaiRec UUID
  -> brpc RecommendRequest.request_id
  -> gateway monotonic uint64 correlation ID
  -> Executor Request.clientId
  -> LlmRequest.mClientId
  -> KVCacheManager sequence
  -> DataSystem Get/Set scope
  -> native completion event joined back by PaiRec UUID
```

Set belongs to the request whose allocation evicts and offloads a block. Get belongs
to the request whose cache reuse onboards the block. Operations without a request
context are reported as unattributed and are never assigned by timestamp proximity.

The tracker closes only after every native Executor sequence for the Recommend has
been removed and every tracked operation has returned. The default TTL is 30 seconds;
expiry emits `attribution_complete=false` instead of silently dropping the sample.

## Build

Run on the machine that owns the patched TensorRT-LLM source and ARM64/GPU runtime:

```bash
cd /home/zcx/workspace/pairec4tigerllm
git pull gitcode pairec-brpc-observability

TRTLLM_DIR=/home/zcx/TensorRT-LLM \
  bash scripts/apply_trtllm_datasystem_request_attribution_patch.sh

cmake --build /home/zcx/TensorRT-LLM/cpp/build -j"$(nproc)"
```

The source tree must already contain the project DataSystem KVC modifications. The
F19 patch is idempotent, but it is not a replacement for those earlier modifications.

Rebuild the TRT-LLM/brpc SDK or runtime image that contains the newly built headers
and `libtensorrt_llm.so`, then rebuild the inference gateway image against that base:

```bash
BASE_IMAGE=zcx-pairec-trtllm-brpc-sdk:v1 \
  bash scripts/build_brpc_trtllm_inference_image.sh \
    docker.io/library/pairec-brpc-inference:k8s-arm64-v1 \
    /home/zcx/pairec-brpc-inference-k8s-arm64-v1.tar
```

Import the rebuilt image into the Kubernetes node's `k8s.io` containerd namespace
using the existing local image flow. Do not transfer the historical 122 GiB tar when
the node can rebuild or receive the changed binary and native libraries directly.

When worker1 already contains the rebuilt gateway and native library under
`/home/zcx/pairec-f19-runtime`, deploy and validate both hostPath overlays from the
master with one command:

```bash
bash scripts/deploy_f19_datasystem_attribution_overlay.sh apply
```

The script backs up the current Deployment, first rolls out with attribution
disabled to verify the executable, hashes, dynamic libraries and Health endpoint,
then enables strict attribution and runs one exact request-id smoke. It is
idempotent. Use `verify` for a read-only recheck, `smoke` to continue from an
already enabled strict Pod without another rollout, and `rollback` to undo one
Deployment revision. The native ready marker is emitted when the first tracked
Recommend constructs the attribution tracker, not during process startup or Health.
Set `RUN_EXACT_SMOKE=0` only when deployment validation must be separated from the
business request.

## Validation

Three-request strict smoke:

```bash
REQUIRE_DATASYSTEM_ATTRIBUTION=1 \
REQUESTS=3 \
WARMUP_REQUESTS=1 \
RUN_HTTP_AB=0 \
BUILD_IMAGES=0 \
IMPORT_IMAGES=0 \
  bash scripts/deploy_and_validate_pairec_brpc_observed.sh
```

Formal run:

```bash
REQUIRE_DATASYSTEM_ATTRIBUTION=1 \
REQUESTS=1000 \
WARMUP_REQUESTS=1 \
RUN_HTTP_AB=0 \
BUILD_IMAGES=0 \
IMPORT_IMAGES=0 \
  bash scripts/deploy_and_validate_pairec_brpc_observed.sh
```

The strict gate requires exactly one native completion event for each workload UUID,
`attribution_complete=true`, and zero failed, pending, unknown, duplicate, missing, or
invalid events. `pipeline_trace_complete.datasystem` remains an immediate snapshot;
the summary joins the later native event from `inference.log` by exact `request_id`.

For a single request, use:

```bash
NAMESPACE=pairec \
PAIREC_TARGET=deploy/pairec-brpc-observed \
BRPC_TARGET=deployment/inference-brpc-trtllm \
REQUIRE_EXACT_DATASYSTEM_ATTRIBUTION=1 \
  bash scripts/trace_single_brpc_datasystem_request.sh
```

## Performance Check

Measure disabled and enabled modes with alternating 1000-request runs after warmup.
Use at least three pairs and compare pair medians. The accepted overhead budget is:

- client E2E average increase: at most 0.1 ms
- client E2E p99 increase: at most 0.5 ms
- throughput loss: at most 1 percent

Both modes must use the same image, workload, CPU limits, DataSystem endpoint, and
cache preparation. Changing only `TRTLLM_DATASYSTEM_REQUEST_ATTRIBUTION` requires an
inference rollout because the native process reads the flag once at startup.
