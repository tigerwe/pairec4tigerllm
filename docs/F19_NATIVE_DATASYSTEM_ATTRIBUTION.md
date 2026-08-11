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

TRTLLM_DIR=/home/zcx/TensorRT-LLM \
  bash scripts/build_f19_attribution_runtime_worker1.sh \
  | tee /tmp/f19-v2-runtime-build.log
```

The source tree must already contain the project DataSystem KVC modifications. The
F19 patch is idempotent, but it is not a replacement for those earlier modifications.

The build script runs on worker1 and writes both overlay artifacts to
`/home/zcx/pairec-f19-runtime/{bin,lib}` only after all build checks pass. It prefers the locally observed
`pairec-brpc-inference:k8s-arm64-trtllm-multisequence-kvc-ctx224-v1` image because
that image matches the current DataSystem/KVC runtime. The exact temporary container
name used by the earlier manual build was not persisted in the project handoff. Override the selected image
only when another local image is known to contain the BRPC, TensorRT-LLM, CUDA and
DataSystem development files:

```bash
BUILD_IMAGE=pairec-brpc-inference:<local-build-capable-tag> \
TRTLLM_DIR=/home/zcx/TensorRT-LLM \
  bash scripts/build_f19_attribution_runtime_worker1.sh
```

The script reproduces the two requirements discovered during the manual build: it
mounts the worker1 NVIDIA driver as `/host-driver/libcuda.so.1`, and adds the CUDA
directory containing `libcudadevrt.a` and `libcudart_static.a` to `LIBRARY_PATH`.
It rejects unresolved gateway libraries, missing V2 attribution markers and missing
output-token trace fields before replacing the overlay. Success ends with
`F19_ATTRIBUTION_RUNTIME_BUILD_OK`.

The full image rebuild remains available as a slower fallback when a hostPath overlay
is not acceptable:

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

That fresh-cache smoke may correctly report `Get=0/Set=0`; it proves identity and
lifecycle closure, but not real DataSystem I/O. To reproduce the previously stable
cache shape and require the same UUID to own exactly three Sets and two Gets, run:

```bash
bash scripts/validate_f19_datasystem_real_io.sh \
  | tee /tmp/f19-datasystem-real-io.log
```

The command verifies the strict runtime, restarts inference before every round,
primes 195 requests, and performs three replay requests. A round passes only when
the legacy offload/onboard trace and the unique native completion both report
`3 Set + 2 Get`, with no failed, pending, or unattributed operation.

## Performance Check

Measure disabled and enabled modes with alternating 1000-request runs after warmup.
Use at least three pairs and compare pair medians. The accepted overhead budget is:

- client E2E average increase: at most 0.1 ms
- client E2E p99 increase: at most 0.5 ms
- throughput loss: at most 1 percent
- every disabled and enabled run runner average: at most 110 ms
- paired output-token average difference: exactly zero

Both modes must use the same image, workload, CPU limits, DataSystem endpoint, and
cache preparation. Changing only `TRTLLM_DATASYSTEM_REQUEST_ATTRIBUTION` requires an
inference rollout because the native process reads the flag once at startup.

Run the formal paired benchmark from the master repository that controls the K8s
Deployment:

```bash
PAIRS=3 \
REQUESTS=1000 \
WARMUP_REQUESTS=1 \
  bash scripts/benchmark_f19_attribution_ab.sh \
  | tee /tmp/f19-attribution-ab.log
```

Each pair runs `disabled` followed by `enabled`. Every round forces an inference
rollout after setting the mode, which resets the native cache before the identical
warmup. Images are neither rebuilt nor imported. The inner validator still checks
business responses, trace closure, BRPC-only protocols, rerank, resource limits and
Pod health; enabled rounds additionally require one exact native completion per
request. Workload throughput is measured from immediately before the first request
until the last response, excluding rollout and completion-wait time.

The final result is written to the printed `summary_json` path. A passing run ends
with both markers:

```text
classification=F19_ATTRIBUTION_AB_PASS
F19_ATTRIBUTION_AB_BENCHMARK_COMPLETE
```

The final enabled round intentionally leaves strict attribution enabled.
Disabled rows report zero DataSystem counts because request attribution is disabled;
the underlying DataSystem KVC path remains configured and is not disabled by this
benchmark.

## Formal Result

The 2026-08-11 ARM64/Kubernetes run completed three 1000-request pairs. Every round
passed the business, trace, BRPC-only, rerank, resource and Pod-health gates. All
enabled rounds joined 1000/1000 exact native completion events and each observed 54
Gets plus 1995 Sets. Pair-median attribution overhead was:

- client E2E average: -0.672 ms
- client E2E p99: -1.947 ms
- closed-loop throughput loss: 0.076 percent
- runner average: -0.188 ms
- runner p99: -1.000 ms

That result is retained as evidence for the tracker statistics and logging path, but
it is no longer accepted as proof for the complete attribution patch. The disabled
mode still set Executor `clientId` and performed the request-map lookup under
`mSequencesMtx` for every generated token. Both modes therefore shared the most
intrusive code path.

The V2 patch makes disabled mode a real no-identity baseline: it does not allocate a
correlation ID, set Executor `clientId`, or access the native request map. Enabled
mode keeps exact attribution but moves the request map to a dedicated mutex so token
accounting cannot contend with the scheduler's sequence critical section. Trace
fields 40 and 41 report output-token count and runner milliseconds per output token.
The benchmark now fails if token counts differ or if either mode remains above 110
ms average runner latency. F19 remains in progress until this stricter benchmark
passes remotely.
