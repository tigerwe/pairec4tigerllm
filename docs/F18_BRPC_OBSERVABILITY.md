# F18: Pure BRPC pipeline observability

## Scope

The isolated experiment keeps the existing HTTP deployment intact and runs:

```text
client -> PaiRec observed
  -> BRPC generative inference -> TensorRT-LLM -> DataSystem
  -> BRPC vector adapter -> localhost DSSM/Milvus backend
  -> BRPC rank adapter -> localhost DeepFM backend
```

The vector and rank adapters are separate services and separate executables. Their Python
backends are reachable only through localhost inside the same Pod. PaiRec has no HTTP fallback
and each BRPC call has one total attempt (zero retries).
The deployment renderer resolves all three BRPC Services to numeric ClusterIPs before creating
the observed PaiRec Pod, so request traffic and readiness do not depend on CoreDNS.

## Trace contract

`proto/pipeline_service.proto` defines the protocol-neutral request context and service trace.
PaiRec owns the authoritative `pipeline_trace_complete` event. Durations use integer
microseconds and process-local monotonic clocks; the collector never subtracts timestamps from
different hosts.

The trace contains these framework stages:

```text
user_feature, recall, filter, general_rank, feature_load, framework_rank,
pipeline_wait, pipeline_merge, sort, response_build, controller_overhead
```

It also contains `generative_recall`, `vector_recall`, and `deepfm_rank`. The original F18
validation emitted `rerank` as explicitly disabled. F20 adds an optional in-process
`source_quota_tail` reranker after DeepFM; it is enabled only by the isolated observed config and
has its own validation gate. See `docs/F20_SOURCE_QUOTA_RERANK.md`.

Service-reported inner phases are carried as span attributes and summarized independently:

```text
generative: rpc_us, inference_total_us, runner_generate_us
vector:     service_total_us, feature_us, compute_us
rank:       service_total_us, feature_us, compute_us, backend_rpc_us
```

The PaiRec closure gate is:

```text
abs(pairec_total_us - accounted_us) <= max(3000us, 5% of pairec_total_us)
```

Prometheus exposes bounded-label histograms and counters under `pairec_pipeline_*`.

## DataSystem attribution

DataSystem timings are accepted only when the native inference response correlates them to the
same request ID. Existing asynchronous or aggregate worker logs are not assigned to a request.
When native TensorRT-LLM KVC does not propagate request identity, the trace records:

```text
attribution_complete=false
reason=native_trt_kvc_request_identity_not_propagated
```

Keep `REQUIRE_DATASYSTEM_ATTRIBUTION=0` for pipeline/BRPC engineering validation. Set it to `1`
only after the deployed inference path returns exact per-request Get/Set counts and durations.
An explicit correlated DataSystem probe can satisfy this gate; inferred `3 Set + 2 Get` counts
cannot.

## Inference prerequisite

The observed PaiRec validates the `request_id` echoed by the C++ generative inference service.
An old `brpc_inference_server` image does not contain fields 30-39 from
`proto/recommend.proto`, so every formal trace will be rejected. Rebuild and roll out the native
TRT-LLM image from this branch before formal validation:

```bash
bash scripts/build_brpc_trtllm_inference_image.sh \
  docker.io/library/pairec-brpc-inference:k8s-arm64-trtllm-observed-v1 \
  /home/zcx/pairec-brpc-inference-k8s-arm64-trtllm-observed-v1.tar
```

Use the existing worker-local image import path. Do not transfer a 100+ GiB image tar over SSH
when the image can be built or imported on worker1.

## One-command deployment and validation

After the updated inference service is Ready:

The existing `zcx-pairec-image:v1.1` does not contain the `pymilvus` packages that were
installed interactively in the `dssm-recall` container. Export those small runtime packages once
instead of rebuilding or transferring the 121 GiB image:

```bash
SOURCE_CONTAINER=dssm-recall \
OUTPUT_DIR=/home/zcx/pairec-python-runtime \
  bash scripts/export_pymilvus_runtime_from_container.sh
```

The vector backend mounts that directory read-only at `/opt/pairec-python-extra`; its adapter
does not become Ready unless the backend explicitly reports `milvus=true`.

```bash
REQUESTS=1000 \
HTTP_BASELINE_REQUESTS=100 \
REQUIRE_DATASYSTEM_ATTRIBUTION=0 \
  bash scripts/deploy_and_validate_pairec_brpc_observed.sh \
  | tee /tmp/pairec-brpc-observed.log
```

The script builds/imports the PaiRec and adapter images, deploys two dual-container BRPC
services and an isolated PaiRec instance, checks true adapter Health RPCs, runs the workload,
validates at least 99.9% complete traces, produces p50/p95/p99 summaries, checks Prometheus,
optionally compares the retained HTTP deployment, and gates restarts, OOM/health events, and
CPU throttled periods (at most 5%) for all five experiment containers.

Formal success is `PAIREC_PURE_BRPC_OBSERVABILITY_OK`. A business response can remain valid
while an experimental trace is invalid; the validator rejects that sample without changing the
recommendation response schema.

## Formal result

The 2026-08-08 ARM64/Kubernetes validation completed 1000/1000 valid pure-BRPC requests with
no missing or invalid traces and no HTTP fallback. Client E2E p50/p95/p99 was
108.774/117.382/120.622ms. The retained HTTP baseline p99 was 118.422ms, so the pure-BRPC
pipeline added 2.200ms at p99. Generative, vector, and DeepFM span p99 values were
108.794/11.923/12.081ms. All five experiment containers stayed healthy with zero restarts;
CPU throttled-period ratios were 0% except PaiRec at 0.068%, below the 5% gate.

DataSystem attribution remained explicitly incomplete for all samples because native TRT KVC
does not yet propagate request identity. Exact native Get/Set attribution is tracked separately
and was not inferred from aggregate worker logs.
