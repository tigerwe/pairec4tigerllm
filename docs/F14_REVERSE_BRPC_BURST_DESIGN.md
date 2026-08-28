# F14 Reverse BRPC Burst Design

Date: 2026-08-28

## Goal

Add two opt-in reverse-direction BRPC pressure stages without changing the
recommendation result:

1. Generation Wrapper on worker1 to Generation Return Sink on master.
2. Rank Wrapper on worker1 to Rank Return Sink on master.

The existing forward bursts remain unchanged. The reverse stages characterize
the latency and variance added by worker1 transmit, the inter-node return path,
and master-side BRPC receive scheduling. They are pressure stages, not queues or
admission-control buffers.

## Topology

```text
master PaiRec -> worker1 Generation Wrapper (existing forward c1000)
  -> generation backend
  -> master Generation Return Sink :18301 (new reverse c1000)
  -> original generation response
  -> worker1 Rank Wrapper (existing forward c1000)
  -> Rank KVC Get and rank backend
  -> master Rank Return Sink :18302 (new reverse c1000)
  -> original rank response -> rerank and HTTP response
```

Both sinks use host networking, are pinned to master, and have fixed requests
and limits of 8 CPU and 2 GiB. Worker Wrappers connect directly to the master
25G address `192.168.100.12`, bypassing ClusterIP, kube-proxy, and the
management network.

## Burst Contract

Each reverse stage owns 1000 long-lived, preconnected sessions and 1000
persistent workers. A request releases one strict barrier containing:

- one business marker;
- 999 Health pressure requests;
- 102400 request bytes per lane;
- a minimal response from the master Sink.

The marker contains the request ID, stage, backend response code, result item
count, and a SHA256 identity for the backend response. The master Sink validates
the payload and echoes its identity.

The Wrapper waits for the marker and fails closed on marker error or its 1000 ms
timeout. It does not wait for the 999 Health requests. Health calls have a
5000 ms timeout and finish asynchronously. Their errors invalidate the measured
round but do not retract a business response that has already returned.

The benchmark waits for both asynchronous completion events before starting the
next measured request. This wait is outside client E2E. Pressure may overlap
later stages within the same request, but may not leak into the next sample.

## Runtime Control

Reverse burst support is disabled unless an endpoint is configured and starts
disarmed. Control and treatment switch it with an explicit arm/disarm command;
no Pod rollout, model reload, connection rebuild, or fallback to lower
concurrency is allowed. Missing configuration preserves current Wrapper
behavior.

## Required Evidence

Every enabled stage must report:

- `connected_sessions=1000` and `armed_workers=1000`;
- marker success, wall time, Sink service time, and front BRPC time;
- 999 pressure requests, 999 successes, and zero errors;
- exactly 102400000 accepted request bytes;
- pressure p95/max, start skew, max active, and tail after marker;
- a request-scoped asynchronous completion event;
- zero restart, OOM, and Sink CPU throttling.

Tail overlap is only calculated inside a shared host clock domain. Generation
Wrapper tail versus Rank Wrapper execution uses worker1 system-clock epochs;
Rank Sink tail versus rerank uses master system-clock epochs. No metric directly
subtracts a worker1 timestamp from a master timestamp.

The recommendation semantic fingerprint must match with reverse burst disabled
and enabled.

## Experiment

Control keeps Generation BRPC c1000, Generation KVC c32, Rank BRPC c1000, and
Rank KVC c32. Treatment changes only the two reverse stages from disarmed to
armed. Run one functional smoke, then five `A B B A` blocks for ten valid Control
and ten valid Treatment samples. The first run establishes latency, CV, MAD, and
paired-delta distributions; performance thresholds are set only after those
measurements exist.

## Implementation State

The coordinator, two Wrapper integration points, strict master Sink, runtime
arm/disarm control, per-request drain gate, Sink runtime gates, n1 smoke, and
five-block ABBA runner are implemented locally. Static verification on
2026-08-28 passed 165 unit tests with one existing conditional skip, Shell
syntax, YAML parsing, embedded Python compilation, and `git diff --check`.
The remaining evidence is an ARM/bRPC SDK image build followed by the cluster
n1 and ABBA runs.
