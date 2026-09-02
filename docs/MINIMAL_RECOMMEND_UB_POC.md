# Minimal RecommendService over UB

This probe isolates the transport from TRT-LLM, DataSystem, and PaiRec. It reuses the repository's
`proto/recommend.proto`, runs a minimal C++ bRPC server and client, and requires UBSocket on both sides.

## Fixed sources

- bRPC: `827db2a9be6a3eac0a1ac3666b4a9cf33b976175`
- UBSComm: `9f80dc9fb5f06ba8b5997064c928b89bda266ffd`

The build script stages the package into the fixed bRPC worktree and links against `//:brpc`. The bRPC
target supplies the matching UBSComm/UMQ implementation selected by `--define brpc_with_urma=true`.

## Build on master

When the machine cannot reach Bazel Central Registry or the SecretFlow registry, use the checked-out
local registry wrapper. It validates the required metadata, repairs the two module-specific registry
overrides, verifies the complete Bzlmod graph with the lock file disabled, and then runs the normal
probe build:

```bash
BRPC_ROOT=/home/zcx/workspace/brpc-827 \
LOCAL_BCR_REGISTRY=/home/zcx/bazel-local-registry/bcr \
LOCAL_SECRET_REGISTRY=/home/zcx/bazel-local-registry/secretflow \
BAZEL_OUTPUT_BASE=/root/.cache/bazel/_bazel_root/0947eeff3cdbdab635f34a3b3ff5f6d1 \
INSTALL_DIR=/opt/pairec-brpc-ub-probe \
BUILD_JOBS=32 \
bash scripts/build_brpc_ub_recommend_probe_local_registry.sh
```

The original `MODULE.bazel` is preserved once as `MODULE.bazel.before-local-registry`. Set
`RUN_BUILD=0` to perform only the registry repair and module graph check.

For a machine with normal registry access, use the standard build entry point:

```bash
BRPC_ROOT=/home/zcx/workspace/brpc-827 \
INSTALL_DIR=/opt/pairec-brpc-ub-probe \
BUILD_JOBS=32 \
bash scripts/build_brpc_ub_recommend_probe.sh
```

Copy `minimal_recommend_client` to node1 after building. Both machines need the same URMA provider
layout used by the successful Echo and KVC UB probes.

## Run

On master:

```bash
SERVER_BIN=/opt/pairec-brpc-ub-probe/bin/minimal_recommend_server \
PORT=18100 \
bash scripts/run_brpc_ub_recommend_server.sh
```

On node1:

```bash
CLIENT_BIN=/opt/pairec-brpc-ub-probe/bin/minimal_recommend_client \
SERVER=141.62.33.105:18100 \
bash scripts/run_brpc_ub_recommend_matrix.sh
```

The matrix covers 0, 1, 4096, 4097, 65536, 1048576, and 3670016 bytes. Every request checks the
server-reported length and SHA-256 plus an exact byte comparison of the response attachment. Passing
also requires a UBSComm `bind jetty success` log. Both programs run with TCP degradation and the
UBSocket backup link disabled.

This proves the standalone RecommendService request/response path over UB. It does not yet prove the
Go PaiRec gateway path, production model inference, or sustained concurrency behavior.
