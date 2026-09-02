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
overrides, isolates the build from remote registry settings in Bazel rc files, and then runs the normal
probe build. The full module graph is not evaluated by default because it expands unrelated publishing
tool extensions such as PyPI dependencies:

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
`RUN_BUILD=0` to perform only registry repair, or additionally set `RUN_MODULE_GRAPH=1` when the
machine has the package-index access needed to diagnose every module extension.

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

### URMA runtime selection

The probe must load the host URMA runtime that matches the installed UB driver. Do not allow a
DataSystem Python virtualenv's bundled `liburma.so` to take precedence: it can be found by the
dynamic loader but not have its matching `urma/` provider directory. The run scripts therefore put
`/usr/lib64:/usr/lib64/urma` first in `LD_LIBRARY_PATH`, verify `/usr/lib64/liburma.so`, and print
`urma_runtime_library_path` before starting either binary. Override these defaults only as a pair:

```bash
URMA_RUNTIME_LIB_DIR=/custom/lib64 \
URMA_PROVIDER_LIB_DIR=/custom/lib64/urma \
URMA_RUNTIME_LD_LIBRARY_PATH=/custom/lib64:/custom/lib64/urma \
bash scripts/run_brpc_ub_recommend_server.sh
```

On 2026-09-02, starting the server from the `ds-ub-poc` virtualenv failed before bRPC began
listening. The decisive log was `dl_addr=/home/zcx/venvs/ds-ub-poc/.../liburma.so`, followed by a
missing provider directory and `urma_get_device_list failed, errno 19`. This is a dynamic-library
selection problem, not evidence of an unavailable UB device. A valid startup must instead report
the host `liburma.so`, complete UMQ initialization, and print `MINIMAL_RECOMMEND_SERVER_READY`.

## Build and Runtime Lessons

- A populated Bazel `output_base` and `--nofetch` do not make Bzlmod resolution offline: module
  metadata is still read from every configured registry.
- Use exact versions from `MODULE.bazel.lock` and local registry contents. This POC needs
  `leveldb@1.23` and `openssl@3.3.2.bcr.1` from the local SecretFlow registry.
- Use `--ignore_all_rc_files` and explicit local `file://` registries to prevent a user or workspace
  Bazel rc file from silently restoring remote BCR access.
- Do not make `bazel mod graph` a normal build gate. It evaluates unrelated module extensions and
  can trigger unavailable PyPI metadata downloads.
- The OpenSSL module uses `rules_foreign_cc`; the root module needs a visible direct
  `rules_foreign_cc` dependency when its preinstalled Make and pkg-config toolchains are selected.
- Bazel sandbox actions do not reliably inherit GCC toolset runtime libraries. The build wrapper
  passes the toolset `lib64` path with `--action_env=LD_LIBRARY_PATH=...` so `ar` can load
  `libbfd-2.42.so`.
- Because `recommend.proto` is staged under the `pairec_ub_probe` Bazel package, C++ includes must
  use `pairec_ub_probe/recommend.pb.h`, not the CMake-style bare filename.

This proves the standalone RecommendService request/response path over UB. It does not yet prove the
Go PaiRec gateway path, production model inference, or sustained concurrency behavior.
