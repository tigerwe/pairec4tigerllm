# Minimal RecommendService over UB

This probe isolates the transport from TRT-LLM, DataSystem, and PaiRec. It reuses the repository's
`proto/recommend.proto`, runs a minimal C++ bRPC server and client, and requires UBSocket on both sides.

## Fixed sources

- bRPC: `827db2a9be6a3eac0a1ac3666b4a9cf33b976175`
- UBSComm: `9f80dc9fb5f06ba8b5997064c928b89bda266ffd`

These identifiers pin the local master environment used by this POC. They must not be attributed to
the public `fanzhaonan/brpc` or openEuler UBSComm repositories without independently verifying object
provenance; the master source trees contain site-specific history and changes.

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

## 2026-09-02 Transport Control Result

The baseline UB attempt reached device discovery on both hosts but failed on the first RPC with
`bthread_setspecific is called on invalid bthread_key_t{index=0 version=0}` and both processes
terminated. A 0-byte TCP call using the same server, client, protobuf service, attachment integrity
checks, and deployment completed with `MINIMAL_RECOMMEND_UB_MATRIX_PASS`. The fault is therefore in
the UBSocket/UB path, not the service contract or the generic bRPC request path. The first focused
UB change removes the Probe's forced `pooled` connection type: it now defaults to bRPC's protocol
default, matching the previously successful `echo_c++_client`; `--probe_connection_type=pooled` is
available for a later explicit comparison.

Before changing bRPC or the Probe again, build and run the official Echo targets from the exact same
bRPC tree, Bzlmod registries, Bazel output base, and UB configuration. The single helper supports all
three host roles:

```bash
# master: build and install the official Echo pair
ACTION=build bash scripts/verify_brpc_ub_echo_baseline.sh

# master: keep the newly built server running
ACTION=server bash /opt/pairec-brpc-ub-echo-baseline/bin/verify_brpc_ub_echo_baseline.sh

# node1: after copying echo_c++_client and this helper into the same bin directory
ACTION=client SERVER=141.62.33.105:18200 \
  bash /opt/pairec-brpc-ub-echo-baseline/bin/verify_brpc_ub_echo_baseline.sh
```

The client is deliberately time-bounded. It passes only after observing both a response and
`bind jetty success`; the known invalid-`bthread_key` failure is reported as
`BRPC_UB_ECHO_BASELINE_CRASH`. If official Echo reproduces it, debug the current bRPC/UB build and
runtime baseline. If Echo passes, compare the Probe and official Echo compile/link actions before
changing transport code.

A passing Echo baseline only proves that the current bRPC/UB build and runtime combination works. A
passing full Recommend matrix is still required to prove the standalone service path; neither result
proves the Go PaiRec gateway path, production model inference, or sustained concurrency behavior.

### Invalid bthread key diagnostic

The current locally built official Echo reproduces the same invalid-`bthread_key` crash as the
Recommend Probe. Collect evidence from the local source trees and binaries without comparing against
public repositories:

```bash
BRPC_ROOT=/home/zcx/workspace/brpc-827 \
BAZEL_OUTPUT_BASE=/root/.cache/bazel/_bazel_root/0947eeff3cdbdab635f34a3b3ff5f6d1 \
SERVER_LOG=/root/brpc-ub-echo-baseline/server-log/echo-server-YYYYMMDD-HHMMSS.log \
bash scripts/diagnose_brpc_ub_bthread_key_crash.sh
```

If the node1 client log has been copied to master, pass it as `CLIENT_LOG=/path/to/echo-client.log`.
The script records local Git state only as metadata, discovers the actual Bazel UBSocket external
tree, finds every bthread key create/get/set/delete site, maps captured stack addresses against both
binaries, records dynamic symbols and dependencies, and packages all evidence under `/tmp`.

### Invalid bthread key root cause and fix

The local-source diagnostic identified a concrete initialization bug in the customized bRPC tree:

- `ubsocket_trace_rpcid_key` was defined as `{0, 0}`, which is exactly `INVALID_BTHREAD_KEY`.
- `ubsocket_trace_call_timestamp` was defined as the hard-coded pair `{1, 0}` instead of being
  allocated by the bthread key registry.
- `Channel::CallMethod()` called `bthread_setspecific()` with both values, while
  `InitializeUBSocket()` registered getters with UBSocket but never called `bthread_key_create()`.
- The observed fatal message named `{index=0 version=0}`, matching the RPC ID key used by
  `Channel::CallMethod()` after it creates the call ID.

Apply the guarded local-tree patch on master:

```bash
cd /home/zcx/workspace/pairec4tigerllm
BRPC_ROOT=/home/zcx/workspace/brpc-827 \
  bash scripts/apply_brpc_ub_trace_key_fix.sh
```

The helper refuses unknown source shapes, saves both changed source files under
`/tmp/brpc-ub-trace-key-fix-backup/<timestamp>/`, and is idempotent. The patch initializes both
globals to `INVALID_BTHREAD_KEY`, allocates them with `bthread_key_create()` before
`ubsocket_init()`, and rolls back the first allocation if the second one fails. Keys remain valid
for process lifetime because UBSocket callbacks and transport threads may continue to read them.

Rebuild and verify the official Echo pair before rebuilding Recommend:

```bash
cd /home/zcx/workspace/pairec4tigerllm
ACTION=build \
BRPC_ROOT=/home/zcx/workspace/brpc-827 \
INSTALL_DIR=/opt/pairec-brpc-ub-echo-baseline \
BUILD_JOBS=32 \
  bash scripts/verify_brpc_ub_echo_baseline.sh
```

The acceptance order is: official Echo UB returns at least one response and records
`bind jetty success`; then the 0-byte Recommend UB request passes; only then run the full payload
matrix. A successful compile alone does not close the runtime defect.
