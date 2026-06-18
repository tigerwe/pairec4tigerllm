# 工作进度

> 最后更新: 2026-06-18 | 当前状态: F14 已回退 `HTTP -> brpc -> HTTP` 的 PaiRec proxy 方案，改走无 HTTP 转发的 C++ brpc inference service 路线；新增 `brpc_inference_server`，直接实现 `RecommendService`，当前 `semantic_map` backend 用于 native brpc/K8s smoke；镜像 tar 默认统一保存在 `/home/zcx` | 下一步: 在 master 构建并导入 `docker.io/library/pairec-brpc-inference:k8s-arm64-v1`，部署 `k8s/deployment-inference-brpc-native.yaml` 并跑 native brpc smoke，然后在 ARM TRT-LLM runtime 内接入 `trtllm_cpp`

## 时间线

| 日期 | 进度 |
|------|------|
| **6/18** | **补强 brpc 镜像离线构建：master 本地已有 `zcx-pairec-image:v1.1` 与 `zcx-pairec-brpc-sdk:v1`，但使用 `docker.io/library/zcx-*` 作为 `BASE_IMAGE` 会触发 daocloud 元数据解析并因私有镜像不在白名单返回 403。已将 brpc build 脚本默认 base image 改成本地 tag；新增 `.dockerignore` 排除 `datasystem/uds` Unix socket，避免 legacy builder 打包 build context 时输出 `archive/tar: sockets not supported`；`docker/Dockerfile.brpc.gateway` 在 build/final stage 清空继承的 `LD_PRELOAD`，避免 DataSystem 预加载路径污染 brpc 镜像构建和 ldd 检查** |
| **6/18** | **统一镜像 tar 保存路径：后续 build/ship 脚本默认把导出的 `.tar` 或 `.tar.gz` 放到 `/home/zcx`，不再使用 `/tmp` 作为镜像中转目录；`ship_brpc_gateway_to_worker.sh` 与 `ship_brpc_inference_to_worker.sh` 会先创建本地 tar 目录，scp 到 worker 时也默认放到 `/home/zcx`，再由 worker 执行 `ctr -n k8s.io images import /home/zcx/*.tar`。已同步 `k8s/README.md` 示例和相关构建脚本输出** |
| **6/18** | **F14 调整方向：已用 `git revert` 生成提交撤销 PaiRec 侧 `brpc_http_proxy` sidecar 方案，因为该形态仍是 `HTTP -> brpc -> HTTP`，对时延优化意义不大；新增 `cpp/brpc_gateway/brpc_inference_server.cpp`，直接实现 `RecommendService`，不调用 Flask HTTP。当前 `backend=semantic_map` 会加载 `/app/data/tenrec/processed/semantic_id_map.json` 并返回确定性 mapped items，用于证明 native C++ brpc service、镜像、K8s 和客户端协议链路；`backend=trtllm_cpp` 当前 fail-fast，作为后续真实 TensorRT-LLM C++ runner 接入口。新增 `k8s/deployment-inference-brpc-native.yaml`、`scripts/build_brpc_inference_image.sh`、`scripts/ship_brpc_inference_to_worker.sh`、`scripts/k8s_apply_inference_brpc_native.sh`、`scripts/test_brpc_native_inference_smoke.sh`。下一步先远程构建/导入 `pairec-brpc-inference:k8s-arm64-v1` 并做 native smoke，再迁移 Python `TRTQwen3Backend` 的 prompt/tokenizer/runner.generate/parse/map trace 到 C++ backend** |
| **6/18** | **F14 brpc phase-1 K8s smoke 验证通过：基于 `docker.io/library/pairec-brpc-gateway:k8s-arm64-v1` sidecar，`scripts/test_brpc_gateway_smoke.sh` 输出 `health ok latency_ms=2 code=200 status=healthy`；`Recommend` 输出 `recommend ok index=1 latency_ms=267 code=200 user_id=brpc_smoke_1 items=5 inference_ms=264.52`，返回 5 个推荐 item，trace 显示 `backend=trt-qwen3`、`runner_calls=1`、`runner_generate_ms=212.35ms`、`total_ms=264.52ms`、`result_cache_source=disabled`。这证明 native brpc/`baidu_std` over TCP -> brpc-gateway sidecar -> localhost Flask `/recommend` -> TRT-LLM 的 phase-1 闭环已跑通；该结果暂不代表最终性能收益，因为 gateway 内部仍转发 HTTP，下一步需要同批 HTTP vs brpc 延迟和结果等价对比** |
| **6/16** | **补强 F14 brpc gateway 镜像构建防线：`scripts/build_brpc_gateway_image.sh` 在 docker build 后自动进入最终镜像，对 `/opt/pairec-brpc/bin/brpc_gateway` 与 `/opt/pairec-brpc/bin/brpc_recommend_client` 执行 `ldd`，若出现 `not found` 立即失败，避免再次出现编译成功但 K8s sidecar 运行时缺 `libbrpc.so`/Abseil/protobuf 等动态库导致 CrashLoopBackOff 的问题；`docs/F14_BRPC_GATEWAY_DESIGN.md` 已同步说明该校验。下一步重新构建 gateway 镜像时应先看到 ldd 全部解析成功，再执行 worker1 导入和 K8s smoke** |
| **6/16** | **F14 brpc 环境搭建流程已固化：新增 `scripts/build_brpc_sdk_image.sh`，基于 `zcx-pairec-image:v1.1` 构建 `docker.io/library/zcx-pairec-brpc-sdk:v1`，在专用 SDK base image 中安装编译依赖并构建/安装 Apache brpc headers/libs；新增 `scripts/ship_brpc_gateway_to_worker.sh`，固化 master 构建 gateway 镜像后的 `docker save`、`scp` 到 worker1、`ctr -n k8s.io images import` 或 `docker load` 导入流程；更新 `docs/F14_BRPC_GATEWAY_DESIGN.md` 为当前 master 构建、worker1 导入的实际 K8s 镜像流转方式。新增脚本本机 `bash -n` 通过，`feature_list.json` JSON parse 和 `git diff --check` 通过。当前远程 master 应使用本地 base tag：`BASE_IMAGE=zcx-pairec-image:v1.1 bash scripts/build_brpc_sdk_image.sh docker.io/library/zcx-pairec-brpc-sdk:v1`、`BASE_IMAGE=zcx-pairec-brpc-sdk:v1 bash scripts/build_brpc_gateway_image.sh`、`WORKER=root@141.61.91.188 bash scripts/ship_brpc_gateway_to_worker.sh`，再 apply/smoke** |
| **6/15** | **F14 brpc phase-1 PoC 设计与实现骨架完成：新增 `proto/recommend.proto` 定义 `RecommendService.Recommend/Health`，字段对齐当前 `/recommend` JSON 并保留 `raw_json` 便于对比；新增 `cpp/brpc_gateway/recommend_gateway.cpp`，作为 native brpc `baidu_std`/TCP server 监听 `18100`，内部通过 brpc HTTP channel 转发到 `127.0.0.1:18000/recommend`；新增 `cpp/brpc_gateway/recommend_client.cpp` 作为 C++ smoke client；新增 `docker/Dockerfile.brpc.gateway`、`k8s/deployment-inference-brpc-image.yaml` 和构建/部署/验证脚本。当前本机验证：`protoc --proto_path=proto --cpp_out=/tmp/pairec-brpc-proto-check proto/recommend.proto` 通过；`bash -n scripts/build_brpc_gateway_image.sh scripts/k8s_apply_inference_brpc_gateway.sh scripts/test_brpc_gateway_smoke.sh` 通过；`python -c "import yaml; list(yaml.safe_load_all(open('k8s/deployment-inference-brpc-image.yaml'))); print('yaml ok')"` 输出 `yaml ok`；`cmake -S cpp/brpc_gateway -B /tmp/pairec-brpc-cmake-check` 找到 Protobuf 3.12.4 后按预期失败于 `brpc SDK was not found`，说明本机缺 brpc headers/libs，需远程或专用 builder 镜像完成编译。已补 `scripts/build_brpc_sdk_image.sh` 固化 brpc SDK base image 构建，并补 `scripts/ship_brpc_gateway_to_worker.sh` 固化 master 构建、tar 导出、scp 到 worker1、containerd/docker 导入流程。F14 仍为 pending，下一步是远程构建 gateway 镜像并在 inference Pod 内做 `Health`/`Recommend` smoke，再输出 HTTP vs brpc 延迟对比** |
| **6/12** | **补充 E2E DataSystem 时延报告指标口径说明：在 `docs/E2E_DATASYSTEM_FINAL_REPORT_2026-06-04.md` 新增“指标口径详细说明”，逐项解释 `client_e2e_ms`、PaiRec `total/user_feature/recall/filter/rank/merge/sort`、GenerativeRecall `cache/history/convert/http/items/http_overhead`、inference `tr_*`/TRT service stages，以及 C++ DataSystem per KV block `offload.create/d2h/set/total`、`onboard.get/h2d/total`。报告明确区分 request 级、服务内阶段级和 KV block 级指标，并标注 `tr_*` 与 inference trace 字段的别名关系，避免将 Python 结果缓存/DataSystem 指标与 TensorRT-LLM C++ KV block Set/Get 混淆** |
| **6/12** | **K8s C++ KV DataSystem offload/onboard 功能闭环验证通过：基于正式 `pairec-inference:k8s-arm64-ds-runtime-v1` 镜像运行 pressure/replay 压测，日志判定显示 KV pool `primaryBlocks=32 secondaryBlocks=28`、scheduler `MAX_UTILIZATION`、`reuse disabled warning seen=False`，新增结构化 DataSystem trace `offload/onboard=1313/3`，`Get Key=3`，说明 C++ 层已真实触发 GPU KV block offload 到 DataSystem 并从 DataSystem onboard 回 HBM。当前样本量仍不足以给 p9999 结论：offload count=1313、onboard count=3，低于脚本建议 minimum=10000/recommended>=100000；下一步只需扩大 replay/onboard 压测样本用于分位数报告，不再需要改 runtime 镜像或 engine** |
| **6/12** | **正式 K8s inference DS runtime 镜像启动验证通过：基于 `scripts/build_datasystem_runtime_base_image.sh` 固化的 patched TensorRT-LLM runtime 重建 `docker.io/library/pairec-inference:k8s-arm64-ds-runtime-v1` 后，inference Pod 启动日志显示 `[TRTQwen3Backend] Scheduler policy: max_utilization, max_kv_tokens=1024, host_cache_size=104857600`、KV pool `primaryBlocks=32 secondaryBlocks=28`、`[TensorRT-LLM][Datasystem] Create Datasystem class`、`Init KvCache Manager DataSystem. host = 141.61.91.188, ip = 18481` 与 `Init KvCache Manager DataSystem success`。Python DataSystem 按预期保持 `PYTHON_DATASYSTEM_ENABLED=0`，当前只验证 C++ KV path。下一步运行 `scripts/test_trt_cpp_kv_offload.py` 的 pressure/replay 压测，确认结构化 `[Datasystem][TRACE] op=offload/onboard` 出现并汇总 C++ Set/Get latency** |
| **6/12** | **确认 K8s C++ DataSystem 差异来自 runtime 镜像而非 engine：用户在历史跑通容器 `3d25ebe028d6` 内用同一 `qwen3_rec_v4` 启动服务，日志出现 `[TensorRT-LLM][Datasystem] Create Datasystem class`、`Init KvCache Manager DataSystem success`，且 `Blocks per window size` 显示 `primaryBlocks=32 secondaryBlocks=28`，说明该容器内 TensorRT-LLM C++ patch 与 `block_ds_consumer.so/stub_gpu.so/libabseil` 预加载链路有效。进一步确认 `tensorrt-llm` 是 editable project，实际位置为 `/TensorRT-LLM`，而该路径是宿主机 `/home/zcx/TensorRT-LLM` 的 bind mount，因此不能直接 `docker commit` 旧容器。已新增 `scripts/build_datasystem_runtime_base_image.sh`，从旧容器复制 `/TensorRT-LLM` 到新的 `docker.io/library/zcx-pairec-ds-runtime:v1` base image；正式 runtime Dockerfile 继续基于该 base 构建，并在镜像内编译 `/opt/pairec/lib/block_ds_consumer.so` 与 `/opt/pairec/lib/stub_gpu.so`；entrypoint 默认 `PYTHONPATH` 同时包含 `/home/TensorRT-LLM`、`/TensorRT-LLM` 与 cutlass python 路径，启动时清理旧 `LD_PRELOAD`，按顺序加载 block/stub/NVML/abseil；K8s manifest 默认 tag 为 `docker.io/library/pairec-inference:k8s-arm64-ds-runtime-v1`。下一步重建/导入新镜像后复验 C++ offload/onboard TRACE** |
| **6/12** | **修正 K8s C++ DataSystem KV 失败判断：用户在历史跑通容器中检查 `trt_engines/qwen3_rec_v4/config.json`，同样显示 `context_fmha=False`、`use_paged_context_fmha=False`、`kv_cache_type=PAGED`，说明不能把当前失败简单归因于 `qwen3_rec_v4` engine 配置。已将 `k8s/deployment-inference-image.yaml` 恢复为默认 `qwen3_rec_v4`，新增 `scripts/k8s_check_trtllm_datasystem_runtime.sh`，用于在当前 inference Pod 内同时打印 engine config、`ModelRunnerCpp.from_dir` 签名，并扫描 TensorRT-LLM `.so` 是否包含 `Create Datasystem class`、`Init KvCache Manager DataSystem`、`op=offload/onboard` 等 C++ DataSystem patch 字符串。新的判断顺序：先确认 runtime 镜像内 C++ DataSystem patch 是否存在并被加载；若不存在，优先重建/替换 TensorRT-LLM runtime；若存在但 upstream block reuse warning 仍阻断路径，再使用 `qwen3_rec_v4_paged_fmha` engine build job 作为后备方案** |
| **6/12** | **远程 verifier 显示 `reuse disabled warning seen=True`、offload/onboard/TRACE 均为 0；进一步检查当前 K8s 挂载的 `/app/trt_engines/qwen3_rec_v4/config.json`，确认 `build_config/plugin_config/context_fmha=False`、`use_paged_context_fmha=False`。该配置会触发 upstream TensorRT-LLM 关闭 `kv_cache_enable_block_reuse` 的 warning，但由于历史容器同样配置曾跑通 C++ DataSystem offload/onboard，当前不能仅凭 engine 配置下结论。为后备验证新增 `k8s/job-build-trt-engine-paged-fmha.yaml` 与 `scripts/k8s_build_trt_engine_paged_fmha.sh`，可在 worker1/GPU 上构建独立 `trt_engines/qwen3_rec_v4_paged_fmha`，构建参数包含 `--kv_cache_type paged --remove_input_padding enable --context_fmha enable --use_paged_context_fmha enable`；旧 `qwen3_rec_v4` 保留为默认 HTTP/TRT 基线 engine。当前本地 `bash -n` 与 YAML parse 通过；下一步优先诊断 inference 镜像内 TRT-LLM C++ DataSystem patch/动态库是否存在** |
| **6/11** | **为 C++ DataSystem KV 验证版 inference 镜像切换独立 tag：默认镜像从 `docker.io/library/pairec-inference:k8s-arm64-runtime` 改为 `docker.io/library/pairec-inference:k8s-arm64-ds-kv-v1`，当前导出 tar 统一保存在 `/home/zcx` 下；`k8s/deployment-inference-image.yaml`、`scripts/build_inference_runtime_image.sh` 与 `k8s/README.md` 已同步，避免覆盖已验证 HTTP/TRT 基线镜像，也避免 kubelet/containerd 因 `IfNotPresent` 使用旧 tag 缓存** |
| **6/11** | **修复 DataSystem K8s Pod 固定 worker 二进制路径错误：远程首次启动 `datasystem` Pod 报 `datasystem_worker not found or not executable: /usr/local/lib64/python3.11/site-packages/yr/datasystem/datasystem_worker`；已将 manifest 改为启动时自动探测 `datasystem_worker`（先 `command -v`，再搜索 `/usr/local`、`/usr`、`/opt`、`/root` 下 DataSystem 路径），找不到时打印 `dscli` 与候选 DataSystem 文件，避免继续猜固定路径。下一步远程重新 apply，若仍找不到 worker 二进制，则根据候选日志决定改用镜像内实际路径或改为 `dscli start` 包装方式** |
| **6/11** | **新增 K8s DataSystem hostNetwork 部署清单：`k8s/deployment-datasystem-hostnetwork.yaml` 固定调度到 worker1，使用 `docker.io/library/zcx-pairec-image:v1.1` 作为 runtime，挂载 worker1 已解压的 `/home/zcx/workspace/pairec4tigerllm/etcd-v3.5.10-linux-arm64`，在容器内以前台进程启动独立 etcd `141.61.91.188:12379/12380` 与 `datasystem_worker --worker_address=141.61.91.188:18481 --shared_memory_size_mb=5120`；为 DataSystem worker 挂载 6Gi memory `emptyDir` 到 `/dev/shm`，避免 K8s 默认 64MiB shm 导致 worker 启动失败；新增 `scripts/k8s_apply_datasystem_hostnetwork.sh`，本地 YAML parse 与 bash 语法检查通过。当前 18481/12379/12380 在 master/worker1 均未占用，下一步远程 apply DataSystem，再重建/apply inference 镜像验证 C++ offload/onboard TRACE** |
| **6/11** | **C++ DataSystem KV trace 验证链路已在 K8s inference runtime 配置侧补齐，待远程重建镜像复验：新增 `TRT_KV_CACHE_HOST_CACHE_SIZE` 传参，构建期 patch `ModelRunnerCpp.from_dir()` 可把 `host_cache_size/onboard_blocks` 转发进 `KvCacheConfig`，用于将 `secondaryBlocks=0` 改为可 offload 的二级池；新增 `PYTHON_DATASYSTEM_ENABLED=0`，允许容器继续向 TensorRT-LLM C++ 暴露非空 `DATASYSTEM_HOST/PORT`，但不启用 Python `yr.datasystem`；`k8s/deployment-inference-image.yaml` 已临时关闭 Python 结果缓存并按当前 `dscli start -w --worker_address ${HOST_IP}:18481` 设置 `DATASYSTEM_HOST=141.61.91.188`、`DATASYSTEM_PORT=18481`、`TRT_KV_CACHE_HOST_CACHE_SIZE=104857600`，下一步远程目标是日志出现 `secondaryBlocks>0`、`[TensorRT-LLM][Datasystem] Init KvCache Manager DataSystem` 和 `[Datasystem][TRACE] op=offload/onboard`** |
| **6/11** | **正式 inference runtime 镜像远程复验通过：重建并导入 `docker.io/library/pairec-inference:k8s-arm64-runtime` 后，`k8s/deployment-inference-image.yaml` 成功启动 TRT 服务；`GET /health` 返回 `status=healthy backend=trt-qwen3 trt_num_samples=1 datasystem=disabled`；直连 `POST /recommend` 返回 `code=200` 和 5 个 recommendations，trace 显示 `runner_calls=1`、`runner_generate_ms≈187ms`、`total_ms≈232ms`；随后 PaiRec 端到端 `GET /ping -> success`，`POST /api/recommend` uid=6312,size=10 返回 `code=200` 和 10 个 `generative_recall` item。正式 inference 镜像已替代启动时 patch/hostPath 代码依赖，当前仅模型/数据仍为 hostPath** |
| **6/11** | **正式 inference runtime 镜像首次远程验证发现镜像缺 `training.rqvae`：`server.py` 导入 `training.decoder.model` 时触发 `training/__init__.py` 导入 `RQVAE`，旧 Dockerfile 只复制 `training/decoder` 导致 `ModuleNotFoundError: No module named 'training.rqvae'`；已修 `docker/Dockerfile.inference.runtime` 为复制完整 `training/` 包，待远程重新构建镜像并复验** |
| **6/10** | **正式 inference runtime 镜像方案已新增：`docker/Dockerfile.inference.runtime` 基于已验证 `zcx-pairec-image:v1.1` runtime，复制 `/app` 业务代码并在构建期 patch TensorRT-LLM `ModelRunnerCpp.from_dir()` 支持 `scheduler_config`；新增 `docker/entrypoint-inference-runtime.sh` 统一设置 gcc-toolset-14 `LD_LIBRARY_PATH`、有效 NVML 和 DataSystem abseil `LD_PRELOAD`；新增 `k8s/deployment-inference-image.yaml` 与 `scripts/build_inference_runtime_image.sh`、`scripts/k8s_apply_inference_image.sh`。本机静态校验 `py_compile`、`bash -n`、YAML parse 通过；待远程 ARM 构建、导入 worker1 并复验 HTTP/TRT 基线** |
| **6/10** | **F13 K8s PaiRec 端到端闭环验证通过：基于 `go build -mod=vendor` 生成 arm64 静态 PaiRec 二进制并打包为 `docker.io/library/pairec-server:k8s-arm64-static`；修复 Alpine 动态链接导致的 `exec /app/pairec-server: no such file or directory`；修复 PaiRec `../data/...` 相对路径，通过 hostPath 将 worker1 `/home/zcx/workspace/pairec4tigerllm/data` 挂到 `/data`，`workingDir=/app`；首次 fallback JSON 触发 1Gi OOM 后将 PaiRec memory limit 提到 8Gi；远程验证 `GET /ping -> success`，`POST /api/recommend` uid=6312,size=10 返回 `code=200` 和 10 个 `generative_recall` item；新增 `k8s/deployment-pairec-hostpath.yaml` 与 `scripts/k8s_apply_pairec_hostpath.sh`** |
| **6/9** | **F13 K8s inference 单服务闭环跑通并固化 hostPath manifest：worker1 已注册 `nvidia.com/gpu=1`；`zcx-pairec-image:v1.1` 作为 TRT/DataSystem runtime，hostPath 挂 `/home/zcx/workspace/pairec4tigerllm`；修正 `LD_LIBRARY_PATH` 使用 gcc-toolset-14、`LD_PRELOAD` 使用有效 NVML + DataSystem abseil，并在启动时临时 patch `ModelRunnerCpp.from_dir()` 转发 `scheduler_config`；远程验证 `/health` 返回 `status=healthy backend=trt-qwen3 trt_num_samples=1`，`/recommend` 返回 `code=200` 和 5 个 item；新增 `k8s/deployment-inference-hostpath.yaml` 与 `scripts/k8s_apply_inference_hostpath.sh`。遗留：Python DataSystem 为 disabled，C++ `CacheTransceiver` disabled，PaiRec Go 镜像/Deployment 待接入** |
| **6/5** | **F13 K8s 最小闭环部署配置已成型：修正 inference `:18000`、PaiRec `:18080`、PaiRec `/ping` 探针、`http://inference:18000` 服务发现、TRT engine/Qwen3/checkpoint/DataSystem/LD_PRELOAD 环境变量、GPU 单副本 `Recreate` 策略；新增 `k8s/README.md` 执行步骤；PaiRec Dockerfile 改为 `-mod=vendor` 离线构建并本机验证 `go build -mod=vendor` 通过** |
| **6/5** | **远端 DataSystem Get 到本地测试完成并补充到 `docs/E2E_DATASYSTEM_FINAL_REPORT_2026-06-04.md`：DS host=`141.61.91.188:18581`，client p50=1050.3ms、p99=1689.2ms；C++ onboard=1346，Get p50=315.572ms、p99=317.038ms；H2D p50=0.405ms，确认当前路径是远端 DS Get 到本机 host 后再本机 H2D，remote H2D 暂未测试** |
| **6/4** | **`docs/E2E_DATASYSTEM_FINAL_REPORT_2026-06-04.md` 补充原始 benchmark stdout 附录，保留 pressure/replay、请求级阶段和 C++ DataSystem block 指标的原始输出，便于复核摘要表** |
| **6/4** | **最终 E2E DataSystem Set/Get 压测完成并生成 `docs/E2E_DATASYSTEM_FINAL_REPORT_2026-06-04.md`：12000 请求成功 11853，client p50=92.8ms、p99=103.2ms；C++ offload=29330，Set p50=0.746ms、p99=1.015ms、p9999=1.689ms；C++ onboard=13421，Get p50=0.623ms、p99=0.818ms、p9999=1.094ms** |
| **6/4** | **新增 `TRT_RESULT_CACHE_ENABLED=0`，可关闭 Python TRT 推荐结果缓存；`scripts/benchmark_e2e_latency.py` 新增 `--uid-file`、pressure/replay 两段流量和按阶段 DataSystem C++ 指标，支持在端到端报告里同时观察 request stages 与 C++ `offload.set_ms` / `onboard.get_ms`** |
| **6/4** | **输出 `docs/E2E_LATENCY_REPORT_2026-06-04.md`，汇总 TRT_NUM_SAMPLES=1 的 PaiRec E2E、缓存分组、当前 E2E 中 DS trace 缺失原因与下一轮充分测 DS Set/Get 的要求** |
| **6/4** | **`scripts/benchmark_e2e_latency.py` 合并 DataSystem C++ KV block 统计：同一份 E2E 报告同时输出请求级阶段耗时和 `offload.set_ms` / `onboard.get_ms` 等 per-block 指标** |
| **6/4** | **TRT_NUM_SAMPLES=1 PaiRec E2E 完成: 60 请求全成功，client p50=3.5ms、p95=90.5ms、p99=92.8ms；报告混入结果缓存命中，冷 miss 对应 p95/p99 尾部约 90ms** |
| **6/4** | **远程 runner 轮数 A/B 完成: 1/2/4/8 轮均 `full_topk=60/60`；HTTP p50 分别为 91.8/180.1/350.2/701.7ms；确认 runner 耗时近似线性缩放** |
| **6/4** | **修复 `scripts/benchmark_trt_runner_samples.py` 使用 `--stop-command 'pkill -f ...'` 时会匹配自身 `--server-cmd` 并被 `Terminated` 的问题；文档改为脚本外手动清理旧服务** |
| **6/3** | **新增 `scripts/benchmark_trt_runner_samples.py`，自动核验 `/health` 的 `trt_num_samples`、运行直连压测、解析 `runner_calls` 并输出 JSON/CSV 对比** |
| **6/2** | **TRT runner 优化准备: 定位每个 miss 请求串行执行 8 次 `runner.generate()`；增加 `TRT_NUM_SAMPLES` 参数和单轮 trace，支持 1/2/4/8 轮质量-时延 A/B** |
| **6/2** | **DataSystem onboard 专项扩样准备: 报告明确区分 host DataSystem API、host 计时同步 D2H/H2D 和总 wall-clock；新增 `--min-onboard-samples` 门槛** |
| **6/2** | **远程 replay 修正验证通过: offload 8237 次，onboard 177 次；Get p50=0.725ms、p99=1.709ms，Get+H2D p50=1.258ms、p99=2.291ms** |
| **6/2** | **远程应用 C++ trace patch 并完成首轮 DataSystem 实测: offload 4181 次，Set p99=1.093ms；onboard/Get 已观测到 1 次，需继续增加读样本** |
| **6/2** | **F12 C++ DataSystem 时延观测补齐: 新增 Create/D2H/Set、Get/H2D 结构化 trace patch；C++ 压测和 PaiRec E2E 汇总新增 p99/p9999/max** |
| **6/1** | **端到端阶段性收口: `dev` 固化为可回退基线；后续从专用分支开展 TRT-LLM C++ DataSystem 与原生 pinned DRAM 的端到端 A/B** |
| **6/1** | **开始 F12 推荐系统时延分析: 补齐 PaiRec 入口、GenerativeRecall、Python TRT 服务和 TRT runner 分阶段 trace；新增端到端压测汇总脚本** |
| **6/1** | **F08 PaiRec 对接完成: 远程复验 Kafka 实时特征用户 `130`、`2184` 均单次触发 TRT 推理并返回完整映射结果** |
| **6/1** | **修复 PaiRec 自定义 recall 启动 panic: 外部注册同步写入配置签名，框架二次加载时正确跳过内置工厂** |
| **6/1** | **修复远程 DNS 不可用: 将完整 Go `vendor/` 纳入仓库，PaiRec 启动固定使用 `-mod=vendor` 离线依赖** |
| **6/1** | **修复 PaiRec 远程启动: 将 `go.sum` 纳入版本控制；启动脚本正确导出 `CONFIG_PATH` 并移除无效 `--port` 参数** |
| **6/1** | **推荐链路对接 — PaiRec 端到端返回成功；修复 Go 客户端 500ms 超时导致的重复推理** |
| **5/30** | C++ KV cache offload/onboard 闭环验证通过; 任务切换到推荐链路工程化 |
| **5/29** | C++ offload/onboard 链路突破 (FMHA crash根因/C++源码绕过/重编.so) |
| **5/28** | KV Cache + offload/onboard 闭环 (TRT/PyTorch双路径/三层缓存全通) |
| **5/27** | 推理打通 (FMHA bf16 kernel SM89非法内存访问根因) |
| **5/23** | ARM 4090D推理部署 + TRT-LLM引擎构建 + 多轮采样去重 |
| **5/22** | 训练完成 epoch 20 (loss 2.49) |
| **5/21** | DDP三卡训练启动 (epoch 11) |

## 当前状态

- **训练**: ✅ epoch 20, loss 2.49
- **TRT-LLM 引擎**: ✅ bfloat16, 1.46 GB, 限制: max_input_len=64, max_new_tokens=32, max_seq_len=96
- **C++ KV offload/onboard**: ✅ 闭环验证通过
- **推理服务**: ✅ /recommend 可用
- **PaiRec 对接**: ✅ Kafka 实时特征、生成式召回、TRT 推理和 item 映射链路已打通
- **时延分析**: ✅ 第一版端到端 trace 和冷请求分解已验证；✅ C++ DataSystem Set/Get 阶段性统计已完成；✅ 关闭结果缓存后的 E2E pressure/replay 最终报告已生成；✅ 远端 DS Get 到本地中等样本已完成；🔄 remote H2D 与 pinned DRAM A/B 待补充
- **K8s 部署**: ✅ ARM worker1 HTTP/TRT 基线闭环已通过：正式 inference runtime 镜像 `/health` + `/recommend` 和 PaiRec `/ping` + `/api/recommend` 均已验证；🔄 仍需将模型/数据 hostPath 替换为 PVC，并单独修 DataSystem Python/C++ KV 路径
- **brpc 改造**: 🔄 F14 已回退 PaiRec proxy 方案，转向 C++ brpc inference service；当前已新增无 HTTP 转发的 `brpc_inference_server` smoke backend，待远程构建/部署验证并接入真实 `trtllm_cpp` backend

## 6/1 探索：推理命中率优化 (5个bug修复)

### 问题链

PaiRec 调用推理服务 → 大部分用户返回 `code:299 "items size not enough"`。

根因是 **推理服务返回的推荐 item 太少**（1个或0个），远低于 PaiRec 期望的 size（10）。

### 修复清单

| # | 提交 | 问题 | 修复 |
|---|------|------|------|
| 1 | `f7beeb4` | 20条历史→117 tokens 超引擎 max_input_len=64 | prompt 自动截断，保留最近9条 |
| 2 | `fb1cb98` | `<s0_X><pad><s1_Y>` 因中间有非语义token被丢弃 | 先过滤有效token再匹配，跳过非语义token |
| 3 | `326e1bb` | 模型输出层序乱 (s2,s1,s0,s3) 连续递增模式找不到 | 改为按层收集+笛卡尔积组合，不要求顺序 |
| 4 | `5533a16` | 20 tokens输出太短+8轮各自为战 | 8轮token合并到一个池子统一组合 |
| 5 | `1544a3f` | max_new_tokens>32 C++层卡死不报错 | 硬限制≤32 (引擎构建时预留) |
| 6 | `b44dbc6` | layer 3永远为0 (1条历史的用户) | 缺失层用{0}填充，由semantic_id_map验证 |

### 引擎硬限制 (重要)

```
max_seq_len = 96
  ├─ max_input_len = 64   (超过报 RuntimeError)
  └─ max_new_tokens = 32  (超过 C++ 层卡死不报错!)
```

必须同时遵守两个限制。

### 核心机制

**组合模式**: 8轮采样 × 32 token = 256 token 池 → 按层收集所有有效语义token → 笛卡尔积组合 → semantic_id_map 验证

**layer 填充**: 当某层缺失时用 {0} 补位 → 组合出候选 → map 验证真假

## 6/1 验证：PaiRec 端到端功能打通

### 验证证据

| 用户 | history_len | PaiRec 响应 |
|------|-------------|-------------|
| `303` | 20 | `code=200`, `size=5`, 5 个 `generative_recall` item |
| `1201` | 1 | `code=200`, `size=5`, 5 个 `generative_recall` item |
| `130` | 1 | `code=200`, `size=5`, 5 个 `generative_recall` item |

### 新发现：Go 客户端超时导致重复推理

单次 TRT miss 约 `667-812ms`，但 Go 客户端默认超时仅 `500ms` 且最多尝试 3 次。同一个 PaiRec 请求会并发触发多次 GPU 推理，再由后续重试命中 HBM 结果缓存。

已在本地修复：`RecallAlgo` 增加 `timeout_ms=3000`、`max_retries=1`，并同步修改默认配置。待远程部署后确认单次 PaiRec 请求只触发一次 `/recommend`。

### 远程启动修复

远程 `go run` 曾因仓库未跟踪 `go.sum` 报依赖校验缺失。已将 `go.sum` 纳入版本控制，并修复 `scripts/start_pairec.sh`：导出 `CONFIG_PATH` 供 `main.go` 首次加载，移除 PaiRec 未定义的 `--port` 参数。

远程环境随后因 DNS 解析失败无法访问 `mirrors.aliyun.com`。已将完整 `vendor/` 纳入版本控制，启动脚本固定使用 `go run -mod=vendor`，不再依赖在线下载。

首次 vendor 提交仍遗漏 128 个文件：根因是 `.gitignore` 中通用 `lib/` 规则误伤 vendor 内 ClickHouse、Apache Thrift 和 PostgreSQL 驱动目录。已增加 `!vendor/**` 例外并补齐文件。验证方式：从 Git 暂存区导出临时副本，在 `GOPROXY=off`、空 `GOMODCACHE` 和空 `GOCACHE` 下执行 `go test -mod=vendor ./services/...`，全部通过且模块缓存文件数为 0。

### 自定义 recall 注册修复

PaiRec 启动时会再次执行 `recall.Load()`。此前 `main.go` 手工注册 `GenerativeRecall` 时只写入实例，没有写入框架的配置签名；二次加载时框架尝试用内置工厂重建自定义类型并 panic：`recall empty, name:generative_recall`。

已在 vendored recall 包增加 `RegisterRecallWithConfig()`，同步写入实例和配置签名，`main.go` 改用该入口。启动级验证已越过注册阶段并输出 `server start`。

## 6/1 F12：推荐系统端到端时延分解

已按当前 Qwen3 TRT 链路补齐 trace：

- PaiRec 入口内部：`user_feature_ms`、`recall_ms`、`filter_ms`、`general_rank_ms`、`feature_ms`、`rank_ms`、`pipeline_wait_ms`、`merge_ms`、`sort_ms`
- GenerativeRecall：`history_ms`、`convert_ms`、`http_ms`、`items_ms`、`http_overhead_ms`
- Python TRT 服务：`prepare_input_ms`、`kv_lookup_ms`、结果缓存查询和异步写入提交耗时
- TRT 后端：`prompt_ms`、8 轮累计 `runner_generate_ms`、`parse_combo_ms`、`output_pad_ms`
- 新增 `scripts/benchmark_e2e_latency.py`：从 PaiRec 入口发请求，用 `request_id` 关联 PaiRec 和 TRT 日志，汇总 p50/p95/p99/p9999/max，并按 `miss` / `hbm_hit` / `ds_hit` 分组
- 修正小样本 percentile 插值：2 个样本的 p50 使用中位数，不再错误取最小值
- 修复 PaiRec trace 采集：`glog` 默认写独立文件，`scripts/start_pairec.sh` 现在默认传入 `--alsologtostderr=true`，保留文件日志并可由 `tee /tmp/pairec.log` 捕获结构化日志

远程冷请求分解已确认：

- fallback 用户 `142`、`211` 均为 TRT `miss`
- 首个 fallback 请求：PaiRec `3392ms`，其中 `history_ms=2573ms`；对应首次加载 `999447` 用户 fallback JSON
- 后续稳态冷请求：PaiRec `687ms`，其中 TRT 服务 `685.3ms`，PaiRec 额外开销约 `2ms`
- 两次 TRT 平均 `750.0ms`，其中 8 轮 `runner_generate` 平均 `686.2ms`，占比约 `91.5%`
- TRT 次要耗时：`output_pad_ms` 平均 `27.3ms`、`prompt_ms` 平均 `10.7ms`
- 结论：fallback JSON 应在启动阶段预加载，避免首请求抖动；稳态冷请求的首要优化目标是 8 轮 TRT runner

本机验证：

```text
python -m py_compile inference/trt_llm/server.py inference/trt_llm/trt_qwen3_backend.py scripts/benchmark_e2e_latency.py
# exit 0

go test -mod=vendor ./services/...
# services / config / feature / recall 均通过

python -c '<trace parser assertions>'
# trace parser OK
```

## 6/2 F12：C++ DataSystem Get/Set 时延与尾延迟指标

已新增 `trtllm-datasystem-latency-trace.patch`，针对外部 TensorRT-LLM
`cpp/tensorrt_llm/batch_manager/kvCacheTransferManager.cpp` 增加结构化日志：

```text
[TensorRT-LLM][Datasystem][TRACE] op=offload ... create_ms=... d2h_ms=... set_ms=... total_ms=...
[TensorRT-LLM][Datasystem][TRACE] op=onboard ... get_ms=... h2d_ms=... total_ms=...
```

统计脚本同步增强：

- `scripts/test_trt_cpp_kv_offload.py`：按 warmup / pressure / replay / overall 汇总 DataSystem C++ 指标，HTTP 和 C++ 指标均输出 `avg/p50/p95/p99/p9999/max`，支持 `--json-output`
- `scripts/benchmark_e2e_latency.py`：PaiRec E2E、各阶段和 TRT 推荐结果缓存分组新增 `p9999`，缓存分组补齐 `p99/p9999/max`
- `p9999` 使用线性插值；样本量不足时仅作方向性观察，正式结论需扩大请求量

本机验证：

```text
git -C /home/vivwimp/TensorRT-LLM apply --check \
  /home/vivwimp/pairec4tigerllm/trtllm-datasystem-latency-trace.patch
# exit 0

python -m py_compile \
  scripts/benchmark_e2e_latency.py scripts/test_trt_cpp_kv_offload.py
# exit 0

python - <<'PY'
# synthetic percentile + DataSystem TRACE parser assertions
PY
# datasystem trace parser OK
```

### 远程 DataSystem 首轮实测

DataSystem runtime 已应用 trace patch 并完成首轮 pressure / replay：

```text
KV pool: primaryBlocks=32 secondaryBlocks=28
offload count=4181
  create_ms p50=0.680 p99=1.025 max=2.171
  d2h_ms    p50=0.413 p99=0.691 max=0.732
  set_ms    p50=0.759 p99=1.093 max=1.381
  total_ms  p50=1.937 p99=2.553 max=4.006
onboard count=1
  get_ms=1.562 h2d_ms=0.345 total_ms=1.961
```

结论：

- DataSystem C++ offload 写路径稳定，单个 3.5 MiB block 的 `Create + D2H + Set`
  总耗时 p50 约 `1.94ms`，p99 约 `2.55ms`
- onboard/Get 已出现 1 次，证明读路径打通；样本不足，暂不能评价 Get 分位数
- 原压测 verdict 依赖 DEBUG 级 `KV cache block reuse is enabled`、`copyBlock entered`
  和 `Set Key` 日志，在 INFO 级日志下会误报失败；脚本已改为优先使用结构化 trace 判定

### 远程 DataSystem 第二轮实测与 replay 修正

扩大 pressure 后采集到：

```text
KV pool: primaryBlocks=32 secondaryBlocks=28
offload count=10440
  create_ms p50=0.765 p99=1.117 p9999=1.783 max=2.311
  d2h_ms    p50=0.493 p99=0.700 p9999=1.475 max=2.087
  set_ms    p50=0.839 p99=1.120 p9999=1.481 max=10.906
  total_ms  p50=2.200 p99=2.796 p9999=4.198 max=12.334
onboard count=0
```

结论：

- `Set` 路径已有 `10440` 个样本，p99 仍约 `1.12ms`；出现一次
  `set_ms=10.906ms` 尾部尖峰，后续 A/B 需保留 max 和原始 JSON
- 原 replay 固定重放最早的 pressure 历史；secondary pool 只有 `28` 个 block，
  增大 pressure 反而会使这些前缀更早被回收，无法稳定触发 `Get`
- `scripts/test_trt_cpp_kv_offload.py` 新增 `--replay-source-count` 和
  `--replay-tail-offset`，默认循环最近一小组 pressure 历史；历史生成器改为
  可逆 32 位混合，避免每 `256` 个 seed 重复，可构造十万级不同请求

### 远程 DataSystem 第三轮实测：Get/onboard 样本补齐

使用近期 pressure 历史循环 replay 后，结构化 trace 判定通过：

```text
PASS: C++ KV offload and DataSystem onboard were both observed.

KV pool: primaryBlocks=32 secondaryBlocks=28
offload count=8237
  set_ms    p50=0.780 p99=1.202 p9999=10.793 max=10.848
  total_ms  p50=2.101 p99=2.760 p9999=12.028 max=12.173
onboard count=177
  get_ms    p50=0.725 p95=1.520 p99=1.709 p9999=1.960 max=1.963
  h2d_ms    p50=0.492 p95=0.607 p99=0.618 p9999=0.624 max=0.624
  total_ms  p50=1.258 p95=2.027 p99=2.291 p9999=2.471 max=2.471
```

其中 replay/onboard wave 单独贡献 `174` 次 onboard，说明新的 replay 策略可以
稳定进入 DataSystem `Get + H2D` 读路径。`Get` 的 p50/p95/p99 已具备阶段性参考
价值；读路径只有 `177` 个样本，p9999 仍接近 max，仅作为观察值。

### Host/device 指标边界与 onboard 专项扩样

已展开 TensorRT-LLM runtime 实现：

```text
create_ms  host steady_clock 包围 DataSystem Create API
set_ms     host steady_clock 包围 DataSystem Set API
get_ms     host steady_clock 包围 DataSystem Get API
d2h_ms     host steady_clock 包围同步 cudaMemcpySanitized/cudaMemcpy DeviceToHost
h2d_ms     host steady_clock 包围同步 cudaMemcpySanitized/cudaMemcpy HostToDevice
total_ms   host steady_clock 包围完整 offload/onboard 流程
```

因此：

- `create_ms/set_ms/get_ms` 是 host 侧 DataSystem API wall-clock
- `d2h_ms/h2d_ms` 是 host 观察到的同步 CPU↔GPU 传输完成耗时，包含 PCIe/DMA
  等待，但不是 CUDA event 计出的纯 device 时间
- 当前没有采集 GPU kernel 或 CUDA-event device-only 指标

`scripts/test_trt_cpp_kv_offload.py` 已增加：

- `--min-onboard-samples`：结构化 onboard trace 少于目标数量时返回失败
- 报告开头逐项打印 metric scope
- 报告结尾提示 p9999 样本门槛：`10000` 是最低尾部观测门槛，稳定结论建议
  `>=100000`

远程下一轮先采集 `>=10000` 个 onboard 样本：

```text
python scripts/test_trt_cpp_kv_offload.py \
  --log /tmp/server_v4.log \
  --warmup-requests 0 \
  --requests 360 --repeat-requests 10000 \
  --replay-source-count 4 --replay-tail-offset 1 \
  --min-onboard-samples 10000 \
  --json-output /tmp/cppkv-datasystem-onboard-10k.json
```

### 远程 DataSystem 第四轮实测：onboard 10k+

onboard 专项 replay 已完成，达到最低 p9999 尾部观测门槛：

```text
replay/onboard wave HTTP:
  count=10000 p50=694ms p95=727ms p99=749ms p9999=832ms max=1552ms

all measured phases:
offload count=166089
  create_ms p50=0.665 p99=0.990 p9999=1.637 max=10.910
  d2h_ms    p50=0.483 p99=0.675 p9999=0.836 max=1.150
  set_ms    p50=0.756 p99=1.084 p9999=1.816 max=11.055
  total_ms  p50=1.937 p99=2.582 p9999=11.877 max=12.513
onboard count=14208
  get_ms    p50=0.740 p99=1.037 p9999=1.570 max=10.801
  h2d_ms    p50=0.407 p99=0.599 p9999=0.804 max=1.021
  total_ms  p50=1.177 p99=1.546 p9999=2.121 max=11.272
```

结论：

- onboard 已有 `14208` 个样本，可观察 p9999；若需要稳定的正式 p9999
  结论，仍建议扩大到 `>=100000`
- onboard 极端 max `11.272ms` 主要来自 `Get max=10.801ms`，不是 H2D；
  `h2d_ms max=1.021ms`
- replay 请求 E2E p50 为 `694ms`，单 block onboard p50 为 `1.177ms`；
  DataSystem 单块读回不是当前推荐请求的主耗时，主耗时仍在 8 轮 TRT runner
- offload 已有 `166089` 个样本，写路径 p9999 具备更好的统计支撑

### TRT runner 耗时根因与减轮数 A/B

当前每个 TRT miss 请求会串行调用 `8` 次 `ModelRunnerCpp.generate()`，每轮最多
生成 `32` 个 token，再合并 token 池做语义四元组组合。端到端基线中：

```text
runner_generate_ms avg=675.2ms
runner calls per request=8
single runner.generate avg≈84.4ms
```

8 轮不是模型推理的硬要求，而是当前无约束解码下为提高候选覆盖率采用的召回
策略。减少轮数预计近似线性降低 runner 耗时，但可能使可映射 item 数下降。

已增加：

- 服务参数 `--trt_num_samples`，环境变量 `TRT_NUM_SAMPLES`，默认 `8`
- TRT trace：`runner_calls`、`runner_avg_ms`、`runner_max_ms`
- `scripts/benchmark_e2e_latency.py` 汇总上述字段
- `scripts/test_trt_cpp_kv_offload.py` 输出每个 phase 的 `full_topk`、item 数
  p50/p95/min/max，便于直连 TRT 做质量-时延 A/B

粗略预估，固定开销按约 `23ms` 计算：

```text
8 rounds: runner≈675ms, request≈698ms  当前基线
4 rounds: runner≈338ms, request≈361ms  待实测
2 rounds: runner≈169ms, request≈192ms  待实测
1 round:  runner≈ 84ms, request≈107ms  待实测
```

远程完整 A/B 已回传，`/health` 与服务 TRACE 均确认轮数生效：

```text
sample health kv_exit calls ok/topk http_p50 http_p99 runner_avg runner_max
1      True   2       1     60/60  91.8     150.7    85.8       177.7
2      True   2       2     60/60  180.1    247.7    85.4       178.6
4      True   2       4     60/60  350.2    423.0    84.8       178.4
8      True   2       8     60/60  701.7    795.6    83.9       178.8
```

结论：

- 对当前直连 TRT、`topk=5`、60 请求样本，1/2/4/8 轮均没有出现 top-k 返回不足
- 1 轮相对 8 轮 HTTP p50 从 `701.7ms` 降到 `91.8ms`，下降约 `87%`
- 单次 `runner.generate()` 平均稳定在 `83.9-85.8ms`，说明主耗时几乎完全随轮数线性缩放
- `kv_exit=2` 来自底层 C++ KV/DataSystem verifier 未通过；该组结果只作为 runner
  轮数 A/B，不作为 DataSystem offload/onboard 结论

远程按 `TRT_NUM_SAMPLES=1/2/4/8` 分别重启服务，再执行相同直连请求序列：

```text
python scripts/test_trt_cpp_kv_offload.py \
  --log /tmp/server_samplesN.log \
  --warmup-requests 0 --requests 60 --repeat-requests 0 \
  --topk 5 --json-output /tmp/runner-samplesN.json
```

脚本默认使用带时间戳的唯一 `user_id` 前缀，避免重复运行命中 Python 结果缓存。
优先比较 `full_topk` 比例和 HTTP p50/p99。默认值暂不下调，需根据远程 A/B
结果选择。

已补充自动编排脚本，推荐后续直接使用：

```text
python scripts/benchmark_trt_runner_samples.py \
  --samples 1,2,4,8 \
  --server-cmd 'python -m inference.trt_llm.server ...' \
  --requests 60 --repeat-requests 0 --topk 5
```

如果需要清理旧服务，先在脚本外单独执行 `pkill -f "inference.trt_llm.server" || true`。
不要把这条 `pkill -f` 放进 `--stop-command`：benchmark 进程自己的
`--server-cmd` 参数也包含 `inference.trt_llm.server`，会被误杀并显示
`Terminated`。

脚本会逐轮注入 `TRT_NUM_SAMPLES`，等待 `/health` 返回匹配的
`trt_num_samples`，运行 `scripts/test_trt_cpp_kv_offload.py`，解析服务 TRACE
中的 `runner_calls/runner_avg_ms/runner_max_ms`，最后输出
`/tmp/trt_runner_samples_ab.json` 和 `/tmp/trt_runner_samples_ab.csv`。
默认情况下，runner A/B 脚本不会因为底层 C++ KV/DataSystem verifier 返回非零而
失败；`kv_exit` 会保留在汇总中作为诊断字段。若需要严格验证 C++ KV/DataSystem，
显式加 `--fail-on-kv-verdict`。

### TRT_NUM_SAMPLES=1 PaiRec E2E 实测

从 PaiRec `:18080` 入口使用 `TRT_NUM_SAMPLES=1`、`size=5`、60 个请求完成
端到端压测：

```text
Client: avg=9.4ms p50=3.5ms p95=90.5ms p99=92.8ms max=94.6ms
PaiRec total: avg=8.4ms p50=3.0ms p95=89.0ms p99=91.6ms max=94.0ms
GenerativeRecall http_ms: avg=8.1ms p50=2.0ms p95=89.0ms p99=91.2ms
TRT trace tr_total_ms: avg=7.5ms p50=1.6ms p95=88.2ms p99=90.4ms
TRT trace tr_runner_ms: avg=5.4ms p50=0.0ms p95=80.0ms p99=81.6ms
ok=60 fail=0
```

该报告混合了推荐结果缓存命中和少量冷 miss。p50 主要代表缓存命中路径；p95/p99
代表仍需进入 TRT runner 的冷 miss，符合 1 轮 runner 约 `84ms` 的预期。由于 TRT
服务日志未按 `request_id` 关联上，`scripts/benchmark_e2e_latency.py` 已补充
item 完整率和基于 GenerativeRecall `tr_result_cache_source` 的缓存分组 fallback。

后续重跑同一个 E2E 脚本时，会追加：

```text
== DataSystem C++ KV block stages ==
  scope: per KV block, from TRT-LLM C++ [Datasystem][TRACE], host wall-clock
  offload.create_ms / offload.d2h_ms / offload.set_ms / offload.total_ms
  onboard.get_ms / onboard.h2d_ms / onboard.total_ms
```

该部分是 per-block C++ KV 传输指标，不是 per-request 请求指标；需要与上面的
请求级 E2E 阶段并排解读。

后续可实验将多轮 prompt 批量提交给 `ModelRunnerCpp.generate()`。当前本地
TensorRT-LLM API 对整批默认共用一个 sampling config，而现有逻辑依赖每轮不同
seed；批量化需要先验证候选多样性，不能直接替换串行循环。

## 下一步

`dev` 已作为端到端阶段性基线保留。后续在专用分支开展 C++ DataSystem A/B：

1. 分开采集 `TRT_NUM_SAMPLES=1` 的纯 cold/miss E2E 与缓存命中 E2E，避免一个报告里 p50/p99 语义混杂
2. 单独恢复 DataSystem C++ KV verifier：当前 runner A/B 中 `kv_exit=2`，不能作为 DataSystem 结论
3. 如需稳定的正式 onboard p9999 结论，再扩大到 `>=100000` 个 onboard 样本
4. 推理服务增加实验开关，关闭 TRT Python 结果缓存和无效的 Python KV Cache 查询，确保请求进入 C++ runner
5. TensorRT-LLM runtime 恢复 KV 配置参数化，确保两组使用相同 primary / secondary block 数
6. 基于同一份 engine 构建原生 pinned DRAM baseline，与当前 DataSystem runtime 对照
7. 后续独立优化 fallback JSON 启动预加载
