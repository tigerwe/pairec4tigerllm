# 工作进度

> 最后更新: 2026-05-30 | 当前阻塞: C++ KV offload/onboard 待压测确认 | 下一步: 运行 `scripts/test_trt_cpp_kv_offload.py` 验证 copyBlock/OffLoad/Set/Get/OnBoard 日志

## 时间线

| 日期 | 进度 |
|------|------|
| **5/30** | **C++ KV offload/onboard 启动前置条件已满足: Scheduler 已切到 MAX_UTILIZATION，reuse disabled warning 消失，primaryBlocks=64/secondaryBlocks=28，DataSystem primary/TMP 均连接 127.0.0.1；新增 `scripts/test_trt_cpp_kv_offload.py`，用于已启动 TRT 服务的复用波、压力波、回放波压测，并解析 C++ KV transfer 日志。** |
| **5/29** | **C++ offload/onboard 链路突破: ① FMHA crash 根因定位 — 非 dtype 问题，是 resize_token_embeddings 扩展词表触发了 XMMA 路径（标准 Qwen3 FP16 FMHA SM 89 正常）；② C++ 源码绕过 — 注释 trtGptModelInflightBatching.cpp:149-155 去掉 paged FMHA 对 enableBlockReuse 的强约束；③ 重编 .so 替换 — secondaryBlocks=28 分配成功，DataSystem C++ Init OK；④ 剩余阻塞: Scheduler 需从 GUARANTEED_NO_EVICT 切到 MAX_UTILIZATION (Python 侧 trt_qwen3_backend.py)** |
| **5/28** | **KV Cache + offload/onboard 闭环: 修复 TRT/PyTorch 双路径 miss loop; TRT 路径加结果缓存(OrderedDict LRU → DataSystem onboard); eviction 触发验证; ds_hit(~3ms) / hbm_hit(~2ms) / miss(~220ms) 三层全通** |
| **5/27** | **推理打通: 根因定位为 FMHA bfloat16 kernel SM 89 非法内存访问(非 cuBLAS 问题); context_fmha disable 绕过; cudaCoreGemm.cu 补 error check; verify_kv.sh 验证脚本; 推理 code=200 hit=5/5** |
| 5/26 | 分析引擎不兼容根因; 创建一键验证脚本 rebuild_and_verify_offload.sh; 推送到 gitcode |
| 5/25 | DataSystem KV Cache 集成：Python 层 + C++ 层连接验证通过；offload/onboard 阻断已绕过 |
| 5/23 | ARM 4090D 推理部署 + TRT-LLM 引擎构建 + 多轮采样去重 |
| 5/22 | 训练完成 epoch 20 (loss 2.49); 仓库规范化 |
| 5/21 | DDP 三卡 3bug 修复; L40S×3 并行训练启动 (epoch 11); Docker 镜像适配; 约束解码实现 |
| 5/20 | 发现 loss 横在 10.27 → modules_to_save 修复 (loss→3.06); lm_head 全序列优化; epoch 10 训练完成(loss 2.65) |
| 5/19 | v1: 设计 Prompt Template 架构; 重写模型类; 修 dtype→torch_dtype 兼容 |
| 5/18 | 启动: 讨论使用 Qwen3-0.6B 替换 GPT2 Decoder |

## 当前状态

- **训练**: ✅ epoch 20, loss 2.49, checkpoint `decoder_epoch_20.pt`
- **PyTorch 推理**: ✅ Flask 服务, hit/miss=5/0
- **TRT-LLM 引擎**: ✅ bfloat16, 1.46 GB
- **约束解码**: ✅ prefix tree (PyTorch) / 多轮采样去重 (TRT)
- **Executor 构造**: ✅ 通过 (stub_gpu.so + block_ds_consumer.so + abseil 三链)
- **DataSystem 连接**: ✅ KvCacheManagerDataSystem Init success
- **/health 接口**: ✅ 200, datasystem=connected
- **推理**: ✅ code=200, hit=5/5, ~220ms (TRT)
- **KV Cache 三层**: ✅ hbm_hit (~2ms) / ds_hit (~3ms) / miss (~220ms)
- **offload/onboard**: ✅ LRU 淘汰 → DataSystem 写入 → 重启后 onboard 回读
- **引擎**: v4 (bfloat16 GEMM + FP16 attention via context_fmha=disable)
- **根因**: `FusedMultiHeadAttentionXMMAKernelV2` bfloat16 路径在 SM 89 上 `cuLaunchKernel` 非法内存访问（已绕过）

---

## ARM 4090D 编译与运行时修复 (2026-05-27)

### 编译环境统一

- 使用 `source /opt/openEuler/gcc-toolset-14/enable` 设置 CC/CXX/LIBRARY_PATH
- cmake 前设置 GCC 14，解决 `__cxa_call_terminate`、`std::ios_base_library_init` 等符号缺失
- 100% 编译通过，无错误

### 三个 LD_PRELOAD stub 链

```bash
export LD_PRELOAD="\
scripts/block_ds_consumer.so:\    # ① ConsumerLoop no-op → 绕过共享内存 SIGSEGV
scripts/stub_gpu.so:\              # ② SwitchToAndGetGpuId 返回 0 → 绕过 GPU ID 空串 substr(4)
/usr/.../libabseil_dll.so.2407.0.0"  # ③ 解决 abseil RegisterFlag 冲突
```

### stub_gpu.c

DataSystem SDK 0.7.7 在 ARM 上 `CudaRH2DDriver::SwitchToAndGetGpuId` 获取 GPU 标识为空串，`substr(4, ...)` 抛 `std::out_of_range`。

GDB 定位（5/27 15:04）：
```
#4  OsXprtPipln::CudaRH2DDriver::SwitchToAndGetGpuId(std::string const&)
    → substr(4, ...) on empty GPU identifier → std::out_of_range
#6  datasystem::object_cache::ObjectClientImpl::Init(bool&, bool)
#7  datasystem::KVClient::Init()
#8  KVCacheManager::KVCacheManager(...)
```

修复: stub 拦截 `_ZN11OsXprtPipln14CudaRH2DDriver19SwitchToAndGetGpuIdE...` 直接返回 0。

### DataSystem 初始化状态

- `KvCacheManagerDataSystem` (host=127.0.0.1:31501): ✅ Init OK
- `KvCacheManagerDataSystemTmp` (host=127.0.0.2:31501): ❌ Init 失败 (`Init KvCache Tmp failed`)，不影响主客户端

### 当前阻塞: CUBLAS_STATUS_EXECUTION_FAILED

**现象**: 推理时 `cublasLtMatmul` 返回 `CUBLAS_STATUS_EXECUTION_FAILED`（`libnvinfer_plugin_tensorrt_llm.so` 中 `GemmPlugin::enqueue`）

**根因**: GCC 14 编译的 `libnvinfer_plugin_tensorrt_llm.so` 生成的 bfloat16 GEMM kernel 与 cuBLAS 运行时（CUDA 12.8 + 4090D SM 89）不兼容。

**修复方向**: 用系统 GCC 12 单独重编译 plugin target（不改其他 .so）

---

## 关键决策

- Prompt Template 而非 inputs_embeds (参照京东方案)
- LoRA + modules_to_save (lm_head + embed_tokens 必须可训)
- DDP 多卡: torchrun + DistributedSampler
- Python KVCacheManager 三层: HBM LRU → DataSystem (TTL 600s) → Prefill
- C++ DataSystem 连接: `KvCacheManagerDataSystem` 单例，KVCacheManager 构造时自动连接
- abseil 冲突: `LD_PRELOAD` 三链 (stub_gpu + stub_consumer + abseil)
- ARM GPU 兼容: stub 绕过 DataSystem SDK 0.7.7 的 GPU ID 解析 bug
- **TRT 路径结果缓存**: TRT-LLM 不暴露 past_key_values → 改用 OrderedDict LRU 缓存推理结果 JSON，二层: HBM → DataSystem onboard (与 PyTorch KV 路径并行)
- **双 DataSystem namespace**: `kv:*` (past_key_values, numpy) 和 `result:*` (JSON) 隔离，避免反序列化冲突

## TRT-LLM ↔ DataSystem 集成架构分析 (2026-05-29)

### Session 总结 (5/29 下午)

#### FMHA crash 根因定位

| 实验 | 配置 | 结果 |
|------|------|------|
| v4 扩展词表 bf16 + fmha on | `--gemm_plugin bfloat16 --context_fmha enable` | ❌ crash: `FusedMultiHeadAttentionXMMAKernelV2::cuLaunchKernel` |
| v5 扩展词表 fp16 + fmha on | `--gemm_plugin float16 --context_fmha enable` | ❌ crash: 同一 kernel, `CUDA_ERROR_INVALID_HANDLE` |
| v_min 扩展词表 fp16 最简 | 不加任何 plugin | ❌ crash（TRT 默认选了 FMHA） |
| vanilla fp16 标准 Qwen3 | `--gemm_plugin float16 --context_fmha enable` | ✅ 不 crash（加载 OOM，非 FMHA 错误） |

**结论：FMHA crash 不是 dtype（bf16/fp16）问题，是 `resize_token_embeddings` 扩展词表（151936→152693）改变了 TensorRT 的图融合策略，导致走了 GPTAttentionPlugin + FMHA XMMA 路径，而标准 Qwen3 的 FP16 FMHA 在 SM 89 上正常工作。**

GCC 12 重编 `.so` 是早期（5/26）基于错误假设（cuBLAS 兼容性）提出的方向，与最终定位的 FMHA kernel 问题无关，已废弃。

#### C++ 源码绕过

改 `trtGptModelInflightBatching.cpp:149-155`：注释掉 `paged_context_fmha` 对 `enableBlockReuse` 的强制检查。让 `context_fmha disable` 的引擎也能激活 paged reuse → secondary pool → offload/onboard。

- **编译**：`make -j$(nproc) tensorrt_llm`（不需要 cmake）
- **替换**：`cp cpp/build/tensorrt_llm/libtensorrt_llm.so` → Python site-packages 路径
- **验证**：`KV cache reuse disabled` warning 消失 ✅；`secondaryBlocks=28` 分配成功 ✅

#### 激活进度

| 步骤 | 状态 |
|------|------|
| paged KV cache（engine: `--paged_kv_cache enable`） | ✅ v4 引擎已开 |
| paged reuse 解锁（C++ 源码绕过） | ✅ secondaryBlocks=28 |
| paged reuse 解锁（DataSystem C++ 连接） | ✅ `KvCacheManagerDataSystem` Init success |
| Scheduler policy: `GUARANTEED_NO_EVICT` → `MAX_UTILIZATION` | ⏳ `trt_qwen3_backend.py` 构造了 `SchedulerConfig` 但未传进 `ModelRunnerCpp.from_dir()` |
| 压测触发 eviction → offload/onboard | ⏳ 待 scheduler 就绪 |
| ConsumerLoop SIGSEGV 修复 | ⏳ DataSystem SDK 0.7.7 ARM 兼容性 |

#### 新增文件

| 文件 | 用途 |
|------|------|
| `diag_offload.sh` | C++ offload/onboard 全链路诊断（启动→压测→日志分析） |
| `test_vanilla_qwen3.sh` | 标准 Qwen3 推理验证（convert→build→推理） |

#### 其他结论

- bf16 → fp16 引擎构建：**不需要重训模型**，只需 `convert_checkpoint --dtype float16` + `trtllm-build --gemm_plugin float16`
- gitcode push 失败原因：token 嵌 URL 不生效，需用 `git -c credential.helper='!f() { echo "username=oauth2"; echo "password=<token>"; }; f' push`

### 两条链路

| 层级 | 实现 | 状态 |
|------|------|------|
| **Python 层** | `inference/kv_cache/manager.py` → `KVCacheManager` (OrderedDict LRU) → `DsClient.kv().get/set` | ✅ 已走通 |
| **C++ 层** | TRT-LLM 内置 `KVCacheManager` (C++) → `KVCacheTransferManager::offload/onboard` → `KVCacheManagerDataSystem` (C++ 单例) | ❌ 两重阻塞 |

#### Python 层（已走通）
- `result:*` namespace: 推理结果 JSON 缓存（TRT 路径用）
- `kv:*` namespace: past_key_values 二进制缓存（PyTorch 路径用）
- 共享同一个 `DsClient` 实例，key 前缀隔离
- 三层延迟: hbm_hit ~2ms / ds_hit ~3ms / miss ~220ms

#### C++ 层（未激活）
- `libtensorrt_llm.so` 已链接 DataSystem SDK（证据：需要 `LD_PRELOAD=block_ds_consumer.so` + stub_gpu.so + libabseil_dll.so）
- `KVCacheManagerDataSystem` Init 成功（连接 127.0.0.1:31501 OK）
- 但实际 offload/onboard 数据流未触发

### C++ 层两重阻塞

```
C++ offload/onboard 链路
    │
    ├─ 需要 paged KV cache mode
    │       │
    │       └─ 需要 context_fmha enable
    │               │
    │               └─ 阻塞①: FMHA bfloat16 kernel SM 89 非法内存访问
    │                   (FusedMultiHeadAttentionXMMAKernelV2::cuLaunchKernel)
    │                   当前绕过: context_fmha disable + use_paged_context_fmha disable
    │
    └─ 需要 ConsumerLoop 异步线程正常运行
            │
            └─ 阻塞②: ConsumerLoop 共享内存 SIGSEGV
                当前绕过: block_ds_consumer.so stub no-op
```

### C++ 层激活路线

| 步骤 | 内容 | 状态 |
|------|------|------|
| ① 解锁 paged reuse | 注释 `trtGptModelInflightBatching.cpp:149-155`，绕过 FMHA 强依赖 | ✅ |
| ② 修 scheduler | `trt_qwen3_backend.py` 传 `SchedulerConfig` 进 `ModelRunnerCpp.from_dir()`，切 `MAX_UTILIZATION` | ⏳ 下一步 |
| ③ 压测触发 eviction | 多用户长历史并发，填满 primary pool → 触发 secondary pool → offload | ⏳ 待② |
| ④ 验证 offload 日志 | `grep -i "offload\|offLoadCopy\|onboard\|onBoardCopy"` 预期有输出 | ⏳ 待②③ |
| ⑤ 修复 ConsumerLoop SIGSEGV | 恢复 DataSystem 异步通信线程（当前 stub no-op） | ⏳ DataSystem SDK 0.7.7 ARM 兼容性 |

---

## 下一步

1. ~~ARM 4090D 推理通过~~ ✅ (context_fmha disable 绕过)
2. ~~KV Cache 验证 + offload/onboard 闭环~~ ✅ (Python 层: hbm_hit/ds_hit/miss 三层全通)
3. ~~FMHA crash 根因定位~~ ✅ (扩展词表触发 XMMA，标准 Qwen3 正常)
4. ~~C++ paged reuse 解锁~~ ✅ (源码绕过)
5. **压测 C++ KV offload/onboard → 验证 copyBlock/OffLoad/Set/Get/OnBoard 日志** ⏳ 下一步
6. Go pairec 联调 (F08)
7. 修复 ConsumerLoop SIGSEGV → 恢复 C++ DataSystem 异步通信

---

## TensorRT-LLM 源码改动记录

| 文件 | 改动 | 状态 |
|------|------|------|
| `cpp/tensorrt_llm/batch_manager/trtGptModelInflightBatching.cpp:149-155` | 注释掉 `paged_context_fmha` 对 `enableBlockReuse` 的强制检查（绕过 SM 89 FMHA crash） | ✅ 已修改，未推送 |
| `cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/cudaCoreGemm.cu` | `cudaCoreGemmTemplateCaller` kernel launch 后加 `cudaGetLastError()` 错误检查 | ✅ 已修改，未推送 |
| `cpp/tensorrt_llm/plugins/gemmPlugin/gemmPlugin.cpp` | 跳过 cudaCoreGemm 路径（`if (false && ...)`，仅排查用） | 排查用，无需推送 |
