# 工作进度

> 最后更新: 2026-05-27 | 当前阻塞: cuBLAS GEMM plugin GCC 14 编译不兼容

## 时间线

| 日期 | 进度 |
|------|------|
| **5/27** | **定位 DataSystem Init substr(4) 根因: CudaRH2DDriver::SwitchToAndGetGpuId ARM 空GPU标识; stub_gpu.so+block_ds_consumer.so+abseil 三链 LD_PRELOAD → Executor OK + DataSystem connected; 推理遇 CUBLAS_STATUS_EXECUTION_FAILED (GCC 14 plugin 问题)** |
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
- **TRT-LLM 引擎**: ✅ bfloat16, 1.46 GB (旧 engine `qwen3_rec` 可用; 新 engine `qwen3_rec_v3` 已构建但 cuBLAS 不兼容)
- **约束解码**: ✅ prefix tree (PyTorch) / 多轮采样去重 (TRT)
- **Executor 构造**: ✅ 通过 (stub_gpu.so + block_ds_consumer.so + abseil 三链)
- **DataSystem 连接**: ✅ KvCacheManagerDataSystem Init success (host=127.0.0.1:31501)
- **/health 接口**: ✅ 200, datasystem=connected
- **推理**: ❌ CUBLAS_STATUS_EXECUTION_FAILED (GCC 14 编译的 libnvinfer_plugin_tensorrt_llm.so)

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

## 下一步

1. **GCC 12 重编 libnvinfer_plugin_tensorrt_llm.so** — 解决 CUBLAS 不兼容 (当前阻塞)
2. ARM 4090D 推理通过 → 批量请求触发 eviction → 验证 DataSystem offload/onboard 日志
3. Go pairec 联调 (F08)
4. TRT 引擎延迟优化 (profiling, KV Cache 池化)
