# 交接日志

> 供 Agent 跨 session 恢复上下文。只记录关键决策和当前任务。

## 最近一次交接 (2026-05-27)

### 当前任务
ARM 4090D 推理已打通，下一步验证 DataSystem KV Cache offload/onboard。
进度: Executor ✅、DataSystem ✅、/health ✅、推理 ✅ (code=200, hit=5/5)。
阻塞: 无。

### 本 session 解决的问题

1. **abseil 冲突 → RegisterFlag 崩溃** → LD_PRELOAD=libabseil_dll.so
2. **ConsumerLoop 线程 SIGSEGV** → block_ds_consumer.so stub no-op
3. **CudaRH2DDriver::SwitchToAndGetGpuId substr(4) 空串** → stub_gpu.so 返回 GPU 0 (新增!)
4. **model_runner_cpp.py: sampling_config_list 未定义** → else 分支补定义
5. **model_runner_cpp.py: from_dir 缺 scheduler_config 参数** → 函数签名+传参
6. **GCC 14 编译环境统一** → source gcc-toolset-14/enable, cmake + make 100% 通过
7. **CUBLAS_STATUS_EXECUTION_FAILED 根因定位** → 非 cuBLAS 问题，是 `FusedMultiHeadAttentionXMMAKernelV2` bfloat16 路径在 SM 89 上非法内存访问；CUDA_LAUNCH_BLOCKING=1 后错误精确定位到 fused_multihead_attention_v2.cpp:379
8. **绕过方案** → trtllm-build 用 `context_fmha disable` + `use_paged_context_fmha disable`，引擎 v4 推理成功 (~595ms, hit=5/5)
9. **cudaCoreGemm.cu 修复** → `cudaCoreGemmTemplateCaller` kernel launch 后加 `cudaGetLastError()` 错误检查，失败回退 cuBLAS（已修改源码，未推送）

### 关键新增文件

| 文件 | 用途 |
|------|------|
| `scripts/stub_gpu.c` | 绕过 DataSystem SDK 0.7.7 ARM GPU ID 解析 bug (SwitchToAndGetGpuId 返回 0) |
| `scripts/rebuild_and_verify_offload.sh` | 一键重建引擎 + offload/onboard 验证脚本 |

### 运行时环境变量

```bash
source /opt/openEuler/gcc-toolset-14/enable

export LD_PRELOAD="\
/workspace/pairec4tigerllm/scripts/block_ds_consumer.so:\
/workspace/pairec4tigerllm/scripts/stub_gpu.so:\
/usr/local/lib/python3.11/site-packages/yr/datasystem/lib/libabseil_dll.so.2407.0.0"
```

### 下一步

1. 用系统 GCC 12 单独重编 `libnvinfer_plugin_tensorrt_llm.so`
2. 推理通过后 → 批量请求 eviction → 验证 offload/onboard 日志
3. Go pairec 联调 (F08)
