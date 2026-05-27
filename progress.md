# 工作进度

> 最后更新: 2026-05-26 | 下一个: ARM 容器内重建引擎 + offload/onboard 验证

## 时间线

| 日期 | 进度 |
|------|------|
| 5/26 | 分析引擎不兼容根因; 创建一键验证脚本 rebuild_and_verify_offload.sh; 推送到 gitcode |
|------|------|
| 5/18 | 启动: 讨论使用 Qwen3-0.6B 替换 GPT2 Decoder |
| 5/19 | v1: 设计 Prompt Template 架构; 重写模型类; 修 dtype→torch_dtype 兼容 |
| 5/20 | 发现 loss 横在 10.27 → modules_to_save 修复 (loss→3.06); lm_head 全序列优化; 梯度检查点+梯度累积; epoch 10 训练完成(loss 2.65) |
| 5/21 | DDP 三卡 3bug 修复; L40S×3 并行训练启动 (epoch 11); Docker 镜像适配; 约束解码实现 |
| 5/22 | 训练完成 epoch 20 (loss 2.49); 仓库规范化 |
| **5/23** | **ARM 4090D 推理部署 + TRT-LLM 引擎构建 + 多轮采样去重** |
| **5/25** | **DataSystem KV Cache 集成：Python 层 + C++ 层连接验证通过；offload/onboard 阻断已绕过** |

## 当前状态

- **训练**: ✅ epoch 20, loss 2.49, checkpoint `decoder_epoch_20.pt`
- **PyTorch 推理**: ✅ Flask 服务, hit/miss=5/0
- **ARM 4090D 推理**: ✅ PyTorch + TRT-LLM 双后端
- **TRT-LLM 引擎**: ✅ bfloat16, 1.46 GB
- **约束解码**: ✅ PyTorch prefix_allowed_tokens_fn / TRT 多轮采样+后处理
- **DataSystem KV Cache**:
  - Python 层: ✅ HBM LRU → DataSystem → Prefill 三层缓存
  - C++ 连接: ✅ `KvCacheManagerDataSystem` Init success (host=127.0.0.1:31501)
  - C++ offload/onboard: ⏳ segfault 已绕过，待重建引擎后验证

## DataSystem 集成核心问题与解决方案

### 问题链

```
问题1: abseil 版本冲突
  libtensorrt_llm.so 链接的 abseil vs DataSystem grpc 自带的 abseil → RegisterFlag 崩溃
  ↓ 解决: LD_PRELOAD=libabseil_dll.so

问题2: LD_PRELOAD 副作用 — consumer 线程崩溃  
  libabseil_dll.so 加载 → 连带加载 libdatasystem.so → 静态构造创建 PipelineRH2DQueueConsumer 线程
  → 线程访问未映射的共享内存 → ShmCircularQueue::UpdateQueueMeta(NULL) → SIGSEGV
  ↓ 解决: stub ConsumerLoop 函数 (见下)

问题3: TRT-LLM 重编译后旧引擎不兼容
  Executor 构造器 substr 越界 → 需重建引擎 (trtllm-build)
```

### 最终 LD_PRELOAD 方案

```bash
# 三个 .so 按序加载：
export LD_PRELOAD="\
./scripts/block_ds_consumer.so:\       # ① stub ConsumerLoop，线程立即返回
/usr/.../libabseil_dll.so.2407.0.0"   # ② 解决 abseil RegisterFlag 冲突
```

**stub 实现** (`scripts/stub_consumer.c`): 用 mangled 符号名定义空的 `PipelineRH2DQueueConsumer::ConsumerLoop()`，编译为 `.so`。consumer 线程启动后立即返回，不访问共享内存。其他 DataSystem 线程（ZMQ 连接管理、ZmqEpoll 等）不受影响。

### 为什么 pthread_create 拦截失败

`libdatasystem.so` 用 C++ `std::thread` 创建线程，Linux 上 `std::thread` 走 `clone` 系统调用，不走 `pthread_create`。GDB backtrace 确认：
```
#8  std::execute_native_thread_routine  ← std::thread 启动点
#1  OsXprtPipln::PipelineRH2DQueueConsumer::ConsumerLoop()
#0  ShmCircularQueue::UpdateQueueMeta() → NULL 指针
```

### 待解决问题：引擎不兼容

重编译后的 TRT-LLM 二进制 (`ModelRunnerCpp`) 无法加载旧版引擎 (`trt_engines/qwen3_rec/rank0.engine`)，错误：
```
IndexError: basic_string::substr: __pos (which is 4) > this->size() (which is 0)
```
需要用当前 TRT-LLM 二进制重新 `trtllm-build` 构建引擎。

## 关键决策

- Prompt Template 而非 inputs_embeds (参照京东方案)
- LoRA + modules_to_save (lm_head + embed_tokens 必须可训)
- DDP 多卡: torchrun + DistributedSampler
- Python KVCacheManager 三层: HBM LRU → DataSystem (TTL 600s) → Prefill
- C++ DataSystem 连接: `KvCacheManagerDataSystem` 单例，KVCacheManager 构造时自动连接
- abseil 冲突: `LD_PRELOAD` + `stub_consumer.so` 双加载

## 下一步

1. ~~重建 TRT-LLM 引擎~~ ← 已完成
2. ~~DataSystem Init substr(4) 崩溃~~ ← 根因定位: CudaRH2DDriver::SwitchToAndGetGpuId ARM GPU 空标识, stub_gpu.so 绕过
3. **ARM 4090D 容器内验证 offload/onboard** — 编译 stub_gpu.so + LD_PRELOAD 三链 → Executor OK → 批量请求触发 eviction
4. Go pairec 联调 (F08)
5. TRT 引擎延迟优化 (profiling, KV Cache 池化)
