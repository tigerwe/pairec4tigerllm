# 工作进度

> 最后更新: 2026-05-25 | 下一个: Go pairec 联调

## 时间线

| 日期 | 进度 |
|------|------|
| 5/18 | 启动: 讨论使用 Qwen3-0.6B 替换 GPT2 Decoder |
| 5/19 | v1: 设计 Prompt Template 架构; 重写模型类; 修 dtype→torch_dtype 兼容 |
| 5/20 | 发现 loss 横在 10.27 → modules_to_save 修复 (loss→3.06); lm_head 全序列优化; 梯度检查点+梯度累积; epoch 10 训练完成(loss 2.65) |
| 5/21 | DDP 三卡 3bug 修复; L40S×3 并行训练启动 (epoch 11); Docker 镜像适配; 约束解码实现 |
| 5/22 | 训练完成 epoch 20 (loss 2.49); 仓库规范化 |
| **5/23** | **ARM 4090D 推理部署 + TRT-LLM 引擎构建 + 多轮采样去重** |
| **5/25** | **DataSystem KV Cache 集成：Python 层 + C++ 层验证通过** |

## 当前状态

- **训练**: ✅ epoch 20, loss 2.49, checkpoint `decoder_epoch_20.pt`
- **PyTorch 推理 (x86 L40S)**: ✅ Flask 服务, hit/miss=5/0
- **ARM 4090D 推理**: ✅ PyTorch + TRT-LLM 双后端, hit/miss=5/0
- **TRT-LLM 引擎**: ✅ bfloat16, 1.46 GB, 构建 12s
- **约束解码**: ✅ PyTorch 路径 prefix_allowed_tokens_fn, TRT 路径多轮采样+后处理过滤
- **DataSystem KV Cache**: ✅ Python 层 + C++ 层全部集成完成
  - Python: HBM LRU → DataSystem → Prefill 三层缓存
  - C++: `KvCacheManagerDataSystem` 单例自动连接，offload/onboard 截获到 DataSystem

## 关键决策

- TRT-LLM 1.0.0 `ModelRunnerCpp.generate()` bug (max_new_tokens 被忽略) → 用多轮采样 (8轮) + 后处理 `_parse_output` 替代 `prefix_allowed_tokens_fn`
- 修复 PEFT export 脚本 bug: `use_lora=False` → `True`, 确保 LoRA 权重 merge 进导出模型
- 修复 `resize_token_embeddings` ARM LAPACK 兼容: 加 `mean_resizing=False`
- 修复 TRT-LLM 源码两处变量未初始化 bug (sampling_config_list, use_sampling_config_for_each_request)
- C++ 层 DataSystem 集成需 `LD_PRELOAD=libabseil_dll.so.2407.0.0` 解决 abseil flag registry 初始化顺序问题（grpc/protobuf 静态初始化在 abseil 就绪前触发）

## 阻塞点

- 无

## 下一步

1. Go pairec 联调 (F08)
2. TRT 引擎延迟优化 (detailed profiling, KV Cache 池化)
3. 预 Tokenize 训练数据 (F10, 低优先级)
