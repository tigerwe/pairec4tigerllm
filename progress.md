# 工作进度

> 最后更新: 2026-05-22 | 下一个: 等待 Phase 3 (TRT-LLM)

## 时间线

| 日期 | 进度 |
|------|------|
| 5/18 | 启动: 讨论使用 Qwen3-0.6B 替换 GPT2 Decoder |
| 5/19 | v1: 设计 Prompt Template 架构; 重写模型类; 修 dtype→torch_dtype 兼容 |
| 5/20 | 发现 loss 横在 10.27 → modules_to_save 修复 (loss→3.06); lm_head 全序列优化; 梯度检查点+梯度累积; epoch 10 训练完成(loss 2.65); 推理服务启动但推荐为空; 修补 checkpoint config; 加 Resume 训练支持 |
| 5/21 | DDP 三卡 3bug 修复; L40S×3 并行训练启动 (epoch 11); Docker 镜像适配; 推理约束解码实现+修复注入bug; loss 2.65→2.57 (epoch 15) |
| 5/22 | 训练完成 epoch 20 (loss 2.49); 仓库规范化 (AGENTS.md, feature_list, progress, handoff); Phase 3 导出脚本 |

## 当前状态

- **训练**: epoch 20 完成, loss 2.49, checkpoint `decoder_epoch_20.pt` 可用
- **推理**: PyTorch Flask 服务正常, 约束解码 hit/miss=5/0
- **Phase 3**: 导出脚本就绪, 待远程 GPU 环境执行 TRT-LLM 引擎构建

## 阻塞点

- 无。可以继续 next step

## 下一步

1. 远程GPU: 执行 Phase 3 导出→TRT-LLM转换→引擎构建
2. 或用 epoch_20 checkpoint 测 Go pairec 联调
