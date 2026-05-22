# 交接日志

> 供 Agent 跨 session 恢复上下文。只记录关键决策和当前任务。

## 最近一次交接 (2026-05-22)

### 当前任务
Phase 3: TensorRT-LLM 引擎构建 + DataSystem 集成

### 已完成
- Qwen3 prompt model 训练完成 (epoch 20, loss 2.49)
- PyTorch 推理服务 + 约束解码 验证通过
- 仓库规范化 (AGENTS.md ≤100行, feature_list.json, progress.md)

### 关键决策
- 用 Prompt Template 而非 inputs_embeds (参照京东方案)
- LoRA + modules_to_save (lm_head + embed_tokens 必须可训)
- 约束解码强制 s0→s1→s2→s3 顺序 + 物品白名单
- DDP 多卡训练: torchrun + DistributedSampler (不除batch_size)
- Phase 1 skip KVCacheManager (prompt 模式用原生 generate)

### 环境
- 代码在本机 `~/pairec4tigerllm`，通过 gitcode 同步到远程
- 训练/推理在远程 L40S `/home/workspace/zcx/pairectest/pairec4tigerllm`

### 下一步命令 (在远程 GPU 机器)
```bash
cd /home/workspace/zcx/pairectest/pairec4tigerllm
git pull gitcode dev

# Phase 3 Step 1: 导出 HF 格式
python scripts/export_for_trtllm.py

# Phase 3 Step 2: TRT-LLM 转换 (在 TensorRT-LLM 目录)
python /path/to/TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py \
    --model_dir ./exported/qwen3_rec --output_dir ./trt_ckpt --dtype bfloat16

# Phase 3 Step 3: 构建引擎
trtllm-build --checkpoint_dir ./trt_ckpt --output_dir ./trt_engines/qwen3_rec \
    --gemm_plugin bfloat16 --gpt_attention_plugin bfloat16 \
    --max_batch_size 32 --max_seq_len 2048 --max_input_len 2048
```
