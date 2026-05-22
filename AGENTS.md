# PaiRec4TigerLLM — AI 代理工作指南

> 最后更新: 2026-05-22 | 分支: `dev`

## 工作纪律

1. **开工读进度** — 先读 `progress.md` 和 `feature_list.json`，了解当前状态
2. **收工写进度** — 更新 `progress.md`，标注完成/阻塞/下一步
3. **一个 session 一个 feature** — 从 `feature_list.json` 取一个 `status: pending` 的任务
4. **验证带证据** — 不说"完成了"，给可复现命令 + 输出
5. **GPU 任务走远程** — 本机 (RTX 3060 6GB) 只能做语法/导入检查；训练/推理在远程 L40S 机器上验证

## 路由规则

| 你想... | 读哪个文件 |
|---------|-----------|
| 知道现在做到哪了 | `progress.md` |
| 知道还有什么待做 | `feature_list.json` |
| 了解架构细节 (Qwen3, DataSystem, TRT-LLM) | `docs/session_context.md` |
| 上次交接的信息 | `HANDOFF.md` |
| 跑命令 | 看下文"速查命令" |

## 关键文件 (精简版)

| 文件 | 一句话 |
|------|--------|
| `training/decoder/qwen3_generative_rec.py` | Qwen3 prompt 模型 (forward + generate + 约束解码) |
| `training/decoder/train.py` | 训练主循环 (DDP + resume + checkpoint) |
| `inference/trt_llm/server.py` | Flask 推理服务 (/recommend, /health) |
| `training/decoder/model.py` | 老 GPT2 Decoder (保留兼容) |
| `inference/kv_cache/manager.py` | Python KVCacheManager (Phase 1 跳过) |

## 速查命令

### 本机验证 (无 GPU，仅语法/导入)

```bash
cd pairec4tigerllm
python -c "import training.decoder.qwen3_generative_rec; print('OK')"
python -c "import inference.trt_llm.server; print('OK')"
python -c "import training.decoder.train; print('OK')"
```

### 远程 GPU 训练 (在 L40S 机器上执行)

```bash
cd /home/workspace/zcx/pairectest/pairec4tigerllm
git pull gitcode dev

# 单卡
CUDA_VISIBLE_DEVICES=6 python -m training.decoder.train \
    --backbone qwen3 --qwen3_model_path ./models/Qwen3-0.6B \
    --train_data ./data/tenrec/processed/train_sequences.json \
    --num_epochs 10 --batch_size 32 --learning_rate 5e-5 \
    --lora_rank 8 --lora_alpha 16 \
    --checkpoint_dir ./checkpoints/decoder_qwen3 --max_seq_len 20

# 三卡
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 \
    -m training.decoder.train ...

# Resume
--load_checkpoint ./checkpoints/decoder_qwen3/decoder_epoch_20.pt --num_epochs 30
```

### 远程推理验证

```bash
pkill -f "inference.trt_llm.server" || true
python -m inference.trt_llm.server \
    --model_path ./checkpoints/decoder_qwen3/decoder_epoch_20.pt \
    --port 18000 --device cuda \
    --qwen3_model_path /home/workspace/zcx/pairecgitcode/pairec4tigerllm/models/Qwen3-0.6B &

curl -X POST http://localhost:18000/recommend \
    -H "Content-Type: application/json" \
    -d '{"user_id":"test","history":[[169,41,0,0],[20,53,0,0],[80,201,0,0]],"topk":10}'
# 预期: code=200, recommendations 非空
```

### Git 操作

```bash
git add <files> && git commit -m "feat/fix: ..."
git push gitcode dev
# token 已经保存在远程会话中，直接 push 即可
```

## 远程环境信息

| 项目 | 本机 | 远程 L40S |
|------|------|-----------|
| GPU | RTX 3060 6GB | 3×L40S 46GB |
| 用途 | 代码编辑/git | 训练 + 推理 |
| 代码路径 | `~/pairec4tigerllm` | `/home/workspace/zcx/pairectest/pairec4tigerllm` |
| 数据路径 | 无 | `/home/workspace/zcx/pairec4tigerllm/data/` |
| 模型路径 | `~/models/Qwen3-0.6B` | `/home/workspace/zcx/pairec4tigerllm/models/Qwen3-0.6B` |
