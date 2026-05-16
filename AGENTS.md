# PaiRec4TigerLLM — AI 代理开发指南

> 最后更新: 2026-05-16 | 分支: `dev` | 状态: Qwen3 改造完成，待远程训练

## 项目概述

PaiRec4TigerLLM 是一个基于生成式模型的推荐系统，当前正在从自研 GPT2 Decoder 升级到 **Qwen3-0.6B**。

| 组件 | 技术栈 |
|------|--------|
| 召回框架 | pairec (Go, 阿里巴巴开源) |
| 语义编码 | RQ-VAE |
| 生成式召回 | ~~GPT2 Decoder~~ → **Qwen3-0.6B + LoRA** |
| 推理加速 | PyTorch (Phase 1) → TensorRT-LLM (Phase 3) |
| KV Cache | KVCacheManager (HBM LRU + DataSystem) |
| 数据流 | Tenrec → Flink/Kafka → RQ-VAE → 语义 ID 序列 |

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                     User Request                         │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                   PaiRec API (Go)                        │
│  ┌─────────────────────────────────────────────────┐    │
│  │  生成式召回 (GenerativeRecall)                    │    │
│  │  → HTTP POST /recommend                         │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│         Python Inference Service (Port 8000)             │
│  ┌──────────────────────────────────────────────────┐   │
│  │  GenerativeInferenceService                       │   │
│  │  ├─ backbone: qwen3 | gpt2                       │   │
│  │  ├─ KVCacheManager (HBM LRU)                     │   │
│  │  └─ recommend() → Qwen3.generate()                │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
┌──────────────────────┐   ┌──────────────────────┐
│   KVCacheManager     │   │   DataSystem Worker  │
│   GPU HBM LRU        │◄─►│   持久化 KV Cache     │
└──────────────────────┘   └──────────────────────┘
```

## 关键源文件

### Python 模型层

| 文件 | 说明 |
|------|------|
| `training/decoder/qwen3_generative_rec.py` | **★ Qwen3-0.6B 模型类** (LoRA + KV Cache) |
| `training/decoder/model.py` | 原 GPT2 Decoder (保留兼容) |
| `training/decoder/train.py` | 训练脚本，`--backbone qwen3\|gpt2` |
| `training/decoder/export.py` | 模型导出 (ONNX) |
| `training/decoder/__init__.py` | 导出 GenerativeDecoder + Qwen3GenerativeRec |
| `training/rqvae/model.py` | RQ-VAE 模型 |

### Python 推理/服务层

| 文件 | 说明 |
|------|------|
| `inference/trt_llm/server.py` | **★ 推理服务** (Qwen3 + GPT2 双 backbone) |
| `inference/trt_llm/build_engine.py` | TensorRT 引擎构建 (GPT2 only) |
| `inference/kv_cache/manager.py` | **★ KVCacheManager** (HBM LRU + 序列化) |
| `inference/kv_cache/__init__.py` | 包导出 |

### Go 集成层 (无需改动)

| 文件 | 说明 |
|------|------|
| `services/recall/trtllm_client.go` | HTTP 客户端 → `/recommend` |
| `services/recall/generative_recall.go` | 生成式召回实现 |
| `services/main.go` | 服务入口 |

### 文档

| 文件 | 说明 |
|------|------|
| `docs/QWEN3_GENERATIVE_RECALL_DESIGN.md` | Qwen3 技术方案 + GitHub 调研 |
| `docs/TRT_LLM_DATASYSTEM_KV_CACHE_INTEGRATION.md` | KV Cache 集成方案 |
| `docs/IMPLEMENTATION_GUIDE.md` | **★ 完整实施指南** (Step-by-step) |
| `docs/P0_DATASYSTEM_LATENCY_GUIDE.md` | DataSystem 延迟指南 |

## 当前状态 (2026-05-16)

### Git

- **分支**: `dev` (Qwen3 改造), `main` (原有 GPT2)
- **远程**: `gitcode` → `https://gitcode.com/weixin_43325008/pairec4tigerllm.git`
- **远程**: `origin` → `git@github.com:tigerwe/pairec4tigerllm.git`
- **最新提交**: `e8d16e4` feat: Qwen3-0.6B generative recall + KV Cache Manager

### 已完成

- [x] Qwen3GenerativeRec 模型类 (修正 KV Cache + RoPE)
- [x] KVCacheManager (HBM LRU + DataSystem 序列化)
- [x] train.py 改造 (`--backbone qwen3`)
- [x] server.py 改造 (Qwen3 加载 + KV Cache 推荐)
- [x] 本地语法/导入验证通过
- [x] 推送到 gitcode `dev` 分支
- [x] 实施指南文档

### 待完成 (下一阶段)

- [ ] 4090 openEuler: 下载 Qwen3-0.6B 权重
- [ ] 训练 (小规模 → 全量)
- [ ] 启动推理服务 + 验证
- [ ] 接入 DataSystem client
- [ ] Go pairec 端到端联调
- [ ] A/B 测试 Recall@K
- [ ] Phase 3: TensorRT-LLM 推理

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Backbone | Qwen3-0.6B | 28层/GQA/RoPE/SwiGLU, 单卡可训可推 |
| ID 接入方式 | `inputs_embeds` (绕过词表) | 不修改 transformers 源码 |
| 输入层 | 4×Embedding(256,1024)→sum | 保留 RQ-VAE 4-quantizer 结构 |
| 输出层 | 4×Linear(1024,256) | 独立预测各分量 |
| 微调策略 | LoRA rank=8, alpha=16 | ~30MB 可训参数, 冻结 1.2GB 基座 |
| KV Cache | Qwen3 原生 `use_cache=True` | 内置支持, 无需额外改动 |
| 推理后端 | Phase 1 PyTorch, Phase 3 TRT | <100ms 延迟已满足推荐场景 |
| Go 服务 | 零改动 | `/recommend` API 接口不变 |

## 已修正的设计文档 Bug

> 以下 Bug 在 `QWEN3_GENERATIVE_RECALL_DESIGN.md` 中存在, 实际代码已修正。

1. **`generate()` KV Cache 使用错误** — 原设计每次 Decode 传完整序列, 正确做法: Prefill 后只传单 token (`seq_len=1`)
2. **RoPE position_ids 缺失** — Decode 步必须显式传入 `past_len + step` 作为绝对位置
3. **LoRA merge 后旧对象未释放** — `merge_and_unload()` 后需 `del peft_model` + `torch.cuda.empty_cache()`

## 常见任务

### 训练 Qwen3

```bash
python -m training.decoder.train \
    --backbone qwen3 \
    --qwen3_model_path ./models/Qwen3-0.6B \
    --train_data ./data/processed/train_sequences.json \
    --num_epochs 10 --batch_size 32 --learning_rate 5e-5 \
    --checkpoint_dir ./checkpoints/decoder_qwen3
```

### 启动推理服务

```bash
python -m inference.trt_llm.server \
    --model_path ./checkpoints/decoder_qwen3/decoder_best.pt \
    --port 8000 --backbone qwen3 \
    --qwen3_model_path ./models/Qwen3-0.6B
```

### 验证推荐接口

```bash
curl -X POST http://localhost:8000/recommend \
    -H "Content-Type: application/json" \
    -d '{"user_id":"u1","history":[[10,20,30,40]],"topk":5}'
```

## 环境信息

| 项目 | 本地 (RTX 3060 Laptop) | 远程 (4090 D) |
|------|----------------------|--------------|
| 显存 | 6 GB | 24 GB |
| 训练 | 不可 (显存不足) | ✅ batch=32 |
| 推理 | ✅ batch=1 | ✅ |
| OS | Ubuntu | openEuler |
| Python | 3.8 | 3.10+ |
| 模型下载方式 | hf-mirror | hf-mirror / ModelScope |

## Git 操作速查

```bash
# 提交到 dev
git add <files>
git commit -m "feat: ..."
git push gitcode dev

# 如果 HTTPS 需要 token
git push https://USER:TOKEN@gitcode.com/weixin_43325008/pairec4tigerllm.git dev
```

## 参考资源

- [Qwen3-0.6B HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B)
- [hf-mirror 镜像站](https://hf-mirror.com)
- [gitcode 仓库](https://gitcode.com/weixin_43325008/pairec4tigerllm)
- [PEFT (LoRA)](https://github.com/huggingface/peft)
- [实施指南](docs/IMPLEMENTATION_GUIDE.md)
- [Qwen3 方案设计](docs/QWEN3_GENERATIVE_RECALL_DESIGN.md)
- [KV Cache 集成方案](docs/TRT_LLM_DATASYSTEM_KV_CACHE_INTEGRATION.md)
