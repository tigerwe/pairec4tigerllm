# Qwen3-0.6B 生成式召回 — 完整实施指南

> 状态：本地代码已就绪（`dev` 分支），待远程 4090 训练 + 部署。
>
> 关联文档：[QWEN3_GENERATIVE_RECALL_DESIGN.md](./QWEN3_GENERATIVE_RECALL_DESIGN.md) | [TRT_LLM_DATASYSTEM_KV_CACHE_INTEGRATION.md](./TRT_LLM_DATASYSTEM_KV_CACHE_INTEGRATION.md)

---

## 一、目标

用 **Qwen3-0.6B** (28层/GQA/RoPE/SwiGLU) 替换自研 GPT2 Decoder，增加 **KVCacheManager** 实现 GPU HBM KV Cache 复用 + DataSystem 持久化。

最终链路：

```
HTTP Request → Go pairec → POST /recommend 
    → Qwen3.generate() → KVCacheManager (HBM LRU) → DataSystem (async) 
    → Response
```

---

## 二、环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA GPU (≥ 24GB 推荐，6GB 可推理) |
| CUDA | 12.x (与 PyTorch 匹配即可) |
| Python | 3.10+ |
| PyTorch | 2.1+ |
| 依赖 | transformers, peft, accelerate, bitsandbytes |

---

## 三、文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `training/decoder/qwen3_generative_rec.py` | **新建** | Qwen3 模型类 |
| `inference/kv_cache/manager.py` | **新建** | KVCacheManager |
| `inference/kv_cache/__init__.py` | **新建** | 包导出 |
| `scripts/verify_changes.py` | **新建** | 验证脚本 |
| `training/decoder/__init__.py` | 修改 | 导出 Qwen3GenerativeRec |
| `training/decoder/train.py` | 修改 | 支持 `--backbone qwen3` |
| `inference/trt_llm/server.py` | 修改 | Qwen3加载 + KV Cache 推荐 |

---

## 四、实施步骤

### Step 1：环境准备（4090 机器）

```bash
# 1.1 克隆代码
git clone https://gitcode.com/weixin_43325008/pairec4tigerllm.git
cd pairec4tigerllm
git checkout dev

# 1.2 安装依赖
pip install peft accelerate bitsandbytes tensorboard

# 1.3 下载 Qwen3-0.6B 模型（三选一）
# A. hf-mirror 镜像（推荐）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B

# B. ModelScope
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', cache_dir='./models/Qwen3-0.6B')"

# C. 离线传输（最后手段）
# 在有网机器下载 → tar → scp 到 4090

# 1.4 验证环境
python scripts/verify_changes.py
```

### Step 2：验证模型加载（Smoke Test）

```bash
python -c "
import torch
from training.decoder.qwen3_generative_rec import Qwen3GenerativeRec

model = Qwen3GenerativeRec(
    model_name_or_path='./models/Qwen3-0.6B',
    use_lora=False,
).cuda().bfloat16().eval()

# Forward
dummy = torch.randint(0, 256, (2, 10, 4)).cuda()
logits, loss, _ = model(dummy)
print(f'Forward OK: {logits.shape}')

# Generate
gen = model.generate(dummy[:1], max_new_tokens=5)
print(f'Generate OK: {gen.shape}')
print(f'VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
"
```

### Step 3：训练

```bash
# 3a. 小规模验证（1 epoch, batch=4）
python -m training.decoder.train \
    --backbone qwen3 \
    --qwen3_model_path ./models/Qwen3-0.6B \
    --train_data ./data/processed/train_sequences.json \
    --num_epochs 1 --batch_size 4 --learning_rate 5e-5 \
    --checkpoint_dir ./checkpoints/decoder_qwen3

# 3b. 全量训练（10 epoch, batch=32）
python -m training.decoder.train \
    --backbone qwen3 \
    --qwen3_model_path ./models/Qwen3-0.6B \
    --train_data ./data/processed/train_sequences.json \
    --val_data ./data/processed/val_sequences.json \
    --num_epochs 10 --batch_size 32 --learning_rate 5e-5 \
    --lora_rank 8 --lora_alpha 16 \
    --checkpoint_dir ./checkpoints/decoder_qwen3
```

### Step 4：启动推理服务

```bash
python -m inference.trt_llm.server \
    --model_path ./checkpoints/decoder_qwen3/decoder_best.pt \
    --port 8000 --device cuda \
    --backbone qwen3 \
    --qwen3_model_path ./models/Qwen3-0.6B

# 验证
curl http://localhost:8000/health
curl -X POST http://localhost:8000/recommend \
    -H "Content-Type: application/json" \
    -d '{"user_id":"u1","history":[[10,20,30,40],[11,21,31,41]],"topk":5}'
```

### Step 5：Go 服务联调

Go 侧 `services/recall/trtllm_client.go` **零改动**。确保配置中 `server_url` 指向推理服务。

---

## 五、关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Backbone | Qwen3-0.6B | 28层/GQA/RoPE，单卡可训可推 |
| ID 接入 | `inputs_embeds` | 绕过 151936 词表 |
| 输入层 | 4×Embedding(256,1024) sum | 保留 RQ-VAE 4-quantizer |
| 输出层 | 4×Linear(1024,256) | 独立预测各分量 |
| 微调 | LoRA rank=8 | 可训参数 ~30MB |
| KV Cache | 原生 `use_cache=True` | Qwen3 内置支持 |
| 推理后端 | Phase 1 PyTorch, Phase 3 TRT | PyTorch <100ms 已可接受 |

## 六、已修正的设计文档 Bug

1. **`generate()` KV Cache 使用错误** — 文档设计每步传完整序列，实际应只传单个 token (seq_len=1)
2. **RoPE position_ids 缺失** — Decode 步必须显式传入 `past_len + step`
3. **LoRA merge 后旧对象未释放** — 需 `del peft_model` + `torch.cuda.empty_cache()`

---

## 七、配置变量（需根据实际环境修改）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `--qwen3_model_path` | `Qwen/Qwen3-0.6B` | 模型路径（HF name 或本地目录） |
| `--train_data` | — | 语义 ID 序列 JSON 文件 |
| `--batch_size` | 32 | 4090 24GB 用 32，6GB 用 4 |
| `--learning_rate` | 5e-5 | Qwen3 推荐值 |
| `KVCacheManager.hbm_capacity` | 50 | HBM 缓存条数 |
| `KVCacheManager.ds_client` | `None` | DataSystem client |
| `server_url` (Go) | `http://localhost:8000` | 指向推理服务 |

---

## 八、远程仓库

| remote | URL | 分支 |
|--------|-----|------|
| `gitcode` | `https://gitcode.com/weixin_43325008/pairec4tigerllm.git` | `dev` |
| `origin` | `git@github.com:tigerwe/pairec4tigerllm.git` | `main` |

---

## 九、下一步工作

- [ ] 4090: 下载 Qwen3-0.6B 权重
- [ ] 4090: 训练 (小规模 → 全量)
- [ ] 4090: 启动推理服务 + 端到端验证
- [ ] 接入 DataSystem client 到 KVCacheManager
- [ ] A/B 测试：GPT2 vs Qwen3-0.6B Recall@K
- [ ] Phase 3: TensorRT-LLM 推理 (等官方 Qwen3 支持)
