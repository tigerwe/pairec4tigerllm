# Qwen3-0.6B 生成式召回：技术方案与 GitHub 调研

> 目标：基于 GitHub 开源实践调研，设计使用 **Qwen3-0.6B** 替换现有自研 GPT2 Decoder 的生成式召回方案，保留现有 RQ-VAE + 语义 ID 数据流，与 `pairec` Go 服务及 DataSystem 兼容。
>
> 日期：2026-05-14

---

## 一、GitHub 调研：LLM-based 生成式推荐主流做法

### 1.1 调研项目总览

| 项目 | 机构 | 核心思路 | 与 Qwen3 场景相关性 |
|------|------|---------|-------------------|
| **[phonism/genrec](https://github.com/phonism/genrec)** | 开源社区 | **RQ-VAE semantic IDs + LLM (LCRec/TIGER)**，Gin-config 驱动，支持多种生成式模型 | **★★★★★** 直接相关：RQ-VAE + LLM 做生成式检索 |
| **[snap-research/GRID](https://github.com/snap-research/GRID)** | Snap Research | **LLM Embedding → RQ 语义 ID → Transformer 生成**，工业级框架 | **★★★★★** 直接相关：LLM 参与语义 ID 生成链路 |
| **[krishnacharya/GLoSS](https://github.com/krishnacharya/GLoSS)** | Gatech | **LoRA + 4-bit 量化 LLaMA-3** 做序列推荐，生成 query 做语义检索 | **★★★★☆** 可参考：小参数 LLM + LoRA 微调 |
| **[Indolent-Kawhi/EAGER-LLM](https://github.com/Indolent-Kawhi/EAGER-LLM)** |  academia | **LLaMA-7B Backbone**，非侵入式行为-语义整合，双源 item index | **★★★★☆** 可参考：Decoder-only LLM 做推荐，Adapter 微调 |
| **[ZY0025/GRLM](https://github.com/ZY0025/GRLM)** | 快手 | **Term IDs (TIDs)** 替代 semantic IDs，LLM 原生理解文本标识符 | **★★★☆☆** 思路参考：用 LLM 友好型 ID 表示物品 |
| **[critical88/TCA4Rec](https://github.com/critical88/TCA4Rec)** | academia | **Collaborative Tokenizer**：将 CF logits 投影到 LLM token 空间，软标签对齐 | **★★★★☆** 可参考：协同信号与 LLM token 空间对齐 |
| **[HestiaSky/E4SRec](https://github.com/HestiaSky/E4SRec)** | academia | **Item ID Embedding → Linear → LLM Embedding**，仅 Linear 可学习 | **★★★★☆** 可参考：最小化侵入的 ID 接入方式 |

### 1.2 关键结论：三条主流技术路线

#### 路线 A：Semantic ID + 生成式检索（TIGER / GRID / LCRec）

```
物品特征 → RQ-VAE → 语义 ID (如 [12, 45, 200, 88])
                              ↓
                    用户历史序列 → Transformer Decoder
                              ↓
                        自回归生成下一个语义 ID
```

- **特点**：保留完整的离散语义 ID 体系，模型输入输出都是整数序列
- **优势**：和现有 `pairec4tigerllm` 架构 100% 兼容，Go 服务层零改动
- **代表**：`phonism/genrec` 中的 `TIGER`、`LCRec`

#### 路线 B：LLM 文本生成 + 语义检索（GLoSS / GPT4Rec）

```
用户历史文本 → LLM 生成推荐 query / 推荐理由
                              ↓
                    语义检索（Dense Retrieval / BM25）
                              ↓
                        返回物品列表
```

- **特点**：利用 LLM 的文本理解能力生成中间表示，再经检索器召回
- **劣势**：需要物品有丰富文本描述；和现有 RQ-VAE 链路脱节
- **适合**：内容推荐（新闻、笔记、商品标题丰富）

#### 路线 C：LLM Backbone + ID Adapter（E4SRec / EAGER-LLM / TokenRec）

```
Item ID (或 Semantic ID) → nn.Embedding / Adapter → LLM Hidden Dim
                              ↓
                    拼接用户行为序列 → 标准 LLM Decoder
                              ↓
                        输出映射回 Item ID 空间
```

- **特点**：保留 LLM 预训练权重，通过少量可学习参数将推荐 ID 映射到 LLM 空间
- **优势**：充分利用 LLM 预训练知识，微调效率高（LoRA/Adapter）
- **代表**：`E4SRec` 用单个 Linear 映射 ID Embedding；`EAGER-LLM` 用 dual-source index

---

### 1.3 对我们的启示

| 设计决策 | 调研结论 | 我们的选择 |
|---------|---------|-----------|
| **是否保留 RQ-VAE 语义 ID？** | TIGER/GRID/LCRec 都保留，这是工业界验证的路径 | ✅ **保留**，与现有数据流兼容 |
| **LLM 如何处理语义 ID？** | E4SRec：Linear 映射；TIGER：直接当 token 用 | ✅ **混合方案**：4 quantizer 各学一个 Embedding，Sum 后投影到 Qwen3 hidden=1024 |
| **微调策略？** | GLoSS/EAGER-LLM：LoRA；E4SRec：只训 Linear | ✅ **LoRA + 输入输出层**，冻结 Qwen3 Backbone |
| **模型规模？** | GLoSS 用 LLaMA-3 (8B) 4-bit；E4SRec 用 7B | ✅ **Qwen3-0.6B** 足够，参数量小但架构完整（28层/GQA/RoPE） |
| **输出层设计？** | TIGER：逐层独立预测；LCRec：标准 LM Head | ✅ **4 个独立 Linear Head**（和现有模型一致），各输出 vocab_size=256 |

---

## 二、Qwen3-0.6B 生成式召回：完整技术方案

### 2.1 Qwen3-0.6B 架构参数

```json
{
  "hidden_size": 1024,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "intermediate_size": 3072,
  "vocab_size": 151936,
  "max_position_embeddings": 40960,
  "head_dim": 128,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16"
}
```

- **GQA**（分组查询注意力）：16 Q heads / 8 KV heads，节省 KV Cache 显存
- **RoPE**：旋转位置编码，支持外推到长序列
- **SwiGLU FFN**：hidden_size × 3 = 3072
- **RMSNorm**：Pre-LN 变体，训练更稳定

### 2.2 模型架构设计：`Qwen3GenerativeRec`

```
语义 ID 输入 [batch, seq_len, num_quantizers=4]
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Semantic Embedding Layer (可学习)               │
│  ├─ emb_0: Embedding(256, 1024)                 │
│  ├─ emb_1: Embedding(256, 1024)                 │
│  ├─ emb_2: Embedding(256, 1024)                 │
│  ├─ emb_3: Embedding(256, 1024)                 │
│  └─ 逐位置相加 → [batch, seq_len, 1024]          │
│  └─ (可选) Dropout + LayerNorm                   │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Qwen3-0.6B Backbone (28层 Decoder)              │
│  ├─ 加载预训练权重 (bfloat16)                     │
│  ├─ GQA + RoPE + SwiGLU + RMSNorm               │
│  └─ 冻结全部参数 (推理时只做前向)                 │
│  └─ (训练时通过 LoRA 微调 Q/K/V/O/Up/Gate/Down) │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  输出投影层 (可学习)                              │
│  ├─ head_0: Linear(1024, 256, bias=False)       │
│  ├─ head_1: Linear(1024, 256, bias=False)       │
│  ├─ head_2: Linear(1024, 256, bias=False)       │
│  ├─ head_3: Linear(1024, 256, bias=False)       │
│  └→ logits: [batch, seq_len, 4, 256]            │
└─────────────────────────────────────────────────┘
    │
    ▼
Softmax → 预测下一个语义 ID 的 4 个分量
```

#### 为什么不用 Qwen3 原生的 `lm_head` 和 `embed_tokens`？

- 原 `embed_tokens`：词表 151936，处理的是文本 token（BPE 分词后的子词单元）
- 原 `lm_head`：输出 151936 类，预测的是下一个文本 token
- 我们的任务：**输入是 4 层离散码本（每层 256 类），输出也是 4 层独立预测**
- 因此必须**替换输入输出层**，但保留中间 28 层 Transformer 的预训练权重

### 2.3 核心代码框架

```python
# training/decoder/qwen3_generative_rec.py
# -*- coding: utf-8 -*-
"""基于 Qwen3-0.6B Backbone 的生成式推荐模型."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Qwen3GenerativeRec(nn.Module):
    """Qwen3-0.6B Backbone + Semantic ID 输入输出的生成式推荐模型."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        vocab_size: int = 256,
        num_quantizers: int = 4,
        max_seq_len: int = 512,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_quantizers = num_quantizers
        self.max_seq_len = max_seq_len

        # 1. 加载 Qwen3-0.6B Backbone (完整模型，含 embed_tokens + lm_head)
        from transformers import AutoModelForCausalLM, AutoConfig
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        # 2. 隐藏原 Embedding 和 LM Head（我们用自己的）
        # 不删除，而是冻结，避免破坏模型结构
        for p in self.base_model.model.embed_tokens.parameters():
            p.requires_grad = False
        if self.base_model.lm_head is not None:
            for p in self.base_model.lm_head.parameters():
                p.requires_grad = False

        # 3. 新建语义 ID 输入层
        # 4 个 [256, 1024] Embedding 表，逐位置相加后作为 Transformer 输入
        self.semantic_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, self.config.hidden_size)
            for _ in range(num_quantizers)
        ])
        self.input_norm = nn.LayerNorm(self.config.hidden_size)
        self.input_dropout = nn.Dropout(0.1)

        # 4. 新建输出层：4 个 [1024, 256] Head
        self.output_heads = nn.ModuleList([
            nn.Linear(self.config.hidden_size, vocab_size, bias=False)
            for _ in range(num_quantizers)
        ])

        # 5. 初始化新层（小 std，避免破坏预训练 backbone）
        for emb in self.semantic_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        for head in self.output_heads:
            nn.init.normal_(head.weight, mean=0.0, std=0.02)

        # 6. LoRA 微调 Backbone
        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            print(f"[Qwen3GenerativeRec] LoRA enabled: rank={lora_rank}, alpha={lora_alpha}")

        # 7. 统计可训练参数
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Qwen3GenerativeRec] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _build_inputs_embeds(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        """将语义 ID 转换为 Qwen3 可接受的输入嵌入.

        Args:
            semantic_ids: [batch, seq_len, num_quantizers]

        Returns:
            hidden_states: [batch, seq_len, hidden_size=1024]
        """
        batch_size, seq_len, _ = semantic_ids.shape
        x = torch.zeros(
            batch_size, seq_len, self.config.hidden_size,
            device=semantic_ids.device, dtype=self.base_model.dtype,
        )
        for i, emb in enumerate(self.semantic_embeddings):
            x = x + emb(semantic_ids[:, :, i])
        return self.input_dropout(self.input_norm(x))

    def forward(
        self,
        semantic_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:
        """前向传播.

        Args:
            semantic_ids: [batch, seq_len, num_quantizers]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len, num_quantizers]
            use_cache: 是否返回 past_key_values
            past_key_values: 用于增量推理的 KV Cache
            position_ids: RoPE 位置编码（自动推导）

        Returns:
            logits: [batch, seq_len, num_quantizers, vocab_size]
            loss: 可选
            past_key_values: 可选（用于 Decode 阶段复用）
        """
        # 1. 构建输入嵌入（替代原 embed_tokens）
        inputs_embeds = self._build_inputs_embeds(semantic_ids)

        # 2. 通过 Qwen3 Backbone
        # 使用 base_model.model (即 Qwen3Model) 的 forward，传入 inputs_embeds
        outputs = self.base_model.model(
            input_ids=None,               # 不传入 input_ids，直接传 embeds
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            output_hidden_states=False,
        )
        hidden_states = outputs[0]        # [batch, seq_len, hidden_size]
        new_past_key_values = outputs[1] if use_cache else None

        # 3. 输出投影（替代原 lm_head）
        logits_list = [head(hidden_states) for head in self.output_heads]
        logits = torch.stack(logits_list, dim=2)  # [batch, seq_len, 4, 256]

        # 4. 计算损失（和现有 train.py 完全兼容）
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels, attention_mask)

        return logits, loss, new_past_key_values

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, num_q, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(-1, -1, num_q).reshape(-1)
            valid = mask.bool()
            logits_flat = logits_flat[valid]
            labels_flat = labels_flat[valid]

        return F.cross_entropy(logits_flat, labels_flat)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """自回归生成（支持 KV Cache 复用）."""
        self.eval()
        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            if generated.shape[1] > self.max_seq_len:
                generated = generated[:, -self.max_seq_len:]
                past_key_values = None  # 超出长度限制，重置 KV Cache

            logits, _, past_key_values = self.forward(
                generated,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )
            next_logits = logits[:, -1, :, :]  # [batch, 4, 256]

            if temperature != 1.0:
                next_logits = next_logits / temperature

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, self.vocab_size))
                next_logits[next_logits < v[:, :, [-1]]] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_tokens = torch.multinomial(
                probs.view(-1, self.vocab_size), num_samples=1
            ).view(-1, self.num_quantizers)

            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

        return generated[:, -max_new_tokens:, :]
```

### 2.4 关键设计决策说明

#### 决策 1：为什么用 `inputs_embeds` 而不是改造 `embed_tokens`？

Qwen3 的 `forward()` 接口支持 `input_ids` 或 `inputs_embeds` 二选一传入。通过 `inputs_embeds`，我们可以：
- 完全绕过 151936 大小的文本词表
- 自由定义语义 ID → hidden_size 的映射逻辑（4 个 embedding sum）
- 无需修改 `transformers` 库内部代码

#### 决策 2：为什么冻结原 `embed_tokens` 和 `lm_head` 而不是删除？

- 保留原层可以**防止 PEFT/LoRA 在包装模型时出错**（某些 LoRA 实现会遍历所有子模块）
- 如果未来需要做**文本-推荐多任务联合训练**，可以解冻复用
- `forward()` 中明确传 `input_ids=None`，不会触发原 embedding 的计算

#### 决策 3：LoRA 目标模块选择

```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
```

- Attention 层：`q/k/v/o_proj` 微调注意力模式，让模型学会关注序列中的协同信号
- MLP 层：`gate/up/down_proj`（SwiGLU 结构）微调 FFN 中的知识表达
- **不微调** `embed_tokens` / `lm_head` / `norm`：这些已被替换或冻结

---

## 三、训练策略

### 3.1 数据流（完全复用现有流程）

```
Tenrec ctr_data_1M.csv
    ↓
scripts/preprocess_data.py  (RQ-VAE 编码)
    ↓
语义 ID 序列 JSON: [[sid_t0], [sid_t1], ..., [sid_tN]]
    ↓
training/decoder/train_qwen3.py
    ↓
Qwen3GenerativeRec.forward(semantic_ids, labels)
```

**不需要改**：
- RQ-VAE 训练
- 数据预处理
- 序列 Dataset 格式
- Go 服务侧的特征获取逻辑

### 3.2 训练超参建议

| 参数 | 自研 GPT2 (当前) | Qwen3-0.6B (建议) | 理由 |
|------|-----------------|-------------------|------|
| 学习率 | 1e-4 | **5e-5** | Backbone 已预训练，避免过大 lr 破坏 |
| Batch Size | 64 | **32** | 显存占用增大（0.6B fp16 ≈ 1.2GB 权重 + 激活） |
| Epochs | 50 | **5~10** | 预训练模型收敛快，观察验证集早停 |
| Warmup | 1000 steps | **500 steps** | 短序列任务不需要长预热 |
| LoRA rank | - | **8** | 平衡表达能力和参数量 |
| 梯度裁剪 | 1.0 | **1.0** | 保持 |
| 优化器 | AdamW | **AdamW (8bit)** | 节省显存：`bitsandbytes` 的 `AdamW8bit` |

### 3.3 显存估算（L40S 24GB）

```
Qwen3-0.6B 权重 (bfloat16):     ~1,200 MB
LoRA 参数 (rank=8):              ~30 MB
Semantic Emb + Heads:            ~5 MB
Optimizer states (8bit AdamW):   ~600 MB
Activation (batch=32, seq=50):   ~800 MB
KV Cache (train):                ~200 MB
─────────────────────────────────────────
总计:                            ~2.8 GB
```

**完全可以在单张 L40S 上训练**，甚至 batch_size 可以调到 64~128。

---

## 四、推理部署

### 4.1 推理流程

```python
# inference/trt_llm/server_qwen3.py

class Qwen3InferenceService:
    def __init__(self, config):
        self.model = Qwen3GenerativeRec(
            model_name="Qwen/Qwen3-0.6B",
            use_lora=False,  # 推理前 merge LoRA 权重
        )
        # 加载微调后的 checkpoint
        ckpt = torch.load(config.model_path, map_location="cpu")
        self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        self.model = self.model.to("cuda").eval().bfloat16()

        # Merge LoRA（可选，减少推理时 overhead）
        if hasattr(self.model.base_model, 'merge_and_unload'):
            self.model.base_model = self.model.base_model.merge_and_unload()
            print("LoRA weights merged into base model")

    def recommend(self, user_history, topk=10, temperature=1.0):
        # 1. 准备输入
        input_ids = self._prepare_input(user_history).cuda()

        # 2. 自回归生成（带 KV Cache）
        generated = self.model.generate(
            input_ids,
            max_new_tokens=topk * 2,
            temperature=temperature,
            use_cache=True,       # 启用 KV Cache 加速
        )

        # 3. 语义 ID → 物品 ID（复用现有逻辑）
        recommendations = self._semantic_to_items(generated)
        return recommendations
```

### 4.2 TensorRT 导出策略

由于 Qwen3 使用 **GQA + RoPE + SwiGLU**，ONNX 导出会有挑战：

| 阶段 | 策略 | 优先级 |
|------|------|--------|
| **Phase 1** | PyTorch 推理（`use_cache=True`，bfloat16） | P0（立即可用） |
| **Phase 2** | `optimum` + ONNX Runtime 加速 | P1 |
| **Phase 3** | TensorRT-LLM 原生支持 Qwen3（社区/官方） | P2（等官方） |

**Phase 1 足够**：Qwen3-0.6B 在 L40S 上，batch=1、seq_len=50 的 Prefill 延迟约 **15~25ms**，每步 Decode 约 **3~5ms**，生成 20 步总延迟 < **100ms**，满足推荐场景要求。

### 4.3 KV Cache 持久化（与 DataSystem 集成）

```python
# 复用已有的 KVCacheManager + DataSystem Bridge
# 在 Qwen3 场景下，KV Cache 体积：
#   28 layers × 2(K+V) × 8 heads × seq_len × 128 head_dim × 2B
#   = 28 * 2 * 8 * 50 * 128 * 2 = 5.7 MB (seq_len=50)
#   = 28 * 2 * 8 * 20 * 128 * 2 = 2.3 MB (生成后 seq_len=70)

# DataSystem Key: pairec4tigerllm:kv:{user}:{history_hash}
# 写入/读取方式与现有 P0 方案完全一致
```

---

## 五、与现有系统的集成

### 5.1 改动清单

| 层级 | 文件 | 改动 | 影响 |
|------|------|------|------|
| **模型训练** | `training/decoder/model.py` | 新增 `Qwen3GenerativeRec` | 新增文件，不破坏旧模型 |
| **训练脚本** | `training/decoder/train.py` | 支持 `--backbone=qwen3` | 可选切换 |
| **导出脚本** | `training/decoder/export.py` | 适配 Qwen3 checkpoint 格式 | 可选切换 |
| **推理服务** | `inference/trt_llm/server.py` | 新增 Qwen3 分支 | 可选切换 |
| **Go 服务** | `services/recall/*.go` | **零改动** | 接口格式不变 |
| **DataSystem** | `docs/P0_DATASYSTEM_LATENCY_GUIDE.md` | **零改动** | KV Cache 格式通用 |

### 5.2 配置切换示例

```json
// configs/pairec_config.json
{
  "recall": {
    "generative_recall": {
      "type": "GenerativeRecall",
      "model_name": "generative_recall",
      "tiger_recall_conf": {
        "tiger_name": "http://localhost:8000",
        "top_k": 50,
        "history_max_length": 20
      }
    }
  }
}
```

```yaml
# configs/generative_config.yaml
# 通过 server_url 区分不同推理后端
server_url: "http://localhost:8000"   # Qwen3 推理服务端口不变
```

---

## 六、GitHub 对标：我们的方案 vs 开源实现

| 维度 | phonism/genrec (LCRec) | E4SRec | **我们的方案** |
|------|------------------------|--------|--------------|
| Backbone | LLaMA-2/3 或 GPT | LLaMA-2/3 | **Qwen3-0.6B** |
| ID 表示 | RQ-VAE Semantic ID | 原始 Item ID + Linear | **RQ-VAE Semantic ID** |
| ID → LLM | 文本化 / 直接嵌入 | nn.Embedding + Linear | **4 × Embedding(256,1024) sum** |
| 微调方式 | Full Fine-tune / LoRA | 只训 Linear | **LoRA + 输入输出层** |
| 输出层 | 标准 LM Head (单维) | Linear → softmax | **4 个独立 Head (多维)** |
| 推理优化 | PyTorch / vLLM | PyTorch | **Phase 1 PyTorch → Phase 2 TensorRT** |
| 外部 KV 存储 | 无 | 无 | **DataSystem Worker** |

**我们的优势**：
1. **最小侵入**：只改 Python 模型层，Go 服务 + DataSystem + 数据流全兼容
2. **工业友好**：Qwen3-0.6B 小参数 + LoRA，单卡可训可推
3. **可扩展**：未来可无缝切换到 Qwen3-8B 或 Qwen3-32B（只需改 model_name）
4. **KV Cache 持久化**：结合 DataSystem 实现跨请求复用（开源项目普遍缺失）

---

## 七、实施 Checklist

- [ ] **Step 1**：环境准备 `pip install transformers>=4.51.0 peft accelerate bitsandbytes`
- [ ] **Step 2**：下载 Qwen3-0.6B 权重到本地或确认 HuggingFace 可访问
- [ ] **Step 3**：实现 `training/decoder/qwen3_generative_rec.py`（本文 2.3 节代码）
- [ ] **Step 4**：改造 `training/decoder/train.py` 支持 `--backbone=qwen3`
- [ ] **Step 5**：小规模验证：batch=4，1 个 epoch，确认 loss 下降
- [ ] **Step 6**：全量训练（5~10 epoch），保存 `decoder_qwen3_best.pt`
- [ ] **Step 7**：改造 `server.py` 加载 Qwen3 模型，跑通 `/recommend` 接口
- [ ] **Step 8**：端到端联调：Go → Python → Qwen3 → DataSystem KV Cache
- [ ] **Step 9**：A/B 测试：对比自研 GPT2 vs Qwen3-0.6B 的 Recall@K

---

## 八、一句话总结

> **调研结论**：GitHub 上 `genrec` (LCRec) 和 `GRID` 验证了 **RQ-VAE Semantic ID + LLM Backbone** 的工业可行性；`E4SRec` 和 `EAGER-LLM` 验证了 **LoRA + 最小侵入 Adapter** 的训练效率。
>
> **我们的方案**：基于这些开源实践，将 **Qwen3-0.6B (28层/GQA)** 作为 Backbone，**替换输入输出层**适配 4-quantizer 语义 ID，**LoRA 微调**，**复用现有数据流和 Go 服务**，并**接入 DataSystem KV Cache 持久化**。单卡 L40S 可训可推，2 周内可完成端到端落地。
