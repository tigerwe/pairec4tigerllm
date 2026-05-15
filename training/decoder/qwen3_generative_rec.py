# -*- coding: utf-8 -*-
"""基于 Qwen3-0.6B Backbone 的生成式推荐模型.

修正要点（vs 设计文档）:
- generate() 中 Prefill 后只传单个 token 做 Decode, 不重复计算全序列
- Decode 阶段显式传入 position_ids (RoPE 需要绝对位置)
- LoRA merge 后显式清理旧对象

与现有系统的兼容性:
- forward() 返回 (logits, loss, past_key_values) — 训练时 past_key_values=None
- generate() 返回 [batch, max_new_tokens, num_quantizers] — 与现有语义 ID 格式一致
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Qwen3GenerativeRec(nn.Module):
    """Qwen3-0.6B Backbone + Semantic ID 输入输出的生成式推荐模型."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-0.6B",
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

        # ── 1. 加载 Qwen3 Backbone ──────────────────────────
        from transformers import AutoModelForCausalLM, AutoConfig

        self.config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.hidden_size = self.config.hidden_size          # 1024
        self.num_layers = self.config.num_hidden_layers      # 28
        self.num_kv_heads = self.config.num_key_value_heads  # 8 (GQA)
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads  # 64
        self.pad_token_id = 0

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=None,
        )

        # 冻结原始 embed_tokens 和 lm_head
        for p in self.base_model.model.embed_tokens.parameters():
            p.requires_grad = False
        if hasattr(self.base_model, 'lm_head') and self.base_model.lm_head is not None:
            for p in self.base_model.lm_head.parameters():
                p.requires_grad = False

        # ── 2. 语义 ID 输入层 ──────────────────────────────
        self.semantic_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, self.hidden_size) for _ in range(num_quantizers)
        ])
        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.input_dropout = nn.Dropout(0.1)

        # ── 3. 输出投影层 ──────────────────────────────────
        self.output_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, vocab_size, bias=False)
            for _ in range(num_quantizers)
        ])

        # ── 4. 初始化新层（小 std，避免破坏预训练 backbone）──
        for emb in self.semantic_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        for head in self.output_heads:
            nn.init.normal_(head.weight, mean=0.0, std=0.02)

        # ── 5. LoRA 微调 ───────────────────────────────────
        self.lora_enabled = use_lora
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

        # ── 6. 统计 ────────────────────────────────────────
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"[Qwen3GenerativeRec] Trainable: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    # ═════════════════════════════════════════════════════════
    #  输入嵌入构建
    # ═════════════════════════════════════════════════════════

    def _build_inputs_embeds(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        """将语义 ID [batch, seq_len, num_quantizers] 转为 inputs_embeds."""
        batch_size, seq_len, _ = semantic_ids.shape
        x = torch.zeros(
            batch_size, seq_len, self.hidden_size,
            device=semantic_ids.device, dtype=self.base_model.dtype,
        )
        for i, emb in enumerate(self.semantic_embeddings):
            x = x + emb(semantic_ids[:, :, i].long())
        return self.input_dropout(self.input_norm(x))

    # ═════════════════════════════════════════════════════════
    #  前向传播
    # ═════════════════════════════════════════════════════════

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
            semantic_ids:   [batch, seq_len, num_quantizers]
            attention_mask: [batch, seq_len]
            labels:         [batch, seq_len, num_quantizers]
            use_cache:      是否返回 past_key_values
            past_key_values: KV Cache (增量推理)
            position_ids:   [batch, seq_len] RoPE 位置 (使用 KV Cache 时必须显式传入)

        Returns:
            logits:          [batch, seq_len, num_quantizers, vocab_size]
            loss:            可选
            past_key_values: 可选
        """
        inputs_embeds = self._build_inputs_embeds(semantic_ids)

        outputs = self.base_model.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            output_hidden_states=False,
        )
        hidden_states = outputs[0]          # [batch, seq_len, hidden_size]
        new_past_key_values = outputs[1] if use_cache else None

        logits_list = [head(hidden_states) for head in self.output_heads]
        logits = torch.stack(logits_list, dim=2)  # [batch, seq_len, num_quantizers, vocab_size]

        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels, attention_mask)

        return logits, loss, new_past_key_values

    def _compute_loss(self, logits, labels, attention_mask):
        batch, seq_len, num_q, vocab = logits.shape
        logits_flat = logits.reshape(-1, vocab)
        labels_flat = labels.reshape(-1)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(-1, -1, num_q).reshape(-1)
            valid = mask.bool()
            logits_flat = logits_flat[valid]
            labels_flat = labels_flat[valid]
        return F.cross_entropy(logits_flat, labels_flat)

    # ═════════════════════════════════════════════════════════
    #  自回归生成 (修正版)
    # ═════════════════════════════════════════════════════════

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """自回归生成（正确的 KV Cache 使用）.

        修正要点:
        - Prefill: 一次性计算全序列 KV Cache
        - Decode:  每步只传单个 token (seq_len=1), 不重复计算
        - 每步显式传入 position_ids (RoPE 需要绝对位置)

        Returns:
            [batch, max_new_tokens, num_quantizers]
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        past_len = input_ids.shape[1]

        # === Prefill ===
        prefill_positions = torch.arange(
            0, past_len, device=device
        ).unsqueeze(0).expand(batch_size, -1)

        logits, _, past_key_values = self.forward(
            input_ids,
            use_cache=use_cache,
            past_key_values=None,
            position_ids=prefill_positions,
        )

        next_tokens = self._sample_token(logits[:, -1, :, :], temperature, top_k)
        generated_tokens = [next_tokens]
        current_position = past_len

        # === Decode ===
        for _ in range(1, max_new_tokens):
            # 关键修正 1: 只传当前 token (seq_len=1)
            current_input = next_tokens.unsqueeze(1)  # [batch, 1, num_quantizers]

            # 关键修正 2: 显式传入 RoPE position_ids
            position_ids = torch.full(
                (batch_size, 1), current_position,
                dtype=torch.long, device=device
            )

            logits, _, past_key_values = self.forward(
                current_input,
                use_cache=use_cache,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

            next_tokens = self._sample_token(logits[:, -1, :, :], temperature, top_k)
            generated_tokens.append(next_tokens)
            current_position += 1

            if current_position >= self.max_seq_len:
                past_key_values = None
                current_position = 0

        return torch.stack(generated_tokens, dim=1)  # [batch, max_new_tokens, num_quantizers]

    def _sample_token(self, logits, temperature, top_k):
        """采样一个 token (num_quantizers 个分量同时采样)."""
        if temperature != 1.0:
            logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, self.vocab_size))
            logits[logits < v[:, :, [-1]]] = float('-inf')

        batch_size = logits.shape[0]
        logits_2d = logits.view(-1, self.vocab_size)
        probs = F.softmax(logits_2d, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1)
        return tokens.view(batch_size, self.num_quantizers)

    # ═════════════════════════════════════════════════════════
    #  LoRA merge
    # ═════════════════════════════════════════════════════════

    def merge_lora(self):
        """推理前将 LoRA 权重合并回基座."""
        if not self.lora_enabled:
            return
        if hasattr(self.base_model, 'merge_and_unload'):
            peft_model = self.base_model
            self.base_model = peft_model.merge_and_unload()
            del peft_model
            torch.cuda.empty_cache()
            self.lora_enabled = False
            print("[Qwen3GenerativeRec] LoRA merged into base model")


__all__ = ["Qwen3GenerativeRec"]
