# -*- coding: utf-8 -*-
"""基于 Qwen3-0.6B + Prompt Template 的生成式推荐模型.

与旧版 (inputs_embeds) 的关键差异:
- 新增 4×256=1024 个语义 ID 专用 token 到 tokenizer 词表
- 用自然语言 Prompt 驱动 LLM，发挥其语言理解和指令跟随能力
- forward 走标准 Causal LM 流程 (tokenizer → model → lm_head)
- generate 用原生 model.generate()，解析输出提取语义 ID
- 训练 loss 仅计算目标语义 ID token 位置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


class Qwen3GenerativeRec(nn.Module):
    """Qwen3-0.6B Backbone + Prompt Template 生成式推荐模型."""

    # ── Prompt 模板 ─────────────────────────────────
    TRAIN_PROMPT = (
        "用户按时间顺序点击过以下商品：\n"
        "{history}\n"
        "他接下来点击了：{target}"
    )
    INFER_PROMPT = (
        "用户按时间顺序点击过以下商品：\n"
        "{history}\n"
        "请预测用户下一个可能点击的商品："
    )
    ITEM_FMT = "<s0_{s0}><s1_{s1}><s2_{s2}><s3_{s3}>"

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-0.6B",
        vocab_size: int = 256,
        num_quantizers: int = 4,
        max_seq_len: int = 2048,
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

        # ── 1. 加载 Tokenizer ──────────────────────────
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        # 新增语义 ID 专用 token
        special_tokens = []
        for i in range(num_quantizers):
            for j in range(vocab_size):
                special_tokens.append(f"<s{i}_{j}>")
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 双向映射: token_id ↔ (quantizer_idx, value)
        self._id_to_sem: Dict[int, Tuple[int, int]] = {}
        self._sem_to_id: Dict[Tuple[int, int], int] = {}
        for i in range(num_quantizers):
            for j in range(vocab_size):
                tok = f"<s{i}_{j}>"
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                self._id_to_sem[tid] = (i, j)
                self._sem_to_id[(i, j)] = tid

        # ── 2. 加载 Qwen3 Backbone ────────────────────
        from transformers import AutoModelForCausalLM, AutoConfig

        self.config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.hidden_size = self.config.hidden_size          # 1024
        self.num_layers = self.config.num_hidden_layers      # 28
        self.num_kv_heads = self.config.num_key_value_heads  # 8 (GQA)
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.pad_token_id = self.tokenizer.pad_token_id

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=None,
        )

        # 扩展词表 embedding 以容纳新增 token
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        # ── 3. LoRA 微调 ──────────────────────────────
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

        # ── 3.5 梯度检查点 (换显存) ──────────────────
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()
            print("[Qwen3GenerativeRec] Gradient checkpointing enabled")

        # ── 3.6 缓存 transformer + lm_head (绕过全序列 logits) ─
        if self.lora_enabled:
            self._transformer = self.base_model.model.model
        else:
            self._transformer = self.base_model.model
        self._lm_head = self.base_model.lm_head

        # ── 4. 统计 ───────────────────────────────────
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"[Qwen3GenerativeRec] Vocab: {len(self.tokenizer)} "
            f"(+{num_quantizers * vocab_size} sem tokens) | "
            f"Trainable: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    # ═════════════════════════════════════════════════════════
    #  Prompt 构造
    # ═════════════════════════════════════════════════════════

    def _format_one(self, sem_ids: List[int]) -> str:
        """将一个语义 ID 格式化为 token 字符串."""
        return self.ITEM_FMT.format(
            s0=sem_ids[0], s1=sem_ids[1],
            s2=sem_ids[2], s3=sem_ids[3],
        )

    def _format_history(self, items: List[List[int]]) -> str:
        """将历史序列格式化为逗号分隔的 token 串."""
        return ",".join(self._format_one(s) for s in items)

    # ═════════════════════════════════════════════════════════
    #  前向传播 (训练)
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
        """训练前向传播.

        Args:
            semantic_ids:  [batch, max_seq_len, 4] — 用户历史序列 (含 padding)
            attention_mask: [batch, max_seq_len]     — 有效位置 mask
            labels:         [batch, max_seq_len, 4] — 目标序列 (shifted, 含 padding)
            use_cache:      是否返回 past_key_values

        Returns:
            logits:          [batch, prompt_len, vocab_size]
            loss:            标量 loss (仅目标位置)
            past_key_values: None
        """
        batch_size = semantic_ids.shape[0]
        device = semantic_ids.device

        # ── 提取每样本的有效历史 + 目标 ──────────────
        train_prompts: List[str] = []
        for b in range(batch_size):
            # 有效历史
            if attention_mask is not None:
                hist_len = int(attention_mask[b].sum().item())
                history = semantic_ids[b, :hist_len].cpu().tolist()
            else:
                history = semantic_ids[b].cpu().tolist()
                history = [h for h in history
                           if not all(v == self.pad_token_id for v in h)]

            # 目标: labels 最后一个有效位置
            if labels is not None and attention_mask is not None:
                lab_len = int(attention_mask[b].sum().item())
                target = labels[b, max(0, lab_len - 1)].cpu().tolist()
            elif labels is not None:
                target = labels[b, -1].cpu().tolist()
            else:
                target = history[-1] if history else [0, 0, 0, 0]

            prompt = self.TRAIN_PROMPT.format(
                history=self._format_history(history),
                target=self._format_one(target),
            )
            train_prompts.append(prompt)

        # ── Tokenize ─────────────────────────────────
        encoded = self.tokenizer(
            train_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        ).to(device)

        input_ids = encoded["input_ids"]   # [B, prompt_len]
        attn = encoded["attention_mask"]

        # ── 找每样本目标 token 位置 ──────────────────
        target_positions: List[List[int]] = []   # [batch, list of 4 positions]
        for b in range(batch_size):
            ids = input_ids[b].tolist()
            sem_pos = [p for p, tid in enumerate(ids) if tid in self._id_to_sem]
            # 最后 num_quantizers 个语义 token = 目标位置
            target_positions.append(sem_pos[-self.num_quantizers:]
                                    if len(sem_pos) >= self.num_quantizers else [])

        # ── Transformer forward (跳过 lm_head) ───────
        embed = self.base_model.get_input_embeddings()
        inputs_embeds = embed(input_ids)
        transformer_out = self._transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            use_cache=False,
        )
        hidden_states = transformer_out[0]  # [B, prompt_len, 1024]

        # ── 只算目标位置的 lm_head + loss ───────────
        # causal LM: hidden_states[pos-1] → lm_head → 预测 token[pos]
        target_logits_list = []
        target_labels_list = []
        for b in range(batch_size):
            pos = target_positions[b]
            if pos and pos[0] > 0:  # 目标前必须有上下文 (pos[0]>0)
                h = hidden_states[b, [p-1 for p in pos]]   # [4, 1024]
                logits_b = self._lm_head(h)                 # [4, vocab]
                target_logits_list.append(logits_b)
                target_labels_list.append(input_ids[b, pos])  # [4]

        if target_logits_list:
            all_logits = torch.cat(target_logits_list, dim=0)  # [N*4, vocab]
            all_labels = torch.cat(target_labels_list, dim=0)  # [N*4]
            loss = F.cross_entropy(all_logits, all_labels)
        else:
            all_logits = torch.zeros(0, len(self.tokenizer), device=device)
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        return all_logits, loss, None

    # ═════════════════════════════════════════════════════════
    #  自回归生成 (推理)
    # ═════════════════════════════════════════════════════════

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 40,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """自回归生成推荐.

        Args:
            input_ids:   [batch, history_len, 4] — 用户历史语义 ID
            max_new_tokens: 生成 token 数上限
            temperature: 采样温度
            top_k:       top-k 采样

        Returns:
            [batch, max_items, 4] — 生成的语义 ID (不足则补 0)
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # ── 构造推理 Prompt ──────────────────────────
        infer_prompts: List[str] = []
        for b in range(batch_size):
            history = input_ids[b].cpu().tolist()
            history = [h for h in history
                       if not all(v == self.pad_token_id for v in h)]
            prompt = self.INFER_PROMPT.format(
                history=self._format_history(history),
            )
            infer_prompts.append(prompt)

        encoded = self.tokenizer(
            infer_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        ).to(device)

        # ── 调用原生 generate ────────────────────────
        generated = self.base_model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k else 50,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # ── 解析生成结果 ─────────────────────────────
        results: List[List[List[int]]] = []
        prompt_len = encoded["input_ids"].shape[1]

        for b in range(batch_size):
            new_tokens = generated[b, prompt_len:].tolist()
            items = self._parse_output(new_tokens)
            results.append(items)

        # ── 填充为规整张量 ──────────────────────────
        max_items = max((len(r) for r in results), default=1)
        padded = torch.zeros(
            batch_size, max_items, self.num_quantizers,
            dtype=torch.long, device=device,
        )
        for b, items in enumerate(results):
            for i, sem in enumerate(items):
                if i < max_items:
                    padded[b, i] = torch.tensor(sem, device=device)

        return padded

    def _parse_output(self, token_ids: List[int]) -> List[List[int]]:
        """从生成 token 流中提取语义 ID 四元组.

        扫描连续的 4 个 token，检查是否构成有效的 <s0_X><s1_Y><s2_Z><s3_W> 序列.
        """
        items: List[List[int]] = []
        i = 0
        n = len(token_ids)
        while i <= n - self.num_quantizers:
            window = token_ids[i:i + self.num_quantizers]
            parsed = [self._id_to_sem.get(tid) for tid in window]
            if all(p is not None for p in parsed):
                layers = [p[0] for p in parsed]
                values = [p[1] for p in parsed]
                if layers == list(range(self.num_quantizers)):
                    items.append(values)
                    i += self.num_quantizers
                    continue
            i += 1
        return items

    # ═════════════════════════════════════════════════════════
    #  工具方法
    # ═════════════════════════════════════════════════════════

    def _sample_token(self, logits, temperature, top_k):
        """兼容旧接口，生成时由 model.generate 内部处理."""
        pass

    def merge_lora(self) -> None:
        """推理前将 LoRA 权重合并回基座."""
        if not self.lora_enabled:
            return
        if hasattr(self.base_model, "merge_and_unload"):
            peft_model = self.base_model
            self.base_model = peft_model.merge_and_unload()
            del peft_model
            torch.cuda.empty_cache()
            self.lora_enabled = False
            print("[Qwen3GenerativeRec] LoRA merged into base model")


__all__ = ["Qwen3GenerativeRec"]
