# -*- coding: utf-8 -*-
"""TRT-LLM Qwen3 后端 — 用 ModelRunnerCpp + 后处理解析替代 generate() 约束解码."""

import torch
from typing import List


class TRTQwen3Backend:
    """封装 TRT-LLM ModelRunnerCpp，对外提供 generate() → 解析后语义ID."""

    def __init__(self, engine_dir: str, tokenizer, num_quantizers: int = 4,
                 vocab_size: int = 256, temperature: float = 0.7, top_k: int = 50):
        from tensorrt_llm.runtime import ModelRunnerCpp
        from tensorrt_llm.bindings.executor import SamplingConfig

        self.runner = ModelRunnerCpp.from_dir(engine_dir)
        self.tokenizer = tokenizer
        self.num_quantizers = num_quantizers
        self.vocab_size = vocab_size
        self.sampling_config = SamplingConfig(temperature=temperature, top_k=top_k)

        # ── 构建 id → (layer, value) 映射 (与 Qwen3GenerativeRec._id_to_sem 一致) ──
        self._id_to_sem = {}
        for i in range(num_quantizers):
            for j in range(vocab_size):
                tid = tokenizer.convert_tokens_to_ids(f"<s{i}_{j}>")
                if tid != tokenizer.unk_token_id:
                    self._id_to_sem[tid] = (i, j)

        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id

        print(f"[TRTQwen3Backend] Engine loaded, id_to_sem={len(self._id_to_sem)} tokens")

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32) -> torch.Tensor:
        """TRT generate → 解析语义ID → [batch, max_items, 4] 张量.

        Args:
            input_ids: [batch, history_len, 4] 用户历史语义 ID
            max_new_tokens: 生成 token 数上限 (会被 TRT 忽略, 仅作软上限)

        Returns:
            [batch, max_items, 4] 解析后的语义 ID (不足补 0)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        all_results = []
        for b in range(batch_size):
            # ── 构造推理 prompt ──
            history = input_ids[b].cpu().tolist()
            history = [h for h in history if not all(v == 0 for v in h)]
            history_str = ",".join(
                f"<s0_{s[0]}><s1_{s[1]}><s2_{s[2]}><s3_{s[3]}>" for s in history
            )
            prompt = (
                "用户按时间顺序点击过以下商品：\n"
                f"{history_str}\n"
                "请预测用户下一个可能点击的商品："
            )

            encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                     max_length=2048).to(device)
            prompt_len = encoded["input_ids"].shape[1]

            # ── TRT generate (max_new_tokens 被引擎忽略, 用后处理控制) ──
            outputs = self.runner.generate(
                [encoded["input_ids"][0]],
                sampling_config=self.sampling_config,
                max_new_tokens=max_new_tokens,
                end_id=self.eos_id,
                pad_id=self.pad_id,
            )

            new_tokens = outputs[0][0][prompt_len:].tolist()

            # ── 解析语义 ID (后处理过滤, 替代 prefix_allowed_tokens_fn) ──
            items = self._parse_output(new_tokens)
            all_results.append(items)

        # ── 填充为规整张量 ──
        max_items = max((len(r) for r in all_results), default=1)
        padded = torch.zeros(batch_size, max_items, self.num_quantizers,
                             dtype=torch.long, device=device)
        for b, items in enumerate(all_results):
            for i, sem in enumerate(items):
                if i < max_items:
                    padded[b, i] = torch.tensor(sem, device=device)

        return padded

    def _parse_output(self, token_ids: List[int]) -> List[List[int]]:
        """从生成 token 流中提取语义 ID 四元组 (s0,s1,s2,s3 严格有序)."""
        items = []
        i = 0
        n = len(token_ids)
        while i <= n - self.num_quantizers:
            window = token_ids[i:i + self.num_quantizers]
            parsed = [self._id_to_sem.get(tid) for tid in window]
            if all(p is not None for p in parsed):
                layers = [p[0] for p in parsed]
                if layers == list(range(self.num_quantizers)):
                    values = [p[1] for p in parsed]
                    items.append(values)
                    i += self.num_quantizers
                    continue
            i += 1
        return items
