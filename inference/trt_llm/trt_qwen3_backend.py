# -*- coding: utf-8 -*-
"""TRT-LLM Qwen3 后端 — 用 ModelRunnerCpp + 后处理解析替代 generate() 约束解码."""

import torch
from typing import List


class TRTQwen3Backend:
    """封装 TRT-LLM ModelRunnerCpp，对外提供 generate() → 解析后语义ID."""

    def __init__(self, engine_dir: str, tokenizer, num_quantizers: int = 4,
                 vocab_size: int = 256, temperature: float = 0.7, top_k: int = 50,
                 max_tokens_in_paged_kv_cache: int = None,
                 scheduler_policy: str = "max_utilization"):
        from tensorrt_llm.runtime import ModelRunnerCpp
        from tensorrt_llm.bindings.executor import SamplingConfig

        # ── 构造 scheduler_config (需 MONKEY-PATCH model_runner_cpp.py) ──
        scheduler_config = None
        try:
            from tensorrt_llm.llmapi import SchedulerConfig, CapacitySchedulerPolicy
            policy = (
                CapacitySchedulerPolicy.MAX_UTILIZATION
                if scheduler_policy == "max_utilization"
                else CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
            )
            scheduler_config = SchedulerConfig(capacity_scheduler_policy=policy)
            print(f"[TRTQwen3Backend] Scheduler policy: {scheduler_policy}, "
                  f"max_kv_tokens={max_tokens_in_paged_kv_cache}")
        except Exception as e:
            print(f"[TRTQwen3Backend] SchedulerConfig unavailable: {e}")

        self.runner = ModelRunnerCpp.from_dir(
            engine_dir,
            scheduler_config=scheduler_config,
            max_tokens_in_paged_kv_cache=max_tokens_in_paged_kv_cache,
        )
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
        self._seed = 42  # multi-sample: incremented per call

        print(f"[TRTQwen3Backend] Engine loaded, id_to_sem={len(self._id_to_sem)} tokens")

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32,
                 num_samples: int = 8) -> torch.Tensor:
        """TRT generate → 多轮采样 + 去重 → [batch, max_items, 4] 张量.

        TRT-LLM generate() 不支持 per-step 约束解码, 用多轮采样弥补命中率.
        4090D 上每轮 ~10ms, 8轮 ≈ 80ms.

        Args:
            input_ids: [batch, history_len, 4]
            max_new_tokens: 每轮生成上限 (TRT 引擎实际生成到 max_seq_len)
            num_samples: 采样轮数

        Returns:
            [batch, max_items, 4] 去重后的语义 ID (不足补 0)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        all_results = []
        for b in range(batch_size):
            # ── 构造 prompt (tokenize 一次) ──
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
            input_id_list = [encoded["input_ids"][0]]

            # ── 多轮采样 + 去重 ──
            from tensorrt_llm.bindings.executor import SamplingConfig
            seen = set()
            items = []
            for s in range(num_samples):
                cfg = SamplingConfig(
                    temperature=self.sampling_config.temperature,
                    top_k=self.sampling_config.top_k,
                    seed=self._seed + s,
                )
                outputs = self.runner.generate(
                    input_id_list,
                    sampling_config=cfg,
                    max_new_tokens=max_new_tokens,
                    end_id=self.eos_id,
                    pad_id=self.pad_id,
                )
                new_tokens = outputs[0][0][prompt_len:].tolist()
                for sem in self._parse_output(new_tokens):
                    key = tuple(sem)
                    if key not in seen:
                        seen.add(key)
                        items.append(sem)
            self._seed += num_samples
            all_results.append(items)

        # ── 填充 ──
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
