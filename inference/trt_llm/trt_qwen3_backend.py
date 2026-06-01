# -*- coding: utf-8 -*-
"""TRT-LLM Qwen3 后端 — 用 ModelRunnerCpp + 后处理解析替代 generate() 约束解码."""

import torch
from typing import List


class TRTQwen3Backend:
    """封装 TRT-LLM ModelRunnerCpp，对外提供 generate() → 解析后语义ID."""

    def __init__(self, engine_dir: str, tokenizer, num_quantizers: int = 4,
                 vocab_size: int = 256, temperature: float = 0.7, top_k: int = 50,
                 max_tokens_in_paged_kv_cache: int = None,
                 scheduler_policy: str = "max_utilization",
                 max_input_len: int = 64):
        from tensorrt_llm.runtime import ModelRunnerCpp
        from tensorrt_llm.bindings.executor import SamplingConfig

        if max_tokens_in_paged_kv_cache is None:
            # Keep the pool intentionally small for offload/onboard diagnostics.
            # Production can pass a larger value once the C++ path is verified.
            max_tokens_in_paged_kv_cache = 2048

        # ── 构造 scheduler_config (需 MONKEY-PATCH model_runner_cpp.py) ──
        scheduler_config = None
        try:
            from tensorrt_llm.bindings.executor import (
                CapacitySchedulerPolicy,
                SchedulerConfig,
            )
            normalized_policy = scheduler_policy.lower().replace("-", "_")
            if normalized_policy in ("max_utilization", "max_util", "max"):
                policy = CapacitySchedulerPolicy.MAX_UTILIZATION
            elif normalized_policy in ("guaranteed_no_evict", "no_evict"):
                policy = CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
            else:
                raise ValueError(f"Unsupported scheduler_policy: {scheduler_policy}")
            scheduler_config = SchedulerConfig(policy)
            print(f"[TRTQwen3Backend] Scheduler policy: {scheduler_policy}, "
                  f"max_kv_tokens={max_tokens_in_paged_kv_cache}")
        except Exception as e:
            print(f"[TRTQwen3Backend] SchedulerConfig unavailable: {e}")
            import traceback; traceback.print_exc()

        if scheduler_config is None:
            raise RuntimeError("SchedulerConfig is required for C++ KV offload diagnostics")

        runner_kwargs = {
            "max_tokens_in_paged_kv_cache": max_tokens_in_paged_kv_cache,
            "kv_cache_enable_block_reuse": True,
            "scheduler_config": scheduler_config,
        }
        try:
            self.runner = ModelRunnerCpp.from_dir(engine_dir, **runner_kwargs)
        except TypeError as e:
            if "scheduler_config" in str(e):
                raise RuntimeError(
                    "TensorRT-LLM ModelRunnerCpp.from_dir does not accept "
                    "scheduler_config yet. Patch "
                    "tensorrt_llm/runtime/model_runner_cpp.py to forward it "
                    "into ExecutorConfig before starting the service."
                ) from e
            raise
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
        self.max_input_len = max_input_len
        self._seed = 42  # multi-sample: incremented per call

        print(f"[TRTQwen3Backend] Engine loaded, id_to_sem={len(self._id_to_sem)} tokens, "
              f"max_input_len={self.max_input_len}")

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
            # ── 构造 prompt (逐条截断直到满足引擎 max_input_len) ──
            history = input_ids[b].cpu().tolist()
            history = [h for h in history if not all(v == 0 for v in h)]
            orig_history_len = len(history)
            print(f"[TRT generate] batch={b}, history_len={len(history)}, "
                  f"vocab check: min={min((min(h) for h in history), default=-1)}, "
                  f"max={max((max(h) for h in history), default=-1)}")

            # 截断循环: prompt 超限时去掉最早的历史条目
            encoded = None
            trunc_history_len = len(history)
            while True:
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
                if prompt_len <= self.max_input_len or len(history) <= 1:
                    break
                # 去掉最早的一条历史，保留最近的
                history = history[1:]
                trunc_history_len = len(history)

            if trunc_history_len < orig_history_len:
                print(f"[TRT generate] batch={b}, history truncated {orig_history_len}→{trunc_history_len} "
                      f"(prompt_len={prompt_len}, max_input_len={self.max_input_len})")

            input_id_list = [encoded["input_ids"][0]]

            # 引擎硬限制: max_new_tokens≤32 (96-64), 总长≤95
            # 超32 C++层分配buffer失败卡死不报错, 总长超95抛RuntimeError
            engine_max_new_tokens = 32  # 引擎构建时预留
            engine_max_seq_len = self.max_input_len + engine_max_new_tokens - 1  # = 95
            if max_new_tokens > engine_max_new_tokens:
                max_new_tokens = engine_max_new_tokens
            if prompt_len + max_new_tokens > engine_max_seq_len:
                max_new_tokens = max(8, engine_max_seq_len - prompt_len)
            print(f"[TRT generate] batch={b}, prompt_len={prompt_len}, "
                  f"max_new_tokens={max_new_tokens}, num_samples={num_samples} "
                  f"(engine limits: max_new≤{engine_max_new_tokens}, total≤{engine_max_seq_len})")

            # ── 多轮采样: 合并所有轮次 token 到统一池子再做组合 ──
            from tensorrt_llm.bindings.executor import SamplingConfig
            all_sampled_tokens = []
            for s in range(num_samples):
                cfg = SamplingConfig(
                    temperature=self.sampling_config.temperature,
                    top_k=self.sampling_config.top_k,
                    seed=self._seed + s,
                )
                try:
                    outputs = self.runner.generate(
                        input_id_list,
                        sampling_config=cfg,
                        max_new_tokens=max_new_tokens,
                        end_id=self.eos_id,
                        pad_id=self.pad_id,
                    )
                except Exception as e:
                    print(f"[TRT generate] ERROR in runner.generate (batch={b}, sample={s}, "
                          f"prompt_len={prompt_len}): {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                new_tokens = outputs[0][0][prompt_len:].tolist()
                all_sampled_tokens.extend(new_tokens)

            # 合并后统一组合解析
            parsed, diag = self._parse_output(all_sampled_tokens, return_diag=True)
            print(f"[TRT parse] merged {num_samples} rounds, "
                  f"total_tokens={diag['total']}, "
                  f"valid_sem={diag['valid_sem']}, "
                  f"layer_counts={diag['layer_counts']}, "
                  f"combo_total={diag.get('combo_total', 0)}, "
                  f"matched={len(parsed)}")
            seen = set()
            items = []
            for sem in parsed:
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

    def _parse_output(self, token_ids: List[int],
                       return_diag: bool = False):
        """从生成 token 流中提取语义 ID 四元组 (s0,s1,s2,s3 层序递增).

        宽松匹配: 跳过非语义 token (pad/eos/普通文字),
        在有效语义 token 子序列中寻找 layer 0→1→2→3 连续递增模式.
        原始 token 流中不要求四个 token 位置连续.
        """
        import itertools

        diag = {
            'total': len(token_ids),
            'valid_sem': 0,
            'layer_counts': [0, 0, 0, 0],
        }
        # 第一步: 按层收集所有 value (去重)
        layer_values = {0: set(), 1: set(), 2: set(), 3: set()}
        for tid in token_ids:
            sem = self._id_to_sem.get(tid)
            if sem is not None:
                layer, val = sem
                diag['valid_sem'] += 1
                if layer < self.num_quantizers:
                    diag['layer_counts'][layer] += 1
                    layer_values[layer].add(val)

        # 第二步: 任一层为空, 用 {0} 填充 (默认值, 由 semantic_id_map 验证)
        filled_layers = []
        for l in range(self.num_quantizers):
            if len(layer_values[l]) == 0:
                filled_layers.append(True)
                layer_values[l] = {0}
                diag['layer_counts'][l] = -1  # 负数标记为填充
            else:
                filled_layers.append(False)

        # 第三步: 笛卡尔积组合, 去重后返回
        sorted_layers = [sorted(layer_values[l]) for l in range(self.num_quantizers)]
        combo_count = 1
        for vals in sorted_layers:
            combo_count *= len(vals)
        items = []
        seen = set()
        for combo in itertools.product(*sorted_layers[:3], sorted_layers[3]):
            key = tuple(combo)
            if key not in seen:
                seen.add(key)
                items.append(list(combo))
        diag['combo_total'] = combo_count
        diag['matched'] = len(items)
        diag['fill_layer3'] = filled_layers[3]  # 标记 layer 3 是否被填充

        if return_diag:
            return items, diag
        return items
