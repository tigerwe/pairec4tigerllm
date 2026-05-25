# -*- coding: utf-8 -*-
"""KV Cache Manager: GPU HBM LRU 缓存 + DataSystem 持久化."""

import io
import time
import threading
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np
import torch


class KVCacheManager:
    """管理 GPU HBM 中的 KV Cache，协调 DataSystem 读写.

    三层缓存架构:
      1. HBM LRU (GPU 显存)  — 最快，容量有限
      2. DataSystem (Host/网络) — 持久化，TTL 600s
      3. Prefill 重算         — 最慢，但保证正确

    使用方式:
      manager = KVCacheManager(num_layers=28, num_kv_heads=8, head_dim=64)
      past_kv, source, ms = manager.query(user_id, history_hash)
      manager.store(user_id, history_hash, past_kv, async_write=True)
    """

    def __init__(
        self,
        num_layers: int = 28,
        num_kv_heads: int = 8,
        head_dim: int = 64,
        max_seq_len: int = 512,
        hbm_capacity: int = 50,
        ds_client=None,
        key_prefix: str = "pairec4tigerllm:kv",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.ds = ds_client
        self.key_prefix = key_prefix

        # HBM 缓存: OrderedDict for LRU
        # key: "{user_id}:{history_hash}"
        # value: {"past_kv": Tuple, "timestamp": float}
        self.hbm_cache: OrderedDict[str, Dict] = OrderedDict()
        self.hbm_capacity = hbm_capacity
        self._lock = threading.RLock()

    # ── 公共接口 ─────────────────────────────────────────

    def query(
        self, user_id: str, history_hash: str
    ) -> Tuple[Optional[Tuple], str, float]:
        """查询 KV Cache.

        Returns:
            (past_kv_tuple, source, lookup_ms)
            source: "hbm_hit" | "ds_hit" | "miss"
        """
        local_key = f"{user_id}:{history_hash}"
        t0 = time.perf_counter()

        # 1. 查 HBM LRU
        with self._lock:
            if local_key in self.hbm_cache:
                self.hbm_cache.move_to_end(local_key)
                entry = self.hbm_cache[local_key]
                entry["timestamp"] = time.time()
                return (
                    entry["past_kv"],
                    "hbm_hit",
                    (time.perf_counter() - t0) * 1000,
                )

        # 2. 查 DataSystem
        if self.ds is not None:
            try:
                raw = self._ds_get(local_key)
                if raw is not None:
                    past_kv = self._deserialize(raw)
                    self._hbm_put(local_key, past_kv)
                    return past_kv, "ds_hit", (time.perf_counter() - t0) * 1000
            except Exception as e:
                print(f"[KVCacheManager] DataSystem query failed: {e}")

        return None, "miss", (time.perf_counter() - t0) * 1000

    def store(
        self,
        user_id: str,
        history_hash: str,
        past_kv: Tuple,
        async_write: bool = True,
    ):
        """存储 KV Cache 到 HBM 和 DataSystem."""
        local_key = f"{user_id}:{history_hash}"

        # 同步写入 HBM
        self._hbm_put(local_key, past_kv)

        # 异步写入 DataSystem（不阻塞推理主路径）
        if self.ds is not None:
            if async_write:
                threading.Thread(
                    target=self._ds_put,
                    args=(local_key, past_kv),
                    daemon=True,
                ).start()
            else:
                self._ds_put(local_key, past_kv)

    def estimate_size_mb(self, seq_len: int, batch_size: int = 1) -> float:
        """估算一条 KV Cache 的显存占用 (MB)."""
        elem_bytes = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        per_layer = (
            2  # K + V
            * batch_size
            * self.num_kv_heads
            * seq_len
            * self.head_dim
            * elem_bytes
        )
        return (self.num_layers * per_layer) / (1024**2)

    # ── 内部方法 ─────────────────────────────────────────

    def _hbm_put(self, local_key: str, past_kv: Tuple):
        with self._lock:
            if local_key in self.hbm_cache:
                self.hbm_cache.move_to_end(local_key)
                return
            # LRU 淘汰
            while len(self.hbm_cache) >= self.hbm_capacity:
                evicted_key, _ = self.hbm_cache.popitem(last=False)
                print(f"[KVCacheManager] HBM evicted: {evicted_key}")
            self.hbm_cache[local_key] = {
                "past_kv": past_kv,
                "timestamp": time.time(),
            }

    def _serialize(self, past_kv: Tuple) -> bytes:
        """将 past_key_values 序列化为 bytes (np.savez_compressed)."""
        arrays = {}
        for i, (k, v) in enumerate(past_kv):
            # k, v: [batch, num_kv_heads, seq_len, head_dim]
            arrays[f"k_{i}"] = k.cpu().numpy()
            arrays[f"v_{i}"] = v.cpu().numpy()
        arrays["meta"] = np.array(
            [len(past_kv), past_kv[0][0].size(-2), past_kv[0][0].size(-1)]
        )
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return buf.getvalue()

    def _deserialize(self, raw) -> Tuple:
        """将 bytes 反序列化为 past_key_values."""
        buf = io.BytesIO(raw if isinstance(raw, bytes) else raw.encode())
        data = np.load(buf)
        meta = data["meta"]
        num_layers, seq_len, head_dim = (
            int(meta[0]),
            int(meta[1]),
            int(meta[2]),
        )

        past_kv = []
        for i in range(num_layers):
            k = torch.from_numpy(data[f"k_{i}"]).to(
                device=self.device, dtype=self.dtype
            )
            v = torch.from_numpy(data[f"v_{i}"]).to(
                device=self.device, dtype=self.dtype
            )
            past_kv.append((k, v))
        return tuple(past_kv)

    # ── DataSystem Bridge ──────────────────────────────────

    def _ds_key(self, local_key: str) -> str:
        return f"{self.key_prefix}:{local_key}"

    def _ds_get(self, local_key: str):
        """从 DataSystem 读取 KV Cache (bytes)."""
        key = self._ds_key(local_key)
        try:
            vals = self.ds.kv().get([key], convert_to_str=False)
            if vals and vals[0] is not None:
                return vals[0]  # bytes
        except Exception:
            pass
        return None

    def _ds_put(self, local_key: str, past_kv: Tuple):
        """写入 DataSystem (带 TTL)."""
        try:
            payload = self._serialize(past_kv)
            self.ds.kv().set(
                self._ds_key(local_key), payload, ttl_second=600
            )
        except Exception as e:
            print(f"[KVCacheManager] DataSystem write failed: {e}")
