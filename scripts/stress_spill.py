#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
压力测试 DataSystem Worker 的自动 Spill 机制.

通过 ObjectClient 循环创建大 Buffer → publish 直到 worker 共享内存满,
触发 C++ 层自动 spill 到磁盘, 然后验证全部数据可正确读回.

用法:
  python scripts/stress_spill.py --count 100 --size-mb 2
  python scripts/stress_spill.py --count 50 --size-mb 4 --verify-samples 10

验证要点:
  1. 所有 publish 是否成功
  2. 随机采样读回, 数据是否匹配 (证明 spill → onboard 闭环)
  3. worker 日志中是否有 spill 记录
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import random
import json
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── 参数 ──────────────────────────────────────────────

DEFAULT_HOST = os.environ.get("DATASYSTEM_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("DATASYSTEM_PORT", "31501"))
PREFIX = "pairec4tigerllm:stress_spill"


def make_test_data(size_mb: int, seed: int) -> bytes:
    """生成可验证的确定性测试数据 (含 seed 和 SHA256)."""
    rng = random.Random(seed)
    header = f"STRESS:{seed}:{size_mb}MB:".encode()
    body_size = max(0, size_mb * 1024 * 1024 - len(header) - 64)
    body = bytes(rng.getrandbits(8) for _ in range(body_size))
    data = header + body
    h = hashlib.sha256(data).hexdigest().encode()
    return data + b":" + h


def verify_data(data: bytes, seed: int, size_mb: int) -> bool:
    """验证数据完整性."""
    h = hashlib.sha256(data[:-65]).hexdigest().encode()
    expected_h = data[-64:]
    return h == expected_h


def fmt_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f}GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f}MB"
    if n >= 1024:
        return f"{n / 1024:.0f}KB"
    return f"{n}B"


def fmt_dt(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60000:
        return f"{ms / 1000:.1f}s"
    return f"{ms / 60000:.1f}m"


def main():
    parser = argparse.ArgumentParser(description="DataSystem Spill 压力测试")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--count", type=int, default=50,
                        help="创建 Buffer 数量 (default: 50)")
    parser.add_argument("--size-mb", type=float, default=2.0,
                        help="每个 Buffer 大小 MB (default: 2)")
    parser.add_argument("--verify-samples", type=int, default=10,
                        help="随机采样验证数量 (default: 10)")
    parser.add_argument("--keep", action="store_true",
                        help="保留数据不清理")
    parser.add_argument("--json-output", default="stress_spill_report.json")
    parser.add_argument("--no-json", action="store_true")
    args = parser.parse_args()

    # ── 导入 ──────────────────────────────────────
    try:
        from yr.datasystem import DsClient
    except ImportError as e:
        print(f"FATAL: yr.datasystem not installed: {e}")
        return 1

    # ── 连接 ──────────────────────────────────────
    print(f"连接 DataSystem {args.host}:{args.port} ...")
    try:
        ds = DsClient(args.host, args.port)
        ds.init()
        print("  OK\n")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    obj_client = ds.object()
    kv_client = ds.kv()

    total_size = args.count * args.size_mb
    print(f"配置: {args.count} 个 Buffer × {args.size_mb}MB = {total_size:.0f}MB 总数据\n")

    # ── 写入阶段 ──────────────────────────────────
    keys: List[str] = []
    seeds: List[int] = []
    write_times: List[float] = []

    t_total = time.perf_counter()
    failed = 0

    for i in range(args.count):
        key = f"{PREFIX}:{int(time.time() * 1000)}:{i:04d}"
        seed = random.randint(0, 2**31)

        try:
            # 生成数据
            t_data = time.perf_counter()
            data = make_test_data(int(args.size_mb), seed)

            # create → copy → publish
            t0 = time.perf_counter()
            buf = obj_client.create(key, len(data))
            buf.memory_copy(data)
            buf.publish()
            dt = (time.perf_counter() - t0) * 1000
            write_times.append(dt)

            keys.append(key)
            seeds.append(seed)

            if (i + 1) % max(1, args.count // 10) == 0:
                pct = (i + 1) / args.count * 100
                avg_ms = sum(write_times) / len(write_times) * 1000 if write_times else 0
                print(f"  写入 {i+1}/{args.count} ({pct:.0f}%)  "
                      f"avg={(avg_ms * 1000):.0f}μs/publish  "
                      f"总数据={fmt_bytes((i+1) * int(args.size_mb) * 1024 * 1024)}")

        except Exception as e:
            failed += 1
            print(f"  [{i}] FAIL: {e}")
            # 可能是内存满了, 继续试
            if i < 5:
                print("  → 早期失败, 退出")
                return 1

    write_total_s = time.perf_counter() - t_total
    success = args.count - failed
    avg_ms = sum(write_times) / len(write_times) * 1000 if write_times else 0

    print(f"\n写入完成: {success}/{args.count} 成功, {failed} 失败  "
          f"耗时 {fmt_dt(write_total_s * 1000)}  "
          f"吞吐 {fmt_bytes(success * int(args.size_mb) * 1024 * 1024 / max(write_total_s, 0.001))}/s")
    print(f"publish 均值 {avg_ms:.0f}μs")

    if success == 0:
        print("FATAL: 全部写入失败")
        return 1

    # ── 验证阶段 (随机采样) ────────────────────────
    sample_count = min(args.verify_samples, success)
    sample_indices = sorted(random.sample(range(success), sample_count))

    print(f"\n验证 {sample_count} 个随机样本 ...")
    verify_ok = 0
    verify_fail = 0
    verify_times: List[float] = []

    for idx in sample_indices:
        key = keys[idx]
        expected_seed = seeds[idx]
        expected_size_mb = args.size_mb

        try:
            t0 = time.perf_counter()
            buf_list = obj_client.get([key], timeout_ms=10000)
            dt = (time.perf_counter() - t0) * 1000
            verify_times.append(dt)

            if buf_list and buf_list[0] is not None:
                raw_data = memoryview(buf_list[0].immutable_data()).tobytes()
                if verify_data(raw_data, expected_seed, int(expected_size_mb)):
                    verify_ok += 1
                    source = "ok"
                else:
                    verify_fail += 1
                    source = "CORRUPT"
            else:
                verify_fail += 1
                source = "None"
                dt = -1

            print(f"  [{idx:04d}] {key[-25:]}: {source}  ({dt:.0f}ms)" if dt > 0 else
                  f"  [{idx:04d}] {key[-25:]}: {source}")

        except Exception as e:
            verify_fail += 1
            print(f"  [{idx:04d}] {key[-25:]}: FAIL — {e}")

    avg_get_ms = sum(verify_times) / len(verify_times) if verify_times else 0

    # ── 清理 ──────────────────────────────────────
    if not args.keep:
        print(f"\n清理 {success} 个 key ...")
        t0 = time.perf_counter()
        # 分批删除 (ObjectClient 无 delete, 用 KVClient)
        batch = 100
        for i in range(0, success, batch):
            batch_keys = keys[i:i + batch]
            try:
                kv_client.delete(batch_keys)
            except Exception as e:
                print(f"  清理 [{i}:{i+batch}] 失败: {e}")
        print(f"  清理完成 ({fmt_dt((time.perf_counter() - t0) * 1000)})")

    # ── 报告 ──────────────────────────────────────
    total_mb = success * args.size_mb
    report = {
        "config": {"count": args.count, "size_mb": args.size_mb,
                   "total_expected_mb": total_mb},
        "write": {"success": success, "failed": failed,
                  "total_s": round(write_total_s, 1),
                  "throughput_mb_s": round(total_mb / max(write_total_s, 0.001), 1),
                  "avg_publish_us": round(avg_ms, 0)},
        "verify": {"samples": sample_count, "ok": verify_ok, "fail": verify_fail,
                   "avg_get_ms": round(avg_get_ms, 1)},
        "spill": {"inferred": total_mb > 100,  # 超过 100MB 很可能触发了 spill
                  "note": "检查 worker 日志中的 spill/eviction 关键字确认"},
        "cleanup": {"kept": args.keep},
    }

    print(f"\n{'='*60}")
    print(f"  写入: {success}/{args.count} OK, {fmt_bytes(total_mb * 1024 * 1024)}  "
          f"{report['write']['throughput_mb_s']:.0f} MB/s")
    print(f"  验证: {verify_ok}/{sample_count} OK, {verify_fail} FAIL  "
          f"avg get={avg_get_ms:.0f}ms")
    spill_likely = total_mb > 100
    print(f"  Spill: {'可能已触发' if spill_likely else '数据量较小, 未必触发'}  "
          f"(总写入 {total_mb:.0f}MB, 检查 worker 日志)")
    print(f"{'='*60}")

    if not args.no_json:
        with open(args.json_output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nJSON → {args.json_output}")

    return 0 if verify_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
