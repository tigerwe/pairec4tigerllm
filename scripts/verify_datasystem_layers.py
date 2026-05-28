#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataSystem 三层 API 验证脚本 (KV → Object → Hetero/Device).

用法:
  # 本地（无 GPU / 无 DataSystem 服务时仅检查导入）
  python scripts/verify_datasystem_layers.py

  # 远程 GPU 机器（ARM 4090D / x86 L40S）
  python scripts/verify_datasystem_layers.py --host 127.0.0.1 --port 31501 --gpu

验证层级:
  L1  KVClient      — 简单 key-value (基础层, 已集成)
  L2  ObjectClient  — 共享内存 Buffer + 自动 Spill (中间层, 未集成)
  L3  HeteroClient  — GPU 设备内存直接管理 (设备层, 未集成)
  L3b TransferEngine — 跨节点 GPU 内存 RDMA 传输 (未集成)

输出:
  - 文本报告 (stdout)
  - JSON 报告 (verify_datasystem_layers_report.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── 常量 ──────────────────────────────────────────────

DS_KEY_PREFIX = "pairec4tigerllm:verify"
TEST_KEY = f"{DS_KEY_PREFIX}:{int(time.time())}"
TEST_VALUE = b"Hello from verify_datasystem_layers.py"


# ── 结果模型 ──────────────────────────────────────────

@dataclass
class CheckResult:
    layer: str
    sub_check: str
    status: str        # "OK" | "FAIL" | "SKIP" | "WARN"
    detail: str = ""
    duration_ms: float = 0.0


@dataclass
class Report:
    host: str = ""
    port: int = 0
    gpu_available: bool = False
    cuda_available: bool = False
    checks: List[CheckResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)


# ── 工具函数 ──────────────────────────────────────────

def _check_import(module_path: str) -> CheckResult:
    t0 = time.perf_counter()
    try:
        __import__(module_path)
        return CheckResult("import", module_path, "OK", "", (time.perf_counter() - t0) * 1000)
    except Exception as e:
        return CheckResult("import", module_path, "FAIL", str(e), (time.perf_counter() - t0) * 1000)


# ── 各层验证 ──────────────────────────────────────────

def verify_kv_layer(host: str, port: int) -> List[CheckResult]:
    """L1: KVClient — 简单 key-value."""
    results = []
    ds = None

    # 导入
    r = _check_import("yr.datasystem")
    results.append(r)
    if r.status != "OK":
        return results

    from yr.datasystem import DsClient

    # 连接
    try:
        t0 = time.perf_counter()
        ds = DsClient(host, port)
        ds.init()
        results.append(CheckResult("KV", "connect", "OK", f"{host}:{port}",
                                    (time.perf_counter() - t0) * 1000))
    except Exception as e:
        results.append(CheckResult("KV", "connect", "FAIL", str(e)))
        return results

    kv = ds.kv()

    # Set
    try:
        t0 = time.perf_counter()
        kv.set(TEST_KEY, TEST_VALUE, ttl_second=60)
        results.append(CheckResult("KV", "set", "OK", f"key={TEST_KEY}, size={len(TEST_VALUE)}B",
                                    (time.perf_counter() - t0) * 1000))
    except Exception as e:
        results.append(CheckResult("KV", "set", "FAIL", str(e)))
        return results

    # Get
    try:
        t0 = time.perf_counter()
        vals = kv.get([TEST_KEY], convert_to_str=False)
        if vals and vals[0] == TEST_VALUE:
            results.append(CheckResult("KV", "get", "OK", "roundtrip matched",
                                        (time.perf_counter() - t0) * 1000))
        else:
            got = vals[0][:50] if vals and vals[0] else None
            results.append(CheckResult("KV", "get", "FAIL",
                                        f"value mismatch, got={got}"))
    except Exception as e:
        results.append(CheckResult("KV", "get", "FAIL", str(e)))

    # 清理
    try:
        kv.delete([TEST_KEY])
    except Exception:
        pass

    return results


def verify_object_layer(host: str, port: int) -> List[CheckResult]:
    """L2: ObjectClient — 共享内存 Buffer + Spill."""
    results = []
    ds = None

    from yr.datasystem import DsClient
    try:
        ds = DsClient(host, port)
        ds.init()
    except Exception as e:
        results.append(CheckResult("Object", "connect", "SKIP", f"KV failed: {e}"))
        return results

    obj_client = ds.object()

    # 检查 ObjectClient 方法
    obj_methods = [m for m in dir(obj_client) if not m.startswith('_')]
    results.append(CheckResult("Object", "methods", "OK",
                                f"{len(obj_methods)} methods: {obj_methods[:8]}..."))

    # Create Buffer
    obj_key = f"{DS_KEY_PREFIX}:obj:{int(time.time())}"
    buffer = None
    try:
        t0 = time.perf_counter()
        buffer = obj_client.create(obj_key, 4096)
        results.append(CheckResult("Object", "create", "OK",
                                    f"key={obj_key}, size=4096B",
                                    (time.perf_counter() - t0) * 1000))
    except Exception as e:
        results.append(CheckResult("Object", "create", "FAIL", str(e)))
        return results

    # MemoryCopy
    try:
        t0 = time.perf_counter()
        buffer.memory_copy(TEST_VALUE)
        results.append(CheckResult("Object", "memory_copy", "OK",
                                    f"wrote {len(TEST_VALUE)}B",
                                    (time.perf_counter() - t0) * 1000))
    except Exception as e:
        results.append(CheckResult("Object", "memory_copy", "FAIL", str(e)))

    # Publish
    try:
        t0 = time.perf_counter()
        buffer.publish()
        results.append(CheckResult("Object", "publish", "OK", "",
                                    (time.perf_counter() - t0) * 1000))
    except Exception as e:
        results.append(CheckResult("Object", "publish", "FAIL", str(e)))

    # Get back
    try:
        t0 = time.perf_counter()
        buf2 = obj_client.get(obj_key)
        data = memoryview(buf2.immutable_data()).tobytes()
        if data[:len(TEST_VALUE)] == TEST_VALUE:
            results.append(CheckResult("Object", "get_roundtrip", "OK",
                                        "data matched",
                                        (time.perf_counter() - t0) * 1000))
        else:
            results.append(CheckResult("Object", "get_roundtrip", "FAIL",
                                        f"data mismatch, got {data[:20]}"))
    except Exception as e:
        results.append(CheckResult("Object", "get_roundtrip", "FAIL", str(e)))

    # 清理
    try:
        obj_client.delete(obj_key)
    except Exception:
        pass

    return results


def verify_hetero_layer(host: str, port: int, gpu: bool) -> List[CheckResult]:
    """L3: HeteroClient — GPU 设备内存直接管理."""
    results = []
    ds = None

    from yr.datasystem import DsClient
    try:
        ds = DsClient(host, port)
        ds.init()
    except Exception as e:
        results.append(CheckResult("Hetero", "connect", "SKIP", f"KV failed: {e}"))
        return results

    hetero = ds.hetero()

    # 检查 HeteroClient 方法
    hetero_methods = [m for m in dir(hetero) if not m.startswith('_')]
    results.append(CheckResult("Hetero", "methods", "OK",
                                f"{len(hetero_methods)} methods: {hetero_methods[:10]}..."))

    # 检查关键方法是否存在
    key_methods = ["create_device_obj", "publish_device_obj", "get_device_obj",
                   "async_get_device_obj", "delete_device_obj"]
    for m in key_methods:
        if hasattr(hetero, m):
            results.append(CheckResult("Hetero", f"has_{m}", "OK", ""))
        else:
            results.append(CheckResult("Hetero", f"has_{m}", "FAIL", "method not found"))

    # GPU 设备对象测试（仅当 GPU 可用时）
    if not gpu:
        results.append(CheckResult("Hetero", "device_obj", "SKIP",
                                    "use --gpu to enable GPU device test"))
        return results

    try:
        import torch
        if not torch.cuda.is_available():
            results.append(CheckResult("Hetero", "device_obj", "SKIP",
                                        "CUDA not available"))
            return results
    except ImportError:
        results.append(CheckResult("Hetero", "device_obj", "SKIP",
                                    "torch not installed"))
        return results

    # ── 构造模拟 paged KV block 的 DeviceBlobList ──
    try:
        from yr.datasystem import Blob, DeviceBlobList

        # 分配一个小的 GPU tensor 模拟一个 KV block
        block_size = 64 * 64  # 模拟: num_kv_heads=8, head_dim=64, seq=8
        gpu_tensor = torch.zeros(block_size, dtype=torch.bfloat16, device="cuda")
        ptr = gpu_tensor.data_ptr()

        blob = Blob()
        blob.ptr = ptr
        blob.size = block_size * 2  # bfloat16 = 2 bytes

        dev_blob_list = DeviceBlobList()
        dev_blob_list.blobs = [blob]
        dev_blob_list.device_id = 0

        results.append(CheckResult("Hetero", "blob_construct", "OK",
                                    f"ptr={hex(ptr)}, size={blob.size}B"))

        # 尝试创建设备对象
        dev_key = f"{DS_KEY_PREFIX}:dev:{int(time.time())}"
        try:
            t0 = time.perf_counter()
            device_buffer = hetero.create_device_obj(dev_key, dev_blob_list)
            results.append(CheckResult("Hetero", "create_device_obj", "OK",
                                        f"key={dev_key}",
                                        (time.perf_counter() - t0) * 1000))
            # 发布
            try:
                t0 = time.perf_counter()
                hetero.publish_device_obj(dev_key, device_buffer)
                results.append(CheckResult("Hetero", "publish_device_obj", "OK",
                                            "",
                                            (time.perf_counter() - t0) * 1000))
            except Exception as e:
                results.append(CheckResult("Hetero", "publish_device_obj", "FAIL",
                                            str(e)))

            # Onboard 回读
            try:
                t0 = time.perf_counter()
                dst_buffer = hetero.get_device_obj([dev_key], device_buffer, 5000)
                results.append(CheckResult("Hetero", "get_device_obj", "OK",
                                            f"onboard success",
                                            (time.perf_counter() - t0) * 1000))
            except Exception as e:
                results.append(CheckResult("Hetero", "get_device_obj", "FAIL",
                                            str(e)))

            # 清理
            try:
                hetero.delete_device_obj(dev_key)
            except Exception:
                pass

        except Exception as e:
            results.append(CheckResult("Hetero", "create_device_obj", "FAIL",
                                        str(e)))

        # 释放 GPU tensor
        del gpu_tensor

    except ImportError as e:
        results.append(CheckResult("Hetero", "blob_import", "FAIL",
                                    f"Blob/DeviceBlobList not in yr.datasystem: {e}"))
    except Exception as e:
        results.append(CheckResult("Hetero", "blob_construct", "FAIL",
                                    str(e)))

    return results


def verify_transfer_engine(host: str, port: int) -> List[CheckResult]:
    """L3b: TransferEngine — 跨节点 GPU 内存 RDMA 传输."""
    results = []

    try:
        from yr.datasystem import TransferEngine, Result, ErrorCode  # noqa: F401
        results.append(CheckResult("TransferEngine", "import", "OK",
                                    "TransferEngine / Result / ErrorCode available"))
    except (ImportError, AttributeError) as e:
        results.append(CheckResult("TransferEngine", "import", "SKIP",
                                    f"not compiled in this build: {e}"))
        return results

    # 检查 TransferEngine 方法
    te_methods = [m for m in dir(TransferEngine) if not m.startswith('_')]
    results.append(CheckResult("TransferEngine", "methods", "OK",
                                f"{te_methods}"))

    results.append(CheckResult("TransferEngine", "full_test", "SKIP",
                                "requires multi-GPU / cross-node setup"))

    return results


def verify_env_info() -> List[CheckResult]:
    """环境信息."""
    results = []

    # Python
    results.append(CheckResult("Env", "python", "OK", sys.version.split()[0]))

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            results.append(CheckResult("Env", "cuda", "OK",
                                        f"torch {torch.__version__}, "
                                        f"CUDA {torch.version.cuda}, "
                                        f"GPU {torch.cuda.get_device_name(0)}, "
                                        f"mem {torch.cuda.get_device_properties(0).total_mem // 1024**3}GB"))
        else:
            results.append(CheckResult("Env", "cuda", "SKIP", "not available"))
    except ImportError:
        results.append(CheckResult("Env", "cuda", "SKIP", "torch not installed"))

    # DataSystem 版本
    try:
        import yr.datasystem
        ds_path = os.path.dirname(yr.datasystem.__file__)
        # 尝试获取版本
        version = getattr(yr.datasystem, '__version__', 'unknown')
        results.append(CheckResult("Env", "datasystem", "OK",
                                    f"version={version}, path={ds_path}"))
    except ImportError:
        results.append(CheckResult("Env", "datasystem", "FAIL", "not installed"))

    # LD_PRELOAD
    ld_preload = os.environ.get("LD_PRELOAD", "")
    if ld_preload:
        so_list = ld_preload.split(":")
        so_short = [s.split("/")[-1] for s in so_list if s]
        results.append(CheckResult("Env", "LD_PRELOAD", "OK", ", ".join(so_short)))
    else:
        results.append(CheckResult("Env", "LD_PRELOAD", "WARN", "not set (may need stub_gpu / block_ds_consumer)"))

    return results


# ── 报告输出 ──────────────────────────────────────────

def print_report(report: Report):
    """打印彩色终端报告."""
    COLOR = {"OK": "\033[32m", "FAIL": "\033[31m", "SKIP": "\033[33m", "WARN": "\033[35m"}
    RESET = "\033[0m"

    header = f"DataSystem 三层 API 验证报告"
    print("=" * 70)
    print(f"  {header}")
    print(f"  Host: {report.host}:{report.port}")
    print(f"  GPU:  {'available' if report.gpu_available else 'not available'}")
    print("=" * 70)

    current_layer = ""
    for c in report.checks:
        if c.layer != current_layer:
            current_layer = c.layer
            print(f"\n── {current_layer} ──")

        flag = COLOR.get(c.status, "")(c.status) + RESET
        line = f"  [{flag}] {c.sub_check}"
        if c.detail:
            line += f"  — {c.detail}"
        if c.duration_ms > 0.5:
            line += f"  ({c.duration_ms:.1f}ms)"
        print(line)

    print("\n" + "=" * 70)
    ok = report.summary.get("OK", 0)
    fail = report.summary.get("FAIL", 0)
    skip = report.summary.get("SKIP", 0)
    warn = report.summary.get("WARN", 0)
    total = ok + fail + skip + warn
    print(f"  总计: {total} | OK={ok} FAIL={fail} SKIP={skip} WARN={warn}")
    if fail > 0:
        print(f"  \033[31m{fail} 项失败, 请检查上方详情\033[0m")
    else:
        print(f"  \033[32m全部通过\033[0m")
    print("=" * 70)


# ── 主入口 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DataSystem 三层 API 验证")
    parser.add_argument("--host", default=os.environ.get("DATASYSTEM_HOST", "127.0.0.1"),
                        help="DataSystem worker host (default: 127.0.0.1 or $DATASYSTEM_HOST)")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("DATASYSTEM_PORT", "31501")),
                        help="DataSystem worker port (default: 31501 or $DATASYSTEM_PORT)")
    parser.add_argument("--gpu", action="store_true",
                        help="启用 GPU 设备对象测试 (HeteroClient)")
    parser.add_argument("--layers", default="all",
                        choices=["all", "kv", "object", "hetero", "transfer"],
                        help="验证层级 (default: all)")
    parser.add_argument("--json-output", default="verify_datasystem_layers_report.json",
                        help="JSON 报告输出路径")
    parser.add_argument("--no-json", action="store_true",
                        help="不生成 JSON 报告")
    args = parser.parse_args()

    # 环境
    import torch
    report = Report(
        host=args.host,
        port=args.port,
        gpu_available=torch.cuda.is_available(),
        cuda_available=torch.cuda.is_available(),
    )

    # 收集各层验证结果
    all_checks = verify_env_info()

    if args.layers in ("all", "kv"):
        all_checks.extend(verify_kv_layer(args.host, args.port))
    if args.layers in ("all", "object"):
        all_checks.extend(verify_object_layer(args.host, args.port))
    if args.layers in ("all", "hetero"):
        all_checks.extend(verify_hetero_layer(args.host, args.port, args.gpu))
    if args.layers in ("all", "transfer"):
        all_checks.extend(verify_transfer_engine(args.host, args.port))

    report.checks = all_checks

    # 统计
    for c in all_checks:
        report.summary[c.status] = report.summary.get(c.status, 0) + 1

    # 输出
    print_report(report)

    # JSON 输出
    if not args.no_json:
        report_dict = {
            "host": report.host,
            "port": report.port,
            "gpu_available": report.gpu_available,
            "summary": report.summary,
            "checks": [
                {
                    "layer": c.layer,
                    "sub_check": c.sub_check,
                    "status": c.status,
                    "detail": c.detail,
                    "duration_ms": round(c.duration_ms, 2),
                }
                for c in all_checks
            ],
        }
        with open(args.json_output, "w") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        print(f"\nJSON 报告已写入: {args.json_output}")

    # 返回码
    return 0 if report.summary.get("FAIL", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
