#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataSystem 三层 API 验证脚本 (KV → Object → Hetero/Device).

用法:
  # 远程 GPU 机器（ARM 4090D / x86 L40S）
  python scripts/verify_datasystem_layers.py --gpu

  # 仅 KV 层
  python scripts/verify_datasystem_layers.py --layers kv

验证层级:
  L1  KVClient      — 简单 key-value (基础层, 已集成)
  L2  ObjectClient  — 共享内存 Buffer + 自动 Spill (中间层, 未集成)
  L3  HeteroClient  — GPU 设备内存 mset_d2h / mget_h2d (设备层, 未集成)
  L3b TransferEngine — 跨节点 GPU 内存 RDMA 传输 (未集成)

输出:
  - 终端彩色报告
  - JSON 报告 (verify_datasystem_layers_report.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── 常量 ──────────────────────────────────────────────

DS_KEY_PREFIX = "pairec4tigerllm:verify"
TEST_KEY_BASE = f"{DS_KEY_PREFIX}:{int(time.time())}"
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
    checks: List[CheckResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)


# ── 工具 ──────────────────────────────────────────────

def _elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


# ── L1: KVClient ──────────────────────────────────────

def verify_kv_layer(host: str, port: int) -> List[CheckResult]:
    results = []
    ds = None

    try:
        from yr.datasystem import DsClient
        from yr.datasystem.kv_client import SetParam
    except ImportError as e:
        results.append(CheckResult("KV", "import", "FAIL", str(e)))
        return results
    results.append(CheckResult("KV", "import", "OK", "yr.datasystem + DsClient"))

    # 连接
    try:
        t0 = time.perf_counter()
        ds = DsClient(host, port)
        ds.init()
        results.append(CheckResult("KV", "connect", "OK", f"{host}:{port}", _elapsed_ms(t0)))
    except Exception as e:
        results.append(CheckResult("KV", "connect", "FAIL", str(e)))
        return results

    kv = ds.kv()
    test_key = f"{TEST_KEY_BASE}:kv"

    # Set
    try:
        t0 = time.perf_counter()
        param = SetParam()
        param.ttl_second = 60
        kv.set(test_key, TEST_VALUE, ttl_second=60)
        results.append(CheckResult("KV", "set", "OK",
                                    f"key={test_key[-20:]}, size={len(TEST_VALUE)}B",
                                    _elapsed_ms(t0)))
    except Exception as e:
        results.append(CheckResult("KV", "set", "FAIL", str(e)))
        return results

    # Get
    try:
        t0 = time.perf_counter()
        vals = kv.get([test_key], convert_to_str=False)
        if vals and vals[0] == TEST_VALUE:
            results.append(CheckResult("KV", "get", "OK", "roundtrip matched",
                                        _elapsed_ms(t0)))
        else:
            got = vals[0][:50] if vals and vals[0] else None
            results.append(CheckResult("KV", "get", "FAIL", f"value mismatch, got={got}"))
    except Exception as e:
        results.append(CheckResult("KV", "get", "FAIL", str(e)))

    # 清理
    try:
        kv.delete([test_key])
    except Exception:
        pass

    return results


# ── L2: ObjectClient ──────────────────────────────────

def verify_object_layer(host: str, port: int) -> List[CheckResult]:
    results = []
    ds = None

    try:
        from yr.datasystem import DsClient
    except ImportError:
        results.append(CheckResult("Object", "import", "SKIP", "yr.datasystem not available"))
        return results

    try:
        ds = DsClient(host, port)
        ds.init()
    except Exception as e:
        results.append(CheckResult("Object", "connect", "FAIL", str(e)))
        return results

    obj_client = ds.object()

    # 方法列表
    obj_methods = [m for m in dir(obj_client) if not m.startswith('_')]
    results.append(CheckResult("Object", "methods", "OK",
                                f"{obj_methods}"))

    # Create Buffer
    obj_key = f"{TEST_KEY_BASE}:obj"
    buffer = None
    try:
        t0 = time.perf_counter()
        buffer = obj_client.create(obj_key, 4096)
        results.append(CheckResult("Object", "create", "OK",
                                    f"key={obj_key[-20:]}, size=4096B",
                                    _elapsed_ms(t0)))
    except Exception as e:
        results.append(CheckResult("Object", "create", "FAIL", str(e)))
        return results

    # memory_copy
    try:
        t0 = time.perf_counter()
        buffer.memory_copy(TEST_VALUE)
        results.append(CheckResult("Object", "memory_copy", "OK",
                                    f"wrote {len(TEST_VALUE)}B",
                                    _elapsed_ms(t0)))
    except Exception as e:
        results.append(CheckResult("Object", "memory_copy", "FAIL", str(e)))

    # publish
    try:
        t0 = time.perf_counter()
        buffer.publish()
        results.append(CheckResult("Object", "publish", "OK", "",
                                    _elapsed_ms(t0)))
    except Exception as e:
        results.append(CheckResult("Object", "publish", "FAIL", str(e)))

    # Get back (ObjectClient.get 返回 list[Buffer])
    try:
        t0 = time.perf_counter()
        buf_list = obj_client.get([obj_key], timeout_ms=5000)
        if buf_list and buf_list[0] is not None:
            data = memoryview(buf_list[0].immutable_data()).tobytes()
            if data[:len(TEST_VALUE)] == TEST_VALUE:
                results.append(CheckResult("Object", "get_roundtrip", "OK",
                                            "data matched",
                                            _elapsed_ms(t0)))
            else:
                results.append(CheckResult("Object", "get_roundtrip", "FAIL",
                                            f"data mismatch, got {data[:20]}"))
        else:
            results.append(CheckResult("Object", "get_roundtrip", "FAIL",
                                        "returned None"))
    except Exception as e:
        results.append(CheckResult("Object", "get_roundtrip", "FAIL", str(e)))

    # 清理 (ObjectClient 无 delete, 通过 KVClient)
    try:
        ds.kv().delete([obj_key])
    except Exception:
        pass

    return results


# ── L3: HeteroClient ──────────────────────────────────

def verify_hetero_layer(host: str, port: int, gpu: bool) -> List[CheckResult]:
    results = []
    ds = None

    try:
        from yr.datasystem import DsClient, Blob, DeviceBlobList
        from yr.datasystem.kv_client import SetParam
    except ImportError as e:
        results.append(CheckResult("Hetero", "import", "FAIL", str(e)))
        return results
    results.append(CheckResult("Hetero", "import", "OK",
                                "DsClient + Blob + DeviceBlobList"))

    try:
        ds = DsClient(host, port)
        ds.init()
    except Exception as e:
        results.append(CheckResult("Hetero", "connect", "FAIL", str(e)))
        return results

    hetero = ds.hetero()

    # 方法列表
    hetero_methods = [m for m in dir(hetero) if not m.startswith('_')]
    results.append(CheckResult("Hetero", "methods", "OK", f"{hetero_methods}"))

    # 检查关键方法
    for m in ["mset_d2h", "mget_h2d", "async_mset_d2h", "delete"]:
        if hasattr(hetero, m):
            results.append(CheckResult("Hetero", f"has_{m}", "OK", ""))
        else:
            results.append(CheckResult("Hetero", f"has_{m}", "FAIL", "method not found"))

    # GPU 设备对象测试
    if not gpu:
        results.append(CheckResult("Hetero", "device_test", "SKIP",
                                    "use --gpu to enable"))
        return results

    try:
        import torch
        if not torch.cuda.is_available():
            results.append(CheckResult("Hetero", "device_test", "SKIP",
                                        "CUDA not available"))
            return results
        device = "cuda:0"
    except ImportError:
        results.append(CheckResult("Hetero", "device_test", "SKIP",
                                    "torch not installed"))
        return results

    # ── mset_d2h: GPU tensor → DataSystem ──
    dev_key = f"{TEST_KEY_BASE}:dev"
    try:
        # 分配 GPU tensor 模拟 paged KV block
        block_elem = 64 * 64  # 模拟: num_kv_heads=8, head_dim=64, seq=8
        gpu_tensor = torch.arange(block_elem, dtype=torch.bfloat16, device=device)
        ptr = gpu_tensor.data_ptr()
        size_bytes = block_elem * 2  # bfloat16 = 2 bytes

        blob = Blob(dev_ptr=ptr, size=size_bytes)
        dev_blob_list = DeviceBlobList(dev_idx=0, blob_list=[blob])

        results.append(CheckResult("Hetero", "blob_construct", "OK",
                                    f"ptr={hex(ptr)}, size={size_bytes}B"))

        # mset_d2h: Device → Host (写入 DataSystem)
        t0 = time.perf_counter()
        set_param = SetParam()
        set_param.ttl_second = 60
        hetero.mset_d2h([dev_key], [dev_blob_list], set_param=set_param)
        results.append(CheckResult("Hetero", "mset_d2h", "OK",
                                    f"key={dev_key[-20:]}",
                                    _elapsed_ms(t0)))

        # mget_h2d: Host → Device (从 DataSystem onboard)
        # 分配目标 tensor
        dst_tensor = torch.zeros(block_elem, dtype=torch.bfloat16, device=device)
        dst_ptr = dst_tensor.data_ptr()
        dst_blob = Blob(dev_ptr=dst_ptr, size=size_bytes)
        dst_blob_list = DeviceBlobList(dev_idx=0, blob_list=[dst_blob])

        t0 = time.perf_counter()
        failed_keys = hetero.mget_h2d([dev_key], [dst_blob_list], sub_timeout_ms=5000)
        if not failed_keys:
            # 验证数据
            if torch.equal(gpu_tensor, dst_tensor):
                results.append(CheckResult("Hetero", "mget_h2d_roundtrip", "OK",
                                            "data matched",
                                            _elapsed_ms(t0)))
            else:
                results.append(CheckResult("Hetero", "mget_h2d_roundtrip", "FAIL",
                                            "data mismatch"))
        else:
            results.append(CheckResult("Hetero", "mget_h2d_roundtrip", "FAIL",
                                        f"failed_keys={failed_keys}"))
    except Exception as e:
        results.append(CheckResult("Hetero", "device_test", "FAIL", str(e)))

    # 清理
    try:
        hetero.delete([dev_key])
    except Exception:
        pass

    # 释放 GPU 内存
    try:
        del gpu_tensor, dst_tensor
    except Exception:
        pass

    return results


# ── L3b: TransferEngine ───────────────────────────────

def verify_transfer_engine(host: str, port: int) -> List[CheckResult]:
    results = []

    try:
        from yr.datasystem import TransferEngine, Result, ErrorCode  # noqa: F401
        results.append(CheckResult("TransferEngine", "import", "OK",
                                    "TransferEngine / Result / ErrorCode"))
    except (ImportError, AttributeError) as e:
        results.append(CheckResult("TransferEngine", "import", "SKIP",
                                    f"not compiled in this build: {e}"))
        return results

    te_methods = [m for m in dir(TransferEngine) if not m.startswith('_')]
    results.append(CheckResult("TransferEngine", "methods", "OK", f"{te_methods}"))

    results.append(CheckResult("TransferEngine", "full_test", "SKIP",
                                "requires multi-GPU / cross-node"))

    return results


# ── 环境信息 ──────────────────────────────────────────

def verify_env_info() -> List[CheckResult]:
    results = []

    # Python
    results.append(CheckResult("Env", "python", "OK", sys.version.split()[0]))

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory // 1024**3
            results.append(CheckResult("Env", "cuda", "OK",
                                        f"torch {torch.__version__}, "
                                        f"CUDA {torch.version.cuda}, "
                                        f"GPU {name}, {mem}GB"))
        else:
            results.append(CheckResult("Env", "cuda", "SKIP", "not available"))
    except ImportError:
        results.append(CheckResult("Env", "cuda", "SKIP", "torch not installed"))

    # DataSystem
    try:
        import yr.datasystem
        ds_path = os.path.dirname(yr.datasystem.__file__)
        version = getattr(yr.datasystem, '__version__', 'unknown')
        results.append(CheckResult("Env", "datasystem", "OK",
                                    f"v{version}, {ds_path}"))
    except ImportError:
        results.append(CheckResult("Env", "datasystem", "FAIL", "not installed"))

    # LD_PRELOAD
    ld_preload = os.environ.get("LD_PRELOAD", "")
    if ld_preload:
        so_short = [s.split("/")[-1] for s in ld_preload.split(":") if s]
        results.append(CheckResult("Env", "LD_PRELOAD", "OK", ", ".join(so_short)))
    else:
        results.append(CheckResult("Env", "LD_PRELOAD", "WARN", "not set"))

    return results


# ── 终端报告 ──────────────────────────────────────────

def print_report(report: Report):
    C = {"OK": "\033[32m", "FAIL": "\033[31m", "SKIP": "\033[33m", "WARN": "\033[35m"}
    R = "\033[0m"

    print("=" * 70)
    print(f"  DataSystem 三层 API 验证报告")
    print(f"  Host: {report.host}:{report.port}  |  GPU: {'yes' if report.gpu_available else 'no'}")
    print("=" * 70)

    current_layer = ""
    for c in report.checks:
        if c.layer != current_layer:
            current_layer = c.layer
            print(f"\n── {current_layer} ──")

        flag = C.get(c.status, c.status) + c.status + R
        line = f"  [{flag}] {c.sub_check}"
        if c.detail:
            line += f"  — {c.detail}"
        if c.duration_ms > 0.5:
            line += f"  ({c.duration_ms:.1f}ms)"
        print(line)

    print("\n" + "=" * 70)
    s = report.summary
    total = sum(s.values())
    print(f"  总计: {total} | OK={s.get('OK',0)} FAIL={s.get('FAIL',0)} "
          f"SKIP={s.get('SKIP',0)} WARN={s.get('WARN',0)}")
    if s.get("FAIL", 0) > 0:
        print(f"  \033[31m{s['FAIL']} 项失败\033[0m")
    else:
        print(f"  \033[32m全部通过\033[0m")
    print("=" * 70)


# ── 主入口 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DataSystem 三层 API 验证")
    parser.add_argument("--host", default=os.environ.get("DATASYSTEM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("DATASYSTEM_PORT", "31501")))
    parser.add_argument("--gpu", action="store_true", help="启用 GPU 设备对象测试")
    parser.add_argument("--layers", default="all",
                        choices=["all", "kv", "object", "hetero", "transfer"])
    parser.add_argument("--json-output", default="verify_datasystem_layers_report.json")
    parser.add_argument("--no-json", action="store_true")
    args = parser.parse_args()

    # GPU 检测
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
    except ImportError:
        gpu_ok = False

    report = Report(host=args.host, port=args.port, gpu_available=gpu_ok)

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
    for c in all_checks:
        report.summary[c.status] = report.summary.get(c.status, 0) + 1

    print_report(report)

    if not args.no_json:
        report_dict = {
            "host": report.host, "port": report.port,
            "gpu_available": report.gpu_available,
            "summary": report.summary,
            "checks": [{"layer": c.layer, "sub_check": c.sub_check,
                        "status": c.status, "detail": c.detail,
                        "duration_ms": round(c.duration_ms, 2)}
                       for c in all_checks],
        }
        with open(args.json_output, "w") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        print(f"\nJSON → {args.json_output}")

    return 0 if report.summary.get("FAIL", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
