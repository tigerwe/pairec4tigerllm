#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""环境检查脚本.

验证 L40S + CUDA 12.2 环境是否满足项目运行要求.
"""

import sys
import subprocess
import importlib
from typing import List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """检查 Python 版本."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        return True, f"Python {version.major}.{version.minor}.{version.micro} ✓"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (需要 3.10+) ✗"


def check_cuda() -> Tuple[bool, str]:
    """检查 CUDA 环境."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA 不可用 ✗"
        
        cuda_version = torch.version.cuda
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # 检查 CUDA 版本 (需要 12.x)
        if cuda_version and cuda_version.startswith("12."):
            return True, f"CUDA {cuda_version}, {device_name} ({device_memory:.1f}GB) ✓"
        return False, f"CUDA {cuda_version} (需要 12.x) ⚠"
    except ImportError:
        return False, "PyTorch 未安装 ✗"
    except Exception as e:
        return False, f"检查失败: {e} ✗"


def check_gpu_compute_capability() -> Tuple[bool, str]:
    """检查 GPU 计算能力."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA 不可用 ✗"
        
        capability = torch.cuda.get_device_capability(0)
        major, minor = capability
        
        # L40S 是 Ada Lovelace 架构，计算能力 8.9
        if major >= 8:
            return True, f"Compute Capability {major}.{minor} ✓"
        return False, f"Compute Capability {major}.{minor} (建议 8.0+) ⚠"
    except Exception as e:
        return False, f"检查失败: {e} ✗"


def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """检查 Python 包."""
    try:
        module = importlib.import_module(package_name.split("[")[0].replace("-", "_"))
        version = getattr(module, "__version__", "未知")
        
        if min_version and version != "未知":
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                return False, f"{package_name} {version} (需要 >= {min_version}) ✗"
        
        return True, f"{package_name} {version} ✓"
    except ImportError:
        return False, f"{package_name} 未安装 ✗"


def check_nvidia_driver() -> Tuple[bool, str]:
    """检查 NVIDIA 驱动."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            driver_version = result.stdout.strip()
            # 535.230.02 是最低要求
            return True, f"NVIDIA Driver {driver_version} ✓"
        return False, "nvidia-smi 执行失败 ✗"
    except FileNotFoundError:
        return False, "nvidia-smi 未找到 (驱动未安装) ✗"
    except Exception as e:
        return False, f"检查失败: {e} ✗"


def check_disk_space() -> Tuple[bool, str]:
    """检查磁盘空间."""
    try:
        import shutil
        stat = shutil.disk_usage(".")
        free_gb = stat.free / (1024**3)
        
        if free_gb >= 50:
            return True, f"可用空间 {free_gb:.1f}GB ✓"
        return False, f"可用空间 {free_gb:.1f}GB (建议 >= 50GB) ⚠"
    except Exception as e:
        return False, f"检查失败: {e} ✗"


def check_tensorrt_llm() -> Tuple[bool, str]:
    """检查 TensorRT-LLM."""
    try:
        import tensorrt_llm
        version = getattr(tensorrt_llm, "__version__", "未知")
        return True, f"TensorRT-LLM {version} ✓"
    except ImportError:
        return False, "TensorRT-LLM 未安装 (可选，将使用 PyTorch 推理) ⚠"


def check_go() -> Tuple[bool, str]:
    """检查 Go 环境."""
    try:
        result = subprocess.run(
            ["go", "version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"{version} ✓"
        return False, "Go 执行失败 ✗"
    except FileNotFoundError:
        return False, "Go 未安装 ✗"


def main():
    """主函数."""
    print("=" * 60)
    print("PaiRec4TigerLLM 环境检查")
    print("=" * 60)
    print()
    
    checks = [
        ("Python 版本", check_python_version),
        ("NVIDIA 驱动", check_nvidia_driver),
        ("CUDA 环境", check_cuda),
        ("GPU 计算能力", check_gpu_compute_capability),
        ("磁盘空间", check_disk_space),
        ("Go 环境", check_go),
        ("PyTorch", lambda: check_package("torch", "2.0.0")),
        ("NumPy", lambda: check_package("numpy", "1.24.0")),
        ("Pandas", lambda: check_package("pandas", "2.0.0")),
        ("Flask", lambda: check_package("flask", "2.3.0")),
        ("TensorRT-LLM", check_tensorrt_llm),
    ]
    
    passed = 0
    failed = 0
    warnings = 0
    
    for name, check_func in checks:
        try:
            success, message = check_func()
            symbol = "✓" if success else "⚠" if "⚠" in message else "✗"
            print(f"{name:.<30} {message}")
            
            if success:
                passed += 1
            elif "⚠" in message:
                warnings += 1
            else:
                failed += 1
        except Exception as e:
            print(f"{name:.<30} 检查异常: {e} ✗")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"结果: {passed} 通过, {warnings} 警告, {failed} 失败")
    print("=" * 60)
    
    if failed > 0:
        print()
        print("请修复上述错误后再运行项目。")
        print("安装依赖: pip install -r requirements.txt")
        sys.exit(1)
    elif warnings > 0:
        print()
        print("环境基本就绪，但存在警告项，可能影响性能。")
        sys.exit(0)
    else:
        print()
        print("✓ 环境检查通过，可以开始运行项目！")
        sys.exit(0)


if __name__ == "__main__":
    main()
