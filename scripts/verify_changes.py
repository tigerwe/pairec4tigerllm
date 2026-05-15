#!/usr/bin/env python3
"""验证所有改动文件的语法和导入."""
import ast
import os
import sys

os.chdir("/home/vivwimp/pairec4tigerllm")
sys.path.insert(0, ".")

files = [
    'training/decoder/qwen3_generative_rec.py',
    'training/decoder/__init__.py',
    'training/decoder/train.py',
    'inference/kv_cache/__init__.py',
    'inference/kv_cache/manager.py',
    'inference/trt_llm/server.py',
]

print("=== Syntax check ===")
for f in files:
    try:
        with open(f) as fh:
            ast.parse(fh.read())
        print(f"OK  {f}")
    except SyntaxError as e:
        print(f"ERR {f}: {e}")
    except FileNotFoundError:
        print(f"MISS {f}")

print("\n=== Import test ===")
try:
    from training.decoder.qwen3_generative_rec import Qwen3GenerativeRec
    print("OK  Qwen3GenerativeRec")
except Exception as e:
    print(f"ERR {e}")

try:
    from inference.kv_cache.manager import KVCacheManager
    print("OK  KVCacheManager")
except Exception as e:
    print(f"ERR {e}")

print("\nDone.")
