#!/usr/bin/env python3
# ============================================================================
# 检查 DataSystem 中是否有 TRT-LLM KV Cache 写入的数据
# 能验证 C++ 层 offload 是否真的发生了
# ============================================================================
import sys
sys.path.insert(0, '/workspace/pairec4tigerllm')

HOST = "127.0.0.1"
PORT = 31501

print(f"=== 连接 DataSystem {HOST}:{PORT} ===")
try:
    import yr.datasystem as ds
    client = ds.KVClient()
    ret = client.Init(HOST, PORT)
    print(f"Init: {ret}")
except Exception as e:
    print(f"❌ 连接失败: {e}")
    sys.exit(1)

# 列出所有 key
print("\n=== Key 列表 ===")
try:
    keys = client.list('')
    print(f"Total keys: {len(keys)}")
    if keys:
        for k in keys[:50]:
            val = client.get(k)
            size = len(val) if val else 0
            print(f"  {k}  ({size} bytes)")
        if len(keys) > 50:
            print(f"  ... +{len(keys) - 50} more")
    else:
        print("  (空 — offload 未触发)")
except Exception as e:
    print(f"❌ list 失败: {e}")

# 搜 pairec/tensorrt/kv 相关的 key
print("\n=== 搜索 TRT/KV 相关 key ===")
for prefix in ["pairec", "tensorrt", "kv", "trt", "block", "KvCache"]:
    try:
        matched = [k for k in keys if prefix.lower() in k.lower()]
        if matched:
            print(f"  prefix='{prefix}': {len(matched)} keys")
            for k in matched[:5]:
                print(f"    {k}")
    except:
        pass

print("\n=== 完成 ===")
