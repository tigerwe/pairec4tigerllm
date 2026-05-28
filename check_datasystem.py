#!/usr/bin/env python3
"""检查 DataSystem 中 Python KVCacheManager 写入的数据."""
import sys
sys.path.insert(0, '/workspace/pairec4tigerllm')

HOST, PORT = "127.0.0.1", 31501
PREFIX = "pairec4tigerllm:kv"

print(f"=== 连接 DataSystem {HOST}:{PORT} ===")
from yr.datasystem import DsClient
client = DsClient(host=HOST, port=PORT)
client.init()
print("✅ 连接成功")

kv = client.kv()

# 用已知 key 试读（模拟 KVCacheManager 的 key 格式）
# KVCacheManager key: pairec4tigerllm:kv:{user_id}:{history_hash}
print(f"\n=== 试读几个 key ===")
test_keys = [
    f"{PREFIX}:smoke",
    f"{PREFIX}:big_test",
    f"{PREFIX}:test",
]
for k in test_keys:
    try:
        vals = kv.get([k], convert_to_str=False)
        if vals and vals[0]:
            print(f"  ✅ {k}: {len(vals[0])} bytes")
        else:
            print(f"  ❌ {k}: 未找到")
    except Exception as e:
        print(f"  ❌ {k}: {e}")

# 尝试 list（如果 DsClient 支持）
print(f"\n=== list 结果 ===")
try:
    keys = kv.list('')
    print(f"Total: {len(keys)} keys")
    for k in keys[:20]:
        print(f"  {k}")
except AttributeError:
    print("  list() 不支持 (DsClient 无此方法)")
except Exception as e:
    print(f"  ❌ {e}")

# 搜 pairec 前缀
print(f"\n=== 搜索 '{PREFIX}' 前缀 ===")
try:
    keys = kv.list(PREFIX)
    print(f"Total: {len(keys)} keys")
    for k in keys[:20]:
        vals = kv.get([k], convert_to_str=False)
        sz = len(vals[0]) if vals and vals[0] else 0
        print(f"  {k} ({sz} bytes)")
except AttributeError:
    print("  kv.list() 不支持")
except Exception as e:
    print(f"  ❌ {e}")

print("\n=== 完成 ===")
