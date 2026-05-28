#!/bin/bash
# 触发结果缓存 LRU 淘汰 → 观察 offload 日志
# 用法: bash test_eviction.sh [PORT] [COUNT]
set -euo pipefail
PORT="${1:-18000}"
COUNT="${2:-55}"  # > hbm_capacity (50)

echo "=========================================="
echo " 触发 LRU 淘汰 (hbm_cap=50, 发送 ${COUNT} 个不同 user)"
echo "=========================================="

for i in $(seq 1 $COUNT); do
    RESP=$(curl -s -m 30 -X POST "http://localhost:$PORT/recommend" \
        -H "Content-Type: application/json" \
        -d "{\"user_id\":\"evict_test_$i\",\"history\":[[$((i*10)),$((i*20)),0,0]],\"topk\":5}")
    KV=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['trace']['kv_source'])" 2>/dev/null || echo "ERR")
    MS=$(echo "$RESP" | python -c "import sys,json; print(f\"{json.load(sys.stdin)['trace']['total_ms']:.0f}\")" 2>/dev/null || echo "ERR")
    echo "  [evict_test_$i] kv=$KV ms=$MS"
done

echo ""
echo "=========================================="
echo " 检查 Python 层 eviction 日志"
echo "=========================================="
grep -i "evicted\|ResultCache" /tmp/server_v4.log 2>/dev/null | tail -20 || echo "(无匹配)"

echo ""
echo "=========================================="
echo " 检查 DataSystem offload/onboard 日志"
echo "=========================================="
grep -i "offload\|onboard\|offLoadCopy\|onBoardCopy" /tmp/server_v4.log 2>/dev/null | tail -20 || echo "(无匹配)"
