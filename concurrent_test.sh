#!/bin/bash
# ============================================================================
# 并发请求打满 KV Cache → 触发 eviction → 验证 offload/onboard
# 前置: 服务已通过 run_full_test.sh 启动
# ============================================================================

PORT="${PORT:-18000}"
LOG="/tmp/server_v4.log"
N=${1:-30}

echo "=== 并发 $N 个请求填 KV Cache (max=256 tokens) ==="
for i in $(seq 1 $N); do
    curl -s -X POST "http://localhost:$PORT/recommend" \
        -H "Content-Type: application/json" \
        -d "{\"user_id\":\"c$i\",\"history\":[[$i,$((i*2)),0,0]],\"topk\":5}" \
        > /dev/null &
done
wait
echo "完成"; sleep 2

echo ""
echo "=== evict ==="
grep -ci "evict" "$LOG" 2>/dev/null | xargs echo "  行数:" || echo "  行数: 0"
grep -i "evict" "$LOG" 2>/dev/null | tail -5

echo ""
echo "=== offload/onboard ==="
grep -ci "offload\|offLoadCopy\|onboard\|onBoardCopy" "$LOG" 2>/dev/null | xargs echo "  行数:" || echo "  行数: 0"
grep -i "offload\|offLoadCopy\|onboard\|onBoardCopy" "$LOG" 2>/dev/null | tail -5

echo ""
echo "=== DataSystem KV ==="
grep -ci "datasystem.*kv\|datasystem.*copy\|datasystem.*get\|datasystem.*set\|datasystem.*put" "$LOG" 2>/dev/null | xargs echo "  行数:" || echo "  行数: 0"
grep -i "datasystem.*kv\|datasystem.*copy\|datasystem.*get\|datasystem.*set" "$LOG" 2>/dev/null | tail -5

echo ""
echo "=== kv_source (最近 10 条) ==="
grep -o '"kv_source":"[^"]*"' "$LOG" 2>/dev/null | tail -10

echo ""
echo "=== 错误 ==="
grep -ci "error\|exception\|Traceback" "$LOG" 2>/dev/null | xargs echo "  行数:" || echo "  行数: 0"
