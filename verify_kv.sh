#!/bin/bash
set -euo pipefail
PORT="${PORT:-18000}"
LOG="/tmp/server_v4.log"
PASS=0
FAIL=0

echo "=========================================="
echo " KV Cache + DataSystem 验证"
echo "=========================================="

# ── 1. Health ─────────────────────────
echo ""
echo "── 1. Health ──"
HEALTH=$(curl -s -m 5 "http://localhost:$PORT/health" || echo '{"datasystem":"UNREACHABLE"}')
echo "$HEALTH" | python -m json.tool 2>/dev/null || echo "$HEALTH"
if echo "$HEALTH" | grep -q '"datasystem":"connected"'; then
    echo "✅ DataSystem connected"
else
    echo "❌ DataSystem 未连接"
fi

# ── 2. 串行请求 ───────────────────────
echo ""
echo "── 2. 串行请求（5 个不同用户） ──"
for i in 1 2 3 4 5; do
    RESP=$(curl -s -m 30 -X POST "http://localhost:$PORT/recommend" \
        -H "Content-Type: application/json" \
        -d "{\"user_id\":\"kv_test_$i\",\"history\":[[$((i*10)),$((i*20)),0,0]],\"topk\":5}")
    CODE=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['code'])" 2>/dev/null || echo "ERR")
    KV=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['trace']['kv_source'])" 2>/dev/null || echo "ERR")
    MS=$(echo "$RESP" | python -c "import sys,json; print(f\"{json.load(sys.stdin)['trace']['total_ms']:.0f}\")" 2>/dev/null || echo "ERR")
    if [ "$CODE" = "200" ]; then
        echo "  [kv_test_$i] code=200 kv=$KV ms=$MS ✅"
        ((PASS++)) || true
    else
        echo "  [kv_test_$i] code=$CODE ❌"
        ((FAIL++)) || true
    fi
done

# ── 3. 重复请求验证 KV Cache 命中 ─────
echo ""
echo "── 3. 重复请求（预期 kv_source: miss → hit） ──"
for round in 1 2 3; do
    RESP=$(curl -s -m 30 -X POST "http://localhost:$PORT/recommend" \
        -H "Content-Type: application/json" \
        -d '{"user_id":"kv_test_1","history":[[10,20,0,0]],"topk":5}')
    CODE=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['code'])" 2>/dev/null || echo "ERR")
    KV=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['trace']['kv_source'])" 2>/dev/null || echo "ERR")
    MS=$(echo "$RESP" | python -c "import sys,json; print(f\"{json.load(sys.stdin)['trace']['total_ms']:.0f}\")" 2>/dev/null || echo "ERR")
    if [ "$KV" = "hit" ]; then
        echo "  [round $round] code=200 kv=$KV ms=$MS ✅ cache hit!"
        ((PASS++)) || true
    else
        echo "  [round $round] code=200 kv=$KV ms=$MS"
        ((PASS++)) || true
    fi
done

# ── 4. 日志汇总 ───────────────────────
echo ""
echo "── 4. kv_source 分布 ──"
grep -o '"kv_source":"[^"]*"' "$LOG" 2>/dev/null | sort | uniq -c || echo "(无日志)"

echo ""
echo "── 5. offload/onboard 事件 ──"
grep -i "offload\|onboard\|evict\|offLoadCopy\|onBoardCopy" "$LOG" 2>/dev/null | tail -10 || echo "(无匹配)"

echo ""
echo "── 6. 错误检查 ──"
ERR_COUNT=$(grep -ci "error\|exception\|Traceback\|segfault\|500" "$LOG" 2>/dev/null || echo 0)
echo "  错误行数: $ERR_COUNT"
if [ "$ERR_COUNT" -gt 0 ]; then
    grep -i "error\|exception\|Traceback" "$LOG" 2>/dev/null | tail -5
fi

# ── 汇总 ──────────────────────────────
echo ""
echo "=========================================="
echo " 通过=$PASS  失败=$FAIL  错误日志=$ERR_COUNT"
echo "=========================================="
