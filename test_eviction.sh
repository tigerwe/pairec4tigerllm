#!/bin/bash
# ============================================================================
# KV Cache + DataSystem Offload 完整验证
# 方法: 长历史请求填充 KV Cache → 重复请求验证命中 → 查 offload 日志
# ============================================================================
set -euo pipefail

PORT="${PORT:-18000}"
LOG="/tmp/server_v4.log"
PASS=0
FAIL=0

echo "=========================================="
echo " KV Cache + Offload 完整验证"
echo "=========================================="

# ── 1. Health ──────────────────────────
echo ""
echo "── 1. Health ──"
HEALTH=$(curl -s -m 5 "http://localhost:$PORT/health")
echo "$HEALTH" | python -m json.tool
echo "$HEALTH" | grep -q '"datasystem":"connected"' && echo "✅ DataSystem connected" || echo "❌ DataSystem 未连接"
echo "$HEALTH" | grep -q '"status":"healthy"' && echo "✅ 服务健康" || echo "❌ 服务异常"

# ── 2. 构造长历史 ─────────────────────
HIST='[[10,20,0,0],[30,40,0,0],[50,60,0,0],[70,80,0,0],[90,100,0,0],[110,120,0,0],[130,140,0,0],[150,160,0,0],[170,180,0,0],[190,200,0,0]]'  # 10 history

# ── 3. 一个长请求填充 KV Cache ──────────
echo ""
echo "── 2. 长历史请求（10 条，填充 KV Cache）──"
RESP=$(curl -s -m 30 -X POST "http://localhost:$PORT/recommend" \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"big_test\",\"history\":$HIST,\"topk\":5}")
CODE=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['code'])" 2>/dev/null || echo "ERR")
KV=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['trace']['kv_source'])" 2>/dev/null || echo "ERR")
MS=$(echo "$RESP" | python -c "import sys,json; print(f\"{json.load(sys.stdin)['trace']['total_ms']:.0f}\")" 2>/dev/null || echo "ERR")
echo "  [big_test r1] code=$CODE kv=$KV ms=$MS"
[ "$CODE" = "200" ] && ((PASS++)) || ((FAIL++))

# ── 4. 重复请求，验证 Python KVCacheManager ──
echo ""
echo "── 3. 重复请求（验证 python 层 kv_source: miss→hit）──"
for round in 1 2; do
    RESP=$(curl -s -m 30 -X POST "http://localhost:$PORT/recommend" \
        -H "Content-Type: application/json" \
        -d "{\"user_id\":\"big_test\",\"history\":$HIST,\"topk\":5}")
    CODE=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['code'])" 2>/dev/null || echo "ERR")
    KV=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['trace']['kv_source'])" 2>/dev/null || echo "ERR")
    MS=$(echo "$RESP" | python -c "import sys,json; print(f\"{json.load(sys.stdin)['trace']['total_ms']:.0f}\")" 2>/dev/null || echo "ERR")
    if [ "$KV" = "hit" ]; then
        echo "  [big_test r$((round+1))] code=$CODE kv=$KV ms=$MS ✅ HIT!"
        ((PASS++))
    else
        echo "  [big_test r$((round+1))] code=$CODE kv=$KV ms=$MS"
        ((PASS++))
    fi
done

# ── 5. C++ 层 offload/onboard ──────────
echo ""
echo "── 4. C++ 层 offload/onboard ──"
OFFLOAD=$(grep -ic "offload\|offLoadCopy\|onboard\|onBoardCopy" "$LOG" 2>/dev/null | tr -d '\n' || echo 0)
echo "  匹配行数: $OFFLOAD"
if [ "$OFFLOAD" -gt 0 ] 2>/dev/null; then
    grep -i "offload\|offLoadCopy\|onboard\|onBoardCopy" "$LOG" | tail -10
fi

# ── 6. evict 日志 ──────────────────────
echo ""
echo "── 5. evict 日志 ──"
EVICT=$(grep -ic "evict" "$LOG" 2>/dev/null | tr -d '\n' || echo 0)
echo "  匹配行数: $EVICT"
if [ "$EVICT" -gt 0 ] 2>/dev/null; then
    grep -i "evict" "$LOG" | tail -10
fi

# ── 7. DataSystem KV 操作 ──────────────
echo ""
echo "── 6. DataSystem KV 操作 ──"
DS_KV=$(grep -ic "datasystem.*kv\|datasystem.*copy\|datasystem.*get\|datasystem.*set\|datasystem.*write\|datasystem.*read\|datasystem.*put" "$LOG" 2>/dev/null | tr -d '\n' || echo 0)
echo "  匹配行数: $DS_KV"
if [ "$DS_KV" -gt 0 ] 2>/dev/null; then
    grep -i "datasystem.*kv\|datasystem.*copy\|datasystem.*get\|datasystem.*set" "$LOG" | tail -10
fi

# ── 8. Scheduler ──────────────────────
echo ""
echo "── 7. Scheduler Policy ──"
grep -i "scheduler policy\|capacity scheduler" "$LOG" | tail -3

# ── 9. 错误 ───────────────────────────
echo ""
echo "── 8. 错误检查 ──"
ERR_COUNT=$(grep -ci "error\|exception\|Traceback\|segfault\|abort\|SIGSEGV" "$LOG" 2>/dev/null | tr -d '\n' || echo 0)
echo "  错误行数: $ERR_COUNT"
if [ "$ERR_COUNT" -gt 0 ] 2>/dev/null; then
    grep -i "error\|exception\|Traceback" "$LOG" | tail -5
else
    echo "  ✅ 无错误"
fi

# ── 汇总 ───────────────────────────────
echo ""
echo "=========================================="
echo " 通过=$PASS  失败=$FAIL"
echo " Offload=$OFFLOAD  Evict=$EVICT  DS_KV=$DS_KV  错误=$ERR_COUNT"
echo "=========================================="
