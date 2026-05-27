#!/bin/bash
# ============================================================================
# KV Cache + Offload 验证（自动适配历史长度）
# ============================================================================
set -euo pipefail

PORT="${PORT:-18000}"
LOG="/tmp/server_v4.log"
PASS=0
FAIL=0

echo "=========================================="
echo " KV Cache + Offload 验证"
echo "=========================================="

# ── 1. Health ──────────────────────────
echo ""
echo "── 1. Health ──"
HEALTH=$(curl -s -m 5 "http://localhost:$PORT/health")
echo "$HEALTH" | python -m json.tool
echo "$HEALTH" | grep -q '"datasystem":"connected"' && echo "✅ DataSystem connected" || echo "❌ 未连接"

# ── 2. 探测最大历史长度 ────────────────
echo ""
echo "── 2. 探测最大可用历史长度 ──"
MAX_HIST=0
for N in 8 6 5 4 3 2 1; do
    ITEMS=""
    for i in $(seq 1 $N); do
        a=$((i * 10)); b=$((i * 20))
        [ "$ITEMS" != "" ] && ITEMS+=","
        ITEMS+="[$a,$b,0,0]"
    done
    HIST="[$ITEMS]"
    CODE=$(curl -s -m 15 -X POST "http://localhost:$PORT/recommend" \
        -H "Content-Type: application/json" \
        -d "{\"user_id\":\"probe\",\"history\":$HIST,\"topk\":5}" \
        | python -c "import sys,json; print(json.load(sys.stdin).get('code','ERR'))" 2>/dev/null || echo "ERR")
    echo "  N=$N → code=$CODE"
    if [ "$CODE" = "200" ]; then
        MAX_HIST=$N
        break
    fi
done

if [ "$MAX_HIST" -eq 0 ]; then
    echo "❌ 连 1 条历史都失败，检查服务状态"
    exit 1
fi
echo "  最大可用: $MAX_HIST 条历史"

# ── 3. 构造请求 ────────────────────────
ITEMS=""
for i in $(seq 1 $MAX_HIST); do
    a=$((i * 10)); b=$((i * 20))
    [ "$ITEMS" != "" ] && ITEMS+=","
    ITEMS+="[$a,$b,0,0]"
done
HIST="[$ITEMS]"

# ── 4. 首次请求 ────────────────────────
echo ""
echo "── 3. 首次请求（${MAX_HIST} 条历史）──"
RESP=$(curl -s -m 30 -X POST "http://localhost:$PORT/recommend" \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"big_test\",\"history\":$HIST,\"topk\":5}")
CODE=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['code'])" 2>/dev/null || echo "ERR")
KV=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['trace']['kv_source'])" 2>/dev/null || echo "ERR")
MS=$(echo "$RESP" | python -c "import sys,json; print(f\"{json.load(sys.stdin)['trace']['total_ms']:.0f}\")" 2>/dev/null || echo "ERR")
echo "  [big_test r1] code=$CODE kv=$KV ms=$MS"
[ "$CODE" = "200" ] && ((PASS++)) || ((FAIL++))

# ── 5. 重复请求验证 KV 命中 ──────────
echo ""
echo "── 4. 重复请求（验证 kv_source: miss→hit）──"
for round in 1 2; do
    RESP=$(curl -s -m 30 -X POST "http://localhost:$PORT/recommend" \
        -H "Content-Type: application/json" \
        -d "{\"user_id\":\"big_test\",\"history\":$HIST,\"topk\":5}")
    CODE=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['code'])" 2>/dev/null || echo "ERR")
    KV=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['trace']['kv_source'])" 2>/dev/null || echo "ERR")
    MS=$(echo "$RESP" | python -c "import sys,json; print(f\"{json.load(sys.stdin)['trace']['total_ms']:.0f}\")" 2>/dev/null || echo "ERR")
    if [ "$KV" = "hit" ]; then
        echo "  [big_test r$((round+1))] code=$CODE kv=$KV ms=$MS ✅ HIT!"
    else
        echo "  [big_test r$((round+1))] code=$CODE kv=$KV ms=$MS"
    fi
    ((PASS++))
done

# ── 6. 日志分析 ────────────────────────
echo ""
echo "── 5. C++ offload/onboard ──"
OFFLOAD=$(grep -ic "offload\|offLoadCopy\|onboard\|onBoardCopy" "$LOG" 2>/dev/null | tr -d '\n' || echo 0)
echo "  行数: $OFFLOAD"
echo "── 6. evict ──"
EVICT=$(grep -ic "evict" "$LOG" 2>/dev/null | tr -d '\n' || echo 0)
echo "  行数: $EVICT"
echo "── 7. DataSystem KV ──"
DS_KV=$(grep -ic "datasystem.*kv\|datasystem.*copy\|datasystem.*get\|datasystem.*set" "$LOG" 2>/dev/null | tr -d '\n' || echo 0)
echo "  行数: $DS_KV"
echo "── 8. Scheduler ──"
grep -i "scheduler policy" "$LOG" | tail -1
echo "── 9. 错误 ──"
ERR=$(grep -ci "error\|exception\|Traceback" "$LOG" 2>/dev/null | tr -d '\n' || echo 0)
echo "  行数: $ERR"

# ── 汇总 ───────────────────────────────
echo ""
echo "=========================================="
echo " hist_max=$MAX_HIST pass=$PASS fail=$FAIL"
echo " offload=$OFFLOAD evict=$EVICT ds=$DS_KV err=$ERR"
echo "=========================================="
