#!/bin/bash
# ============================================================================
# KV Cache Eviction + DataSystem Offload/Onboard 验证
#
# 原理: 并发请求填满 KV Cache → MAX_UTILIZATION 驱逐 → DataSystem offload
# ============================================================================
set -euo pipefail

PORT="${PORT:-18000}"
LOG="/tmp/server_v4.log"
PASS=0
FAIL=0

echo "=========================================="
echo " KV Cache Eviction 压力测试"
echo "=========================================="

# ── 1. 清掉旧请求痕迹 ──────────────────
echo ""
echo "── 1. 初始状态 ──"
HEALTH=$(curl -s "http://localhost:$PORT/health")
echo "$HEALTH" | python -m json.tool 2>/dev/null
echo ""

# ── 2. 并发 50 个不同用户请求 ──────────
#    每个请求 ~32 tokens，max_kv_tokens=256 → 8 个就满 → 后续必然驱逐
echo "── 2. 并发发送 50 个请求 (预期触发 eviction) ──"
START_TIME=$(date +%s)

for i in $(seq 1 50); do
    s0=$((RANDOM % 200 + 1))
    s1=$((RANDOM % 200 + 1))
    s2=$((RANDOM % 200 + 1))
    s3=$((RANDOM % 200 + 1))
    (
        RESULT=$(curl -s -m 30 -X POST "http://localhost:$PORT/recommend" \
            -H "Content-Type: application/json" \
            -d "{\"user_id\":\"evict_$i\",\"history\":[[$s0,$s1,$s2,$s3]],\"topk\":5}" 2>/dev/null)
        CODE=$(echo "$RESULT" | python -c "import sys,json; print(json.load(sys.stdin)['code'])" 2>/dev/null || echo "ERR")
        echo "  [evict_$i] code=$CODE" >> /tmp/evict_results.txt
    ) &
done

echo "  等待所有请求完成..."
wait

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "  耗时: ${ELAPSED}s"

# ── 3. 统计分析 ─────────────────────────
echo ""
echo "── 3. 请求统计 ──"
if [ -f /tmp/evict_results.txt ]; then
    SUCCESS=$(grep -c "code=200" /tmp/evict_results.txt || echo 0)
    ERRORS=$(grep -c "code=ERR\|code=500" /tmp/evict_results.txt || echo 0)
    echo "  成功: $SUCCESS / 50, 失败: $ERRORS"
    rm -f /tmp/evict_results.txt
fi

# 给 offload 一些时间
sleep 2

# ── 4. DataSystem offload/onboard 日志 ──
echo ""
echo "── 4. C++ 层 offload/onboard ──"
OFFLOAD_COUNT=$(grep -ci "offload\|offLoadCopy\|onboard\|onBoardCopy" "$LOG" 2>/dev/null || echo 0)
echo "  offload/onboard 行数: $OFFLOAD_COUNT"
if [ "$OFFLOAD_COUNT" -gt 0 ]; then
    grep -i "offload\|offLoadCopy\|onboard\|onBoardCopy" "$LOG" | tail -20
else
    echo "  (无 offload/onboard 日志)"
fi

echo ""
echo "── 5. evict 相关日志 ──"
EVICT_COUNT=$(grep -ci "evict" "$LOG" 2>/dev/null || echo 0)
echo "  evict 行数: $EVICT_COUNT"
if [ "$EVICT_COUNT" -gt 0 ]; then
    grep -i "evict" "$LOG" | tail -20
else
    echo "  (无 evict 日志)"
fi

echo ""
echo "── 6. DataSystem KV 操作 ──"
DS_COUNT=$(grep -ci "datasystem.*kv\|datasystem.*copy\|datasystem.*get\|datasystem.*set\|datasystem.*write\|datasystem.*read" "$LOG" 2>/dev/null || echo 0)
echo "  DataSystem KV 操作行数: $DS_COUNT"
if [ "$DS_COUNT" -gt 0 ]; then
    grep -i "datasystem.*kv\|datasystem.*copy\|datasystem.*get\|datasystem.*set" "$LOG" | tail -20
else
    echo "  (无 DataSystem KV 操作日志)"
fi

echo ""
echo "── 7. 最终 Health ──"
HEALTH=$(curl -s "http://localhost:$PORT/health")
echo "$HEALTH" | python -m json.tool

echo ""
echo "── 8. 服务日志尾巴 ──"
tail -20 "$LOG"

echo ""
echo "=========================================="
echo " 完成"
echo "=========================================="
