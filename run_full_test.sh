#!/bin/bash
# ============================================================================
# 完整验证流程: 启动推理服务 → 探测历史长度 → KV Cache 命中 → Offload 日志
# ============================================================================
set -euo pipefail

PORT="${PORT:-18000}"
LOG="/tmp/server_v4.log"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$PROJECT_DIR"

# ── 0. 杀旧服务 ────────────────────────
echo "── 0. 清理旧服务 ──"
pkill -f "inference.trt_llm.server" 2>/dev/null || true
sleep 2
lsof -i ":$PORT" 2>/dev/null && { echo "端口 $PORT 仍被占用"; exit 1; } || echo "  端口 $PORT 已释放"

# ── 1. 启动服务 ────────────────────────
echo ""
echo "── 1. 启动推理服务 ──"

# 设置 GCC 14 toolchain + LD_PRELOAD (DataSystem SDK ARM bug workaround)
source /opt/openEuler/gcc-toolset-14/enable 2>/dev/null || true
export LD_PRELOAD="\
/workspace/pairec4tigerllm/scripts/block_ds_consumer.so:\
/workspace/pairec4tigerllm/scripts/stub_gpu.so:\
/usr/local/lib/python3.11/site-packages/yr/datasystem/lib/libabseil_dll.so.2407.0.0"

python -m inference.trt_llm.server \
    --model_path ./checkpoints/decoder_qwen3/decoder_epoch_20.pt \
    --qwen3_model_path ./models/Qwen3-0.6B \
    --trt_engine_dir ./trt_engines/qwen3_rec_v4 \
    --port "$PORT" --device cuda \
    --datasystem_host 127.0.0.1 --datasystem_port 31501 \
    > "$LOG" 2>&1 &
SERVER_PID=$!
echo "  PID=$SERVER_PID"

# ── 2. 等待就绪 ────────────────────────
echo -n "  等待就绪"
for i in $(seq 1 30); do
    if curl -s -m 2 "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo " ✅ (${i}s)"
        break
    fi
    sleep 1
    echo -n "."
done

# ── 3. Health ──────────────────────────
echo ""
echo "── 2. Health ──"
HEALTH=$(curl -s -m 5 "http://localhost:$PORT/health")
echo "$HEALTH" | python -m json.tool
echo "$HEALTH" | grep -q '"datasystem":"connected"' && echo "✅ DataSystem connected" || echo "❌ 未连接"

# ── 4. 基础请求验证 ────────────────────
echo ""
echo "── 3. 基础请求验证（确认服务可推理）──"
RESP=$(curl -s -m 30 -X POST "http://localhost:$PORT/recommend" \
    -H "Content-Type: application/json" \
    -d '{"user_id":"smoke","history":[[10,20,0,0]],"topk":5}')
CODE=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin).get('code','ERR'))" 2>/dev/null || echo "ERR")
if [ "$CODE" != "200" ]; then
    echo "❌ 基础请求失败: code=$CODE"
    echo "$RESP" | head -c 200
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
KV=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['trace']['kv_source'])" 2>/dev/null)
MS=$(echo "$RESP" | python -c "import sys,json; print(f\"{json.load(sys.stdin)['trace']['total_ms']:.0f}\")" 2>/dev/null)
echo "  code=200 kv=$KV ms=$MS ✅"

# ── 5. 探测最大历史长度 ────────────────
echo ""
echo "── 4. 探测最大历史长度 ──"
MAX_HIST=0
for N in 6 5 4 3 2 1; do
    ITEMS=""
    for i in $(seq 1 $N); do
        [ "$ITEMS" != "" ] && ITEMS+=","
        ITEMS+="[$((i*10)),$((i*20)),0,0]"
    done
    CODE=$(curl -s -m 15 -X POST "http://localhost:$PORT/recommend" \
        -H "Content-Type: application/json" \
        -d "{\"user_id\":\"probe_$N\",\"history\":[$ITEMS],\"topk\":5}" \
        | python -c "import sys,json; print(json.load(sys.stdin).get('code','ERR'))" 2>/dev/null || echo "ERR")
    echo "  N=$N → code=$CODE"
    if [ "$CODE" = "200" ]; then
        MAX_HIST=$N
        break
    fi
done

if [ "$MAX_HIST" -eq 0 ]; then
    echo "❌ 探测失败"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
echo "  最大可用: $MAX_HIST 条"

# ── 6. 构造请求参数 ────────────────────
ITEMS=""
for i in $(seq 1 $MAX_HIST); do
    [ "$ITEMS" != "" ] && ITEMS+=","
    ITEMS+="[$((i*10)),$((i*20)),0,0]"
done
HIST="[$ITEMS]"

# ── 7. 重复请求验证 KV 命中 ──────────
echo ""
echo "── 5. KV Cache 命中验证（${MAX_HIST} 条历史 × 3 轮）──"
for round in 1 2 3; do
    RESP=$(curl -s -m 30 -X POST "http://localhost:$PORT/recommend" \
        -H "Content-Type: application/json" \
        -d "{\"user_id\":\"big_test\",\"history\":$HIST,\"topk\":5}")
    CODE=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin).get('code','ERR'))" 2>/dev/null || echo "ERR")
    KV=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['trace']['kv_source'])" 2>/dev/null || echo "ERR")
    MS=$(echo "$RESP" | python -c "import sys,json; print(f\"{json.load(sys.stdin)['trace']['total_ms']:.0f}\")" 2>/dev/null || echo "ERR")
    [ "$KV" = "hit" ] && ICON="✅ HIT!" || ICON="  MISS"
    echo "  [r$round] code=$CODE kv=$KV ms=$MS $ICON"
done

# ── 8. 日志汇总 ────────────────────────
echo ""
echo "── 6. C++ offload/onboard ──"
grep -c "offload\|offLoadCopy\|onboard\|onBoardCopy" "$LOG" 2>/dev/null | xargs echo "  行数:" || echo "  行数: 0"
echo "── 7. evict ──"
grep -c "evict" "$LOG" 2>/dev/null | xargs echo "  行数:" || echo "  行数: 0"
echo "── 8. Scheduler ──"
grep -i "scheduler policy" "$LOG" | tail -1
echo "── 9. 错误 ──"
grep -c "error\|exception\|Traceback" "$LOG" 2>/dev/null | xargs echo "  行数:" || echo "  行数: 0"

echo ""
echo "=========================================="
echo " 服务 PID=$SERVER_PID, 日志=$LOG"
echo "=========================================="
