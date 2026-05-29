#!/bin/bash
# ============================================================================
# C++ 层 TRT-LLM ↔ DataSystem offload/onboard 链路诊断
#
# 运行环境: ARM 4090D 容器
# 前置条件:
#   1. v4 引擎在 ./trt_engines/qwen3_rec_v4/
#   2. checkpoint 在 ./checkpoints/decoder_qwen3/decoder_epoch_20.pt
#   3. DataSystem Worker 在 DATASYSTEM_HOST:31501 运行
#   4. LD_PRELOAD 三链已设置
#
# 用法:
#   bash diag_offload.sh
# ============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
step()  { echo -e "\n${CYAN}═══ $1 ═══${NC}"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; }

PORT="${PORT:-18000}"
LOG="/tmp/cpp_offload_diag.log"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ═══════════════════════════════════════════════════════════════
# 0. 清理
# ═══════════════════════════════════════════════════════════════

step "0. 清理旧服务"
pkill -f "inference.trt_llm.server" 2>/dev/null || true
sleep 2
if lsof -i ":$PORT" &>/dev/null; then
    echo "  端口 $PORT 仍被占用，强制释放"
    fuser -k "$PORT/tcp" 2>/dev/null || true
    sleep 1
fi
> "$LOG"
ok "ready"

# ═══════════════════════════════════════════════════════════════
# 1. 启动服务
# ═══════════════════════════════════════════════════════════════

step "1. 启动推理服务"

DATASYSTEM_HOST="${DATASYSTEM_HOST:-127.0.0.1}"
DATASYSTEM_PORT="${DATASYSTEM_PORT:-31501}"
echo "  DataSystem: $DATASYSTEM_HOST:$DATASYSTEM_PORT"

# 检查前置条件
[ -d "./trt_engines/qwen3_rec_v4" ] || { fail "引擎目录不存在"; exit 1; }
[ -f "./checkpoints/decoder_qwen3/decoder_epoch_20.pt" ] || { fail "checkpoint 不存在"; exit 1; }
ok "前置条件通过"

python -m inference.trt_llm.server \
    --model_path ./checkpoints/decoder_qwen3/decoder_epoch_20.pt \
    --qwen3_model_path ./models/Qwen3-0.6B \
    --trt_engine_dir ./trt_engines/qwen3_rec_v4 \
    --port "$PORT" --device cuda \
    --datasystem_host "$DATASYSTEM_HOST" \
    --datasystem_port "$DATASYSTEM_PORT" \
    >> "$LOG" 2>&1 &
SERVER_PID=$!
echo "  PID=$SERVER_PID"

# 等待就绪
echo "  等待服务就绪..."
for i in $(seq 1 60); do
    if curl -s -m 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then
        ok "服务就绪 (${i}s)"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        fail "进程已退出"
        echo "--- 最后 50 行 ---"; tail -50 "$LOG"; exit 1
    fi
    [ $((i % 10)) -eq 0 ] && echo "    ... ${i}s"
    sleep 1
done

# ═══════════════════════════════════════════════════════════════
# 2. 启动日志分析
# ═══════════════════════════════════════════════════════════════

step "2. paged KV cache 状态"
grep -i "paged\|block.*pool\|secondary\|maxNumSequences\|max_tokens_in_paged\|kv_cache_config\|KvCacheConfig" "$LOG" 2>/dev/null | head -20 || echo "  (无匹配 — paged 模式可能未激活)"

echo ""
echo "=== ③ DataSystem C++ 层连接 ==="
grep -i "datasystem\|KvCache.*DataSystem\|Init KvCache\|Create Datasystem\|Datasystem.*class" "$LOG" 2>/dev/null | head -20 || echo "  (无匹配 — C++ DataSystem 未链接或未初始化)"

echo ""
echo "=== ④ KVCacheManager / TransferManager ==="
grep -i "KVCacheManager\|kvCacheManager\|TransferManager\|offload\|onboard" "$LOG" 2>/dev/null | head -20 || echo "  (无匹配)"

# ═══════════════════════════════════════════════════════════════
# 3. Health
# ═══════════════════════════════════════════════════════════════

step "3. Health"
HEALTH=$(curl -s "http://localhost:$PORT/health")
echo "$HEALTH" | python -m json.tool 2>/dev/null || echo "$HEALTH"
if echo "$HEALTH" | grep -q '"backend":"trt-qwen3"'; then
    ok "backend=trt-qwen3"
else
    fail "后端不是 TRT-LLM"
fi

# ═══════════════════════════════════════════════════════════════
# 4. 压测 — 制造 primary pool 压力触发 eviction
# ═══════════════════════════════════════════════════════════════

step "4. 压测 (20 个不同用户 + 长历史，触发 primary pool 紧张)"

BEFORE_EVICT=$(grep -ci "evict" "$LOG" 2>/dev/null || echo 0)
echo "  当前 evict 行数: $BEFORE_EVICT"

for i in $(seq 1 20); do
    # 构造 30 个商品的长历史
    HIST="["
    for j in $(seq 1 30); do
        s0=$(( (i*7 + j*13) % 256 ))
        s1=$(( (i*11 + j*19) % 256 ))
        HIST+="[$s0,$s1,0,0]"
        [ $j -lt 30 ] && HIST+=","
    done
    HIST+="]"

    RESP=$(curl -s -m 60 -X POST "http://localhost:$PORT/recommend" \
        -H "Content-Type: application/json" \
        -d "{\"user_id\":\"load_$i\",\"history\":$HIST,\"topk\":5}")
    CODE=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('code','ERR'))" 2>/dev/null || echo "ERR")
    KV=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['trace']['kv_source'])" 2>/dev/null || echo "?")
    MS=$(echo "$RESP" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin)['trace']['total_ms']:.0f}\")" 2>/dev/null || echo "?")
    echo "  [load_$i] code=$CODE kv=$KV ms=$MS"
done

# ═══════════════════════════════════════════════════════════════
# 5. 重复请求 — 检验缓存行为
# ═══════════════════════════════════════════════════════════════

step "5. 重复请求 (load_1 × 3，检验缓存命中)"

for r in 1 2 3; do
    HIST="[[$((1*7+1*13)%256),$((1*11+1*19)%256),0,0]"
    for j in $(seq 2 30); do
        HIST+=",[$((1*7+j*13)%256),$((1*11+j*19)%256),0,0]"
    done
    RESP=$(curl -s -m 60 -X POST "http://localhost:$PORT/recommend" \
        -H "Content-Type: application/json" \
        -d "{\"user_id\":\"load_1\",\"history\":[$HIST],\"topk\":5}")
    KV=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['trace']['kv_source'])" 2>/dev/null || echo "?")
    MS=$(echo "$RESP" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin)['trace']['total_ms']:.0f}\")" 2>/dev/null || echo "?")
    echo "  [round $r] kv=$KV ms=$MS"
done

# ═══════════════════════════════════════════════════════════════
# 6. 日志汇总
# ═══════════════════════════════════════════════════════════════

step "6. 日志汇总"

EVICT_COUNT=$(grep -ci "evict" "$LOG" 2>/dev/null || echo 0)
OFFLOAD_COUNT=$(grep -ci "offload\|offLoadCopy\|onboard\|onBoardCopy" "$LOG" 2>/dev/null || echo 0)
DS_COUNT=$(grep -ci "datasystem.*kv\|datasystem.*copy\|datasystem.*set\|datasystem.*get" "$LOG" 2>/dev/null || echo 0)
ERR_COUNT=$(grep -ci "error\|exception\|segfault\|SIGSEGV\|Traceback" "$LOG" 2>/dev/null || echo 0)

echo "  Evict:      $EVICT_COUNT 行"
echo "  Offload:    $OFFLOAD_COUNT 行"
echo "  DS KV:      $DS_COUNT 行"
echo "  Error:      $ERR_COUNT 行"

if [ "$EVICT_COUNT" -gt 0 ]; then
    echo ""
    echo "  —— 最近 10 条 evict ——"
    grep -i "evict" "$LOG" 2>/dev/null | tail -10
else
    echo ""
    echo "  ⚠ 没有 eviction 日志 — primary pool 未被填满"
    echo "    可能原因: maxNumSequences 设得过大，或 batch/seq 数不够"
fi

if [ "$OFFLOAD_COUNT" -gt 0 ]; then
    echo ""
    echo "  —— 最近 10 条 offload/onboard ——"
    grep -i "offload\|offLoadCopy\|onboard\|onBoardCopy" "$LOG" 2>/dev/null | tail -10
else
    echo ""
    echo "  ⚠ 没有 offload/onboard 日志"
    if [ "$EVICT_COUNT" -gt 0 ]; then
        echo "    evict 发生了但 offload 没触发 → secondary pool 可能未初始化 / DRAM 分配被注释"
    else
        echo "    先确认 eviction 能发生"
    fi
fi

echo ""
echo "============================================"
echo " Server PID=$SERVER_PID"
echo " 日志:  $LOG"
echo "============================================"
echo ""
echo "持续监控: tail -f $LOG | grep -i 'offload\|onboard\|evict\|datasystem'"
