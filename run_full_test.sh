#!/bin/bash
# ============================================================================
# 完整验证: 启动服务 → 探测历史 → KV 命中 → offload 日志
# 用法: ./run_full_test.sh
# ============================================================================

PORT="${PORT:-18000}"
LOG="/tmp/server_v4.log"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

red(){ echo -e "\033[31m$1\033[0m"; }
green(){ echo -e "\033[32m$1\033[0m"; }
cyan(){ echo -e "\033[36m$1\033[0m"; }
step(){ echo ""; cyan "── $1 ──"; }

# ── 0. 清理 ────────────────────────────
step "0. 清理旧服务"
pkill -f "inference.trt_llm.server" 2>/dev/null; sleep 2
if lsof -i ":$PORT" &>/dev/null; then
    red "端口 $PORT 仍被占用，强制 kill"; fuser -k "$PORT/tcp" 2>/dev/null; sleep 1
fi
green "端口 $PORT 已释放"

# ── 1. 环境 ────────────────────────────
step "1. 设置环境变量"
export LD_PRELOAD="/workspace/pairec4tigerllm/scripts/block_ds_consumer.so:/workspace/pairec4tigerllm/scripts/stub_gpu.so:/usr/local/lib/python3.11/site-packages/yr/datasystem/lib/libabseil_dll.so.2407.0.0"
echo "LD_PRELOAD=$LD_PRELOAD"

# GCC toolchain (可选，非致命)
if [ -f /opt/openEuler/gcc-toolset-14/enable ]; then
    source /opt/openEuler/gcc-toolset-14/enable
    echo "GCC toolset 14 enabled"
else
    echo "GCC toolset 14 not found (skipped)"
fi

# ── 2. 启动 ────────────────────────────
step "2. 启动推理服务"
python -m inference.trt_llm.server \
    --model_path ./checkpoints/decoder_qwen3/decoder_epoch_20.pt \
    --qwen3_model_path ./models/Qwen3-0.6B \
    --trt_engine_dir ./trt_engines/qwen3_rec_v4 \
    --port "$PORT" --device cuda \
    --datasystem_host 127.0.0.1 --datasystem_port 31501 \
    > "$LOG" 2>&1 &
PID=$!
echo "PID=$PID"

# ── 3. 等待就绪 ────────────────────────
step "3. 等待服务就绪"
for i in $(seq 1 60); do
    if curl -s -m 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then
        green "就绪 (${i}s)"
        break
    fi
    if ! kill -0 $PID 2>/dev/null; then
        red "服务进程已退出"; echo "--- 日志 ---"; tail -30 "$LOG"; exit 1
    fi
    [ $((i % 10)) -eq 0 ] && echo "  ... ${i}s"
    sleep 1
done

# ── 4. Health ──────────────────────────
step "4. Health"
HEALTH=$(curl -s "http://localhost:$PORT/health")
echo "$HEALTH" | python -m json.tool
echo "$HEALTH" | grep -q '"datasystem":"connected"' && green "✅ DataSystem connected" || red "❌ DataSystem 未连接"

# ── 5. Smoketest ───────────────────────
step "5. Smoketest"
RESP=$(curl -s -m 30 -X POST "http://localhost:$PORT/recommend" \
    -H "Content-Type: application/json" \
    -d '{"user_id":"smoke","history":[[10,20,0,0]],"topk":5}')
CODE=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('code','ERR'))" 2>/dev/null || echo "ERR")
if [ "$CODE" != "200" ]; then
    red "失败: code=$CODE"; echo "$RESP" | head -c 300; echo; exit 1
fi
KV=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['trace']['kv_source'])" 2>/dev/null)
MS=$(echo "$RESP" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin)['trace']['total_ms']:.0f}\")" 2>/dev/null)
green "code=200 kv=$KV ms=$MS"

# ── 6. 相同请求再发两次 ────────────────
step "6. KV Cache 命中 (相同用户 × 2)"
for r in 1 2; do
    RESP=$(curl -s -m 30 -X POST "http://localhost:$PORT/recommend" \
        -H "Content-Type: application/json" \
        -d '{"user_id":"smoke","history":[[10,20,0,0]],"topk":5}')
    CODE=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('code','ERR'))" 2>/dev/null)
    KV=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['trace']['kv_source'])" 2>/dev/null)
    MS=$(echo "$RESP" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin)['trace']['total_ms']:.0f}\")" 2>/dev/null)
    [ "$KV" = "hit" ] && green "  [r$((r+1))] code=$CODE kv=$KV ms=$MS ✅ HIT!" || echo "  [r$((r+1))] code=$CODE kv=$KV ms=$MS"
done

# ── 7. 日志汇总 ────────────────────────
step "7. 日志分析"
echo "Scheduler: $(grep -i 'scheduler policy' "$LOG" | tail -1)"
echo "Offload:   $(grep -ci 'offload\|offLoadCopy\|onboard\|onBoardCopy' "$LOG" 2>/dev/null || echo 0) 行"
echo "Evict:     $(grep -ci 'evict' "$LOG" 2>/dev/null || echo 0) 行"
echo "DS KV:     $(grep -ci 'datasystem.*kv\|datasystem.*copy' "$LOG" 2>/dev/null || echo 0) 行"
echo "Error:     $(grep -ci 'error\|exception\|Traceback' "$LOG" 2>/dev/null || echo 0) 行"
echo ""
green "完成. PID=$PID, 日志=$LOG"
