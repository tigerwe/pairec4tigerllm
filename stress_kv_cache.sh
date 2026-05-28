#!/bin/bash
# 构造大体积 KV Cache 条目，验证 offload/onboard 性能
# 
# 前提: 推理服务需以 PyTorch 模式启动 (不加 --trt_engine_dir)
#   python -m inference.trt_llm.server \
#       --model_path ./checkpoints/decoder_qwen3/decoder_epoch_20.pt \
#       --port 18000 --device cuda --qwen3_model_path ./models/Qwen3-0.6B
#
# Qwen3-0.6B past_kv 每 token ≈ 56KB
#   8MB ≈ 146 tokens ≈ 10-15 条历史 (取决于每条历史的 token 数)
#
# 用法: bash stress_kv_cache.sh [PORT] [HIST_LEN]
set -euo pipefail
PORT="${1:-18000}"
HIST_LEN="${2:-30}"  # 30 条历史 ≈ 15-20MB 每条

echo "=========================================="
echo " 大体积 KV Cache 构造 (hist_len=$HIST_LEN)"
echo "=========================================="

# 构造历史: [[10,20,0,0], [30,40,0,0], ...]
HIST_ITEMS=""
for i in $(seq 1 $HIST_LEN); do
    a=$((i * 10 % 256))
    b=$((i * 20 % 256))
    [ "$HIST_ITEMS" != "" ] && HIST_ITEMS+=","
    HIST_ITEMS+="[$a,$b,0,0]"
done

# 第一步: 发请求，触发 store
echo ""
echo "── 1. 首次请求（触发 store）──"
t0=$(date +%s%N)
RESP=$(curl -s -m 120 -X POST "http://localhost:$PORT/recommend" \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"big_kv_test\",\"history\":[$HIST_ITEMS],\"topk\":5}")
KV=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['trace']['kv_source'])" 2>/dev/null || echo "ERR")
MS=$(echo "$RESP" | python -c "import sys,json; print(f\"{json.load(sys.stdin)['trace']['total_ms']:.0f}\")" 2>/dev/null || echo "ERR")
echo "  kv_source=$KV total_ms=$MS ms"

# 第二步: 重复请求，验证 HBM hit
echo ""
echo "── 2. 重复请求（HBM hit）──"
for round in 1 2 3; do
    RESP=$(curl -s -m 30 -X POST "http://localhost:$PORT/recommend" \
        -H "Content-Type: application/json" \
        -d "{\"user_id\":\"big_kv_test\",\"history\":[$HIST_ITEMS],\"topk\":5}")
    KV=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['trace']['kv_source'])" 2>/dev/null || echo "ERR")
    MS=$(echo "$RESP" | python -c "import sys,json; print(f\"{json.load(sys.stdin)['trace']['total_ms']:.0f}\")" 2>/dev/null || echo "ERR")
    echo "  [round $round] kv_source=$KV ms=$MS"
done

# 第三步: kv-stats
echo ""
echo "── 3. KV Cache 统计 ──"
curl -s "http://localhost:$PORT/kv-stats" | python -m json.tool 2>/dev/null || echo "(kv-stats 端点不可用)"

echo ""
echo "── 4. offload/onboard 日志 ──"
grep -i "offload\|onboard\|evict\|KVCacheManager" /tmp/server_v4.log 2>/dev/null | tail -10 || echo "(无匹配)"
