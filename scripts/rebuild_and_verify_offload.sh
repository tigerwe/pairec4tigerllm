#!/bin/bash
# ============================================================================
# 重建 TRT-LLM 引擎 + DataSystem offload/onboard 验证
# 
# 运行环境: ARM 4090D 容器 (pairec-inference:v1.0)
# 前置条件:
#   1. import tensorrt_llm 成功 (1.0.0)
#   2. import yr.datasystem 成功
#   3. LD_PRELOAD 已设置 (见 AGENTS.md)
#   4. checkpoint decoder_epoch_20.pt 在 ./checkpoints/decoder_qwen3/
#   5. DataSystem worker 在 DATASYSTEM_HOST:31501 运行中
#
# 用法:
#   chmod +x scripts/rebuild_and_verify_offload.sh
#   ./scripts/rebuild_and_verify_offload.sh
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_step()  { echo -e "\n${CYAN}═══ $1 ═══${NC}"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_fail()  { echo -e "${RED}[FAIL]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PORT="${PORT:-18000}"

# ── Step 0: 环境检查 ─────────────────────────────────────

log_step "Step 0: 环境预检"

cd "$PROJECT_DIR"

# 检查 tensorrt_llm
if python -c "import tensorrt_llm" 2>/dev/null; then
    TRT_VER=$(python -c "print(getattr(__import__('tensorrt_llm'), '__version__', 'OK'))" 2>/dev/null || echo "OK")
    log_ok "tensorrt_llm $TRT_VER 可用"
else
    log_fail "tensorrt_llm 不可用，请 pip install -e /TensorRT-LLM"
    exit 1
fi

# 检查 datasystem
if python -c "import yr.datasystem; print('OK')" 2>/dev/null | grep -q "OK"; then
    log_ok "yr.datasystem 可用"
else
    log_warn "yr.datasystem 不可用，DataSystem offload/onboard 将无法验证"
fi

# 检查 checkpoint
CKPT="./checkpoints/decoder_qwen3/decoder_epoch_20.pt"
if [ ! -f "$CKPT" ]; then
    log_fail "checkpoint 不存在: $CKPT"
    exit 1
fi
log_ok "checkpoint: $CKPT"

# 检查 Qwen3 基础模型
QWEN3_MODEL="${QWEN3_MODEL_PATH:-./models/Qwen3-0.6B}"
if [ ! -d "$QWEN3_MODEL" ]; then
    log_warn "Qwen3 基础模型不存在: $QWEN3_MODEL"
    log_warn "请设置环境变量 QWEN3_MODEL_PATH"
fi

# ── Step 1: 导出模型 ─────────────────────────────────────

log_step "Step 1: 导出模型 (merge LoRA → HuggingFace)"

EXPORT_DIR="./exported/qwen3_rec"

if [ -d "$EXPORT_DIR" ] && [ -f "$EXPORT_DIR/config.json" ]; then
    log_warn "$EXPORT_DIR 已存在，跳过导出"
    echo "  如需重新导出: rm -rf $EXPORT_DIR"
else
    QWEN3_MODEL_PATH="$QWEN3_MODEL" python scripts/export_for_trtllm.py
    log_ok "模型已导出到 $EXPORT_DIR"

    # 验证导出产物
    if [ -f "$EXPORT_DIR/config.json" ] && [ -f "$EXPORT_DIR/tokenizer.json" ]; then
        VOCAB=$(python -c "import json; c=json.load(open('$EXPORT_DIR/config.json')); print(c.get('vocab_size','N/A'))")
        HIDDEN=$(python -c "import json; c=json.load(open('$EXPORT_DIR/config.json')); print(c.get('hidden_size','N/A'))")
        log_ok "导出验证: vocab_size=$VOCAB, hidden_size=$HIDDEN"
    else
        log_fail "导出不完整，缺少 config.json 或 tokenizer.json"
        exit 1
    fi
fi

# ── Step 2: convert_checkpoint ────────────────────────────

log_step "Step 2: convert_checkpoint (HF → TRT-LLM checkpoint)"

TRT_CKPT="./trt_ckpt"

if [ -d "$TRT_CKPT" ] && [ -f "$TRT_CKPT/config.json" ]; then
    log_warn "$TRT_CKPT 已存在，跳过 convert"
    echo "  如需重新 convert: rm -rf $TRT_CKPT"
else
    # 找到 convert_checkpoint.py 路径
    CONVERT_SCRIPT=""
    for candidate in \
        "/TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py" \
        "$(python -c 'import tensorrt_llm; import os; print(os.path.dirname(tensorrt_llm.__file__))' 2>/dev/null)/../examples/models/core/qwen/convert_checkpoint.py" \
        "/opt/TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py"; do
        if [ -f "$candidate" ]; then
            CONVERT_SCRIPT="$candidate"
            break
        fi
    done

    if [ -z "$CONVERT_SCRIPT" ]; then
        log_fail "找不到 convert_checkpoint.py"
        log_fail "请确认 TensorRT-LLM 源码路径"
        exit 1
    fi

    echo "使用 convert 脚本: $CONVERT_SCRIPT"
    python "$CONVERT_SCRIPT" \
        --model_dir "$EXPORT_DIR" \
        --output_dir "$TRT_CKPT" \
        --dtype bfloat16 \
        --tp_size 1 \
        --pp_size 1

    log_ok "convert_checkpoint 完成 → $TRT_CKPT"
fi

# ── Step 3: 构建引擎 ─────────────────────────────────────

log_step "Step 3: trtllm-build (构建新引擎)"

ENGINE_DIR="./trt_engines/qwen3_rec"

if [ -d "$ENGINE_DIR" ] && [ -f "$ENGINE_DIR/rank0.engine" ]; then
    log_warn "$ENGINE_DIR 已存在，跳过构建"
    echo "  如需重新构建: rm -rf $ENGINE_DIR"
else
    rm -rf "$ENGINE_DIR"
    trtllm-build \
        --checkpoint_dir "$TRT_CKPT" \
        --output_dir "$ENGINE_DIR" \
        --gemm_plugin bfloat16 \
        --gpt_attention_plugin bfloat16 \
        --max_batch_size 32 \
        --max_input_len 20 \
        --max_seq_len 30 \
        --max_beam_width 1 \
        --remove_input_padding enable \
        --context_fmha enable

    if [ -f "$ENGINE_DIR/rank0.engine" ]; then
        ENGINE_SIZE=$(du -h "$ENGINE_DIR/rank0.engine" | cut -f1)
        log_ok "引擎构建成功: $ENGINE_SIZE"
    else
        log_fail "引擎构建失败: rank0.engine 未生成"
        exit 1
    fi
fi

# ── Step 4: 启动推理服务 ─────────────────────────────────

log_step "Step 4: 启动推理服务 (TRT-LLM + DataSystem)"

# 检查端口
if lsof -i :$PORT &>/dev/null; then
    log_warn "端口 $PORT 已被占用，尝试 kill"
    fuser -k $PORT/tcp 2>/dev/null || true
    sleep 2
fi

# 设置 DataSystem 环境变量 (如果没设就用默认值)
DATASYSTEM_HOST="${DATASYSTEM_HOST:-127.0.0.1}"
DATASYSTEM_PORT="${DATASYSTEM_PORT:-31501}"

echo "DataSystem: $DATASYSTEM_HOST:$DATASYSTEM_PORT"
echo "Engine dir: $ENGINE_DIR"

# 后台启动服务
python -m inference.trt_llm.server \
    --model_path "$CKPT" \
    --qwen3_model_path "$QWEN3_MODEL" \
    --trt_engine_dir "$ENGINE_DIR" \
    --port "$PORT" \
    --device cuda \
    --datasystem_host "$DATASYSTEM_HOST" \
    --datasystem_port "$DATASYSTEM_PORT" \
    > /tmp/trt_server_offload_test.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "日志文件: /tmp/trt_server_offload_test.log"

# 等待服务就绪
echo -n "等待服务就绪..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo ""
        log_ok "服务就绪 (${i}s)"
        break
    fi
    sleep 1
    echo -n "."
done

if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    log_fail "服务启动超时 (60s)"
    echo "--- 最近日志 ---"
    tail -50 /tmp/trt_server_offload_test.log
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# 检查 health 中的 DataSystem 状态
HEALTH=$(curl -s "http://localhost:$PORT/health")
echo "Health: $HEALTH" | head -c 300
echo ""

# ── Step 5: 批量请求触发 eviction ─────────────────────────

log_step "Step 5: 批量请求触发 KV Cache eviction (验证 offload/onboard)"

# 用 100+ 不同 user_id 发请求，max_tokens_in_paged_kv_cache=256 会快速填满
# eviction 触发后，C++ 层 offload 到 DataSystem + 后续 onboard 回 HBM
BATCH_COUNT="${BATCH_COUNT:-80}"

echo "发送 $BATCH_COUNT 个不同用户的推荐请求..."

SUCCESS=0
FAIL=0
TOTAL_MS=0

for i in $(seq 1 $BATCH_COUNT); do
    UID="evict_test_${i}"
    # 随机历史序列 (不同长度模拟真实场景)
    HIST_LEN=$(( (i % 5) + 1 ))
    HISTORY="["
    for j in $(seq 1 $HIST_LEN); do
        s0=$((RANDOM % 200 + 1))
        s1=$((RANDOM % 200 + 1))
        s2=$((RANDOM % 200 + 1))
        s3=$((RANDOM % 200 + 1))
        if [ $j -gt 1 ]; then HISTORY+=","; fi
        HISTORY+="[$s0,$s1,$s2,$s3]"
    done
    HISTORY+="]"

    RESP=$(curl -s -m 30 -X POST "http://localhost:$PORT/recommend" \
        -H "Content-Type: application/json" \
        -d "{\"user_id\":\"$UID\",\"history\":$HISTORY,\"topk\":5}" 2>/dev/null)

    if echo "$RESP" | grep -q '"recommendations"'; then
        MS=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['trace']['total_ms'])" 2>/dev/null || echo "0")
        TOTAL_MS=$(python -c "print($TOTAL_MS + $MS)" 2>/dev/null || echo "$TOTAL_MS")
        ((SUCCESS++)) || true
        if [ $((i % 20)) -eq 0 ]; then
            echo "  [$i/$BATCH_COUNT] 成功=$SUCCESS, 上次耗时=${MS}ms"
        fi
    else
        ((FAIL++)) || true
        echo "  [$i/$BATCH_COUNT] 失败: $(echo "$RESP" | head -c 100)"
    fi

    # 小间隔避免压垮服务
    sleep 0.02
done

echo ""
log_ok "请求完成: 成功=$SUCCESS, 失败=$FAIL"

if [ $SUCCESS -gt 0 ]; then
    AVG_MS=$(python -c "print(f'{$TOTAL_MS}/{$SUCCESS}')" 2>/dev/null | python -c "import sys; print(round(eval(sys.stdin.read()), 1))" 2>/dev/null || echo "N/A")
    echo "平均延迟: ${AVG_MS}ms"
fi

# ── Step 6: 验证 offload/onboard 日志 ─────────────────────

log_step "Step 6: 验证 DataSystem offload/onboard 日志"

echo "--- 搜索关键日志 ---"

# C++ 层 offload/onboard 关键字
echo ""
echo "=== offLoadCopy / onBoardCopy (C++ 层) ==="
grep -i "offload\|onboard\|offLoadCopy\|onBoardCopy" /tmp/trt_server_offload_test.log | head -20 || echo "(无匹配)"

echo ""
echo "=== DataSystem 连接 / KV Cache 操作 ==="
grep -i "datasystem\|KvCache\|kv_cache\|evict" /tmp/trt_server_offload_test.log | head -20 || echo "(无匹配)"

echo ""
echo "=== Python 层 KVCacheManager ==="
grep "KVCacheManager\|kv_source\|kv_lookup" /tmp/trt_server_offload_test.log | head -20 || echo "(无匹配)"

echo ""
echo "=== 错误 / 异常 ==="
grep -i "error\|exception\|traceback\|segfault\|fail" /tmp/trt_server_offload_test.log | head -10 || echo "(无匹配)"

# ── Step 7: 汇总 ─────────────────────────────────────────

log_step "Step 7: 汇总"

echo ""
echo "┌────────────────────────────────────────────┐"
echo "│          Offload/Onboard 验证报告           │"
echo "├────────────────────────────────────────────┤"
echo "│ 引擎路径:  $ENGINE_DIR"
echo "│ 成功率:    $SUCCESS/$BATCH_COUNT"
echo "│ 服务 PID:  $SERVER_PID"
echo "│ 日志文件:  /tmp/trt_server_offload_test.log"
echo "└────────────────────────────────────────────┘"
echo ""

echo "后续操作:"
echo "  tail -f /tmp/trt_server_offload_test.log     # 实时查看日志"
echo "  curl http://localhost:$PORT/health            # 健康检查"
echo "  kill $SERVER_PID                              # 停止服务"
echo ""
echo "验证通过标准:"
echo "  1. 批量请求成功率 > 95%"
echo "  2. 日志中出现 [Datasystem] Init KvCache Manager DataSystem success"
echo "  3. 日志中出现 offLoadCopy / onBoardCopy (C++ eviction 触发时)"
echo "  4. /health 中 datasystem 状态为 connected"
