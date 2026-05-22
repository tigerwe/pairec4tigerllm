#!/bin/bash
# PaiRec4TigerLLM — 基线验证 (本机无 GPU)
# 只做语法/导入检查，不跑 GPU 推理
set -e

echo "=== PaiRec4TigerLLM Baseline Check ==="
echo ""

# 1. Python 语法检查
echo "[1/4] Python syntax..."
python -c "import training.decoder.qwen3_generative_rec; print('  qwen3_generative_rec OK')"
python -c "import training.decoder.train; print('  train OK')"
python -c "import training.decoder.model; print('  model OK')"
python -c "import inference.trt_llm.server; print('  server OK')"
echo ""

# 2. JSON 有效性
echo "[2/4] JSON validation..."
python -c "
import json
with open('feature_list.json') as f:
    data = json.load(f)
print(f'  feature_list.json: {len(data[\"features\"])} features')
"
echo ""

# 3. Git status
echo "[3/4] Git status..."
echo "  branch: $(git branch --show-current)"
echo "  remote: $(git remote get-url gitcode 2>/dev/null || echo 'not set')"
echo ""

# 4. Dockerfile 存在
echo "[4/4] Dockerfiles..."
[ -f docker/Dockerfile.train ] && echo "  Dockerfile.train OK" || echo "  Dockerfile.train MISSING"
[ -f docker/Dockerfile.inference ] && echo "  Dockerfile.inference OK" || echo "  Dockerfile.inference MISSING"
echo ""

echo "=== Baseline PASSED ==="
echo "NOTE: GPU training/inference must be verified on remote machine."
echo "See AGENTS.md for remote commands."
