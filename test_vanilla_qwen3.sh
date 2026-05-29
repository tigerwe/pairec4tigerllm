#!/bin/bash
# ============================================================================
# 标准 Qwen3-0.6B 推理测试：验证 FMHA crash 是否与扩展词表模型有关
#
# 如果这个脚本不 crash → 根因在扩展词表模型的特异性
# 如果这个脚本也 crash → SM 89 + TRT-LLM 1.0.0 的 FMHA 就是个死路
# ============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; }

QWEN3_MODEL="${QWEN3_MODEL_PATH:-./models/Qwen3-0.6B}"
if [ ! -d "$QWEN3_MODEL" ]; then
    fail "Qwen3 基础模型不存在: $QWEN3_MODEL"
    exit 1
fi

echo "模型: $QWEN3_MODEL"
echo ""

# ═══ 1. convert_checkpoint ═══
echo "── 1. convert_checkpoint ──"
rm -rf ./trt_ckpt_vanilla
python /TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py \
    --model_dir "$QWEN3_MODEL" \
    --output_dir ./trt_ckpt_vanilla \
    --dtype float16 \
    --tp_size 1 --pp_size 1
ok "convert_checkpoint done"

# ═══ 2. trtllm-build ═══
echo ""
echo "── 2. trtllm-build ──"
rm -rf ./trt_engines/qwen3_vanilla
trtllm-build \
    --checkpoint_dir ./trt_ckpt_vanilla \
    --output_dir ./trt_engines/qwen3_vanilla \
    --max_batch_size 8 \
    --max_input_len 256 \
    --max_seq_len 320
ok "trtllm-build done"

# ═══ 3. 极简推理测试 ═══
echo ""
echo "── 3. 推理测试 ──"
python3 -c "
import torch
from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunnerCpp

tokenizer = AutoTokenizer.from_pretrained('$QWEN3_MODEL', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print('Loading engine...')
runner = ModelRunnerCpp.from_dir('./trt_engines/qwen3_vanilla')
print('Engine loaded')

prompt = '你好，请介绍一下自己'
encoded = tokenizer(prompt, return_tensors='pt')
input_ids = [encoded['input_ids'][0].cuda()]

print(f'Generating... prompt_len={input_ids[0].shape[0]}')
outputs = runner.generate(
    input_ids,
    max_new_tokens=20,
    end_id=tokenizer.eos_token_id,
    pad_id=tokenizer.pad_token_id,
)

result = tokenizer.decode(outputs[0][0])
print(f'Output: {result}')
print('')
print('SUCCESS — 标准 Qwen3-0.6B 推理不 crash')
"
ok "推理成功"
