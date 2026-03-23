#!/bin/bash
# 检查 TensorRT-LLM 环境脚本

echo "===== Python 环境检查 ====="
echo "Python 路径: $(which python)"
echo "Python 版本: $(python --version)"
echo ""

echo "===== 检查 TensorRT-LLM ====="
pip show tensorrt-llm 2>/dev/null || echo "❌ pip 中未找到 tensorrt-llm"
echo ""

echo "===== 检查 conda 环境 ====="
conda info --envs 2>/dev/null || echo "无 conda 环境或 conda 未激活"
echo ""

echo "===== 检查 Python 路径 ====="
python -c "import sys; print('\n'.join(sys.path[:5]))"
echo ""

echo "===== 尝试导入 TensorRT-LLM ====="
python -c "import tensorrt_llm; print(f'✅ TensorRT-LLM 版本: {tensorrt_llm.__version__}')" 2>/dev/null || echo "❌ 无法导入 tensorrt_llm"
echo ""

echo "===== 检查 trtllm-build 命令 ====="
which trtllm-build 2>/dev/null || echo "❌ 未找到 trtllm-build 命令"
