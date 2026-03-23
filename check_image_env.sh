#!/bin/bash
# 检查 TensorRT-LLM 镜像环境

echo "=== 系统信息 ==="
cat /etc/os-release | head -3

echo ""
echo "=== 所有可用的 Python ==="
which -a python python3 2>/dev/null
ls -la /usr/bin/python* 2>/dev/null | head -5

echo ""
echo "=== Python 版本详情 ==="
for py in /usr/bin/python3.* /opt/conda/bin/python* 2>/dev/null; do
    if [ -x "$py" ]; then
        echo "$py: $($py --version 2>&1)"
    fi
done

echo ""
echo "=== 当前 PATH ==="
echo $PATH

echo ""
echo "=== conda 环境 ==="
ls -la /opt/conda/envs/ 2>/dev/null || echo "无 conda 环境目录"

echo ""
echo "=== tensorrt-llm 安装位置 ==="
pip show tensorrt-llm 2>/dev/null | grep -E "Location|Requires|Version"

echo ""
echo "=== 正确的 Python 应该在哪里 ==="
echo "检查 /opt/python 或 /usr/local/bin/python:"
ls -la /opt/python*/bin/python* 2>/dev/null || echo "无 /opt/python"
ls -la /usr/local/bin/python* 2>/dev/null | head -5 || echo "无 /usr/local/bin/python"
