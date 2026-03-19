# TensorRT-LLM 1.0.0 迁移指南

本文档说明从旧版本 TensorRT-LLM 迁移到 1.0.0 版本的注意事项。

## 版本要求

- **TensorRT-LLM**: 1.0.0
- **CUDA**: 12.2
- **Python**: 3.10+
- **NVIDIA Driver**: 535.230.02+

## 安装

```bash
# 卸载旧版本
pip uninstall tensorrt-llm

# 安装 1.0.0 版本
pip install tensorrt-llm==1.0.0 --extra-index-url https://pypi.nvidia.com

# 验证安装
python -c "import tensorrt_llm; print(f'TensorRT-LLM: {tensorrt_llm.__version__}')"
```

## 主要变化

### 1. 构建 API 变化

**旧版本 (0.8.x):**
```python
from tensorrt_llm.builder import Builder

builder = Builder()
config = builder.create_builder_config(...)
engine = builder.build_engine(network, config)
```

**新版本 (1.0.0):**
```python
from tensorrt_llm.builder import Builder, BuildConfig

build_config = BuildConfig(
    max_batch_size=32,
    max_input_len=256,
    max_output_len=256,
)
builder = Builder()
engine = builder.build_engine(model, build_config)
```

### 2. 推荐使用命令行工具

TensorRT-LLM 1.0.0 推荐使用 `trtllm-build` 命令行工具：

```bash
trtllm-build \
    --checkpoint ./checkpoints/decoder/decoder_best.pt \
    --output ./exported/decoder/decoder.engine \
    --max-batch-size 32 \
    --max-input-len 256 \
    --max-output-len 256 \
    --dtype float16 \
    --use-gpt-attention-plugin \
    --use-gemm-plugin
```

### 3. 模型配置变化

**旧版本:**
```python
from tensorrt_llm.models import GPTConfig

config = GPTConfig(
    vocab_size=256,
    num_layers=6,
    num_heads=8,
    hidden_size=256,
)
```

**新版本:**
```python
from tensorrt_llm.models import GPTConfig

config = GPTConfig(
    vocab_size=256,
    num_hidden_layers=6,  # 参数名变化
    num_attention_heads=8,
    hidden_size=256,
    ffn_hidden_size=1024,  # 新增参数
)
```

## 引擎构建

### Python API 方式

```python
from inference.trt_llm.build_engine import build_gpt_decoder_engine

success = build_gpt_decoder_engine(
    checkpoint_path='./checkpoints/decoder/decoder_best.pt',
    output_path='./exported/decoder/decoder.engine',
    max_batch_size=32,
    max_seq_len=512,
    dtype='float16',
    use_gpt_attention_plugin=True,
    use_gemm_plugin=True,
)
```

### 命令行方式

```bash
python inference/trt_llm/build_engine.py \
    --checkpoint_path ./checkpoints/decoder/decoder_best.pt \
    --output_path ./exported/decoder/decoder.engine \
    --max_batch_size 32 \
    --max_seq_len 512 \
    --dtype float16 \
    --use_gpt_attention_plugin \
    --use_gemm_plugin \
    --verbose
```

## 精度配置

TensorRT-LLM 1.0.0 支持更多精度选项：

| 精度 | 参数 | 说明 |
|------|------|------|
| FP32 | `--dtype float32` | 默认精度 |
| FP16 | `--dtype float16` | 推荐，速度提升 2x |
| BF16 | `--dtype bfloat16` | Ampere+ 架构 |
| INT8 | `--use_weight_only --weight_only_precision int8` | 量化加速 |
| INT4 | `--use_weight_only --weight_only_precision int4` | 最大压缩 |

## 插件系统

### GPT Attention 插件

```bash
--use_gpt_attention_plugin
```

优势：
- 优化的 attention 计算
- 支持更长的序列
- 更低的显存占用

### GEMM 插件

```bash
--use_gemm_plugin
```

优势：
- 优化的矩阵乘法
- 更好的性能

## 故障排查

### 问题 1: ImportError

```
ImportError: cannot import name 'Builder' from 'tensorrt_llm.builder'
```

**解决:**
```bash
# 确保安装的是 1.0.0 版本
pip show tensorrt-llm

# 如果版本不对，重新安装
pip install --upgrade tensorrt-llm==1.0.0 --extra-index-url https://pypi.nvidia.com
```

### 问题 2: CUDA 版本不匹配

```
RuntimeError: CUDA version mismatch
```

**解决:**
```bash
# 检查 CUDA 版本
nvidia-smi

# 确保安装的是 CUDA 12.2 版本
pip install tensorrt-llm==1.0.0 --extra-index-url https://pypi.nvidia.com
```

### 问题 3: 构建失败

```
Error: Engine build failed
```

**解决:**
1. 检查显存是否足够（建议 >= 24GB）
2. 使用更小的 batch size
3. 使用 FP16 精度
4. 查看详细日志：`--verbose`

### 问题 4: trtllm-build 命令未找到

```bash
trtllm-build: command not found
```

**解决:**
```bash
# 确保 tensorrt-llm 正确安装
pip install tensorrt-llm==1.0.0 --extra-index-url https://pypi.nvidia.com

# 检查 PATH
which trtllm-build

# 如果没有找到，尝试使用 Python 方式
python -m tensorrt_llm.commands.build --help
```

## 性能对比

| 配置 | 延迟 (P99) | 吞吐 (QPS) | 显存占用 |
|------|-----------|-----------|---------|
| PyTorch | ~50ms | ~200 | ~8GB |
| TensorRT-LLM 1.0.0 FP16 | ~15ms | ~800 | ~6GB |
| TensorRT-LLM 1.0.0 INT8 | ~10ms | ~1200 | ~4GB |

*测试环境: NVIDIA L40S, CUDA 12.2, batch_size=32*

## 回退方案

如果 TensorRT-LLM 1.0.0 构建失败，可以回退到 PyTorch 推理：

```bash
# 使用 PyTorch 推理服务（无需 TensorRT-LLM）
./scripts/start_trt_server.sh
```

PyTorch 推理服务功能完整，只是性能略低于 TensorRT-LLM。

## 参考文档

- [TensorRT-LLM 1.0.0 Release Notes](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.0.0)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [CUDA 12.2 Documentation](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

## 技术支持

如有问题，请提交 Issue 或联系维护团队。
