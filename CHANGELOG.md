# Changelog

所有版本的变更记录。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

---

## [1.0.0] - 2026-03-19

### 🎉 首个正式版本发布

PaiRec4TigerLLM 1.0.0 是一个完整的端到端生成式推荐系统，适配 NVIDIA L40S + CUDA 12.2 环境，支持 TensorRT-LLM 1.0.0。

### ✨ 核心功能

#### 模型训练
- **RQ-VAE 训练**: 完整的残差量化 VAE 训练流程
  - 4 层量化器，每层 256 个码本向量
  - 支持早停、TensorBoard 监控、检查点保存
  - 导出 ONNX 格式用于推理
  
- **GPT2 Decoder 训练**: 生成式推荐模型
  - 6 层 Transformer，8 头注意力
  - 支持 Temperature 采样和 Beam Search
  - 支持 warm-up 学习率调度

#### 推理服务
- **PyTorch 推理服务**: HTTP API (端口 8000)
  - `/health` 健康检查
  - `/recommend` 推荐接口
  - 支持批量推理
  
- **TensorRT-LLM 1.0.0 支持**: GPU 加速推理（可选）
  - 适配 CUDA 12.2
  - 支持 FP16/BF16/INT8 精度
  - 支持 GPT Attention 插件
  - 引擎构建脚本

#### pairec 集成
- **生成式召回**: Go 语言实现
  - 复用 TigerRecallConf 配置
  - 支持本地/Redis 缓存
  - 完整的语义 ID 映射加载
  
- **HTTP API**: RESTful 接口 (端口 8080)
  - `/api/rec/feed` 推荐接口
  - 统一的响应格式
  - 完整的错误处理

### 🛠️ 基础设施

#### Docker 支持
- Dockerfile (CUDA 12.2, Ubuntu 22.04)
- docker-compose.yml (多服务编排)
- 健康检查配置

#### 环境验证
- `scripts/check_environment.py`: 环境检查脚本
- 自动检测 CUDA、GPU、依赖版本
- 清晰的错误提示

#### 测试覆盖
- 单元测试: RQ-VAE、Decoder 模型测试
- 集成测试: 端到端流程验证
- 测试数据生成

### 📦 依赖要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 基础环境 |
| CUDA | 12.2 | GPU 计算 |
| NVIDIA Driver | 535.230.02+ | 驱动支持 |
| PyTorch | 2.1.2+cu121 | 深度学习框架 |
| TensorRT-LLM | 1.0.0 | 推理加速（可选） |
| Go | 1.21+ | pairec 服务 |

### 🔧 TensorRT-LLM 1.0.0 特别说明

#### 安装方式
```bash
pip install tensorrt-llm==1.0.0 --extra-index-url https://pypi.nvidia.com
```

#### 主要变化
- 新的构建 API (BuildConfig)
- 推荐使用 `trtllm-build` 命令行工具
- 改进的插件系统
- 更好的 CUDA 12.2 支持

#### 构建引擎
```bash
# 方式 1: 使用 Python API
python inference/trt_llm/build_engine.py \
    --checkpoint_path ./checkpoints/decoder/decoder_best.pt \
    --output_path ./exported/decoder/decoder.engine \
    --dtype float16

# 方式 2: 使用命令行工具
trtllm-build \
    --checkpoint ./checkpoints/decoder/decoder_best.pt \
    --output ./exported/decoder/decoder.engine \
    --max-batch-size 32 \
    --dtype float16
```

#### 已知限制
- TensorRT-LLM 1.0.0 需要 Python 3.10+
- 某些旧版 GPU 可能不完全支持
- 首次构建时间较长（5-30分钟）

### 🔧 已知限制

1. **TensorRT-LLM 引擎构建**: 提供的脚本为基础实现，生产环境可能需要针对特定模型调优
2. **语义 ID 生成**: 首次预处理使用哈希方式，建议训练 RQ-VAE 后重新预处理以获得更优效果
3. **批处理优化**: 当前推理服务支持批量请求，但内部实现为串行处理

### 📋 快速开始

```bash
# 1. 环境检查
python scripts/check_environment.py

# 2. 数据预处理
python scripts/preprocess_data.py --input /path/to/ctr_data_1M.csv

# 3. 训练模型
./scripts/train_rqvae.sh
python scripts/preprocess_data.py --rqvae_model ./checkpoints/rqvae/rqvae_best.pt
./scripts/train_decoder.sh

# 4. 构建 TensorRT 引擎（可选，需要 TensorRT-LLM 1.0.0）
python inference/trt_llm/build_engine.py

# 5. 启动服务
./scripts/run_all.sh

# 6. 测试
./scripts/test_api.sh
```

### 🐛 修复的问题

- 修复语义 ID 映射加载失败的问题
- 修复 Go 召回层配置解析错误
- 修复 TensorRT-LLM 导出脚本的占位符问题
- 修复数据预处理脚本的类型转换问题
- 适配 TensorRT-LLM 1.0.0 API 变化

### 📝 文档

- 完整的 README.md 快速开始指南
- AGENTS.md AI 代理开发指南
- CONSTITUTION.md 项目章程
- 代码内完整的中文文档字符串
- TensorRT-LLM 1.0.0 迁移指南

---

## [0.1.0] - 2026-03-18

### 🚀 初始版本

- 项目初始化
- 基础架构搭建
- RQ-VAE 和 Decoder 模型定义
- 基础训练脚本
- 推理服务原型

---

[1.0.0]: https://github.com/yourorg/pairec4tigerllm/releases/tag/v1.0.0
[0.1.0]: https://github.com/yourorg/pairec4tigerllm/releases/tag/v0.1.0
