# 部署检查清单

在迁移到新环境 (NVIDIA L40S + CUDA 12.2) 时，请按此清单检查。

## 环境准备

- [ ] NVIDIA Driver >= 535.230.02
- [ ] CUDA 12.2 安装正确
- [ ] Python 3.10+ 可用
- [ ] Go 1.21+ 可用
- [ ] 显存 >= 24GB (L40S 46GB 满足)

## 文件检查

### Python 模块
- [x] `training/__init__.py` - 训练模块入口
- [x] `training/rqvae/__init__.py` - RQ-VAE 模块
- [x] `training/decoder/__init__.py` - Decoder 模块
- [x] `inference/__init__.py` - 推理模块
- [x] `data/__init__.py` - 数据模块
- [x] `data/utils/__init__.py` - 数据工具模块

### 脚本文件
- [x] `scripts/preprocess_data.py` - 数据预处理
- [x] `scripts/generate_user_features.py` - 用户特征生成
- [x] `scripts/check_environment.py` - 环境检查
- [x] `scripts/train_rqvae.sh` - RQ-VAE 训练
- [x] `scripts/train_decoder.sh` - Decoder 训练
- [x] `scripts/build_trt_engine.sh` - TensorRT 引擎构建
- [x] `scripts/start_trt_server.sh` - 推理服务启动
- [x] `scripts/start_pairec.sh` - pairec 服务启动
- [x] `scripts/run_all.sh` - 一键启动
- [x] `scripts/test_api.sh` - API 测试

### 配置文件
- [x] `configs/pairec_config.json` - pairec 主配置
- [x] `configs/generative_config.yaml` - 生成式召回配置
- [x] `requirements.txt` - Python 依赖
- [x] `docker/Dockerfile` - Docker 镜像
- [x] `docker-compose.yml` - 容器编排

### 文档
- [x] `README.md` - 项目说明
- [x] `AGENTS.md` - 开发指南
- [x] `CONSTITUTION.md` - 项目章程
- [x] `CHANGELOG.md` - 版本历史
- [x] `docs/TENSORRT_LLM_1.0_MIGRATION.md` - 迁移指南
- [x] `docs/DEPLOYMENT_CHECKLIST.md` - 本文件

## 功能检查

### 1. 数据流程
- [x] Tenrec 数据加载 (data_loader.py)
- [x] 数据预处理 (preprocessor.py)
- [x] 语义 ID 映射生成
- [x] 用户特征生成 (generate_user_features.py)

### 2. 模型训练
- [x] RQ-VAE 模型定义
- [x] RQ-VAE 训练脚本
- [x] RQ-VAE 导出脚本 (ONNX + 语义 ID)
- [x] Decoder 模型定义
- [x] Decoder 训练脚本
- [x] Decoder 导出脚本 (ONNX + TensorRT 配置)

### 3. 推理服务
- [x] PyTorch 推理 (server.py)
- [x] TensorRT-LLM 1.0.0 推理支持 (server.py)
- [x] HTTP API (/health, /recommend)
- [x] 语义 ID 到物品 ID 映射
- [x] Python 客户端

### 4. pairec 集成
- [x] Go 服务入口 (main.go)
- [x] 生成式召回实现 (generative_recall.go)
- [x] TRT-LLM 客户端 (trtllm_client.go)
- [x] 配置定义 (generative_config.go)
- [x] HTTP API (/api/rec/feed)
- [x] 缓存支持

### 5. TensorRT-LLM 1.0.0
- [x] 引擎构建脚本 (build_engine.py)
- [x] 支持 FP16/BF16/INT8 精度
- [x] 支持 GPT Attention 插件
- [x] 支持 GEMM 插件
- [x] 备用构建方案 (ONNX + TensorRT)

### 6. 测试
- [x] RQ-VAE 单元测试
- [x] Decoder 单元测试
- [x] 端到端集成测试
- [x] API 测试脚本

## 部署步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd pairec4tigerllm

# 2. 环境检查
python scripts/check_environment.py

# 3. 准备数据
mkdir -p data/tenrec
ln -s /path/to/ctr_data_1M.csv data/tenrec/
python scripts/preprocess_data.py

# 4. 生成用户特征
python scripts/generate_user_features.py

# 5. 训练模型
./scripts/train_rqvae.sh
python scripts/preprocess_data.py --rqvae_model ./checkpoints/rqvae/rqvae_best.pt
./scripts/train_decoder.sh

# 6. 构建 TensorRT 引擎（可选）
python inference/trt_llm/build_engine.py \
    --checkpoint_path ./checkpoints/decoder/decoder_best.pt \
    --output_path ./exported/decoder/decoder.engine \
    --dtype float16

# 7. 测试
./scripts/test_api.sh

# 8. Docker 部署（可选）
docker-compose up -d
```

## 已知限制

1. **TensorRT-LLM 1.0.0**: 推理代码提供基本框架，具体实现需根据实际 API 调整
   - 已提供 PyTorch 回退机制
   - 见 `inference/trt_llm/server.py` 中的 `TensorRTLLMInference` 类

2. **首次构建**: TensorRT 引擎首次构建需要 5-30 分钟

3. **语义 ID**: 首次预处理使用哈希方式，建议训练 RQ-VAE 后重新预处理

## 故障排查

### 问题: TensorRT-LLM 导入失败
```bash
# 检查版本
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

# 重新安装
pip install tensorrt-llm==1.0.0 --extra-index-url https://pypi.nvidia.com
```

### 问题: 模型文件不存在
```bash
# 检查训练输出
ls -la checkpoints/rqvae/
ls -la checkpoints/decoder/

# 重新训练
./scripts/train_rqvae.sh
./scripts/train_decoder.sh
```

### 问题: 语义 ID 映射不存在
```bash
# 重新预处理数据
python scripts/preprocess_data.py --rqvae_model ./checkpoints/rqvae/rqvae_best.pt
```

### 问题: 用户特征文件不存在
```bash
# 生成用户特征
python scripts/generate_user_features.py
```

## 验证命令

```bash
# 健康检查
curl http://localhost:8080/health
curl http://localhost:8000/health

# 推荐测试
curl -X POST http://localhost:8080/api/rec/feed \
  -H "Content-Type: application/json" \
  -d '{"uid": "76295990", "size": 10}'

# 推理测试
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "history": [[100,50,25,10], [101,51,26,11]],
    "topk": 5
  }'
```

---

**最后更新**: 2026-03-19  
**版本**: 1.0.0
