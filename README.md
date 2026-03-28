# PaiRec4TigerLLM

基于 pairec 和 TensorRT-LLM 的生成式推荐系统。

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](VERSION)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.2-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

## 项目简介

PaiRec4TigerLLM 是一个端到端推荐系统，集成了生成式召回技术。项目特点：

- **生成式召回**：使用 RQ-VAE + GPT2 模型生成语义 ID 进行推荐
- **TensorRT-LLM 推理**：高性能 GPU 推理加速（可选）
- **pairec 框架**：基于阿里巴巴开源的 pairec 推荐服务框架
- **Tenrec 数据集**：使用 Tenrec 的 `ctr_data_1M.csv` 数据集
- **Docker 支持**：完整的容器化部署方案

## 系统架构

### 基础架构

```
用户请求 -> pairec API (Go) -> 多路召回（生成式召回 + 传统召回）-> 精排 -> 重排 -> 返回结果
                          ↓
                   TensorRT-LLM 推理服务 (Python)
```

### 实时特征架构（新增）

```
用户点击流 -> Kafka (user-clicks) -> Flink Processor -> Kafka (user-features)
                                                               ↓
用户请求 -> pairec API (Go) -> GenerativeRecall -> 实时特征缓存 -> TensorRT-LLM
```

**特性**：
- **实时特征**：Flink 实时计算用户点击序列，分钟级延迟
- **降级保障**：Kafka 不可用时自动回退到离线 JSON
- **配置开关**：通过 `feature_source` 配置切换实时/离线模式

## 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 基础环境 |
| CUDA | 12.2 | GPU 计算 |
| NVIDIA Driver | 535.230.02+ | 驱动支持 |
| GPU | NVIDIA L40S 或同等算力 | 推荐显存 >= 24GB |
| PyTorch | 2.1.2+cu121 | 深度学习框架 |
| Go | 1.21+ | pairec 服务 |
| TensorRT-LLM | 1.0.0 | 推理加速（可选） |

## 项目结构

```
pairec4tigerllm/
├── data/                    # 数据集
│   ├── tenrec/             # Tenrec 数据集
│   └── utils/              # 数据加载和预处理
├── training/               # 模型训练
│   ├── rqvae/              # RQ-VAE 训练
│   └── decoder/            # GPT2 Decoder 训练
├── inference/              # 推理服务
│   ├── trt_llm/            # TensorRT-LLM 服务
│   └── client/             # 客户端
├── services/               # pairec 服务 (Go)
│   ├── config/             # 配置定义
│   ├── recall/             # 召回实现
│   └── main.go             # 服务入口
├── configs/                # 配置文件
├── scripts/                # 工具脚本
├── tests/                  # 测试用例
├── docker/                 # Docker 配置
└── docker-compose.yml      # 容器编排
```

## 快速开始

### 方式一：Docker 部署（推荐）

```bash
# 1. 克隆项目
git clone <repository-url>
cd pairec4tigerllm

# 2. 准备数据
mkdir -p data/tenrec
ln -s /path/to/Tenrec/ctr_data_1M.csv data/tenrec/

# 3. 构建镜像
docker-compose build

# 4. 启动服务
docker-compose up -d

# 5. 验证服务
curl http://localhost:8080/health
```

### 方式二：本地部署

#### 1. 环境检查

```bash
# 检查环境是否满足要求
python scripts/check_environment.py
```

#### 2. 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 安装 TensorRT-LLM（可选）
pip install tensorrt-llm==0.8.0 --extra-index-url https://pypi.nvidia.com

# 安装 Go 依赖
cd services && go mod tidy
```

#### 3. 数据准备

```bash
# 确保 Tenrec 数据在正确位置
ln -s /path/to/Tenrec/ctr_data_1M.csv data/tenrec/

# 数据预处理
python scripts/preprocess_data.py --input /path/to/ctr_data_1M.csv
```

#### 4. 模型训练

```bash
# 4.1 训练 RQ-VAE
./scripts/train_rqvae.sh

# 4.2 重新预处理数据（使用 RQ-VAE 生成语义 ID）
python scripts/preprocess_data.py \
    --input /path/to/ctr_data_1M.csv \
    --rqvae_model ./checkpoints/rqvae/rqvae_best.pt

# 4.3 训练 Decoder
./scripts/train_decoder.sh

# 4.4 导出模型
python training/decoder/export.py \
    --model_path ./checkpoints/decoder/decoder_best.pt \
    --output_dir ./exported/decoder

# 4.5 构建 TensorRT 引擎（可选）
./scripts/build_trt_engine.sh
```

#### 5. 启动服务

```bash
# 方式 1: 分别启动
./scripts/start_trt_server.sh  # 推理服务 (端口 8000)
./scripts/start_pairec.sh      # 推荐服务 (端口 8080)

# 方式 2: 一键启动
./scripts/run_all.sh
```

## API 接口

### 健康检查

```bash
curl http://localhost:8080/health
curl http://localhost:8000/health
```

### 推荐接口

**请求：**

```bash
curl -X POST http://localhost:8080/api/rec/feed \
  -H "Content-Type: application/json" \
  -d '{
    "uid": "76295990",
    "size": 10,
    "scene_id": "home_feed"
  }'
```

**响应：**

```json
{
    "code": 200,
    "msg": "success",
    "request_id": "e332fe9c-7d99-45a8-a047-bc7ec33d07f6",
    "size": 10,
    "experiment_id": "",
    "items": [
        {
            "item_id": "248791390",
            "score": 0.9991594902203332,
            "retrieve_id": "generative_recall"
        }
    ]
}
```

### 推理服务接口

**直接调用推理服务：**

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "history": [[100, 50, 25, 10], [101, 51, 26, 11]],
    "topk": 10,
    "temperature": 1.0,
    "beam_width": 1
  }'
```

## 模型架构

### RQ-VAE

- **输入**：物品特征
- **输出**：语义 ID（4 层，每层 256 个码本向量）
- **架构**：Encoder (256->128->64) + Residual Quantizer + Decoder
- **损失**：重建损失 + 承诺损失 + 码本损失

### GPT2 Decoder

- **架构**：Decoder-only Transformer
  - 6 层 Transformer
  - 8 头注意力
  - 256 维嵌入
  - 1024 维 FFN
- **输入**：语义 ID 序列
- **输出**：下一个语义 ID 的概率分布
- **生成方式**：Temperature 采样 / Beam Search

## 配置说明

### pairec 配置

编辑 `configs/pairec_config.json`：

```json
{
  "recall": {
    "generative_recall": {
      "type": "GenerativeRecall",
      "model_name": "generative_recall",
      "recall_count": 50,
      "cache_adapter": "local",
      "cache_time": 300,
      "tiger_recall_conf": {
        "tiger_name": "http://localhost:8000",
        "top_k": 50,
        "temperature": 1.0,
        "beam_width": 1,
        "history_from": "user_feature",
        "history_feature_name": "click_history",
        "history_delimiter": ",",
        "history_max_length": 20
      }
    }
  }
}
```

### 生成式召回配置

编辑 `configs/generative_config.yaml`：

```yaml
server_url: "http://localhost:8000"
timeout: 500ms
max_retries: 3
topk: 50
temperature: 1.0
beam_width: 1
history_max_length: 20
cache_enable: true
cache_time: 300
```

## 测试

```bash
# 运行单元测试
python -m pytest tests/unit/ -v

# 运行集成测试
python -m pytest tests/integration/ -v

# 运行所有测试
python -m pytest tests/ -v

# 测试 API
./scripts/test_api.sh
```

## 性能优化

### GPU 推理优化

1. **TensorRT-LLM 1.0.0 加速**：
   ```bash
   # 构建 FP16 引擎
   python inference/trt_llm/build_engine.py \
       --checkpoint_path ./checkpoints/decoder/decoder_best.pt \
       --output_path ./exported/decoder/decoder.engine \
       --dtype float16 \
       --use_gpt_attention_plugin \
       --use_gemm_plugin
   ```

2. **调整批大小**：
   ```bash
   export MAX_BATCH_SIZE=64
   ./scripts/start_trt_server.sh
   ```

3. **调整序列长度**：
   ```bash
   export MAX_SEQ_LEN=256
   ./scripts/start_trt_server.sh
   ```

### 缓存优化

1. **启用 Redis 缓存**：
   ```json
   {
     "cache_adapter": "redis",
     "cache_time": 600
   }
   ```

2. **调整缓存时间**：根据数据更新频率设置合理的 TTL

## 监控与日志

### 日志格式

```
requestId={uuid}\tmodule={module_name}\tname={model_name}\tcount={item_count}\tcost={time_ms}
```

### 监控指标

- 请求 QPS
- 响应延迟（P50/P99）
- 缓存命中率
- 模型推理时间

### TensorBoard

```bash
# 训练时自动记录
tensorboard --logdir=./logs --port=6006
```

## 故障排查

### 常见问题

1. **CUDA out of memory**
   - 减小 `batch_size`
   - 减小 `max_seq_len`
   - 使用 FP16 精度

2. **TensorRT-LLM 构建失败**
   - 检查 CUDA 版本是否匹配
   - 确保有足够的磁盘空间
   - 查看详细日志：`python build_engine.py --verbose`

3. **语义 ID 映射加载失败**
   - 检查 `semantic_id_map.json` 是否存在
   - 重新运行数据预处理

## 版本历史

详见 [CHANGELOG.md](CHANGELOG.md)

## 许可证

MIT License

## 致谢

- [pairec](https://github.com/alibaba/pairec) - 阿里巴巴开源推荐框架
- [RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender) - 生成式推荐实现
- [Tenrec](https://github.com/yuangh-x/2022-M10-Tenrec) - 推荐数据集
- [TensorRT-LLM 1.0.0](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.0.0) - NVIDIA GPU 推理加速

---

**版本**: 1.0.0 | **更新日期**: 2026-03-19
