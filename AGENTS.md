# PaiRec4TigerLLM - AI 代理开发指南

## 项目概述

PaiRec4TigerLLM 是一个基于生成式模型的推荐系统，集成了：
- **pairec**：阿里巴巴开源的 Go 语言推荐框架
- **RQ-VAE**：残差量化变分自编码器，用于生成语义 ID
- **GPT2 Decoder**：生成式推荐模型
- **TensorRT-LLM 1.0.0**：NVIDIA GPU 推理加速库 (CUDA 12.2)

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      User Request                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PaiRec API (Go)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  召回层       │  │  精排层       │  │  重排层       │       │
│  │ ┌──────────┐ │  │              │  │              │       │
│  │ │生成式召回 │ │  │              │  │              │       │
│  │ │(Generative)│ │  │              │  │              │       │
│  │ └──────────┘ │  │              │  │              │       │
│  │ ┌──────────┐ │  │              │  │              │       │
│  │ │ 其他召回  │ │  │              │  │              │       │
│  │ └──────────┘ │  │              │  │              │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│           TensorRT-LLM Inference Service (Python)           │
│                    Port: 8000                               │
└─────────────────────────────────────────────────────────────┘
```

## 关键代码文件

### 数据处理层 (Python)

| 文件 | 说明 |
|------|------|
| `data/utils/data_loader.py` | Tenrec 数据加载器 |
| `data/utils/preprocessor.py` | 数据预处理器 |

### 模型训练层 (Python)

| 文件 | 说明 |
|------|------|
| `training/rqvae/model.py` | RQ-VAE 模型定义 |
| `training/rqvae/train.py` | RQ-VAE 训练脚本 |
| `training/rqvae/export.py` | RQ-VAE 导出脚本 |
| `training/decoder/model.py` | GPT2 Decoder 模型定义 |
| `training/decoder/train.py` | Decoder 训练脚本 |
| `training/decoder/export.py` | Decoder 导出脚本 |

### 推理服务层 (Python)

| 文件 | 说明 |
|------|------|
| `inference/trt_llm/server.py` | TensorRT-LLM 推理服务 |
| `inference/client/python_client.py` | Python 客户端 |

### pairec 集成层 (Go)

| 文件 | 说明 |
|------|------|
| `services/config/generative_config.go` | 配置定义 |
| `services/recall/trtllm_client.go` | TRT-LLM 客户端 |
| `services/recall/generative_recall.go` | 生成式召回实现 |
| `services/main.go` | 服务入口 |

## 开发规范

### Python 代码规范

1. **文件头**：每个文件必须包含编码声明
   ```python
   # -*- coding: utf-8 -*-
   """模块文档字符串."""
   ```

2. **导入顺序**：
   - 标准库
   - 第三方库
   - 项目内部模块

3. **命名规范**：
   - 类名：PascalCase（`GenerativeDecoder`）
   - 函数/变量：snake_case（`get_recommendations`）
   - 常量：UPPER_SNAKE_CASE（`MAX_SEQ_LEN`）

4. **文档字符串**：
   ```python
   def recommend(self, user_id: str, topk: int = 10) -> List[Item]:
       """获取推荐.

       Args:
           user_id: 用户 ID
           topk: 推荐数量

       Returns:
           推荐物品列表
       """
   ```

### Go 代码规范

1. **命名规范**：
   - 导出标识符：PascalCase（`GenerativeRecall`）
   - 私有标识符：camelCase（`historyFrom`）

2. **错误处理**：
   ```go
   if err != nil {
       return fmt.Errorf("operation failed: %w", err)
   }
   ```

3. **日志格式**：
   ```go
   log.Info(fmt.Sprintf("requestId=%s\tmodule=%s\tname=%s\tcount=%d",
       ctx.RecommendId, "GenerativeRecall", r.modelName, len(items)))
   ```

## 常见任务

### 添加新的召回路

1. 在 `services/recall/` 下创建新的召回文件
2. 实现召回接口：
   ```go
   type NewRecall struct {
       *recall.BaseRecall
       // 自定义字段
   }

   func (r *NewRecall) GetCandidateItems(user *module.User, ctx *context.RecommendContext) []*module.Item {
       // 实现召回逻辑
   }
   ```
3. 在配置中注册新的召回类型

### 修改模型架构

1. **RQ-VAE**：编辑 `training/rqvae/model.py`
   - `ResidualVectorQuantizer`：修改量化器
   - `Encoder`/`Decoder`：修改编解码器

2. **GPT2 Decoder**：编辑 `training/decoder/model.py`
   - `GenerativeDecoder`：修改 Transformer 层数、头数等

### 添加新的 API 接口

1. Python 推理服务：编辑 `inference/trt_llm/server.py`
   ```python
   @app.route('/new_endpoint', methods=['POST'])
   def new_endpoint():
       # 实现逻辑
   ```

2. Go 客户端：编辑 `services/recall/trtllm_client.go`
   ```go
   func (c *TRTLLMClient) NewMethod() (*Response, error) {
       // 实现逻辑
   }
   ```

## 调试技巧

### Python 调试

```python
# 添加日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查模型输出
print(f"Logits shape: {logits.shape}")
```

### Go 调试

```go
// 使用 fmt.Printf 打印调试信息
fmt.Printf("Debug: user_id=%s, history=%v\n", user.Id, history)

// 检查错误
if err != nil {
    log.Error(fmt.Sprintf("Error: %v", err))
}
```

## 测试

### 单元测试

```bash
# Python
python -m pytest tests/unit/

# Go
go test ./services/...
```

### 集成测试

```bash
# 启动服务
./scripts/start_trt_server.sh &
./scripts/start_pairec.sh &

# 运行测试
curl -v http://localhost:8080/api/rec/feed \
  -d '{"uid": "123", "size": 10}'
```

## 性能优化

### 推理优化

1. **TensorRT-LLM 1.0.0**：
   - 使用 FP16 量化: `--dtype float16`
   - 启用 GPT Attention 插件: `--use_gpt_attention_plugin`
   - 启用 GEMM 插件: `--use_gemm_plugin`
   - 调整 max_seq_len
   - 构建命令:
     ```bash
     python inference/trt_llm/build_engine.py \
         --checkpoint_path ./checkpoints/decoder/decoder_best.pt \
         --output_path ./exported/decoder/decoder.engine \
         --dtype float16 \
         --use_gpt_attention_plugin \
         --use_gemm_plugin
     ```

2. **缓存**：
   - 启用 Redis 缓存
   - 设置合理的 TTL

### 训练优化

1. **混合精度**：
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

2. **数据加载**：
   - 使用 `num_workers > 0`
   - 启用 `pin_memory`

## 故障排查

### 常见问题

1. **TensorRT-LLM 1.0.0 启动失败**
   - 检查 CUDA 版本是否为 12.2
   - 检查 TensorRT-LLM 版本: `pip show tensorrt-llm`
   - 检查模型文件是否存在
   - 查看详细日志: `docker-compose logs inference`

2. **推理结果为空**
   - 检查语义 ID 映射是否正确
   - 检查输入历史是否为空

3. **pairec 集成失败**
   - 检查配置文件格式
   - 检查服务地址是否可达
   - 检查 user_features.json 是否存在

## 已知限制

1. **TensorRT-LLM 1.0.0 推理**: 
   - 由于 TensorRT-LLM 1.0.0 API 可能变化，`server.py` 中的 TensorRT-LLM 推理代码提供基本框架
   - 系统已提供完整的 PyTorch 回退机制，确保服务可用
   - 生产环境建议根据实际 API 调整 `TensorRTLLMInference` 类的实现

2. **语义 ID 映射**:
   - 首次数据预处理使用哈希方式生成临时语义 ID
   - 建议在 RQ-VAE 训练完成后重新预处理数据以获得最优效果

3. **用户特征**:
   - pairec 配置依赖 `data/user_features.json`
   - 使用 `scripts/generate_user_features.py` 生成

## 参考资源

- [pairec 文档](https://github.com/alibaba/pairec)
- [RQ-VAE 论文](https://arxiv.org/abs/2305.05065)
- [TensorRT-LLM 1.0.0 Release](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.0.0)
- [TensorRT-LLM 文档](https://nvidia.github.io/TensorRT-LLM/)
- [Tenrec 数据集](https://github.com/yuangh-x/2022-M10-Tenrec)
- [迁移指南](docs/TENSORRT_LLM_1.0_MIGRATION.md)

## 联系

如有问题，请查阅项目文档或提交 Issue。
