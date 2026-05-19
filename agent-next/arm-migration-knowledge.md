# PaiRec4TigerLLM ARM 移植知识记录

> 记录时间: 2026-04-20~21
> 源环境: x86_64 (训练+开发)
> 目标环境: ARM64 (aarch64) + NVIDIA RTX 4090 D
> 操作系统: openEuler 24.03 LTS SP3 (容器)
> 目标容器: `openeuler-24.03-lts-sp3:TensorRT-LLM-v1.0.0` (116GB)

---

## 1. 项目概述

PaiRec4TigerLLM 是一个基于生成式模型的推荐系统，核心组件：
- **RQ-VAE**: 生成语义 ID
- **GPT2 Decoder**: 生成式推荐模型
- **TensorRT-LLM 1.0.0**: GPU 推理加速
- **pairec (Go)**: 阿里巴巴开源推荐框架

本次任务: 将 x86 上端到端跑通的环境，完整移植到 ARM 机器。

---

## 2. 核心结论

| 组件 | x86 → ARM 是否可复用 | 方案 |
|------|---------------------|------|
| 训练好的模型权重 (`.pt`) | ✅ 直接拷贝 | 无需重新训练 |
| PyTorch 推理服务 | ✅ 可用 | 容器内装 PyTorch 即可 |
| TensorRT-LLM 加速 | ⚠️ 需重新构建引擎 | x86 引擎不能跨架构，ARM 需重新 build |
| Go 服务 (pairec) | ✅ 可编译运行 | 需 Go 1.24+，推荐 vendor 模式 |
| 数据预处理产物 | ✅ 直接拷贝 | `processed/` 目录完整迁移 |

---

## 3. 关键踩坑记录

### 3.1 Go 依赖与网络问题

**问题**: ARM 机器上 `go mod tidy` 报公司代理超时 (`141.5.74.26:3128`)。

**解决方案演进**:
1. 尝试传 834M 的 `go-mod-cache.tar.gz` → 传输中断失败
2. 尝试 `go mod vendor` → 在 x86 A 上因 `pairec` 模块缺失报错
3. **最终方案**: 在 x86 A 上 `go mod download` + `go mod vendor` 成功后再打包 vendor 目录传到 ARM

**教训**:
- `go/pkg/mod` (834M) 包含所有历史版本，远大于 `vendor/` (通常 < 150M)
- 优先传 `vendor/` 而非整个 mod cache
- `github.com/alibaba/pairec/v2 v2.6.2` 要求 **Go 1.24+**，Go 1.21 无法编译

### 3.2 Docker 镜像选择

**问题**: 有两个 TensorRT-LLM 镜像可选:
- `TensorRT-LLM-1.0.0` (63.6GB) → PyTorch 缺 7 个核心 CUDA 库，libstdc++ 也不够
- `TensorRT-LLM-v1.0.0` (116GB) → **最终选用**，环境完整

**教训**: 
- 大镜像通常更完整，63GB 镜像是阉割版
- ARM 上 NVIDIA 生态依赖容器内完整 CUDA runtime

### 3.3 容器内 CUDA/PyTorch 环境修复

**问题 1**: `import torch` 报 `libcudart.so.12` 找不到
- **原因**: `LD_LIBRARY_PATH` 指向 `/usr/local/cuda-12.9/lib64`，但实际 CUDA 库在 `/usr/local/lib/python3.11/site-packages/nvidia/*/lib/`
- **修复**:
  ```bash
  export LD_LIBRARY_PATH=$(find /usr/local/lib/python3.11/site-packages/nvidia -type d -name "lib" | tr '\n' ':')$LD_LIBRARY_PATH
  ```

**问题 2**: `import torch` 报 `libstdc++.so.6: version GLIBCXX_3.4.32 not found`
- **原因**: 系统默认 libstdc++ 只到 GLIBCXX_3.4.30，PyTorch 需要 3.4.32
- **修复**:
  ```bash
  export LD_LIBRARY_PATH=/opt/openEuler/gcc-toolset-14/root/usr/lib64:$LD_LIBRARY_PATH
  ```

**问题 3**: `import torch` 报 `libcufile.so.0` 等缺失
- **原因**: 容器 PyTorch 编译时链接了 cuFile，但运行时库缺失
- **结论**: 63GB 镜像环境残缺，直接换 116GB 镜像解决

### 3.4 TensorRT 引擎构建 (build_engine.py)

**问题**: `build_engine.py` 使用旧版 TensorRT API，在新版容器中批量报错:

| 报错 | 原因 | 修复 |
|------|------|------|
| `max_workspace_size` 不存在 | TensorRT 8.6+ 移除 | `set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ...)` |
| `max_batch_size` 不存在 | EXPLICIT_BATCH 模式下废弃 | 直接删除该行 |
| `build_engine()` 不存在 | TensorRT 10.x 移除 | `build_serialized_network()` + `deserialize_cuda_engine()` |
| `engine.serialize()` 返回类型变 | TensorRT 10.x | `bytes(engine.serialize())` |

**修复脚本**: 见 `fix_trt_api.py` (项目根目录)

### 3.5 Python `__init__.py` 导入链断裂

**问题**: 打包时 `--exclude='training/*/train.py'` 和 `--exclude='training/*/export.py'`，但 `training/rqvae/__init__.py` 和 `training/decoder/__init__.py` 在初始化时导入了这些文件。

**修复**:
```bash
sed -i 's/^\(from \.train import\|from \.export import\)/# \1/' training/rqvae/__init__.py
sed -i 's/^\(from \.train import\|from \.export import\)/# \1/' training/decoder/__init__.py
```

**教训**: 排除文件时要检查 `__init__.py` 的依赖链。

### 3.6 打包体积优化

**原始体积**: 31GB → **优化后**: ~2GB

| 排除项 | 体积 | 说明 |
|--------|------|------|
| `data/tenrec/processed/cache/tenrec_processed.pkl` | 18GB | DataLoader 缓存，运行时重建 |
| `data/tenrec/ctr_data_1M.csv` | 10GB | 原始数据，processed/ 已够用 |
| `services/go-mod-cache.tar.gz` | 834MB | 残留垃圾 |
| `checkpoints/*/decoder_epoch_*.pt` | ~600MB | 中间 checkpoint，只留 best |
| `*.engine` | 11MB | x86 引擎，ARM 不可用 |

**必须保留**:
- `training/rqvae/model.py` 和 `training/decoder/model.py`（`server.py` 会 import）
- `checkpoints/*/*_best.pt`
- `data/tenrec/processed/` (train/test sequences, semantic maps)
- `data/user_features.json`

---

## 4. 最终部署架构

```
┌─────────────────────────────────────────────┐
│           ARM Host (openEuler 24.03)         │
│  ┌───────────────────────────────────────┐  │
│  │  Docker Container                     │  │
│  │  openeuler-24.03-lts-sp3:TensorRT-LLM │  │
│  │  -v1.0.0 (116GB)                      │  │
│  │                                       │  │
│  │  Python 推理服务 (Port 8000)          │  │
│  │  Backend: TensorRT-LLM 1.0.0          │  │
│  │  Device: RTX 4090 D                   │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  Go 服务 (Port 8080)                        │
│  pairec + GenerativeRecall                  │
└─────────────────────────────────────────────┘
```

---

## 5. 命令速查表

### 5.1 容器启动

```bash
docker run -it --rm \
  --gpus all \
  --privileged \
  -v /opt/pairec4tigerllm:/app \
  -w /app \
  -p 8000:8000 \
  -e LD_LIBRARY_PATH="/opt/openEuler/gcc-toolset-14/root/usr/lib64:$(find /usr/local/lib/python3.11/site-packages/nvidia -type d -name 'lib' | tr '\n' ':')$LD_LIBRARY_PATH" \
  --name pairec-inference \
  openeuler-24.03-lts-sp3:TensorRT-LLM-v1.0.0 \
  /bin/bash
```

### 5.2 容器内构建 TensorRT 引擎

```bash
cd /app
export PYTHONPATH=/app

# 修复 API (只需一次)
python fix_trt_api.py

# 构建引擎
python inference/trt_llm/build_engine.py \
    --checkpoint_path ./checkpoints/decoder/decoder_best.pt \
    --output_path ./exported/decoder/decoder.engine \
    --max_batch_size 32 \
    --max_seq_len 512 \
    --dtype float16 \
    --use_gpt_attention_plugin \
    --use_gemm_plugin
```

### 5.3 容器内启动推理服务

```bash
python inference/trt_llm/server.py \
    --model_path ./checkpoints/decoder/decoder_best.pt \
    --port 8000 \
    --device cuda \
    --use_trt_llm \
    --max_batch_size 32 \
    --max_seq_len 512
```

### 5.4 宿主机启动 Go 服务

```bash
cd /opt/pairec4tigerllm
./pairec-server --config ./configs/pairec_config.json --port 8080
```

### 5.5 验证

```bash
# 推理服务
curl http://localhost:8000/health

# pairec 服务
curl http://localhost:8080/health

# 端到端推荐
curl -X POST http://localhost:8080/api/rec/feed \
    -H "Content-Type: application/json" \
    -d '{"uid": "test", "size": 5}'
```

---

## 6. 关键文件变更

| 文件 | 变更 | 原因 |
|------|------|------|
| `inference/trt_llm/build_engine.py` | TensorRT API 兼容性修复 | 新版 TensorRT 移除旧 API |
| `training/rqvae/__init__.py` | 注释 train/export import | 打包时排除了这些文件 |
| `training/decoder/__init__.py` | 注释 train/export import | 同上 |
| `fix_trt_api.py` | 新增 | 自动化修复 build_engine.py |
| `check_excludables.sh` | 新增 | 打包前体积检查工具 |

---

## 7. 经验教训

1. **模型权重跨平台**: PyTorch `.pt` 文件是平台无关的，x86 训练 ARM 直接用，无需重训。
2. **TensorRT 引擎不跨平台**: x86 构建的 `.engine` 不能在 ARM 上用，必须重新 build。
3. **Go vendor 优于 mod cache**: `vendor/` 体积小、传输稳、部署时无需网络。
4. **Docker 镜像要选对**: ARM NVIDIA 生态要用完整版镜像（116GB > 63GB），阉割版缺库严重。
5. **libstdc++ 版本陷阱**: 高版本 PyTorch 需要 GCC 13+ 的 libstdc++，系统默认的可能不够。
6. **打包时检查 `__init__.py`**: 排除 `.py` 文件时要确认不会被 `__init__.py` 导入。
7. **LD_LIBRARY_PATH 是神器**: ARM 容器里 CUDA 库路径可能分散在各个角落，需要手动拼接。

---

## 8. 后续优化方向

- [ ] 将 `fix_trt_api.py` 的修复逻辑合并到主分支 `build_engine.py`，支持 TensorRT 8.6+/10.x
- [ ] 为 `training/__init__.py` 添加条件导入，避免部署时依赖 train.py/export.py
- [ ] 制作一个精简的 ARM Docker 镜像（基于官方 nvidia/cuda ARM64）
- [ ] 将 `check_excludables.sh` 纳入 CI，自动检查打包体积

---

## 9. Kafka 连接踩坑记录（2026-04-24）

### 9.1 问题现象

pairec-server 启动后，Kafka FeatureConsumer 报错：
```
Read error: fetching message: failed to dial: failed to open connection to kafka:9092:
dial tcp: lookup kafka on [::1]:53: read udp [::1]:48010->[::1]:53: read: connection refused
```

但配置文件 `pairec_config.kafka.json` 里明确写的是：
```json
"kafka_config":{"brokers":["141.61.91.188:9092"],...}
```

### 9.2 排查过程（三阶段）

**阶段一：配置地址错误**
- 初始配置写的是 `localhost:9092`，直接改成目标 Kafka IP `141.61.91.188:9092`

**阶段二：二进制是旧的 x86，代码有 bug**
- `file pairec-server` 发现是 `ELF 64-bit LSB executable, x86-64`
- 说明在 ARM 上跑的是 x86 旧二进制（靠 box64/QEMU 模拟）
- 代码有两处 bug：
  1. `recallConfigJSON` 结构体**缺少 `FeatureSource` 和 `KafkaConfig` 字段**，`json.Unmarshal` 时这两个字段被静默丢弃
  2. `feature.NewConsumer` 后来加了 `maxMessages int` 参数，但 `generative_recall.go` 调用时只传了一个参数，**编译不过**
- 修复代码后重新编译 ARM 原生二进制

**阶段三：配置解析正确，但连接地址还是不对**
- 新二进制运行后，日志确认 `KafkaConfig.Brokers=[141.61.91.188:9092]` 已正确解析
- 但 consumer 启动后仍然报错 `lookup kafka on [::1]:53`
- 这说明客户端在尝试 DNS 解析 `kafka` 这个 hostname，而不是直接用配置里的 IP

### 9.3 根因分析

**Kafka 的 `advertised.listeners` 机制**。

客户端连接流程：
1. 客户端先连接配置的 broker 地址：`141.61.91.188:9092`
2. Kafka broker 返回集群元数据，其中包含 `advertised.listeners` 配置的地址
3. 如果 Kafka 服务器上 `advertised.listeners=PLAINTEXT://kafka:9092`，客户端会转而连接 `kafka:9092`
4. 客户端本地没有 `kafka` 的 DNS 解析，报错

### 9.4 解决方案（三选一）

| 方案 | 操作 | 适用场景 |
|------|------|----------|
| **改 hosts**（最快） | ARM 机器执行：`echo "141.61.91.188 kafka" >> /etc/hosts` | 快速验证，不动 Kafka 服务端 |
| **改 Kafka 配置**（根治） | Kafka 服务器修改 `server.properties`：`advertised.listeners=PLAINTEXT://141.61.91.188:9092`，重启 Kafka | 长期方案，需要服务端权限 |
| **双 listener** | 配置 `INTERNAL://kafka:9092` + `EXTERNAL://141.61.91.188:9092` | 内外网同时访问 |

**实际采用方案 1（改 hosts），立即解决。**

### 9.5 经验教训

1. **报错地址 ≠ 配置地址**：Kafka consumer 报错里的地址不一定是配置文件写的地址，可能是 Kafka broker 返回的 `advertised.listeners`。
2. **`lookup xxx on [::1]:53` = DNS 解析失败**：看到这种报错，直接检查 `/etc/hosts` 或 Kafka 的 `advertised.listeners` 配置。
3. **ARM 上必须重编二进制**：x86 二进制在 ARM 上靠模拟运行，会出现各种诡异行为（配置不生效、旧逻辑残留等），务必重新编译 ARM 原生版本。
4. **Go `json.Unmarshal` 静默丢弃字段**：结构体字段和 JSON 配置不同步时，不会报错，只会静默忽略，排查时容易忽略。
