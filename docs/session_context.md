# PaiRec4TigerLLM — Qwen3 Prompt Template 训练与推理 完整工作记录

> 自动生成于 2026-05-21 | 项目路径: `~/pairec4tigerllm` | 远端: gitcode.com/weixin_43325008/pairec4tigerllm (dev 分支)

---

## 目录
1. [项目目标](#一项目目标)
2. [架构演进三阶段](#二架构演进三阶段)
3. [当前实现细节](#三当前实现细节)
4. [训练数据格式](#四训练数据格式)
5. [关键 Bug 与修复日志](#五关键-bug-与修复日志)
6. [详细问题分析与根因](#六详细问题分析与根因)
7. [当前进展与时间线](#七当前进展与时间线)
8. [训练命令](#八训练命令)
9. [推理服务](#九推理服务)
10. [多卡 DDP 架构](#十多卡-ddp-架构)
11. [Docker 容器化方案](#十一docker-容器化方案)
12. [8MB KV Cache 与 DataSystem](#十二8mb-kv-cache-与-datasystem)
13. [京东方案对比](#十三京东方案对比)
14. [关键文件清单](#十四关键文件清单)
15. [待办事项](#十五待办事项)

---

## 一、项目目标

用 Qwen3-0.6B 替换自研 GPT2 Decoder，跑通生成式召回全链路。参照京东做法使用 Prompt Template 发挥 LLM 能力。DataSystem KV Cache 单条上限 2MB/8MB。

## 二、架构演进三阶段

### v0: GPT2 Decoder（原始）
- 6层自研 Transformer，4×256 Embedding sum 输入，4×Linear(256→256) 输出。~4M参数。

### v1: Qwen3 inputs_embeds（废弃）
- 28层冻结+LoRA，绕过词表走 inputs_embeds。废弃原因：没用到 LLM 语言能力。

### v2: Qwen3 Prompt Template（当前）
- 新增 1024 个语义ID token (`<s0_0>~<s3_255>`) 到词表 (151936+1024=152693)
- Prompt: "用户按时间顺序点击过以下商品：{history}\n他接下来点击了：{target}"
- forward: 绕开 lm_head 全序列计算，只算 4 个目标位置（~250x lm_head 节省）
- LoRA r=8 + modules_to_save=["lm_head","model.embed_tokens"]
- 训练参数: ~317M/915M (34.7%)
- generate: 原生 model.generate() + _parse_output 提取四元组

## 三、当前实现细节

### 模型初始化
1. Tokenizer → 注册1024特殊token → _id_to_sem/_sem_to_id双向映射
2. AutoModelForCausalLM.from_pretrained(torch_dtype=bf16)
3. resize_token_embeddings(152693)
4. LoRA: target_modules=[q/k/v/o/gate/up/down_proj], modules_to_save=[lm_head, model.embed_tokens]
5. gradient_checkpointing_enable()
6. 缓存 _transformer(Qwen3Model) 和 _lm_head(lm_head层) 引用

### forward 训练
```
inputs: semantic_ids[B, max_seq_len, 4], attention_mask[B, max_seq_len], labels[B, max_seq_len, 4]

1. 提取历史 + 目标 (target=labels最后有效位置)
2. 构造prompt → tokenizer(2048截断)
3. 找目标位置 (最后4个语义token)
4. embed → inputs_embeds.requires_grad_(True)
5. _transformer → hidden_states[B, prompt_len, 1024]
6. h = hidden_states[b, p-1]  # causal: pos-1预测pos
   logits = _lm_head(h)       # [4, 152693]
   loss = F.cross_entropy     # 仅4个目标位置
```

### generate 推理
```
input_ids[B, history_len, 4] → 构造推理prompt → tokenizer
→ base_model.generate(max_new_tokens, do_sample=True)
→ _parse_output: 扫描连续4token, layers==[0,1,2,3] 有效
→ 返回 [B, max_items, 4]
```

### Checkpoint 格式
```python
{'epoch','model_state_dict','optimizer_state_dict','loss',
 'config': {'backbone': 'qwen3', 'model_name_or_path':'', 'vocab_size':256,
            'num_quantizers':4, 'max_seq_len':2048, 'hidden_size':1024,
            'num_layers':28, 'num_kv_heads':8, 'head_dim':64},
 '_id_to_sem': {int: (int,int)}, '_sem_to_id': {(int,int): int}}
```

## 四、训练数据格式

```json
[[[169,41,0,0], [20,53,0,0], ...], ...]  // 790K用户序列
```
每物品四元组 [s0,s1,s2,s3]，每维0~255。数据量和值可能偏稀疏（高层量化器多为0）。

训练时 `--max_seq_len` 控制数据集保留条数（默认50），模型 tokenizer 截断在 __init__ 中固定为 2048，两者独立。

## 五、关键 Bug 与修复日志

| # | 问题 | 根因 | 修复 | 影响 |
|---|------|------|------|------|
| 1 | Loss 横在10.27 | PEFT冻结了全部参数，新增token的embedding/lm_head行随机且不可训 | `modules_to_save=["lm_head","model.embed_tokens"]` 共 ~317M 参数参与训练 | loss: 10.27→3.06 |
| 2 | OOM batch=16 | lm_head [B,1000,152693] logits ~5GB | forward绕开全序列lm_head，只算4目标位置 | batch: 4→16~32 |
| 3 | OOM modules_to_save后 | +300M参数 optimizer states +2.5GB | max_seq_len从20→10缩短序列 | 显存足够 |
| 4 | `dtype` 参数报错 | 老transformers不要 `dtype=` | 改回 `torch_dtype=` | x86/ARM兼容 |
| 5 | epoch checkpoint 缺 config | 周期保存没写 config | 补 config dict | server 加载不报 KeyError |
| 6 | server 报 HFValidationError | model_name_or_path='' 不触发 dict.get default | `.get('x') or self.config.xxx` | 推理正常启动 |
| 7 | loss.backward no grad_fn | inputs_embeds.requires_grad=False → 梯度检查点短路 | `inputs_embeds.requires_grad_(True)` | 梯度流通 |
| 8 | DDP pad_token_id 报错 | DDP包装后 model.xxx→model.module.xxx | `_raw = model.module if is_ddp else model` | DDP正常 |
| 9 | DDP is_rank0 UnboundLocalError | epoch loop内定义, writer初始化在loop外 | 移到epoch loop前 | DDP正常 |
| 10 | DDP 三卡 it/s 反降 1.49 | batch_size//WORLD_SIZE→每卡10利用率低 | 删掉除号,DistributedSampler已分数据 | it/s: 1.49→3.53 |
| 11 | DDP max_steps 死锁 | rank0保存后return,非rank0 continue→all-reduce等待 | barrier()后一起return | 早停正常 |

## 六、详细问题分析与根因

### 6.1 Loss 横在 10.27（最严重，耗时最长）

**现象**: 修复前 epoch 1~10 loss 全在 10.27，不降不动。

**分析步骤**:
1. 排除 NaN/inf — loss 值稳定，模型在输出有效概率
2. 计算基线: 随机152693类=11.94 loss; 只在1024语义token猜=6.93 loss。10.27介于两者→模型不是完全随机，但学不到语义token
3. 怀疑梯度: 开了 `detect_anomaly` 发现梯度极小 (~1e-6)
4. 最终定位: 遍历 `model.named_parameters()` 打印 `requires_grad`，发现 `embed_tokens.weight[151936:]` (新增1024行) 和 `lm_head.weight[151936:]` (新增1024行) 都是 `False`
5. **根因**: `get_peft_model()` 冻结了全部基座参数。`resize_token_embeddings` 在 PEFT 之前调用，新增的行到 PEFT 包装时仍然被冻结
6. **修复**: `LoraConfig(modules_to_save=["lm_head","model.embed_tokens"])` — PEFT 官方机制，标记这些模块不受冻结
7. **验证**: 修复后 loss 3.06→2.81→2.65，持续下降。30%+ 可训参数（新增 312M embedding+lm_head）

**教训**: PEFT 的 `modules_to_save` 不是可选项，是必须。任何需要新增 token/输出类别的微调都必须用它。

### 6.2 推理返回空推荐

**现象**: loss 2.65 时 `/recommend` 返回 `recommendations=[]`，服务正常。

**分析**:
- 打印生成 token: 20 个 token 中无有效四元组序列
- 数学: loss 2.65 → P(单token)≈7% → P(四元组全对)≈0.0024% → 5次尝试命中率≈0.012%
- **结论**: 不是代码 bug，模型loss不够低
- loss < 2.0 时命中率 ≈ 0.8%; loss < 1.5 时 ≈ 6%

### 6.3 OOM 原因链

```
初期: batch 16 OOM → 梯度检查点(batch→8) + 绕开lm_head(batch→32)
中期: modules_to_save 后 OOM → max_seq_len 10→省显存
L40S: batch 32 + max_seq_len 20 不 OOM（46GB够用）
```

### 6.4 训练速度分析

| | inputs_embeds 旧 | prompt 新 |
|---|---|---|
| Transformer | [B,50,1024] ~2.5 TFLOPS | [B,~440,1024] ~8 TFLOPS |
| 输出头 | 4×Linear(256) ~0.2 GFLOPS | lm_head(4pos) ~0.04 TFLOPS |
| 总 | ~2.5 TFLOPS | ~8 TFLOPS |

**根因**: 不是代码效率问题，是 prompt 模式序列从 50 条语义ID → ~440 tokens，Transformer 计算量倍乘。这是用 LLM 语言能力的代价。

**已做优化**: lm_head 绕开(250x)、梯度检查点(40%内存)、DDP(3x)、num_workers=4

**可做**: 预tokenize (~20-30%加速)、torch.compile

### 6.5 x86 vs ARM 环境差异

| 问题 | x86 | ARM | 原因 |
|------|-----|-----|------|
| dtype参数 | 报错 TypeError | 无问题(旧版代码) | transformers 版本不同 |
| 梯度检查点 | 需 requires_grad_(True) | 无问题(旧版无检查点) | x86 版加了优化 |
| CUDA驱动 | 正常 | CPU mode | 需重装匹配CUDA版的PyTorch |

### 6.6 DDP 多卡踩坑记录

1. **属性访问**: DDP 包装后 `model.pad_token_id` → `model.module.pad_token_id`
2. **batch双重缩小**: `batch_size//WORLD_SIZE` + DistributedSampler 导致每卡只10条 → 利用率低
3. **is_rank0 作用域**: epoch 循环内定义，循环外使用 → UnboundLocalError
4. **max_steps 死锁**: rank0 return, 其他rank continue → all-reduce 等 rank0 → 死锁

### 6.7 Checkpoint 加载链路

三次修复才跑通:
1. epoch save 缺 config → KeyError
2. model_name_or_path='' → HF validation
3. _id_to_sem 缺失 → token映射丢失

**根因**: 三种保存路径（max_steps/best/epoch）格式未统一。已统一。

## 七、当前进展与时间线

### 时间线

| 日期 | 进度 |
|------|------|
| 5/19 | 设计 Prompt Template 架构，重写模型类；修 dtype→torch_dtype |
| 5/20 | 发现 loss 横盘 → modules_to_save 修复(loss 10.27→3.06)；lm_head 全序列优化；加梯度检查点+梯度累积；epoch 10 训练完成(loss 2.65)；推理服务启动成功但推荐为空；修补 checkpoint config；加 Resume 训练 |
| 5/21 | DDP 三卡 3 个 bug 修复；L40S×3 并行训练启动(epoch 11)；Docker 训练/推理镜像适配 Qwen3 |

### 当前运行

```
环境: ubuntu-server, 3×L40S (46GB), CUDA 12.x
训练: max_seq_len=20, batch=32×3=96, lr=5e-5, modules_to_save
进度: epoch 11/20 (从10 resume), loss 2.65→目标 2.0 以下
速度: ~3.5 it/s per GPU, 总 ~336 samples/s
```

## 八、训练命令

### 单卡
```bash
CUDA_VISIBLE_DEVICES=N python -m training.decoder.train \
    --backbone qwen3 --qwen3_model_path ./models/Qwen3-0.6B \
    --train_data ./data/tenrec/processed/train_sequences.json \
    --num_epochs 10 --batch_size 32 --learning_rate 5e-5 \
    --lora_rank 8 --lora_alpha 16 \
    --checkpoint_dir ./checkpoints/decoder_qwen3 --max_seq_len 20
```

### 三卡 DDP
```bash
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 \
    -m training.decoder.train \
    --backbone qwen3 --qwen3_model_path ./models/Qwen3-0.6B \
    --train_data ./data/tenrec/processed/train_sequences.json \
    --num_epochs 20 --batch_size 32 --learning_rate 5e-5 \
    --lora_rank 8 --lora_alpha 16 \
    --checkpoint_dir ./checkpoints/decoder_qwen3 --max_seq_len 20
```

### Resume
```bash
--num_epochs 20 --load_checkpoint ./checkpoints/decoder_qwen3/decoder_epoch_10.pt
```

## 九、推理服务

```bash
python -m inference.trt_llm.server \
    --model_path ./checkpoints/decoder_qwen3/decoder_epoch_N.pt \
    --port 18000 --device cuda --qwen3_model_path ./models/Qwen3-0.6B

curl -X POST http://localhost:18000/recommend \
    -H "Content-Type: application/json" \
    -d '{"user_id":"test","history":[[169,41,0,0],[20,53,0,0]],"topk":10}'
```

## 十、多卡 DDP 架构

torchrun 自动设置 LOCAL_RANK/WORLD_SIZE。DistributedDataParallel 包装模型。DistributedSampler 分数据。`_raw = model.module if is_ddp else model` 统一属性访问。rank-0 保存 checkpoint/日志，barrier() 同步退出。

## 十一、Docker 容器化方案

- `docker/Dockerfile.train`: CUDA 12.4 + PyTorch + transformers/peft/accelerate
- `docker/Dockerfile.inference`: 推理镜像(Qwen3适配)
- `scripts/run_train_docker.sh`: build/train/shell/export 一键
- x86 镜像不能直接在 ARM 运行，需在 ARM 重新构建

## 十二、8MB KV Cache 与 DataSystem

- KV: `28层 × 2(K+V) × 8头 × 64维 × 2bytes = 57KB/token`。8MB≈140 tokens≈35条历史
- DataSystem: 华为分布式共享内存KV缓存。TRT-LLM 内部 KvCacheManagerDataSystem 单例，KvCacheTransferManager 负责零拷贝搬移
- 我们 Python 版 KVCacheManager: Phase 1 暂不用，prompt 模式用原生 generate(use_cache=True)

## 十三、京东方案对比

| | 京东 | 我们 |
|---|---|---|
| 商品表示 | RQ-VAE 语义ID | 同 |
| 模型 | LLM 0.5B~7B | Qwen3-0.6B |
| 微调 | - | LoRA r=8 + modules_to_save |
| Token | `<a_99><b_225><c_67><d_242>` | `<s0_99><s1_225><s2_67><s3_242>` |
| 推理加速 | TRT-LLM | Phase 1 PyTorch → Phase 3 TRT-LLM |
| KV Cache | DataSystem 2M/8M | Phase 3 接 DataSystem |

## 十四、关键文件清单

| 文件 | 职责 | 行数 |
|------|------|------|
| `training/decoder/qwen3_generative_rec.py` | 模型定义 (forward/generate/parse) | ~389 |
| `training/decoder/train.py` | 训练主循环 (DDP,resume,checkpoint) | ~637 |
| `training/decoder/model.py` | 老 GPT2 Decoder | ~462 |
| `inference/trt_llm/server.py` | Flask 推理服务 | ~788 |
| `inference/kv_cache/manager.py` | Python KVCacheManager | ~199 |
| `docker/Dockerfile.train` | 训练镜像 | ~59 |
| `docker/Dockerfile.inference` | 推理镜像 | ~47 |
| `docker/entrypoint-inference.sh` | 推理启动脚本 | ~65 |
| `scripts/run_train_docker.sh` | Docker 一键脚本 | ~82 |

## 十五、待办事项

- [ ] 训练 loss 降到 2.0 以下，推理验证
- [ ] loss<1.5 测 Recall@K
- [ ] ARM 推理部署
- [ ] Phase 3: TRT-LLM + DataSystem
- [ ] 预 tokenize 训练数据 (~30%加速)
- [ ] Go pairec 联调
- [ ] Docker ARM 构建
