# PaiRec4TigerLLM — Qwen3 Prompt Template 架构分析

> 2026-05-22 | 基于 dev 分支 commit `6719140`

---

## 一、整体架构

```
用户请求 (语义ID序列)
  │
  ▼
┌──────────────────────────────────────────────────┐
│  Qwen3GenerativeRec (training/decoder/qwen3_generative_rec.py)  │
│                                                    │
│  训练路径 (forward):                                │
│    semantic_ids → _build_prompt → tokenizer         │
│    → embed → _transformer (28层 Qwen3Model)         │
│    → _lm_head(仅4个目标位置) → CrossEntropy loss    │
│                                                    │
│  推理路径 (generate):                               │
│    history → _build_infer_prompt → tokenizer        │
│    → model.generate() + 约束解码                     │
│    → _parse_output → [max_items, 4]                │
└───────────────────┬──────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────┐
│  Flask Inference Service (inference/trt_llm/server.py) │
│                                                    │
│  /recommend POST                                   │
│    → _prepare_input → model.generate()             │
│    → _tokens_to_items → semantic_id_map lookup     │
│    → [{"item_id": ..., "semantic_id": ...}, ...]   │
└──────────────────────────────────────────────────┘
```

## 二、模型类 Qwen3GenerativeRec

### 2.1 初始化 (__init__)

```
1. 加载 Qwen3 Tokenizer
2. 注册 1024 个特殊 token: <s0_0> ~ <s3_255>
3. 构建双向映射: _id_to_sem (token_id → (layer, value))
                 _sem_to_id ((layer, value) → token_id)
4. 加载 Qwen3ForCausalLM (torch_dtype=bf16)
5. resize_token_embeddings(152693)  # 151936 + 1024
6. LoRA: r=8, alpha=16
   - target_modules: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
   - modules_to_save: [lm_head, model.embed_tokens]  ← ★ 关键
7. gradient_checkpointing_enable()
8. 缓存 _transformer (Qwen3Model) 和 _lm_head 引用
9. _item_prefix = None (推理时注入)
```

**关键设计点**:
- `modules_to_save` 确保新增的 1024 个 token embedding 和 lm_head 输出行可训练
- `_transformer` 引用绕开 CausalLM wrapper，避免 forward 时计算全序列 lm_head

### 2.2 Prompt 模板

```python
# 训练模板
TRAIN_PROMPT = (
    "用户按时间顺序点击过以下商品：\n"
    "{history}\n"
    "他接下来点击了：{target}"
)

# 推理模板
INFER_PROMPT = (
    "用户按时间顺序点击过以下商品：\n"
    "{history}\n"
    "请预测用户下一个可能点击的商品："
)

# 物品格式: 四元组 → 四个连续 token
ITEM_FMT = "<s0_{s0}><s1_{s1}><s2_{s2}><s3_{s3}>"
```

**示例**:
```
用户历史: [[169,41,0,0], [20,53,0,0]]
目标:     [80,201,0,0]

训练 prompt:
"用户按时间顺序点击过以下商品：
<s0_169><s1_41><s2_0><s3_0>,<s0_20><s1_53><s2_0><s3_0>
他接下来点击了：<s0_80><s1_201><s2_0><s3_0>"
```

### 2.3 训练前向 (forward)

```
输入: semantic_ids [B, max_seq_len, 4]  ← SequenceDataset 提供
      attention_mask [B, max_seq_len]
      labels [B, max_seq_len, 4]

步骤:
1. 提取每样本有效历史 (利用 attention_mask 过滤 padding)
2. 取 labels 最后一个有效位置作为目标
3. 构造训练 prompt 字符串列表
4. tokenizer(prompts, padding=True, truncation=True, max_length=2048)
   → input_ids [B, prompt_len]

5. 在 tokenized 序列中找语义 token 位置
   target_positions = 每个样本的语义 token 位置中取最后 4 个

6. 绕过 CausalLM，直接调 transformer backbone:
   embed = base_model.get_input_embeddings()   ← 152693×1024 嵌入表
   inputs_embeds = embed(input_ids)             ← [B, prompt_len, 1024]
   inputs_embeds.requires_grad_(True)           ← 梯度检查点需要
   hidden_states = _transformer(inputs_embeds)  ← 28层 Qwen3Model
   hidden_states[0]                             ← [B, prompt_len, 1024]

7. 只在目标位置计算 lm_head:
   for each sample:
     h = hidden_states[b, [p-1 for p in target_pos]]  ← causal: pos-1→pos
     logits = _lm_head(h)                               ← [4, 152693]
     labels = input_ids[b, target_pos]                  ← [4]

8. loss = F.cross_entropy(all_logits, all_labels)   ← 仅4个位置
```

**为什么绕过 CausalLM**: 标准 `base_model(input_ids, labels=...)` 内部会计算 `lm_head([B, prompt_len, 1024]) → [B, prompt_len, 152693]`。100个位置的 prompt 中只有 4 个目标位置需要 loss，另外 96 个位置的 lm_head 计算是浪费的。绕过 CausalLM 直接调 transformer + 手工算 lm_head，lm_head 计算量减少 250× (1000→4)。

**因果 LM 偏移**: `hidden_states[pos-1]` 通过 lm_head 预测 `token[pos]`。第一个目标 token (如 `<s0_80>`) 的前一个是 prompt 中的冒号"："，第二个目标 token (`<s1_201>`) 的前一个是已经生成的 `<s0_80>`，依次自回归。

### 2.4 推理生成 (generate)

```
输入: input_ids [B, history_len, 4]  ← 用户历史语义ID

步骤:
1. 构造推理 prompt 字符串列表
2. tokenizer(prompts, padding=True, max_length=2048)
   → encoded {"input_ids": [B, prompt_len], "attention_mask": ...}

3. 约束解码函数 prefix_fn:
   def prefix_fn(batch_id, sent):
       last = sent[-1].item()
       info = _id_to_sem.get(last)
       pfx = _item_prefix  ← 物品前缀树 (推理时注入)

       if info is None:
           # prompt 结束，从 <s0_X> 开始
           return [sem_to_id[(0,v)] for v in pfx['s0_set']]  ← 白名单
    
       layer, value = info
    
       if layer == 0:   # s0 → s1
           allowed = pfx['s01_map'].get(value, set())
           return [sem_to_id[(1,v)] for v in allowed]
    
       if layer == 1:   # s1 → s2 (需回溯 s0)
           s0_val = ...
           allowed = pfx['s012_map'].get((s0_val, value), set())
           return [sem_to_id[(2,v)] for v in allowed]
    
       if layer == 2:   # s2 → s3 (需回溯 s0, s1)
           ...
           allowed = pfx['s0123_map'].get((s0_val, s1_val, value), set())
           return [sem_to_id[(3,v)] for v in allowed]
    
       # s3 → 下一轮 s0
       return [sem_to_id[(0,v)] for v in pfx['s0_set']]

4. base_model.generate(
       **encoded,
       max_new_tokens=topk*2,      ← 如 topk=10 → 20 tokens → 5 items
       temperature, top_k, do_sample=True,
       prefix_allowed_tokens_fn=prefix_fn,  ← 约束解码
   )

5. 解析输出: 扫描新生成的 token，每4个一组提取语义ID
   → padded [B, max_items, 4]
```

**两层约束**:
1. 格式约束: 强制 s0→s1→s2→s3 顺序，不允许生成非语义 token
2. 物品白名单: 只在 semantic_id_map 中存在的四元组内选择

### 2.5 解析输出 (_parse_output)

```python
def _parse_output(token_ids: List[int]) -> List[List[int]]:
    items = []
    i = 0
    while i <= len(token_ids) - 4:
        window = token_ids[i:i+4]
        parsed = [self._id_to_sem.get(tid) for tid in window]
        if all(p is not None for p in parsed):
            layers = [p[0] for p in parsed]
            values = [p[1] for p in parsed]
            if layers == [0, 1, 2, 3]:  ← 严格有序
                items.append(values)
                i += 4
                continue
        i += 1
    return items
```

## 三、训练系统

### 3.1 数据集 (SequenceDataset)

```python
# train_sequences.json: List[List[List[int]]]
# 每个用户一个序列 [[169,41,0,0], [20,53,0,0], ...]

class SequenceDataset:
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq = seq[-self.max_seq_len:]       # 截断到 max_seq_len
        input_seq = seq[:-1]                # items[0:N-2]
        label_seq = seq[1:]                 # items[1:N-1] (shifted)
        # 填充到 max_seq_len
        return (padded_input, padded_labels, attention_mask)
```

**关键参数**:
- `max_seq_len`: 数据集保留多少条历史 (CLI `--max_seq_len`，默认50，与模型 tokenizer 截断独立)
- 模型 tokenizer 截断: `__init__` 中 `max_seq_len=2048` (token 数量)

### 3.2 训练循环 (train_decoder)

```
for epoch in range(start_epoch, num_epochs):
    sampler.set_epoch(epoch)          # DDP: 每epoch不同shuffle
    for batch in DataLoader:
        input_ids, labels, attn = batch.to(device)
        logits, loss, _ = model(input_ids, attn, labels)  → forward()
        loss = loss / grad_accum_steps                  ← 梯度累积
        loss.backward()
        
        if accumulated_enough:
            clip_grad_norm_(max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
```

### 3.3 DDP 多卡

```
torchrun --nproc_per_node=3 -m training.decoder.train ...

每个进程:
  LOCAL_RANK = int(os.environ['LOCAL_RANK'])    ← 0/1/2
  init_process_group('nccl')
  device = f'cuda:{LOCAL_RANK}'
  model = DDP(model, device_ids=[LOCAL_RANK])   ← 分布式包装
  _raw = model.module if is_ddp else model      ← 访问原始属性
  DistributedSampler(dataset)                   ← 数据分片

  rank 0 负责: 日志、TensorBoard、checkpoint保存
  barrier() 同步: 早停时所有rank一起退出
```

### 3.4 Checkpoint 格式

```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'loss': float,
    'config': {
        'backbone': 'qwen3', 'model_name_or_path': '',
        'vocab_size': 256, 'num_quantizers': 4,
        'max_seq_len': 2048, 'hidden_size': 1024,
        'num_layers': 28, 'num_kv_heads': 8, 'head_dim': 64,
    },
    '_id_to_sem': {int: (int, int)},     ← token_id → (layer, value)
    '_sem_to_id': {(int, int): int},     ← (layer, value) → token_id
}
```

## 四、推理服务

### 4.1 服务启动

```
GenerativeInferenceService.__init__():
  1. 加载 checkpoint → 读 config
  2. _load_qwen3_model(checkpoint):
     - 创建 Qwen3GenerativeRec (use_lora=False)
     - load_state_dict(checkpoint) + merge_lora()
     - 恢复 _id_to_sem / _sem_to_id 映射
  3. _load_semantic_id_mapping():
     - 加载 semantic_id_map.json (231万条)
     - 构建 semantic_to_item_tuple (查表)
     - 构建 _item_prefix (约束解码用前缀树)
  4. 注入 _item_prefix → model._item_prefix ← ★ 顺序关键
```

### 4.2 前缀树索引 (_item_prefix)

```
从 semantic_id_map.json 构建四层前缀索引:

s0_set:    {所有出现的 s0 值}              → 256 个
s01_map:   {s0 → {s1值的集合}}             → 256 个 key
s012_map:  {(s0,s1) → {s2值的集合}}        → 65536 个 key
s0123_map: {(s0,s1,s2) → {s3值的集合}}     → 2310087 个 key

生成时:
  步骤1 (<s0_>): 只能在 s0_set 里选 (256 选 1)
  步骤2 (<s1_>): 只能在 s01_map[s0] 里选
  步骤3 (<s2_>): 只能在 s012_map[(s0,s1)] 里选
  步骤4 (<s3_>): 只能在 s0123_map[(s0,s1,s2)] 里选
```

### 4.3 /recommend 请求流

```
POST /recommend {user_id, history: [[s0,s1,s2,s3],...], topk}

1. _prepare_input(history):
   → tensor [1, hist_len, 4]

2. model.generate(input_ids, max_new_tokens=topk*2):
   → 构造推理prompt → tokenizer → 约束解码生成
   → [1, max_items, 4] tensor

3. _tokens_to_items(tokens[0], topk):
   → 逐个四元组查 semantic_to_item_tuple
   → 去重取 topk

4. 返回:
   {"code":200, "recommendations": [{"item_id":..., "semantic_id":[...], "score":1.0}, ...],
    "inference_time_ms": ..., "trace": {...}}
```

## 五、KV Cache 分析

### 5.1 单 token 计算

```
Qwen3-0.6B 配置:
  num_layers  = 28     (config.num_hidden_layers)
  num_kv_heads = 8     (config.num_key_value_heads, GQA)
  head_dim     = 64    (hidden_size // num_attention_heads = 1024/16)
  dtype        = bf16  (2 bytes)

单 token KV:
  每层: 2(K+V) × 8 heads × 64 dim × 2 bytes = 2048 bytes
  28层: 28 × 2048 = 57,344 bytes ≈ 56 KB

请求分析 (max_seq_len=20):
  prompt模板:  ~30 tokens
  历史物品:    20 × 5 = 100 tokens  (4个s token + 逗号)
  总计:        ~130 tokens
  KV cache:    130 × 56 KB ≈ 7.3 MB

DataSystem 8MB 上限:
  7.3 MB < 8MB → 单请求可容纳
  如果 max_seq_len=30: ~190 tokens → ~10.6 MB → 超限
```

### 5.2 GQA 优化

Qwen3 用 Grouped Query Attention: 16 个 Q 头配 8 个 KV 头。如果标准 MHA (16 KV 头)，每 token KV 为 112 KB。GQA 省了一半。

## 六、关键 Bug 记录

### 6.1 modules_to_save 缺失 (最严重)
- 现象: loss 横在 10.27，10 个 epoch 不降
- 根因: PEFT 冻结了所有参数，resize_token_embeddings 新增的 1024 行 embedding 和 lm_head 行被冻结
- 修复: `modules_to_save=["lm_head", "model.embed_tokens"]`

### 6.2 lm_head 全序列计算 (OOM)
- 现象: batch=16 OOM，lm_head [B,1000,152693] ~5GB
- 修复: forward 绕过 CausalLM，只在 4 个目标位置算 lm_head

### 6.3 物品前缀注入顺序
- 现象: pfx_injected=False，约束解码未生效
- 根因: _load_semantic_id_mapping 在 _load_qwen3_model 之后调用
- 修复: 注入移到 mapping 加载之后

### 6.4 梯度检查点短路
- 现象: loss.backward() "no grad_fn"
- 根因: inputs_embeds.requires_grad=False → gradient checkpoint 跳过
- 修复: `inputs_embeds.requires_grad_(True)`

### 6.5 DDP 属性访问
- 现象: DistributedDataParallel has no attribute 'pad_token_id'
- 修复: `_raw = model.module if is_ddp else model`

## 七、训练数据格式

```json
// train_sequences.json
[
  [[169,41,0,0], [20,53,0,0], [80,201,0,0], ...],   // 用户1, N个物品
  [[249,0,0,0], [99,54,0,0], ...],                     // 用户2
  // ...790,156 个用户序列
]

// 每物品: [s0, s1, s2, s3], 每维 0~255
// 值可能偏稀疏 (高层量化器多为0)
```

## 八、环境依赖

| 组件 | 版本/要求 |
|------|----------|
| Python | 3.10+ |
| PyTorch | 2.1+ cu124 |
| transformers | 4.51+ |
| peft | 0.14+ |
| accelerate | 0.33+ |
| bitsandbytes | 0.44+ |
| CUDA | 12.4 |
| GPU | L40S 46GB × 3 (训练), 任意 (推理) |
