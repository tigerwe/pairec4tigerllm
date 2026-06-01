# 工作进度

> 最后更新: 2026-06-01 | 当前状态: 推荐链路工程化 — PaiRec接通过程中 | 下一步: 验证 max_new_tokens≤32 + 组合 + layer填充 后 PaiRec 端到端

## 时间线

| 日期 | 进度 |
|------|------|
| **6/1** | **推荐链路对接 — 推理服务命中率优化: 发现并修复5个瓶颈 (input截断/max_new上限/宽松匹配/组合模式/layer3填充)** |
| **5/30** | C++ KV cache offload/onboard 闭环验证通过; 任务切换到推荐链路工程化 |
| **5/29** | C++ offload/onboard 链路突破 (FMHA crash根因/C++源码绕过/重编.so) |
| **5/28** | KV Cache + offload/onboard 闭环 (TRT/PyTorch双路径/三层缓存全通) |
| **5/27** | 推理打通 (FMHA bf16 kernel SM89非法内存访问根因) |
| **5/23** | ARM 4090D推理部署 + TRT-LLM引擎构建 + 多轮采样去重 |
| **5/22** | 训练完成 epoch 20 (loss 2.49) |
| **5/21** | DDP三卡训练启动 (epoch 11) |

## 当前状态

- **训练**: ✅ epoch 20, loss 2.49
- **TRT-LLM 引擎**: ✅ bfloat16, 1.46 GB, 限制: max_input_len=64, max_new_tokens=32, max_seq_len=96
- **C++ KV offload/onboard**: ✅ 闭环验证通过
- **推理服务**: ✅ /recommend 可用
- **PaiRec 对接**: 🔄 进行中 — 推理命中率已改善，待验证端到端

## 6/1 探索：推理命中率优化 (5个bug修复)

### 问题链

PaiRec 调用推理服务 → 大部分用户返回 `code:299 "items size not enough"`。

根因是 **推理服务返回的推荐 item 太少**（1个或0个），远低于 PaiRec 期望的 size（10）。

### 修复清单

| # | 提交 | 问题 | 修复 |
|---|------|------|------|
| 1 | `f7beeb4` | 20条历史→117 tokens 超引擎 max_input_len=64 | prompt 自动截断，保留最近9条 |
| 2 | `fb1cb98` | `<s0_X><pad><s1_Y>` 因中间有非语义token被丢弃 | 先过滤有效token再匹配，跳过非语义token |
| 3 | `326e1bb` | 模型输出层序乱 (s2,s1,s0,s3) 连续递增模式找不到 | 改为按层收集+笛卡尔积组合，不要求顺序 |
| 4 | `5533a16` | 20 tokens输出太短+8轮各自为战 | 8轮token合并到一个池子统一组合 |
| 5 | `1544a3f` | max_new_tokens>32 C++层卡死不报错 | 硬限制≤32 (引擎构建时预留) |
| 6 | `b44dbc6` | layer 3永远为0 (1条历史的用户) | 缺失层用{0}填充，由semantic_id_map验证 |

### 引擎硬限制 (重要)

```
max_seq_len = 96
  ├─ max_input_len = 64   (超过报 RuntimeError)
  └─ max_new_tokens = 32  (超过 C++ 层卡死不报错!)
```

必须同时遵守两个限制。

### 核心机制

**组合模式**: 8轮采样 × 32 token = 256 token 池 → 按层收集所有有效语义token → 笛卡尔积组合 → semantic_id_map 验证

**layer 填充**: 当某层缺失时用 {0} 补位 → 组合出候选 → map 验证真假

## 下一步

1. 验证当前版本 (max_new≤32 + 组合 + layer填充) 在 PaiRec 端到端的表现
2. 如果 history_len=1 用户仍不足: 重建引擎放大 max_seq_len (256) + max_new_tokens (128)
3. 继续 F08 接通 PaiRec (写测试 curl、trace 验证)
4. F12 推荐系统各阶段时延分析
