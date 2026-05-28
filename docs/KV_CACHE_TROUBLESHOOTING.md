# KV Cache + offload/onboard 问题记录

> 日期: 2026-05-28 | 关联: F11 (DataSystem KV Cache 集成)
> 最终状态: hbm_hit(~2ms) / ds_hit(~3ms) / miss(~220ms) 三层全通

## 问题索引

| # | 问题 | 类型 | 影响 |
|---|------|------|------|
| 1 | KV Cache 永无命中 (miss loop) | 逻辑缺陷 | 所有请求 kv_source=miss |
| 2 | past_key_values 含生成 token 导致位置错位 | 语义 bug | 缓存回注后推理错误 |
| 3 | TRT-LLM 不暴露 past_key_values | 架构约束 | Python KVCacheManager 对 TRT 路径无用 |
| 4 | DataSystem `kv().set()` API 签名错误 | API 误用 | 后台线程 TypeError，写入失败 |
| 5 | `np.load` 须 `allow_pickle=True` | 依赖变更 | numpy >=1.24 默认禁用 pickle |
| 6 | 首次请求查 DataSystem 必然 miss | 日志噪音 | 干扰排错 |
| 7 | result cache 与 KV cache DataSystem key 冲突 | 命名空间污染 | JSON 被当作 numpy pickle 反序列化失败 |

---

## 1. KV Cache 永无命中 (miss loop)

### 现象
`verify_kv.sh` 重复请求 3 次同一用户，`kv_source` 始终为 `miss`，从未 hit。

### 根因
`server.py` 的 `recommend()` 方法有三条推理分支，但只有第三条（已废弃的旧 inputs_embeds 路径）实际使用了 `past_kv` 并捕获 `final_past_kv`：

```python
if TRT backend:              # ① past_kv 不传, final_past_kv 不捕获
elif _id_to_sem 存在:        # ② past_kv 不传, final_past_kv 不捕获 ← 每次都走这里
elif hasattr generate:       # ③ past_kv 传入, final_past_kv 捕获  ← 死代码
```

`_id_to_sem` 从 checkpoint 恢复后永远存在，所以分支 ② 永远命中，`kv_manager.store()` 永不执行。

### 修复
- PyTorch 路径：给 `Qwen3GenerativeRec.generate()` 加 `past_key_values` 参数和 `return_past_kv` 参数
- server.py 中在分支 ② 传 `past_key_values=past_kv, return_past_kv=True`，解包 `(tokens, final_past_kv)`

### 知识点
> **Python `hasattr` 做分支选择是脆弱的**。属性存在不代表语义正确。应该用显式的状态变量（如 `self._backend_mode = "prompt_template"`）代替隐式的特征检测。

---

## 2. past_key_values 含生成 token 导致位置错位

### 现象
如果直接把 HuggingFace `generate()` 返回的完整 `past_key_values`（覆盖 prompt + 生成的所有 token）存下来，下次传回去时，模型会从 prompt+gen 之后的位开始 decoding，解析输出时 `new_tokens = generated[b, prompt_len:]` 会包含上一轮的生成 token。

### 根因
HuggingFace `generate()` 的 `past_key_values` 维度是 `[batch, kv_heads, total_seq_len, head_dim]`，其中 `total_seq_len = prompt_len + generated_len`。

### 修复
存储前切片仅保留 prompt 部分：
```python
prompt_len = encoded["input_ids"].shape[1]
prompt_past_kv = tuple(
    (k[:, :, :prompt_len, :], v[:, :, :prompt_len, :])
    for k, v in full_past_kv
)
```

### 知识点
> **缓存 KV 时只缓存 prompt 部分，而不是完整序列**。下次请求回注后，模型从 prompt 末尾开始生成，等同于 "跳过 Prefill，重新 Decode"。

---

## 3. TRT-LLM 不暴露 past_key_values

### 现象
TRT-LLM 后端下 Python KVCacheManager 完全无用，因为 `_trt_backend.generate()` 不返回 `past_key_values`。

### 根因
TRT-LLM 内部使用 paged KV cache，Python 层无法获取和注入张量级的 past_key_values。

### 修复
为 TRT 路径单独实现**结果缓存**（cache 推理结果 JSON 而非 KV 张量）：
- 内存层：`OrderedDict` (LRU, cap=50)
- 持久层：DataSystem (TTL 600s，JSON bytes)
- 独立 key 前缀：`pairec4tigerllm:result:{user_id}:{hash}`

### 知识点
> **不同推理后端的缓存策略不同**。TRT-LLM 内部有自己的 KV cache 管理，Python 层应缓存结果（JSON）而非中间张量。PyTorch 路径则可以缓存 `past_key_values` 来跳过 Prefill。

---

## 4. DataSystem `kv().set()` API 签名错误

### 现象
每次推理后后台线程报 `TypeError: The input of key has invalid type, valid type: [<class 'str'>]`。

### 根因
DataSystem SDK 的 API 不对称：
```python
kv().get([key1, key2], ...)            # 接受 List[str]
kv().set(key, value, ttl_second=600)   # 接受 str（非 List）
```
代码错误地用 `kv().set([ds_key], [payload], ...)` 传了 list。

### 修复
```python
# 错误
self.kv_manager.ds.kv().set([ds_key], [payload], ttl_second=600)
# 正确
self.kv_manager.ds.kv().set(ds_key, payload, ttl_second=600)
```

### 知识点
> **不要假设 SDK 的 get/set 签名对称**。`get` 常支持批量（List），`set` 可能只支持单条。遇到 `TypeError` 第一时间检查 SDK 源码或 `help()` 确认签名。

---

## 5. `np.load` 须 `allow_pickle=True`

### 现象
KVCacheManager 从 DataSystem 读回 past_key_values 时报：
`Cannot load file containing pickled data when allow_pickle=False`

### 根因
numpy >= 1.24 默认 `allow_pickle=False`（安全原因），但 `np.savez_compressed` 内部使用 pickle 序列化。

### 修复
```python
# 错误
data = np.load(buf)
# 正确
data = np.load(buf, allow_pickle=True)
```

### 知识点
> **numpy >= 1.24 默认禁用 pickle 加载**。所有 `np.load()` 调用处如果数据来自 `np.save`/`np.savez`/`np.savez_compressed`，必须显式加 `allow_pickle=True`。

---

## 6. 首次请求查 DataSystem 必然 miss

### 现象
每个首次请求都打印：
```
[ResultCache] DataSystem onboard failed: code: [Key not found], msg: [...]
```

### 根因
首次请求 DataSystem 中还没有数据，`kv().get()` 返回 "Key not found" 错误（抛异常或返回错误码），被 except 捕获后打印。

### 修复
区分 "Key not found"（正常 miss）和真正的错误：
```python
except Exception as e:
    msg = str(e)
    if "Key not found" not in msg and "not found" not in msg.lower():
        print(f"[ResultCache] DataSystem onboard error: {e}")
```

### 知识点
> **区分 "Key not found" 和真正的错误**。缓存系统里 miss 是正常的，不应以 ERROR 级别打印。使用 `if "not found" in msg` 判断并静默。

---

## 7. result cache 与 KV cache DataSystem key 冲突

### 现象
KVCacheManager 的 `_ds_get` 报错：
`Failed to interpret file <_io.BytesIO object at 0x...> as a pickle`

### 根因
Result cache 写入 DataSystem 时使用了 `kv_manager._ds_key(cache_key)`，key 前缀是 `pairec4tigerllm:kv:*`。这和 KVCacheManager 的 past_key_values 共用同一 namespace。KVCacheManager 尝试 `np.load` 反序列化 JSON 数据，报 pickle 错误。

### 修复
Result cache 使用独立的 key 前缀：
```python
ds_key = f"{self._result_ds_prefix}:{cache_key}"  # pairec4tigerllm:result:*
```

### 知识点
> **不同子系统在共享存储（DataSystem/Redis）中必须用不同 key 前缀或 namespace 隔离**。尤其当序列化格式不同（numpy pickle vs JSON）时，同名空间的跨格式反序列化会报难以诊断的错误。

---

## 可复现验证

```bash
# 完整验证（需远程 ARM 4090D + DataSystem）
git pull gitcode dev
pkill -f "inference.trt_llm.server" || true
python -m inference.trt_llm.server ... &
bash verify_kv.sh      # 验证 hbm_hit
bash test_eviction.sh  # 触发 LRU 淘汰
curl ...               # 验证被淘汰的 key → ds_hit
```

## 涉及文件

| 文件 | 改动 |
|------|------|
| `training/decoder/qwen3_generative_rec.py` | `generate()` 支持 `past_key_values` / `return_past_kv` |
| `inference/trt_llm/server.py` | TRT 结果缓存 + DataSystem onboard + PyTorch 路径 KV 注入 |
| `inference/kv_cache/manager.py` | `np.load(allow_pickle=True)` |
| `test_eviction.sh` | LRU 淘汰触发脚本 |
