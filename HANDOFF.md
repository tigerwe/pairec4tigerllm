# 交接日志

> 供 Agent 跨 session 恢复上下文。只记录关键决策和当前任务。

## 最近一次交接 (2026-05-25)

### 当前任务
DataSystem KV Cache 集成完成 (Python + C++ 双验证) → 下一步: Go pairec 联调 (F08)

### 本次完成
- **C++ 层 DataSystem 验证通过**:
  - 在容器 `pairec-ds-build` 内重新编译 TRT-LLM (对接 DataSystem SDK 0.7.7)
  - 日志确认: `[TensorRT-LLM][Datasystem] Init KvCache Manager DataSystem success`
  - Worker 地址: `127.0.0.1:31501`
- **踩坑记录**:
  - cmake 版本检测替换为硬编码 `set(TRTLLM_VERSION "1.0.0")`
  - `executorWorker` 链接失败忽略（不影响推理）
  - `import tensorrt_llm` segfault → `LD_PRELOAD=libabseil_dll.so.2407.0.0` 解决初始化顺序
- **Python 层 DataSystem 对接完成** (F11):
  - `inference/kv_cache/manager.py`: API 修正
  - `inference/trt_llm/server.py`: `_init_datasystem_client()` + CLI 参数

### 容器运行时环境变量 (已写入 ~/.bashrc)
```bash
export LD_LIBRARY_PATH="/opt/openEuler/gcc-toolset-14/root/usr/lib64:/usr/local/lib/python3.11/site-packages/yr/datasystem/lib:/TensorRT-LLM/tensorrt_llm:/TensorRT-LLM/cpp/build/tensorrt_llm/thop:..."
export PYTHONPATH=/TensorRT-LLM:$PYTHONPATH
export LD_PRELOAD=/usr/local/lib/python3.11/site-packages/yr/datasystem/lib/libabseil_dll.so.2407.0.0
```

### 关键决策
- 用 Prompt Template 而非 inputs_embeds (参照京东方案)
- LoRA + modules_to_save (lm_head + embed_tokens 必须可训)
- DataSystem KV Cache 三层: HBM LRU → DataSystem → Prefill
- C++ 层: `KvCacheManagerDataSystem` 单例在 KVCacheManager 构造时自动连接
- DataSystem 连接: `127.0.0.1:31501`, Worker 自动启动

### 下一步
1. Go pairec 联调 (F08)
2. TRT 引擎延迟优化 (detailed profiling, KV Cache 池化)
3. 预 Tokenize 训练数据 (F10, 低优先级)

### 快速验证
```bash
# 容器内
python -c "import tensorrt_llm; print(tensorrt_llm.__file__)"  # → /TensorRT-LLM/tensorrt_llm/__init__.py
python -m inference.trt_llm.server --model_path ... --port 18000 --device cuda --trt_engine_dir ... \
    2>&1 | grep -E "Datasystem|TRT"
# 预期: [Datasystem] Init KvCache Manager DataSystem success
```
