# TensorRT-LLM + DataSystem ARM64 容器编译完整步骤

> 环境: `pairec-inference:v1.0` (基于 `openeuler-24.03-lts-sp3:TensorRT-LLM-v1.0.0`, 116GB)
> 架构: ARM64 (aarch64) + NVIDIA RTX 4090 D (SM 89)
> Python: 3.11
> 目标: 编译 AspirationWang/TensorRT-LLM release/1.0 分支（集成 openYuanrong DataSystem）

---

## 前置准备

### 1. 宿主机准备

确认以下文件在宿主机上可用：
- `openyuanrong_datasystem-0.7.7-cp311-cp311-manylinux_2_38_aarch64.whl`（DataSystem wheel）
- `TensorRT-LLM` 源码目录（AspirationWang 的 release/1.0 分支）
- `TensorRT-10.11.0.*.Linux.aarch64-gnu.cuda-12.8.tar.gz`（TensorRT C++ 头文件，解压用）

### 2. 启动容器

```bash
# 根据实际路径调整挂载
WHEEL_DIR="/path/to/wheel/directory"
TRTLLM_PATH="/home/vivwimp/TensorRT-LLM"
TRT_TAR="/path/to/TensorRT-10.11.0.41.Linux.aarch64-gnu.cuda-12.8.tar.gz"

docker run -it --rm \
  --gpus all \
  --privileged \
  --ipc host \
  -v /opt/pairec4tigerllm:/app \
  -v ${TRTLLM_PATH}:/TensorRT-LLM \
  -v ${WHEEL_DIR}:/wheels \
  -v ${TRT_TAR}:/tmp/tensorrt.tar.gz \
  -w /app \
  -p 8000:8000 \
  -e LD_LIBRARY_PATH="/opt/openEuler/gcc-toolset-14/root/usr/lib64:$(find /usr/local/lib/python3.11/site-packages/nvidia -type d -name 'lib' | tr '\n' ':')$LD_LIBRARY_PATH" \
  --name pairec-ds-build \
  pairec-inference:v1.0 \
  /bin/bash
```

---

## Step 1: 安装 DataSystem

```bash
# 安装 openyuanrong_datasystem wheel
pip install /wheels/openyuanrong_datasystem-0.7.7-cp311-cp311-manylinux_2_38_aarch64.whl

# 验证安装位置
python -c "import yr.datasystem; print(yr.datasystem.__file__)"
# 预期输出: /usr/local/lib/python3.11/site-packages/yr/datasystem/__init__.py
```

### 验证 C++ SDK

```bash
# 确认头文件和库存在
ls /usr/local/lib/python3.11/site-packages/yr/datasystem/include/datasystem/kv_client.h
ls /usr/local/lib/python3.11/site-packages/yr/datasystem/lib/libdatasystem.so
```

---

## Step 2: 准备 TensorRT C++ 头文件

容器里只有 TensorRT Python wheel 的运行时库，缺少 C++ 开发头文件。

```bash
cd /tmp
tar -xzf tensorrt.tar.gz

# 复制头文件到标准位置
cp -r TensorRT-10.11.0.41/include/* /usr/local/include/

# 验证
ls /usr/local/include/NvInfer.h
ls /usr/local/include/NvOnnxParser.h
```

---

## Step 3: 初始化 TensorRT-LLM 子模块

```bash
cd /TensorRT-LLM
git submodule update --init --recursive
```

---

## Step 4: 修改 CMake 配置

### 4.1 修改 FindDATASYSTEM.cmake

修复 include 路径指向问题（让 `datasystem/kv_client.h` 能被正确找到）：

```bash
sed -i 's|"${DATASYSTEM_INCLUDE_DIR}"|"${DATASYSTEM_INCLUDE_DIR}/../"|g' \
  /TensorRT-LLM/cpp/cmake/modules/FindDATASYSTEM.cmake
```

### 4.2 修改 cpp/CMakeLists.txt

全局添加 DataSystem 的 include 路径，确保所有 target（包括测试代码）都能解析：

```bash
sed -i '375s|set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIR})|set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIR} /usr/local/lib/python3.11/site-packages/yr/datasystem/include)|' \
  /TensorRT-LLM/cpp/CMakeLists.txt
```

> 第 375 行原始内容：`set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIR})`

---

## Step 5: 安装系统依赖

```bash
yum install -y zeromq-devel tbb-devel spdlog-devel 2>/dev/null || \
dnf install -y zeromq-devel tbb-devel spdlog-devel
```

---

## Step 6: CMake 配置

```bash
cd /TensorRT-LLM/cpp/build
rm -rf *

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=$(which python3) \
  -DPython3_EXECUTABLE=$(which python3) \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89" \
  -DTensorRT_ROOT=/usr/local/lib/python3.11/site-packages/tensorrt_libs \
  -DTensorRT_INCLUDE_DIR=/usr/local/include \
  -DTensorRT_LIBRARY=/usr/local/lib/python3.11/site-packages/tensorrt_libs/libnvinfer.so \
  -DBUILD_TESTS=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DBUILD_MICRO_BENCHMARKS=OFF
```

### 验证 CMake 输出

确认以下关键行存在：
```
-- TensorRT LLM version: 1.0.0
-- Found TensorRT: ... (found suitable version "10.11.0.33")
-- Found DATASYSTEM: /usr/local/lib/python3.11/site-packages/yr/datasystem/lib/libdatasystem.so;...
-- Configuring done
```

---

## Step 7: 编译 C++ 后端

```bash
cd /TensorRT-LLM/cpp/build
make -j$(nproc) 2>&1 | tee make.log
```

> ARM64 上编译时间约 30~90 分钟。如果内存不足，改用 `make -j4` 或 `make -j2`。

### 编译产物确认

编译完成后（即使 executorWorker 链接失败），确认以下产物存在：

```bash
ls /TensorRT-LLM/cpp/build/tensorrt_llm/libtensorrt_llm.so
ls /TensorRT-LLM/cpp/build/tensorrt_llm/pybind/bindings.cpython-311-aarch64-linux-gnu.so
ls /TensorRT-LLM/cpp/build/tensorrt_llm/thop/libth_common.so 2>/dev/null
```

---

## Step 8: 安装 Python 包

### 8.1 复制编译产物到 Python 包目录

```bash
cd /TensorRT-LLM

# 创建 bindings 目录（存放 stub 文件）
mkdir -p tensorrt_llm/bindings

# 核心：bindings.so 必须放在 tensorrt_llm/ 根目录，不是 bindings/ 子目录
cp cpp/build/tensorrt_llm/pybind/bindings.cpython-311-aarch64-linux-gnu.so tensorrt_llm/
cp cpp/build/tensorrt_llm/libtensorrt_llm.so tensorrt_llm/
cp cpp/build/tensorrt_llm/thop/libth_common.so tensorrt_llm/ 2>/dev/null || true
```

### 8.2 pip 安装

```bash
cd /TensorRT-LLM
pip install -e . --no-build-isolation
```

> ⚠️ **注意**：pip 可能会尝试降级 `torch`。如果降级了，需要从虚拟环境恢复（见 Step 9）。

---

## Step 9: 恢复 PyTorch CUDA 版本（如被降级）

如果 `pip install -e .` 卸载了 torch 2.8.0a0 并安装了 torch 2.7.1，需要从 `build_wheel.py` 创建的虚拟环境恢复：

```bash
# 检查是否缺失 CUDA 库
ls /usr/local/lib64/python3.11/site-packages/torch/lib/libc10_cuda.so 2>/dev/null || echo "MISSING"
ls /usr/local/lib64/python3.11/site-packages/torch/lib/libtorch_cuda.so 2>/dev/null || echo "MISSING"

# 如果缺失，从虚拟环境恢复
rm -rf /usr/local/lib64/python3.11/site-packages/torch
cp -r /home/TensorRT-LLM/.venv-3.11/lib64/python3.11/site-packages/torch /usr/local/lib64/python3.11/site-packages/
cp -r /home/TensorRT-LLM/.venv-3.11/lib64/python3.11/site-packages/torchgen /usr/local/lib64/python3.11/site-packages/ 2>/dev/null || true

# 验证
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Step 10: 设置运行时环境变量

```bash
export LD_LIBRARY_PATH="\
/opt/openEuler/gcc-toolset-14/root/usr/lib64:\
/usr/local/lib/python3.11/site-packages/yr/datasystem/lib:\
/TensorRT-LLM/tensorrt_llm:\
/TensorRT-LLM/cpp/build/tensorrt_llm/thop:\
$(find /usr/local/lib/python3.11/site-packages/nvidia -type d -name 'lib' | tr '\n' ':')\
$LD_LIBRARY_PATH"
```

建议写入 `.bashrc` 或启动脚本：
```bash
echo 'export LD_LIBRARY_PATH="/opt/openEuler/gcc-toolset-14/root/usr/lib64:/usr/local/lib/python3.11/site-packages/yr/datasystem/lib:/TensorRT-LLM/tensorrt_llm:/TensorRT-LLM/cpp/build/tensorrt_llm/thop:'"'"'$(find /usr/local/lib/python3.11/site-packages/nvidia -type d -name "lib" | tr "\n" ":")'"'"'$LD_LIBRARY_PATH"' >> ~/.bashrc
```

---

## Step 11: 验证

### 验证 DataSystem
```bash
python -c "import yr.datasystem; print('DataSystem OK')"
```

### 验证 TensorRT-LLM（理想状态）
```bash
python -c "import tensorrt_llm; print(tensorrt_llm.__file__); print(tensorrt_llm.__version__)"
```

> ⚠️ 当前环境在 SM 89 (RTX 4090 D) 上会遇到 `libtensorrt_llm.so` FMHA 内核符号缺失问题，导致 `import tensorrt_llm` 失败。这是 TensorRT-LLM release/1.0 在 ARM64 + SM 89 上的已知编译缺陷，与 DataSystem 集成无关。

---

## 快速命令汇总（一次性复制版）

```bash
# ===== 容器内执行（假设已挂载好目录） =====

# 1. 安装 DataSystem
pip install /wheels/openyuanrong_datasystem-0.7.7-cp311-cp311-manylinux_2_38_aarch64.whl

# 2. 解压 TensorRT 头文件
cd /tmp && tar -xzf tensorrt.tar.gz && cp -r TensorRT-10.11.0.41/include/* /usr/local/include/

# 3. 初始化子模块
cd /TensorRT-LLM && git submodule update --init --recursive

# 4. 修改 CMake
sed -i 's|"${DATASYSTEM_INCLUDE_DIR}"|"${DATASYSTEM_INCLUDE_DIR}/../"|g' cpp/cmake/modules/FindDATASYSTEM.cmake
sed -i '375s|set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIR})|set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIR} /usr/local/lib/python3.11/site-packages/yr/datasystem/include)|' cpp/CMakeLists.txt

# 5. 安装系统依赖
yum install -y zeromq-devel tbb-devel spdlog-devel 2>/dev/null || dnf install -y zeromq-devel tbb-devel spdlog-devel

# 6. CMake
cd /TensorRT-LLM/cpp/build && rm -rf *
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=$(which python3) \
  -DPython3_EXECUTABLE=$(which python3) \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89" \
  -DTensorRT_ROOT=/usr/local/lib/python3.11/site-packages/tensorrt_libs \
  -DTensorRT_INCLUDE_DIR=/usr/local/include \
  -DTensorRT_LIBRARY=/usr/local/lib/python3.11/site-packages/tensorrt_libs/libnvinfer.so \
  -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_MICRO_BENCHMARKS=OFF

# 7. 编译
make -j$(nproc)

# 8. 复制产物
cd /TensorRT-LLM
mkdir -p tensorrt_llm/bindings
cp cpp/build/tensorrt_llm/pybind/bindings.cpython-311-aarch64-linux-gnu.so tensorrt_llm/
cp cpp/build/tensorrt_llm/libtensorrt_llm.so tensorrt_llm/
cp cpp/build/tensorrt_llm/thop/libth_common.so tensorrt_llm/ 2>/dev/null || true

# 9. 安装 Python 包
pip install -e . --no-build-isolation

# 10. 如 torch 被降级，恢复（检查后再执行）
# cp -r /home/TensorRT-LLM/.venv-3.11/lib64/python3.11/site-packages/torch /usr/local/lib64/python3.11/site-packages/
# cp -r /home/TensorRT-LLM/.venv-3.11/lib64/python3.11/site-packages/torchgen /usr/local/lib64/python3.11/site-packages/ 2>/dev/null || true

# 11. 设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/opt/openEuler/gcc-toolset-14/root/usr/lib64:/usr/local/lib/python3.11/site-packages/yr/datasystem/lib:/TensorRT-LLM/tensorrt_llm:/TensorRT-LLM/cpp/build/tensorrt_llm/thop:$(find /usr/local/lib/python3.11/site-packages/nvidia -type d -name 'lib' | tr '\n' ':')$LD_LIBRARY_PATH"

# 12. 验证
python -c "import yr.datasystem; print('DataSystem OK')"
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"  # SM 89 上可能失败
```

---

## 关键注意事项

1. **bindings.so 位置**: 必须放在 `tensorrt_llm/` 根目录，不能放 `tensorrt_llm/bindings/` 子目录
2. **torch 降级陷阱**: `pip install -e .` 会按 `requirements.txt` 重新解析依赖，可能降级 torch。务必检查后再恢复
3. **libstdc++ 版本**: 容器需要 `/opt/openEuler/gcc-toolset-14/root/usr/lib64` 的 libstdc++（GCC 14），否则会有 `std::ios_base_library_init()` 等符号缺失
4. **DataSystem 运行时**: 必须部署 ETCD + DataSystem Worker（端口 31501），并设置 `DATASYSTEM_HOST` / `DATASYSTEM_PORT` 环境变量
5. **SM 89 限制**: RTX 4090 D (SM 89) 在 TensorRT-LLM release/1.0 上存在 FMHA 内核编译问题，如果是 H100 (SM 90) 可能不存在此问题
