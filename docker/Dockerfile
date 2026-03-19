# PaiRec4TigerLLM - Docker 镜像
# 基础环境: CUDA 12.2, Ubuntu 22.04
# TensorRT-LLM: 1.0.0
# 硬件要求: NVIDIA GPU (L40S/Ada Lovelace 架构)

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    jq \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# 安装 PyTorch (CUDA 12.1 版本兼容 CUDA 12.2)
RUN pip3 install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 安装基础依赖
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    flask==2.3.3 \
    requests==2.31.0 \
    werkzeug==2.3.7 \
    tqdm==4.66.1 \
    tensorboard==2.14.0 \
    onnx==1.15.0 \
    onnxruntime-gpu==1.16.3 \
    pyyaml==6.0.1 \
    psutil==5.9.5

# 安装 TensorRT-LLM 1.0.0
# 参考: https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.0.0
RUN pip3 install --no-cache-dir \
    tensorrt-llm==1.0.0 \
    --extra-index-url https://pypi.nvidia.com || \
    (echo "TensorRT-LLM 1.0.0 安装失败，将使用 PyTorch 推理" && \
     echo "如需 TensorRT-LLM，请手动安装或检查 CUDA 版本兼容性")

# 安装 Go 1.21 (用于 pairec 服务)
RUN wget -q https://go.dev/dl/go1.21.5.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz && \
    rm go1.21.5.linux-amd64.tar.gz
ENV PATH=/usr/local/go/bin:${PATH}
ENV GOPROXY=https://goproxy.cn,direct

# 创建工作目录
WORKDIR /app

# 复制项目文件
COPY . /app/

# 创建必要目录
RUN mkdir -p /app/data/tenrec/processed \
    /app/checkpoints/rqvae \
    /app/checkpoints/decoder \
    /app/exported/rqvae \
    /app/exported/decoder \
    /app/logs/rqvae \
    /app/logs/decoder

# 设置权限
RUN chmod +x /app/scripts/*.sh

# 验证安装
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')" && \
    (python3 -c "import tensorrt_llm; print(f'TensorRT-LLM: {tensorrt_llm.__version__}')" 2>/dev/null || echo "TensorRT-LLM not installed") && \
    go version

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 暴露端口
EXPOSE 8080 8000 6006

# 默认命令
CMD ["/bin/bash"]
