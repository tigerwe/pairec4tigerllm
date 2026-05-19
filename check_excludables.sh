#!/bin/bash

PROJECT_DIR="${1:-$(pwd)}"
cd "$PROJECT_DIR" || { echo "无法进入目录: $PROJECT_DIR"; exit 1; }

echo "=========================================="
echo "PaiRec4TigerLLM 移植打包 - 可排除项检查"
echo "项目路径: $(pwd)"
echo "=========================================="
echo ""

# 1. 总体积
TOTAL_KB=$(du -sk . | awk '{print $1}')
echo "【1】项目总体积: $(du -sh . | awk '{print $1}')"
echo ""

# 2. 一级目录体积排行
echo "【2】目录体积排行:"
du -sh */ .[^.]* 2>/dev/null | grep -v '^total' | sort -rh | head -15 | sed 's/^/    /'
echo ""

# 3. 超大文件扫描 (>30MB)
echo "【3】超过 30MB 的文件:"
find . -maxdepth 4 -type f -size +30M -not -path './.git/*' 2>/dev/null | while read f; do
    printf "    %-80s %s\n" "$(echo "$f" | cut -c1-80)" "$(ls -lh "$f" 2>/dev/null | awk '{print $5}')"
done
echo ""

# 辅助函数：统计并累加
check_excludable() {
    local label="$1"
    local path_spec="$2"
    local reason="$3"
    local flag="${4:-[✓]}"  # [✓] 表示安全排除, [⚠] 表示需谨慎

    local count=0
    local size_kb=0

    if [ -d "$path_spec" ]; then
        size_kb=$(du -sk "$path_spec" 2>/dev/null | awk '{print $1}')
        count=1
    elif [ -f "$path_spec" ]; then
        size_kb=$(du -sk "$path_spec" 2>/dev/null | awk '{print $1}')
        count=1
    else
        # 尝试 glob
        local files=$(find . -path "./$path_spec" 2>/dev/null)
        if [ -n "$files" ]; then
            count=$(echo "$files" | wc -l)
            size_kb=$(echo "$files" | xargs -I{} du -sk {} 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
        fi
    fi

    if [ "$count" -gt 0 ] && [ "$size_kb" -gt 0 ]; then
        local size_human=$(echo "$size_kb" | awk '{printf "%.1fMB", $1/1024}')
        printf "    %-8s %-22s %8s -> %s\n" "$flag" "$label" "$size_human" "$reason"
        echo "$size_kb"
    else
        echo "0"
    fi
}

echo "【4】可安全排除的项目（不影响 ARM 运行）:"
SAVED=0

# 4.1 版本控制
VAL=$(check_excludable ".git/" ".git" "版本控制，部署不需要" "[✓]"); SAVED=$((SAVED + VAL))

# 4.2 Python缓存
VAL=$(check_excludable "__pycache__/" "__pycache__" "运行时重建" "[✓]"); SAVED=$((SAVED + VAL))
PCOUNT=$(find . -name "*.pyc" 2>/dev/null | wc -l)
[ "$PCOUNT" -gt 0 ] && printf "    [✓] %-22s %8s -> %s\n" "*.pyc 文件" "${PCOUNT}个" "运行时重建"

# 4.3 文档
for doc in AGENTS.md README.md CHANGELOG.md CONSTITUTION.md; do
    [ -f "$doc" ] && {
        SIZE=$(stat -c%s "$doc" 2>/dev/null | awk '{printf "%.1fKB", $1/1024}')
        printf "    [✓] %-22s %8s -> %s\n" "$doc" "$SIZE" "部署不需要"
    }
done

# 4.4 Docker
VAL=$(check_excludable "docker/" "docker" "ARM不用Docker方案" "[✓]"); SAVED=$((SAVED + VAL))
[ -f "docker-compose.yml" ] && printf "    [✓] %-22s %8s -> %s\n" "docker-compose.yml" "-" "ARM不用Docker"

# 4.5 测试/文档目录
VAL=$(check_excludable "docs/" "docs" "部署文档不需要" "[✓]"); SAVED=$((SAVED + VAL))
VAL=$(check_excludable "tests/" "tests" "测试代码不需要" "[✓]"); SAVED=$((SAVED + VAL))

# 4.6 TensorRT引擎
ENGINE_KB=0
if find . -name "*.engine" 2>/dev/null | grep -q .; then
    ENGINE_KB=$(find . -name "*.engine" 2>/dev/null -exec du -sk {} + | awk '{sum+=$1} END {print sum}')
    ECOUNT=$(find . -name "*.engine" 2>/dev/null | wc -l)
    printf "    [✓] %-22s %8s -> %s\n" "*.engine (${ECOUNT}个)" "$(echo $ENGINE_KB | awk '{printf "%.1fMB", $1/1024}')" "ARM不支持TensorRT"
    SAVED=$((SAVED + ENGINE_KB))
fi

# 4.7 中间 checkpoint
MID_KB=0
if find checkpoints -name "checkpoint_epoch_*.pt" 2>/dev/null | grep -q .; then
    MID_KB=$(find checkpoints -name "checkpoint_epoch_*.pt" 2>/dev/null -exec du -sk {} + | awk '{sum+=$1} END {print sum}')
    MCOUNT=$(find checkpoints -name "checkpoint_epoch_*.pt" 2>/dev/null | wc -l)
    printf "    [⚠] %-22s %8s -> %s\n" "中间epoch(${MCOUNT}个)" "$(echo $MID_KB | awk '{printf "%.1fMB", $1/1024}')" "只保留*_best.pt"
    SAVED=$((SAVED + MID_KB))
fi

# 4.8 原始CSV
CSV_KB=0
if find data -name "*.csv" 2>/dev/null | grep -q .; then
    CSV_KB=$(find data -name "*.csv" 2>/dev/null -exec du -sk {} + | awk '{sum+=$1} END {print sum}')
    CCOUNT=$(find data -name "*.csv" 2>/dev/null | wc -l)
    printf "    [⚠] %-22s %8s -> %s\n" "原始CSV(${CCOUNT}个)" "$(echo $CSV_KB | awk '{printf "%.1fMB", $1/1024}')" "processed/已够用"
    SAVED=$((SAVED + CSV_KB))
fi

# 4.9 训练脚本
for f in training/rqvae/train.py training/rqvae/export.py training/decoder/train.py training/decoder/export.py; do
    [ -f "$f" ] && printf "    [✓] %-22s %8s -> %s\n" "$(basename $f)" "-" "模型已训好"
done

# 4.10 TRT相关脚本
for f in scripts/build_trt_engine.sh scripts/start_trt_server.sh scripts/check_environment.py; do
    [ -f "$f" ] && printf "    [✓] %-22s %8s -> %s\n" "$(basename $f)" "-" "TensorRT相关"
done

# 4.11 Kafka测试脚本
for f in scripts/e2e_test.sh scripts/test_api.sh scripts/test_e2e_mock.sh scripts/setup_kafka.sh; do
    [ -f "$f" ] && printf "    [✓] %-22s %8s -> %s\n" "$(basename $f)" "-" "测试链路"
done

# 4.12 Flink
VAL=$(check_excludable "flink/" "flink" "如不需要实时链路可删" "[⚠]"); SAVED=$((SAVED + VAL))

# 4.13 其他根目录垃圾
for f in VERSION .gitignore check_image_env.sh check_trtllm_env.sh build_engine_simple.py test_trtllm.py; do
    [ -f "$f" ] && printf "    [✓] %-22s %8s -> %s\n" "$f" "-" "运行时不需要"
done

echo ""
echo "【5】⚠️  ARM 运行时真正需要的核心路径（别删！）:"
echo "    checkpoints/rqvae/rqvae_best.pt"
echo "    checkpoints/decoder/decoder_best.pt"
echo "    data/tenrec/processed/ (train_sequences.json, test_sequences.json, ...)"
echo "    data/user_features.json"
echo "    training/rqvae/model.py        ← inference/server.py 会 import"
echo "    training/decoder/model.py      ← inference/server.py 会 import"
echo "    inference/                     ← 推理服务入口"
echo "    services/                      ← Go 服务代码"
echo "    configs/                       ← 配置文件"
echo "    go.mod go.sum                  ← 模块定义"
echo "    exported/                      ← 如有导出模型映射"
echo ""

echo "【6】📊 体积预估:"
echo "    当前总体积:     $(echo "$TOTAL_KB" | awk '{printf "%.1fMB", $1/1024}')"
echo "    可节省约:       $(echo "$SAVED" | awk '{printf "%.1fMB", $1/1024}')"
echo "    打包后约:       $(echo $((TOTAL_KB - SAVED)) | awk '{printf "%.1fMB", $1/1024}')"
echo ""

echo "【7】📦 生成的 tar 命令:"
echo "---------------------------------------------------"
echo "tar czvf pairec4tigerllm-arm.tar.gz \\"
echo "    --exclude='.git' \\"
echo "    --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \\"
[ -d "docker" ] && echo "    --exclude='docker' \\"
[ -f "docker-compose.yml" ] && echo "    --exclude='docker-compose.yml' \\"
[ -d "docs" ] && echo "    --exclude='docs' \\"
[ -d "tests" ] && echo "    --exclude='tests' \\"
[ "$ENGINE_KB" -gt 0 ] && echo "    --exclude='*.engine' \\"
[ "$MID_KB" -gt 0 ] && echo "    --exclude='checkpoints/*/checkpoint_epoch_*.pt' \\"
[ "$CSV_KB" -gt 0 ] && echo "    --exclude='data/tenrec/*.csv' \\"
echo "    --exclude='training/rqvae/train.py' --exclude='training/rqvae/export.py' \\"
echo "    --exclude='training/decoder/train.py' --exclude='training/decoder/export.py' \\"
echo "    --exclude='scripts/build_trt_engine.sh' \\"
echo "    --exclude='scripts/start_trt_server.sh' --exclude='scripts/check_environment.py' \\"
echo "    --exclude='scripts/train_rqvae.sh' --exclude='scripts/train_decoder.sh' \\"
echo "    --exclude='scripts/e2e_test.sh' --exclude='scripts/test_api.sh' \\"
echo "    --exclude='scripts/test_e2e_mock.sh' --exclude='scripts/setup_kafka.sh' \\"
[ -d "flink" ] && echo "    --exclude='flink' \\"
echo "    --exclude='*.tar.gz' --exclude='*.log' --exclude='logs/*' \\"
echo "    --exclude='VERSION' --exclude='.gitignore' \\"
echo "    --exclude='check_image_env.sh' --exclude='check_trtllm_env.sh' \\"
echo "    --exclude='build_engine_simple.py' --exclude='test_trtllm.py' \\"
echo "    --exclude='AGENTS.md' --exclude='README.md' --exclude='CHANGELOG.md' --exclude='CONSTITUTION.md' \\"
echo "    ."
echo "---------------------------------------------------"
