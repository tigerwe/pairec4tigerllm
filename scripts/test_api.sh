#!/bin/bash
# API 测试脚本
# 验证 pairec4tigerllm 服务是否正常工作

set -e

echo "========================================="
echo "PaiRec4TigerLLM API 测试"
echo "========================================="
echo ""

# 配置
PAIREC_URL="${PAIREC_URL:-http://localhost:8080}"
INFERENCE_URL="${INFERENCE_URL:-http://localhost:8000}"
TIMEOUT="${TIMEOUT:-10}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 计数器
PASSED=0
FAILED=0

# 测试函数
test_health() {
    local url=$1
    local name=$2
    
    echo -n "测试 $name 健康检查 ... "
    
    if curl -s -m $TIMEOUT "$url/health" > /dev/null 2>&1; then
        echo -e "${GREEN}通过${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}失败${NC}"
        ((FAILED++))
        return 1
    fi
}

test_inference() {
    echo -n "测试推理服务推荐接口 ... "
    
    local response=$(curl -s -m $TIMEOUT -X POST "$INFERENCE_URL/recommend" \
        -H "Content-Type: application/json" \
        -d '{
            "user_id": "test_user",
            "history": [[100, 50, 25, 10], [101, 51, 26, 11]],
            "topk": 5,
            "temperature": 1.0,
            "beam_width": 1
        }' 2>/dev/null)
    
    if [ -n "$response" ] && echo "$response" | grep -q "recommendations"; then
        echo -e "${GREEN}通过${NC}"
        echo "  响应预览: $(echo "$response" | jq -c '.recommendations[:2]' 2>/dev/null || echo "N/A")"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}失败${NC}"
        echo "  响应: $response"
        ((FAILED++))
        return 1
    fi
}

test_pairec() {
    echo -n "测试 pairec 推荐接口 ... "
    
    local response=$(curl -s -m $TIMEOUT -X POST "$PAIREC_URL/api/rec/feed" \
        -H "Content-Type: application/json" \
        -d '{
            "uid": "76295990",
            "size": 5,
            "scene_id": "home_feed"
        }' 2>/dev/null)
    
    if [ -n "$response" ] && echo "$response" | grep -q '"code":200'; then
        echo -e "${GREEN}通过${NC}"
        local item_count=$(echo "$response" | jq '.items | length' 2>/dev/null || echo "0")
        echo "  返回物品数: $item_count"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}失败${NC}"
        echo "  响应: $response"
        ((FAILED++))
        return 1
    fi
}

test_performance() {
    echo ""
    echo "性能测试 (10 次请求):"
    
    local total_time=0
    local success_count=0
    
    for i in {1..10}; do
        local start_time=$(date +%s%N)
        
        local response=$(curl -s -m $TIMEOUT -X POST "$PAIREC_URL/api/rec/feed" \
            -H "Content-Type: application/json" \
            -d '{
                "uid": "76295990",
                "size": 10,
                "scene_id": "home_feed"
            }' 2>/dev/null)
        
        local end_time=$(date +%s%N)
        local elapsed=$(( (end_time - start_time) / 1000000 ))  # 转换为毫秒
        
        if echo "$response" | grep -q '"code":200'; then
            ((success_count++))
            total_time=$((total_time + elapsed))
            echo "  请求 $i: ${elapsed}ms"
        else
            echo "  请求 $i: 失败"
        fi
    done
    
    if [ $success_count -gt 0 ]; then
        local avg_time=$((total_time / success_count))
        echo ""
        echo "  成功率: $success_count/10"
        echo "  平均延迟: ${avg_time}ms"
        
        if [ $avg_time -lt 100 ]; then
            echo -e "  ${GREEN}性能良好${NC}"
        elif [ $avg_time -lt 200 ]; then
            echo -e "  ${YELLOW}性能一般${NC}"
        else
            echo -e "  ${RED}性能较差${NC}"
        fi
    fi
}

# 主测试流程
main() {
    echo "测试配置:"
    echo "  pairec 服务: $PAIREC_URL"
    echo "  推理服务: $INFERENCE_URL"
    echo "  超时时间: ${TIMEOUT}s"
    echo ""
    
    # 1. 健康检查
    echo "1. 健康检查"
    echo "----------------------------------------"
    test_health "$INFERENCE_URL" "推理服务"
    test_health "$PAIREC_URL" "pairec 服务"
    echo ""
    
    # 2. 功能测试
    echo "2. 功能测试"
    echo "----------------------------------------"
    
    # 检查推理服务是否可用
    if curl -s -m $TIMEOUT "$INFERENCE_URL/health" > /dev/null 2>&1; then
        test_inference
    else
        echo -e "${YELLOW}推理服务不可用，跳过推理测试${NC}"
    fi
    
    # 检查 pairec 是否可用
    if curl -s -m $TIMEOUT "$PAIREC_URL/health" > /dev/null 2>&1; then
        test_pairec
    else
        echo -e "${YELLOW}pairec 服务不可用，跳过推荐测试${NC}"
    fi
    
    echo ""
    
    # 3. 性能测试
    echo "3. 性能测试"
    echo "----------------------------------------"
    if curl -s -m $TIMEOUT "$PAIREC_URL/health" > /dev/null 2>&1; then
        test_performance
    else
        echo -e "${YELLOW}pairec 服务不可用，跳过性能测试${NC}"
    fi
    
    echo ""
    echo "========================================="
    echo "测试结果: $PASSED 通过, $FAILED 失败"
    echo "========================================="
    
    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}✓ 所有测试通过！${NC}"
        exit 0
    else
        echo -e "${RED}✗ 部分测试失败${NC}"
        echo ""
        echo "故障排查:"
        echo "  1. 检查服务是否启动:"
        echo "     curl $INFERENCE_URL/health"
        echo "     curl $PAIREC_URL/health"
        echo "  2. 查看服务日志:"
        echo "     docker-compose logs -f inference"
        echo "     docker-compose logs -f pairec"
        echo "  3. 检查模型文件是否存在"
        exit 1
    fi
}

# 运行测试
main
