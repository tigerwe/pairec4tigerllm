#!/bin/bash
# e2e_test.sh
#
# 端到端测试脚本 - 验证完整数据流：
#   Kafka (user-clicks) -> Flink Processor -> Kafka (user-features) -> Pairec -> TensorRT-LLM
#
# 使用方法:
#   ./scripts/e2e_test.sh [start|stop|status|test]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_DIR="/tmp/pairec4tigerllm"

mkdir -p "$PID_DIR"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置
KAFKA_SERVERS="localhost:9092"
PAIREC_PORT=18080
TRTLLM_PORT=8000

# 帮助信息
show_help() {
    cat << EOF
端到端测试脚本

Usage: $0 COMMAND

Commands:
    start       启动所有服务 (Kafka + Processor + Pairec + TRT-LLM)
    stop        停止所有服务
    status      检查服务状态
    test        运行完整测试流程
    quick-test  快速测试 (仅验证 Kafka 链路)

Examples:
    # 启动完整环境
    $0 start

    # 检查状态
    $0 status

    # 运行测试
    $0 test

    # 停止所有服务
    $0 stop

EOF
}

# 检查依赖
check_deps() {
    echo -e "${BLUE}Checking dependencies...${NC}"
    
    # 检查 Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker not found. Please install Docker first.${NC}"
        exit 1
    fi
    
    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python3 not found. Please install Python3.${NC}"
        exit 1
    fi
    
    # 检查 Go
    if ! command -v go &> /dev/null; then
        echo -e "${RED}Go not found. Please install Go.${NC}"
        exit 1
    fi
    
    # 检查 Python 依赖
    python3 -c "import kafka" 2>/dev/null || {
        echo -e "${YELLOW}Installing kafka-python...${NC}"
        pip3 install kafka-python requests -q
    }
    
    echo -e "${GREEN}All dependencies OK${NC}"
}

# 启动服务
start_services() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}    Starting All Services               ${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    check_deps
    
    # 1. 启动 Kafka
    echo -e "\n${YELLOW}[1/4] Starting Kafka...${NC}"
    "$SCRIPT_DIR/setup_kafka.sh" docker-up || {
        echo -e "${RED}Failed to start Kafka${NC}"
        exit 1
    }
    
    # 2. 创建 topics
    echo -e "\n${YELLOW}[2/4] Creating Kafka topics...${NC}"
    sleep 5
    "$SCRIPT_DIR/setup_kafka.sh" setup-topics
    
    # 3. 启动特征处理器
    echo -e "\n${YELLOW}[3/4] Starting Feature Processor...${NC}"
    cd "$PROJECT_DIR"
    nohup python3 flink/user_feature_job.py \
        --mode consumer \
        --kafka "$KAFKA_SERVERS" > "$PID_DIR/processor.log" 2>&1 &
    echo $! > "$PID_DIR/processor.pid"
    echo -e "${GREEN}Feature Processor started (PID: $(cat "$PID_DIR/processor.pid"))${NC}"
    
    # 4. 检查 TRT-LLM 服务
    echo -e "\n${YELLOW}[4/4] Checking TRT-LLM service...${NC}"
    if curl -s "http://localhost:$TRTLLM_PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}TRT-LLM is running on port $TRTLLM_PORT${NC}"
    else
        echo -e "${YELLOW}Warning: TRT-LLM not detected on port $TRTLLM_PORT${NC}"
        echo "Please start TRT-LLM manually:"
        echo "  python inference/trt_llm/server.py --model_path checkpoints/decoder/decoder_best.pt"
    fi
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}    All Services Started!               ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Service Status:"
    echo "  Kafka:          Running (localhost:9092)"
    echo "  Feature Proc:   Running (PID: $(cat "$PID_DIR/processor.pid" 2>/dev/null || echo 'N/A'))"
    echo "  TRT-LLM:        $(curl -s "http://localhost:$TRTLLM_PORT/health" 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo 'Not running')"
    echo ""
    echo "Next steps:"
    echo "  1. Start Pairec: cd services && go run main.go"
    echo "  2. Run test:     $0 test"
}

# 停止服务
stop_services() {
    echo -e "${GREEN}Stopping all services...${NC}"
    
    # 停止特征处理器
    if [ -f "$PID_DIR/processor.pid" ]; then
        PID=$(cat "$PID_DIR/processor.pid")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping Feature Processor (PID: $PID)..."
            kill "$PID" 2>/dev/null || true
        fi
        rm -f "$PID_DIR/processor.pid"
    fi
    
    # 停止 Kafka
    "$SCRIPT_DIR/setup_kafka.sh" docker-down 2>/dev/null || true
    
    echo -e "${GREEN}All services stopped${NC}"
}

# 检查状态
status() {
    echo -e "${BLUE}Service Status:${NC}"
    echo ""
    
    # Kafka
    if docker ps | grep -q "kafka"; then
        echo -e "  Kafka:        ${GREEN}Running${NC} (localhost:9092)"
    else
        echo -e "  Kafka:        ${RED}Stopped${NC}"
    fi
    
    # Feature Processor
    if [ -f "$PID_DIR/processor.pid" ]; then
        PID=$(cat "$PID_DIR/processor.pid")
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "  Feature Proc: ${GREEN}Running${NC} (PID: $PID)"
        else
            echo -e "  Feature Proc: ${RED}Stopped${NC} (stale PID file)"
        fi
    else
        echo -e "  Feature Proc: ${RED}Stopped${NC}"
    fi
    
    # Pairec
    if curl -s "http://localhost:$PAIREC_PORT/health" > /dev/null 2>&1; then
        echo -e "  Pairec:       ${GREEN}Running${NC} (port: $PAIREC_PORT)"
    else
        echo -e "  Pairec:       ${RED}Stopped${NC} (port: $PAIREC_PORT)"
    fi
    
    # TRT-LLM
    if curl -s "http://localhost:$TRTLLM_PORT/health" > /dev/null 2>&1; then
        echo -e "  TRT-LLM:      ${GREEN}Running${NC} (port: $TRTLLM_PORT)"
    else
        echo -e "  TRT-LLM:      ${RED}Stopped${NC} (port: $TRTLLM_PORT)"
    fi
}

# 快速测试 (仅 Kafka 链路)
quick_test() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}    Quick Test (Kafka Only)             ${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    cd "$PROJECT_DIR"
    
    # 启动消费者（后台）
    echo -e "\n${YELLOW}Starting feature processor...${NC}"
    python3 flink/user_feature_job.py --mode consumer --kafka "$KAFKA_SERVERS" &
    CONSUMER_PID=$!
    sleep 2
    
    # 启动生产者（短时间）
    echo -e "\n${YELLOW}Producing test events (5s)...${NC}"
    python3 flink/user_feature_job.py --mode producer --kafka "$KAFKA_SERVERS" --users 5 --interval 0.1 --duration 5
    
    # 等待处理
    sleep 2
    
    # 停止消费者
    kill $CONSUMER_PID 2>/dev/null || true
    
    echo -e "\n${GREEN}Quick test completed!${NC}"
    echo ""
    echo "Check logs above for:"
    echo "  - 'Sent X events' from producer"
    echo "  - 'User=X, History=Y' from consumer"
}

# 完整测试
run_test() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}    Full End-to-End Test                ${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    cd "$PROJECT_DIR"
    
    # 1. 检查服务状态
    echo -e "\n${YELLOW}[Step 1] Checking service status...${NC}"
    status
    
    # 2. 生产测试数据
    echo -e "\n${YELLOW}[Step 2] Producing test events to user-clicks...${NC}"
    python3 flink/user_feature_job.py \
        --mode producer \
        --kafka "$KAFKA_SERVERS" \
        --users 50 \
        --interval 0.5 \
        --duration 30 &
    PRODUCER_PID=$!
    
    # 3. 等待生产者完成
    wait $PRODUCER_PID
    echo -e "${GREEN}Test data production completed${NC}"
    
    # 4. 等待 Pairec 消费
    echo -e "\n${YELLOW}[Step 3] Waiting for Pairec to consume features...${NC}"
    sleep 5
    
    # 5. 发送推荐请求
    echo -e "\n${YELLOW}[Step 4] Testing recommendation API...${NC}"
    
    TEST_USER="user_00001"
    
    # 生成一些点击数据给这个用户
    python3 << EOF
import json
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='$KAFKA_SERVERS',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# 为用户生成点击历史
for item_id in [10001, 10002, 10003, 10004, 10005]:
    event = {
        'user_id': '$TEST_USER',
        'item_id': item_id,
        'timestamp': 1234567890,
        'category': 'test'
    }
    producer.send('user-clicks', value=event)

producer.flush()
producer.close()
print(f"Generated click history for user: $TEST_USER")
EOF
    
    sleep 3
    
    # 6. 调用推荐 API
    echo -e "\n${YELLOW}[Step 5] Calling recommendation API...${NC}"
    
    RESPONSE=$(curl -s -X POST "http://localhost:$PAIREC_PORT/api/rec/feed" \
        -H "Content-Type: application/json" \
        -d "{\"uid\":\"$TEST_USER\",\"size\":10}" 2>/dev/null || echo '{"error":"connection failed"}')
    
    echo "Response:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
    
    # 7. 验证结果
    if echo "$RESPONSE" | grep -q "items\|recommendations"; then
        echo -e "\n${GREEN}✓ Test PASSED: Recommendation API returned results${NC}"
    else
        echo -e "\n${RED}✗ Test FAILED: No recommendations returned${NC}"
        echo "Please check:"
        echo "  1. Is Pairec running? (cd services && go run main.go)"
        echo "  2. Is TRT-LLM running?"
        echo "  3. Check logs in $PID_DIR/"
    fi
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}    End-to-End Test Completed!          ${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# 主逻辑
case "${1:-}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    status)
        status
        ;;
    test)
        run_test
        ;;
    quick-test)
        quick_test
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac
