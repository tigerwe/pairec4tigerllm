#!/bin/bash
# setup_kafka.sh
#
# 一键设置 Kafka 环境并启动端到端测试.
# 支持两种模式：
#   1. Docker 模式：自动启动 Kafka 容器
#   2. 本地模式：使用已存在的 Kafka

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
KAFKA_VERSION="7.4.0"
COMPOSE_FILE="docker-compose.kafka.yml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 帮助信息
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    docker-up       使用 Docker 启动 Kafka
    docker-down     停止 Docker Kafka
    setup-topics    创建必要的 Kafka topics
    test-producer   启动测试数据生产者
    test-consumer   启动特征消费者
    test-full       运行完整端到端测试
    status          检查 Kafka 状态

Options:
    -h, --help      显示帮助信息
    -l, --local     使用本地 Kafka (默认: localhost:9092)
    -k, --kafka     指定 Kafka 地址 (默认: localhost:9092)

Examples:
    # 使用 Docker 启动完整环境
    $0 docker-up

    # 创建 topics
    $0 setup-topics

    # 运行完整测试
    $0 test-full

    # 使用本地 Kafka
    $0 -k localhost:9092 test-full

EOF
}

# 解析参数
KAFKA_SERVERS="localhost:9092"
USE_LOCAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -l|--local)
            USE_LOCAL=true
            shift
            ;;
        -k|--kafka)
            KAFKA_SERVERS="$2"
            shift 2
            ;;
        *)
            COMMAND="$1"
            shift
            ;;
    esac
done

# 检查命令
if [ -z "$COMMAND" ]; then
    show_help
    exit 1
fi

# 检查 Python 依赖
check_python_deps() {
    python3 -c "import kafka" 2>/dev/null || {
        echo -e "${YELLOW}Installing kafka-python...${NC}"
        pip3 install kafka-python -q
    }
}

# Docker 启动 Kafka
docker_up() {
    echo -e "${GREEN}Starting Kafka with Docker...${NC}"
    
    cd "$PROJECT_DIR"
    
    # 创建 Docker Compose 文件
    cat > "$COMPOSE_FILE" << 'EOF'
version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:${KAFKA_VERSION:-7.4.0}
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:${KAFKA_VERSION:-7.4.0}
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
    healthcheck:
      test: ["CMD", "kafka-broker-api-versions", "--bootstrap-server", "localhost:9092"]
      interval: 10s
      timeout: 5s
      retries: 5
EOF

    docker-compose -f "$COMPOSE_FILE" up -d
    
    echo -e "${YELLOW}Waiting for Kafka to be ready...${NC}"
    sleep 10
    
    # 检查健康状态
    for i in {1..30}; do
        if docker-compose -f "$COMPOSE_FILE" exec -T kafka kafka-broker-api-versions --bootstrap-server localhost:9092 >/dev/null 2>&1; then
            echo -e "${GREEN}Kafka is ready!${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
    done
    
    echo -e "${RED}Kafka failed to start${NC}"
    return 1
}

# Docker 停止 Kafka
docker_down() {
    echo -e "${GREEN}Stopping Kafka...${NC}"
    cd "$PROJECT_DIR"
    docker-compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true
    rm -f "$COMPOSE_FILE"
    echo -e "${GREEN}Kafka stopped${NC}"
}

# 创建 Topics
setup_topics() {
    echo -e "${GREEN}Creating Kafka topics...${NC}"
    check_python_deps
    
    cd "$PROJECT_DIR"
    python3 flink/user_feature_job.py --mode setup --kafka "$KAFKA_SERVERS"
}

# 启动测试生产者
test_producer() {
    echo -e "${GREEN}Starting test producer...${NC}"
    check_python_deps
    
    cd "$PROJECT_DIR"
    python3 flink/user_feature_job.py \
        --mode producer \
        --kafka "$KAFKA_SERVERS" \
        --users 100 \
        --interval 0.5
}

# 启动特征消费者
test_consumer() {
    echo -e "${GREEN}Starting feature consumer...${NC}"
    check_python_deps
    
    cd "$PROJECT_DIR"
    python3 flink/user_feature_job.py \
        --mode consumer \
        --kafka "$KAFKA_SERVERS"
}

# 完整端到端测试
test_full() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}    Running Full End-to-End Test        ${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # 1. 创建 topics
    setup_topics
    
    # 2. 启动消费者（后台）
    echo -e "\n${YELLOW}Starting feature processor in background...${NC}"
    cd "$PROJECT_DIR"
    python3 flink/user_feature_job.py \
        --mode consumer \
        --kafka "$KAFKA_SERVERS" &
    CONSUMER_PID=$!
    
    # 等待消费者启动
    sleep 3
    
    # 3. 启动生产者（后台）
    echo -e "\n${YELLOW}Starting test producer...${NC}"
    python3 flink/user_feature_job.py \
        --mode producer \
        --kafka "$KAFKA_SERVERS" \
        --users 10 \
        --interval 0.2 \
        --duration 30 &
    PRODUCER_PID=$!
    
    # 4. 等待生产者完成
    echo -e "\n${YELLOW}Waiting for producer to complete (30s)...${NC}"
    wait $PRODUCER_PID
    
    # 5. 给消费者一些时间处理
    echo -e "\n${YELLOW}Waiting for consumer to process...${NC}"
    sleep 5
    
    # 6. 停止消费者
    echo -e "\n${YELLOW}Stopping consumer...${NC}"
    kill $CONSUMER_PID 2>/dev/null || true
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}    End-to-End Test Completed!          ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Test Summary:"
    echo "  - Produced events to: user-clicks"
    echo "  - Consumed and processed to: user-features"
    echo "  - Pairec service can now consume from: user-features"
    echo ""
    echo "Next steps:"
    echo "  1. Update pairec_config.json to use Kafka:"
    echo "     \"feature_source\": \"kafka\""
    echo "  2. Start pairec service: cd services && go run main.go"
    echo "  3. Send recommendation request"
}

# 检查状态
status() {
    echo -e "${GREEN}Checking Kafka status...${NC}"
    
    # 检查 Docker 容器
    if docker ps | grep -q "kafka"; then
        echo -e "${GREEN}Docker Kafka: Running${NC}"
    else
        echo -e "${YELLOW}Docker Kafka: Not running${NC}"
    fi
    
    # 检查 topics
    check_python_deps
    python3 -c "
from kafka import KafkaAdminClient
try:
    client = KafkaAdminClient(bootstrap_servers='$KAFKA_SERVERS')
    topics = client.list_topics()
    print(f'Available topics: {topics}')
    if 'user-clicks' in topics and 'user-features' in topics:
        print('Required topics: OK')
    else:
        print('Required topics: MISSING (run: setup-topics)')
except Exception as e:
    print(f'Cannot connect to Kafka: {e}')
"
}

# 执行命令
case $COMMAND in
    docker-up)
        docker_up
        ;;
    docker-down)
        docker_down
        ;;
    setup-topics)
        setup_topics
        ;;
    test-producer)
        test_producer
        ;;
    test-consumer)
        test_consumer
        ;;
    test-full)
        test_full
        ;;
    status)
        status
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        show_help
        exit 1
        ;;
esac
