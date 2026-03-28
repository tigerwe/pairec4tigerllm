# Flink 用户特征实时计算

本目录包含用户特征的实时流处理作业，用于将用户点击流转换为推荐服务可用的特征。

## 架构

```
用户点击流 (user-clicks) 
        │
        ▼
┌─────────────────┐
│ Feature Processor│  (Python Flink 简化版)
│                 │
│  - 消费点击事件  │
│  - 维护用户状态  │
│  - 输出特征更新  │
└─────────────────┘
        │
        ▼
用户特征流 (user-features)
        │
        ▼
┌─────────────────┐
│  Pairec Service  │
│                 │
│  - 消费特征更新  │
│  - 本地缓存     │
│  - 生成推荐     │
└─────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
pip install kafka-python
```

### 2. 启动 Kafka（Docker）

```bash
# 使用脚本一键启动
./scripts/setup_kafka.sh docker-up

# 创建 topics
./scripts/setup_kafka.sh setup-topics
```

### 3. 启动特征处理器

```bash
# 方式1：使用脚本
./scripts/setup_kafka.sh test-consumer

# 方式2：直接运行
python flink/user_feature_job.py --mode consumer --kafka localhost:9092
```

### 4. 生产测试数据

```bash
# 方式1：使用脚本
./scripts/setup_kafka.sh test-producer

# 方式2：直接运行
python flink/user_feature_job.py --mode producer --kafka localhost:9092 --users 100 --interval 0.5
```

### 5. 完整端到端测试

```bash
# 一键运行完整测试
./scripts/e2e_test.sh start    # 启动所有服务
./scripts/e2e_test.sh test     # 运行测试
./scripts/e2e_test.sh stop     # 停止所有服务
```

## 配置 Pairec

编辑 `configs/pairec_config.json`，启用 Kafka 特征源：

```json
{
  "recall_algo": "{
    \"feature_source\": \"kafka\",
    \"kafka_config\": {
      \"brokers\": [\"localhost:9092\"],
      \"topic\": \"user-features\",
      \"group_id\": \"pairec-generative\"
    }
  }"
}
```

## 命令参考

### user_feature_job.py

```bash
# 创建 Kafka topics
python flink/user_feature_job.py --mode setup --kafka localhost:9092

# 启动特征消费者
python flink/user_feature_job.py --mode consumer --kafka localhost:9092

# 启动测试生产者
python flink/user_feature_job.py --mode producer \
    --kafka localhost:9092 \
    --users 100 \
    --interval 0.5 \
    --duration 60
```

### setup_kafka.sh

```bash
# Docker 启动 Kafka
./scripts/setup_kafka.sh docker-up

# 创建 topics
./scripts/setup_kafka.sh setup-topics

# 运行完整测试
./scripts/setup_kafka.sh test-full

# 停止 Kafka
./scripts/setup_kafka.sh docker-down
```

### e2e_test.sh

```bash
# 启动完整环境
./scripts/e2e_test.sh start

# 检查状态
./scripts/e2e_test.sh status

# 运行测试
./scripts/e2e_test.sh test

# 快速测试（仅 Kafka 链路）
./scripts/e2e_test.sh quick-test

# 停止所有服务
./scripts/e2e_test.sh stop
```

## 数据格式

### 输入：Click Event

```json
{
  "user_id": "user_00001",
  "item_id": 12345,
  "timestamp": 1699123456789,
  "category": "sports"
}
```

### 输出：User Features

```json
{
  "user_id": "user_00001",
  "click_history": [10001, 10002, 10003, 10004, 10005],
  "update_time": "2024-03-27T10:30:00",
  "realtime_ctr": 0.15,
  "short_term_tags": ["sports", "tech"]
}
```

## Java 版本（可选）

如果需要使用 Flink 集群，可以使用 Java 版本：

```bash
cd flink
mvn clean package

# 提交到 Flink 集群
flink run -c com.pairec.flink.UserFeatureJob \
  target/user-feature-job-1.0-SNAPSHOT.jar \
  --kafka.bootstrap.servers localhost:9092
```

## 监控

### 查看 Kafka Topics

```bash
# 使用脚本
./scripts/setup_kafka.sh status

# 或使用 Kafka 命令行
kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic user-features --from-beginning
```

### 检查特征处理器日志

```bash
tail -f /tmp/pairec4tigerllm/processor.log
```

### 检查 Pairec 特征缓存

```bash
# 查看缓存统计
curl http://localhost:18080/debug/vars 2>/dev/null | grep feature
```

## 故障排查

### 问题1：无法连接 Kafka

```bash
# 检查 Kafka 是否运行
docker ps | grep kafka

# 检查端口
netstat -an | grep 9092

# 测试连接
python -c "from kafka import KafkaClient; c = KafkaClient('localhost:9092'); print(c.cluster.brokers())"
```

### 问题2：Pairec 收不到特征更新

```bash
# 检查 consumer 是否运行
ps aux | grep user_feature_job

# 检查 topic 是否有数据
kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic user-features

# 检查 Pairec 日志
```

### 问题3：特征延迟高

- 检查网络延迟：`ping localhost`
- 检查 Kafka consumer lag
- 考虑增加 Pairec 消费者线程数

## 扩展

### 添加更多特征

编辑 `flink/user_feature_job.py` 中的 `UserFeatureProcessor.process_click`：

```python
features = {
    'user_id': user_id,
    'click_history': list(self.user_histories[user_id]),
    'update_time': datetime.now().isoformat(),
    # 添加新特征
    'click_count_1h': self.calculate_hourly_clicks(user_id),
    'favorite_category': self.get_favorite_category(user_id),
}
```

### 使用 Flink SQL

对于复杂特征计算，可以使用 Flink SQL：

```sql
CREATE TABLE user_clicks (
  user_id STRING,
  item_id BIGINT,
  ts TIMESTAMP(3),
  WATERMARK FOR ts AS ts - INTERVAL '5' SECOND
) WITH (
  'connector' = 'kafka',
  'topic' = 'user-clicks',
  'properties.bootstrap.servers' = 'localhost:9092',
  'format' = 'json'
);

INSERT INTO user_features
SELECT 
  user_id,
  COLLECT_LIST(item_id) as click_history,
  COUNT(*) as click_count
FROM user_clicks
GROUP BY 
  user_id,
  TUMBLE(ts, INTERVAL '1' MINUTE);
```
