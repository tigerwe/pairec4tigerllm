#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户特征实时计算作业 (Python 简化版).

功能：
1. 消费用户点击流 (Kafka topic: user-clicks)
2. 实时聚合每个用户的最近 N 个点击物品
3. 输出到 Kafka topic: user-features

依赖：
    pip install kafka-python

使用方式：
    # 1. 启动消费者（持续运行）
    python flink/user_feature_job.py --mode consumer
    
    # 2. 生产测试数据（另一个终端）
    python flink/user_feature_job.py --mode producer --users 100 --interval 1

"""

import argparse
import json
import time
import random
from datetime import datetime
from collections import defaultdict, deque
from threading import Thread
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 可选依赖
try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: kafka-python not installed. Run: pip install kafka-python")


# ============ 配置 ============
DEFAULT_KAFKA_SERVERS = ["localhost:9092"]
INPUT_TOPIC = "user-clicks"
OUTPUT_TOPIC = "user-features"
MAX_HISTORY = 50  # 保留最近50个点击


class UserFeatureProcessor:
    """用户特征处理器（简化版流处理）."""
    
    def __init__(self, kafka_servers):
        self.kafka_servers = kafka_servers
        # 内存中的用户状态：user_id -> deque of item_ids
        self.user_histories = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
        self.producer = None
        
    def start_consumer(self):
        """启动 Kafka 消费者."""
        if not KAFKA_AVAILABLE:
            print("Error: kafka-python not installed")
            return
            
        print(f"Starting consumer...")
        print(f"  Kafka: {self.kafka_servers}")
        print(f"  Input: {INPUT_TOPIC}")
        print(f"  Output: {OUTPUT_TOPIC}")
        
        # 创建生产者（用于输出特征）
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda v: v.encode('utf-8') if v else None
        )
        
        # 创建消费者
        consumer = KafkaConsumer(
            INPUT_TOPIC,
            bootstrap_servers=self.kafka_servers,
            group_id='python-feature-processor',
            auto_offset_reset='latest',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        print(f"\nConsuming from {INPUT_TOPIC}...")
        
        try:
            for message in consumer:
                try:
                    self.process_click(message.value)
                except Exception as e:
                    print(f"Error processing message: {e}")
        except KeyboardInterrupt:
            print("\nStopping consumer...")
        finally:
            consumer.close()
            if self.producer:
                self.producer.close()
    
    def process_click(self, event):
        """处理单个点击事件."""
        user_id = event.get('user_id')
        item_id = event.get('item_id')
        
        if not user_id or not item_id:
            return
        
        # 更新用户历史
        self.user_histories[user_id].append(item_id)
        
        # 构建特征
        features = {
            'user_id': user_id,
            'click_history': list(self.user_histories[user_id]),
            'update_time': datetime.now().isoformat(),
            'realtime_ctr': 0.0,  # 简化处理
            'short_term_tags': []
        }
        
        # 发送到 Kafka
        if self.producer:
            self.producer.send(OUTPUT_TOPIC, key=user_id, value=features)
            self.producer.flush()
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] User={user_id}, "
              f"History={len(features['click_history'])}, Items={features['click_history'][-5:]}")


class ClickEventProducer:
    """点击事件生产者（用于测试）."""
    
    def __init__(self, kafka_servers):
        self.kafka_servers = kafka_servers
        self.producer = None
        
    def start(self, num_users=100, interval=1.0, duration=None):
        """开始生产测试数据."""
        if not KAFKA_AVAILABLE:
            print("Error: kafka-python not installed")
            return
            
        print(f"Starting producer...")
        print(f"  Kafka: {self.kafka_servers}")
        print(f"  Target: {INPUT_TOPIC}")
        print(f"  Users: {num_users}")
        print(f"  Interval: {interval}s")
        
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # 预生成用户池
        user_pool = [f"user_{i:05d}" for i in range(num_users)]
        item_pool = list(range(10000, 20000))  # 10000个物品
        
        print(f"\nProducing events to {INPUT_TOPIC}...")
        print("Press Ctrl+C to stop\n")
        
        start_time = time.time()
        event_count = 0
        
        try:
            while True:
                # 随机选择一个用户点击一个物品
                user_id = random.choice(user_pool)
                item_id = random.choice(item_pool)
                
                event = {
                    'user_id': user_id,
                    'item_id': item_id,
                    'timestamp': int(time.time() * 1000),
                    'category': random.choice(['sports', 'tech', 'news', 'entertainment'])
                }
                
                self.producer.send(INPUT_TOPIC, value=event)
                event_count += 1
                
                if event_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = event_count / elapsed if elapsed > 0 else 0
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Sent {event_count} events, Rate: {rate:.1f} events/s")
                
                # 检查是否达到持续时间
                if duration and (time.time() - start_time) >= duration:
                    break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nStopping producer...")
        finally:
            self.producer.close()
            print(f"Total events sent: {event_count}")


def setup_kafka_topics(kafka_servers):
    """创建必要的 Kafka topics."""
    if not KAFKA_AVAILABLE:
        print("Error: kafka-python not installed")
        return False
        
    from kafka.admin import KafkaAdminClient, NewTopic
    
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=kafka_servers)
        
        topics = [INPUT_TOPIC, OUTPUT_TOPIC]
        existing_topics = admin_client.list_topics()
        
        for topic in topics:
            if topic in existing_topics:
                print(f"Topic already exists: {topic}")
                continue
                
            new_topic = NewTopic(
                name=topic,
                num_partitions=3,
                replication_factor=1
            )
            admin_client.create_topics([new_topic])
            print(f"Created topic: {topic}")
        
        admin_client.close()
        return True
        
    except Exception as e:
        print(f"Error creating topics: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='User Feature Job (Python)')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['consumer', 'producer', 'setup'],
                       help='运行模式: consumer(消费处理), producer(生产测试数据), setup(创建topics)')
    parser.add_argument('--kafka', type=str, default='localhost:9092',
                       help='Kafka 服务器地址')
    parser.add_argument('--users', type=int, default=100,
                       help='测试用户数（producer模式）')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='事件发送间隔秒数（producer模式）')
    parser.add_argument('--duration', type=int, default=None,
                       help='运行持续时间秒数（producer模式）')
    
    args = parser.parse_args()
    
    kafka_servers = args.kafka.split(',')
    
    if args.mode == 'setup':
        # 创建 Kafka topics
        setup_kafka_topics(kafka_servers)
        
    elif args.mode == 'consumer':
        # 启动消费者（特征处理器）
        processor = UserFeatureProcessor(kafka_servers)
        processor.start_consumer()
        
    elif args.mode == 'producer':
        # 启动生产者（测试数据）
        producer = ClickEventProducer(kafka_servers)
        producer.start(
            num_users=args.users,
            interval=args.interval,
            duration=args.duration
        )


if __name__ == '__main__':
    main()
