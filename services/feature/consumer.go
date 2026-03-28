// feature/consumer.go
//
// Kafka 特征消费者.
// 持续消费 Flink 生成的用户特征更新，维护本地内存缓存.

package feature

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/segmentio/kafka-go"
)

// Consumer Kafka 特征消费者.
type Consumer struct {
	config *KafkaConfig
	reader *kafka.Reader
	cache  *sync.Map // map[string]*UserFeatures
	ctx    context.Context
	cancel context.CancelFunc
}

// NewConsumer 创建 Kafka 消费者.
//
// Args:
//     config: Kafka 配置，包含 brokers、topic、group_id
//
// Returns:
//     消费者实例
func NewConsumer(config *KafkaConfig) *Consumer {
	ctx, cancel := context.WithCancel(context.Background())

	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers:     config.Brokers,
		Topic:       config.Topic,
		GroupID:     config.GroupID,
		MinBytes:    10e3, // 10KB
		MaxBytes:    10e6, // 10MB
		MaxWait:     time.Second,
		ReadBackoffMin: 100 * time.Millisecond,
		ReadBackoffMax: time.Second,
	})

	return &Consumer{
		config: config,
		reader: reader,
		cache:  &sync.Map{},
		ctx:    ctx,
		cancel: cancel,
	}
}

// Start 启动消费循环（非阻塞）.
// 在后台 goroutine 中持续消费 Kafka 消息.
func (c *Consumer) Start() error {
	go c.consume()
	log.Printf("[FeatureConsumer] Started, topic=%s, group=%s", c.config.Topic, c.config.GroupID)
	return nil
}

// Stop 停止消费者.
// 调用此方法会停止消费循环并关闭 Kafka 连接.
func (c *Consumer) Stop() error {
	c.cancel()
	return c.reader.Close()
}

// Get 获取用户特征.
//
// Args:
//     userID: 用户 ID
//
// Returns:
//     用户特征，如果不存在返回 nil
func (c *Consumer) Get(userID string) *UserFeatures {
	if v, ok := c.cache.Load(userID); ok {
		return v.(*UserFeatures)
	}
	return nil
}

// GetCacheSize 获取缓存中的用户数量（用于监控）.
func (c *Consumer) GetCacheSize() int {
	count := 0
	c.cache.Range(func(_, _ interface{}) bool {
		count++
		return true
	})
	return count
}

// consume 消费循环.
// 持续读取 Kafka 消息并更新本地缓存.
func (c *Consumer) consume() {
	for {
		select {
		case <-c.ctx.Done():
			return
		default:
		}

		msg, err := c.reader.ReadMessage(c.ctx)
		if err != nil {
			if c.ctx.Err() != nil {
				return
			}
			log.Printf("[FeatureConsumer] Read error: %v", err)
			time.Sleep(time.Second)
			continue
		}

		var features UserFeatures
		if err := json.Unmarshal(msg.Value, &features); err != nil {
			log.Printf("[FeatureConsumer] Unmarshal error: %v", err)
			continue
		}

		// 更新本地缓存（全量替换）
		c.cache.Store(features.UserID, &features)
		
		// 调试日志（可选）
		if len(features.ClickHistory) > 0 {
			log.Printf("[FeatureConsumer] Updated user=%s, history_len=%d",
				features.UserID, len(features.ClickHistory))
		}
	}
}

// IsHealthy 检查消费者健康状态.
func (c *Consumer) IsHealthy() bool {
	// 简单检查：如果能获取到缓存大小，说明运行正常
	_ = c.GetCacheSize()
	return c.ctx.Err() == nil
}

// String 返回消费者状态信息.
func (c *Consumer) String() string {
	return fmt.Sprintf("Consumer{topic=%s, group=%s, cache_size=%d}",
		c.config.Topic, c.config.GroupID, c.GetCacheSize())
}
