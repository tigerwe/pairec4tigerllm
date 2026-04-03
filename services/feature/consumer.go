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
	"strconv"
	"sync"
	"time"

	"github.com/segmentio/kafka-go"
)

// KafkaUserFeature 适配 recall-sink topic 的数据格式
type KafkaUserFeature struct {
	UserID        int    `json:"user_id"`        // 数字类型
	Gender        int    `json:"gender"`         // 性别
	Age           int    `json:"age"`            // 年龄
	ClickHistory  string `json:"click_history"`  // JSON 字符串，如 "[\"3035268\"]"
}

// Consumer Kafka 特征消费者.
type Consumer struct {
	config        *KafkaConfig
	reader        *kafka.Reader
	cache         *sync.Map // map[string]*UserFeatures
	ctx           context.Context
	cancel        context.CancelFunc
	maxMessages   int       // 最大消费消息数（0表示无限制）
	msgCount      int       // 已消费消息计数
}

// NewConsumer 创建 Kafka 消费者.
//
// Args:
//     config: Kafka 配置，包含 brokers、topic、group_id
//     maxMessages: 最大消费消息数（0表示无限制）
//
// Returns:
//     消费者实例
func NewConsumer(config *KafkaConfig, maxMessages int) *Consumer {
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
		config:      config,
		reader:      reader,
		cache:       &sync.Map{},
		ctx:         ctx,
		cancel:      cancel,
		maxMessages: maxMessages,
		msgCount:    0,
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

		// 检查是否达到最大消费数量
		if c.maxMessages > 0 && c.msgCount >= c.maxMessages {
			log.Printf("[FeatureConsumer] Reached max messages limit (%d), stopping consumption", c.maxMessages)
			return
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

		// 解析 recall-sink 格式的消息
		var kafkaFeature KafkaUserFeature
		if err := json.Unmarshal(msg.Value, &kafkaFeature); err != nil {
			log.Printf("[FeatureConsumer] Unmarshal error: %v, raw=%s", err, string(msg.Value))
			continue
		}

		// 转换为内部 UserFeatures 格式
		features := c.convertToUserFeatures(&kafkaFeature)
		if features == nil {
			log.Printf("[FeatureConsumer] Failed to convert feature for user=%d", kafkaFeature.UserID)
			continue
		}

		// 更新本地缓存（全量替换）
		c.cache.Store(features.UserID, features)
		c.msgCount++
		
		// 调试日志：显示详细的特征信息
		log.Printf("[FeatureConsumer] [KAFKA→CACHE] User=%s, History=%v, HistoryLen=%d, Count=%d/%d",
			features.UserID, 
			features.ClickHistory,
			len(features.ClickHistory),
			c.msgCount, c.maxMessages)
	}
}

// convertToUserFeatures 将 Kafka 格式转换为内部 UserFeatures 格式
func (c *Consumer) convertToUserFeatures(kf *KafkaUserFeature) *UserFeatures {
	// 1. 转换 user_id: int -> string
	userID := strconv.Itoa(kf.UserID)

	// 2. 解析 click_history: JSON 字符串 -> []int
	// click_history 格式: "[]" 或 "[\"3035268\"]" 或 "[\"123\",\"456\"]"
	var historyStrs []string
	if err := json.Unmarshal([]byte(kf.ClickHistory), &historyStrs); err != nil {
		log.Printf("[FeatureConsumer] Failed to parse click_history string: %v, raw=%s", err, kf.ClickHistory)
		// 如果解析失败，返回空历史而不是 nil
		historyStrs = []string{}
	}

	// 3. 字符串数组转 int 数组
	clickHistory := make([]int, 0, len(historyStrs))
	for _, s := range historyStrs {
		if s == "" {
			continue
		}
		id, err := strconv.Atoi(s)
		if err != nil {
			log.Printf("[FeatureConsumer] Failed to convert item_id to int: %s", s)
			continue
		}
		clickHistory = append(clickHistory, id)
	}

	return &UserFeatures{
		UserID:        userID,
		ClickHistory:  clickHistory,
		UpdateTime:    time.Now().Format(time.RFC3339), // 使用当前时间
		RealtimeCTR:   0.0,
		ShortTermTags: []string{},
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
