// config/generative_config.go
//
// 生成式召回配置定义.
// 用于配置 TensorRT-LLM 推理服务和召回参数.

package config

import (
	"fmt"
	"time"
)

// KafkaConfig Kafka 配置.
type KafkaConfig struct {
	Brokers []string `json:"brokers" yaml:"brokers"` // Kafka 地址列表
	Topic   string   `json:"topic" yaml:"topic"`     // 订阅的 topic
	GroupID string   `json:"group_id" yaml:"group_id"` // 消费者组 ID
}

// GenerativeRecallConfig 生成式召回配置.
type GenerativeRecallConfig struct {
	// 服务配置
	ServerURL    string        `json:"server_url" yaml:"server_url"`       // TensorRT-LLM 服务地址
	Timeout      time.Duration `json:"timeout" yaml:"timeout"`             // 请求超时
	MaxRetries   int           `json:"max_retries" yaml:"max_retries"`     // 最大重试次数
	MaxBatchSize int           `json:"max_batch_size" yaml:"max_batch_size"` // 最大批次大小

	// 推理参数
	TopK        int     `json:"topk" yaml:"topk"`               // 推荐数量
	Temperature float64 `json:"temperature" yaml:"temperature"` // 采样温度
	BeamWidth   int     `json:"beam_width" yaml:"beam_width"`   // Beam search 宽度

	// 特征配置
	HistoryFrom        string `json:"history_from" yaml:"history_from"`               // 历史来源: "user_feature" 或 "context"
	HistoryFeatureName string `json:"history_feature_name" yaml:"history_feature_name"` // 历史特征字段名
	HistoryDelimiter   string `json:"history_delimiter" yaml:"history_delimiter"`     // 历史序列分隔符
	HistoryMaxLength   int    `json:"history_max_length" yaml:"history_max_length"`   // 最大历史长度

	// 缓存配置
	CacheEnable  bool          `json:"cache_enable" yaml:"cache_enable"`   // 是否启用缓存
	CacheType    string        `json:"cache_type" yaml:"cache_type"`       // 缓存类型: "local" 或 "redis"
	CacheTime    int           `json:"cache_time" yaml:"cache_time"`       // 缓存时间（秒）
	CachePrefix  string        `json:"cache_prefix" yaml:"cache_prefix"`   // 缓存键前缀

	// 新增：Kafka 实时特征配置
	FeatureSource string       `json:"feature_source" yaml:"feature_source"`       // 特征源: "file" 或 "kafka"
	KafkaConfig   *KafkaConfig `json:"kafka_config,omitempty" yaml:"kafka_config,omitempty"` // Kafka 配置
}

// DefaultGenerativeRecallConfig 返回默认配置.
func DefaultGenerativeRecallConfig() *GenerativeRecallConfig {
	return &GenerativeRecallConfig{
		ServerURL:          "http://localhost:8000",
		Timeout:            500 * time.Millisecond,
		MaxRetries:         3,
		MaxBatchSize:       32,
		TopK:               50,
		Temperature:        1.0,
		BeamWidth:          1,
		HistoryFrom:        "user_feature",
		HistoryFeatureName: "click_history",
		HistoryDelimiter:   ",",
		HistoryMaxLength:   20,
		CacheEnable:        true,
		CacheType:          "local",
		CacheTime:          300,
		CachePrefix:        "gen_recall_",
		FeatureSource:      "file", // 默认使用文件
	}
}

// Validate 验证配置.
func (c *GenerativeRecallConfig) Validate() error {
	if c.ServerURL == "" {
		return fmt.Errorf("server_url is required")
	}

	if c.TopK <= 0 {
		c.TopK = 50
	}

	if c.Temperature <= 0 {
		c.Temperature = 1.0
	}

	if c.HistoryFeatureName == "" {
		return fmt.Errorf("history_feature_name is required")
	}

	if c.HistoryMaxLength <= 0 {
		c.HistoryMaxLength = 20
	}

	if c.Timeout <= 0 {
		c.Timeout = 500 * time.Millisecond
	}

	return nil
}

// MergeWithDefault 与默认配置合并.
func (c *GenerativeRecallConfig) MergeWithDefault() {
	defaultCfg := DefaultGenerativeRecallConfig()

	if c.ServerURL == "" {
		c.ServerURL = defaultCfg.ServerURL
	}
	if c.Timeout == 0 {
		c.Timeout = defaultCfg.Timeout
	}
	if c.MaxRetries == 0 {
		c.MaxRetries = defaultCfg.MaxRetries
	}
	if c.TopK == 0 {
		c.TopK = defaultCfg.TopK
	}
	if c.Temperature == 0 {
		c.Temperature = defaultCfg.Temperature
	}
	if c.HistoryDelimiter == "" {
		c.HistoryDelimiter = defaultCfg.HistoryDelimiter
	}
	if c.HistoryMaxLength == 0 {
		c.HistoryMaxLength = defaultCfg.HistoryMaxLength
	}
	if c.CacheTime == 0 {
		c.CacheTime = defaultCfg.CacheTime
	}
	if c.CachePrefix == "" {
		c.CachePrefix = defaultCfg.CachePrefix
	}
}
