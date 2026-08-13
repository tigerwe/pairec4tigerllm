// config/generative_config.go
//
// 生成式召回配置定义.
// 用于配置 TensorRT-LLM 推理服务和召回参数.

package config

import (
	"fmt"
	"strings"
	"time"
)

// KafkaConfig Kafka 配置.
type KafkaConfig struct {
	Brokers []string `json:"brokers" yaml:"brokers"`   // Kafka 地址列表
	Topic   string   `json:"topic" yaml:"topic"`       // 订阅的 topic
	GroupID string   `json:"group_id" yaml:"group_id"` // 消费者组 ID
}

// GenerativeRecallConfig 生成式召回配置.
type GenerativeRecallConfig struct {
	// 服务配置
	ServerURL                string        `json:"server_url" yaml:"server_url"`                       // TensorRT-LLM HTTP 服务地址
	Protocol                 string        `json:"protocol" yaml:"protocol"`                           // 推理服务协议: "http" 或 "brpc"
	BRPCEndpoint             string        `json:"brpc_endpoint" yaml:"brpc_endpoint"`                 // brpc/TCP 服务地址, 例如 inference-brpc-trtllm:18100
	BRPCServiceName          string        `json:"brpc_service_name" yaml:"brpc_service_name"`         // brpc service full name
	BRPCFallbackToHTTP       bool          `json:"brpc_fallback_to_http" yaml:"brpc_fallback_to_http"` // brpc 失败时是否回退 HTTP
	BRPCPayloadBytes         int           `json:"brpc_payload_bytes" yaml:"brpc_payload_bytes"`       // brpc 压测用额外 payload 字节数, 默认 0
	BRPCBurstEnabled         bool          `json:"brpc_burst_enabled" yaml:"brpc_burst_enabled"`
	BRPCBurstConcurrency     int           `json:"brpc_burst_concurrency" yaml:"brpc_burst_concurrency"`
	BRPCBurstPoolSize        int           `json:"brpc_burst_pool_size" yaml:"brpc_burst_pool_size"`
	BRPCBurstActive          int           `json:"brpc_burst_active_connections" yaml:"brpc_burst_active_connections"`
	BRPCBurstCPUShards       []int         `json:"brpc_burst_cpu_shards" yaml:"brpc_burst_cpu_shards"`
	BRPCBurstPayloadBytes    int           `json:"brpc_burst_payload_bytes" yaml:"brpc_burst_payload_bytes"`
	BRPCBurstPreconnect      bool          `json:"brpc_burst_preconnect" yaml:"brpc_burst_preconnect"`
	BRPCBurstPressureTimeout time.Duration `json:"brpc_burst_pressure_timeout" yaml:"brpc_burst_pressure_timeout"`
	Timeout                  time.Duration `json:"timeout" yaml:"timeout"`               // 请求超时
	MaxRetries               int           `json:"max_retries" yaml:"max_retries"`       // 最大重试次数
	MaxBatchSize             int           `json:"max_batch_size" yaml:"max_batch_size"` // 最大批次大小

	// 推理参数
	TopK        int     `json:"topk" yaml:"topk"`               // 推荐数量
	Temperature float64 `json:"temperature" yaml:"temperature"` // 采样温度
	BeamWidth   int     `json:"beam_width" yaml:"beam_width"`   // Beam search 宽度

	// 特征配置
	HistoryFrom        string `json:"history_from" yaml:"history_from"`                 // 历史来源: "user_feature" 或 "context"
	HistoryFeatureName string `json:"history_feature_name" yaml:"history_feature_name"` // 历史特征字段名
	HistoryDelimiter   string `json:"history_delimiter" yaml:"history_delimiter"`       // 历史序列分隔符
	HistoryMaxLength   int    `json:"history_max_length" yaml:"history_max_length"`     // 最大历史长度

	// 缓存配置
	CacheEnable bool   `json:"cache_enable" yaml:"cache_enable"` // 是否启用缓存
	CacheType   string `json:"cache_type" yaml:"cache_type"`     // 缓存类型: "local" 或 "redis"
	CacheTime   int    `json:"cache_time" yaml:"cache_time"`     // 缓存时间（秒）
	CachePrefix string `json:"cache_prefix" yaml:"cache_prefix"` // 缓存键前缀

	// 新增：Kafka 实时特征配置
	FeatureSource string       `json:"feature_source" yaml:"feature_source"`                 // 特征源: "file" 或 "kafka"
	KafkaConfig   *KafkaConfig `json:"kafka_config,omitempty" yaml:"kafka_config,omitempty"` // Kafka 配置
}

// DefaultGenerativeRecallConfig 返回默认配置.
func DefaultGenerativeRecallConfig() *GenerativeRecallConfig {
	return &GenerativeRecallConfig{
		ServerURL:          "http://localhost:8000",
		Protocol:           "http",
		BRPCServiceName:    "pairec.inference.RecommendService",
		BRPCFallbackToHTTP: true,
		Timeout:            3 * time.Second,
		MaxRetries:         1,
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
	c.Protocol = strings.ToLower(strings.TrimSpace(c.Protocol))
	if c.Protocol == "" {
		c.Protocol = "http"
	}
	if c.BRPCServiceName == "" {
		c.BRPCServiceName = "pairec.inference.RecommendService"
	}
	if c.BRPCPayloadBytes < 0 {
		c.BRPCPayloadBytes = 0
	}
	if c.Timeout <= 0 {
		c.Timeout = 3 * time.Second
	}
	if c.BRPCBurstEnabled {
		if c.Protocol != "brpc" {
			return fmt.Errorf("brpc burst requires protocol=brpc")
		}
		if c.BRPCBurstConcurrency <= 0 || c.BRPCBurstConcurrency > 1000 {
			return fmt.Errorf("brpc_burst_concurrency must be in [1,1000]")
		}
		if c.BRPCBurstPoolSize == 0 {
			c.BRPCBurstPoolSize = c.BRPCBurstConcurrency
		}
		if c.BRPCBurstActive == 0 {
			c.BRPCBurstActive = c.BRPCBurstConcurrency
		}
		if c.BRPCBurstPoolSize < 1 || c.BRPCBurstPoolSize > 10000 {
			return fmt.Errorf("brpc_burst_pool_size must be in [1,10000]")
		}
		if c.BRPCBurstActive < 1 || c.BRPCBurstActive > 1000 || c.BRPCBurstActive > c.BRPCBurstPoolSize {
			return fmt.Errorf("brpc_burst_active_connections must be in [1,min(1000,pool_size)]")
		}
		seenShards := make(map[int]struct{}, len(c.BRPCBurstCPUShards))
		for _, shard := range c.BRPCBurstCPUShards {
			if shard < 0 {
				return fmt.Errorf("brpc_burst_cpu_shards must be non-negative")
			}
			if _, exists := seenShards[shard]; exists {
				return fmt.Errorf("brpc_burst_cpu_shards must be unique")
			}
			seenShards[shard] = struct{}{}
		}
		if c.BRPCBurstPayloadBytes < 0 || c.BRPCBurstPayloadBytes > 1<<20 {
			return fmt.Errorf("brpc_burst_payload_bytes must be in [0,1048576]")
		}
		if !c.BRPCBurstPreconnect {
			return fmt.Errorf("brpc burst requires brpc_burst_preconnect=true")
		}
		if c.BRPCBurstPressureTimeout <= 0 {
			c.BRPCBurstPressureTimeout = c.Timeout
		}
	}

	switch c.Protocol {
	case "http":
	case "brpc":
		if strings.TrimSpace(c.BRPCEndpoint) == "" {
			return fmt.Errorf("brpc_endpoint is required when protocol=brpc")
		}
	default:
		return fmt.Errorf("unsupported protocol: %s", c.Protocol)
	}

	if c.ServerURL == "" && (c.Protocol == "http" || c.BRPCFallbackToHTTP) {
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

	return nil
}

// MergeWithDefault 与默认配置合并.
func (c *GenerativeRecallConfig) MergeWithDefault() {
	defaultCfg := DefaultGenerativeRecallConfig()

	if c.ServerURL == "" {
		c.ServerURL = defaultCfg.ServerURL
	}
	if c.Protocol == "" {
		c.Protocol = defaultCfg.Protocol
	}
	if c.BRPCServiceName == "" {
		c.BRPCServiceName = defaultCfg.BRPCServiceName
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
