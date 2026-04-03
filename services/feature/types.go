// feature/types.go
//
// 特征模块共享类型定义.
// 包含用户特征结构和 Kafka 配置.

package feature

// UserFeatures 用户实时特征.
// Flink 计算后通过 Kafka 推送到本服务.
type UserFeatures struct {
	UserID        string   `json:"user_id"`
	ClickHistory  []int    `json:"click_history"`  // 实时点击序列
	UpdateTime    string   `json:"update_time"`    // 更新时间（字符串格式，避免时区解析问题）
	RealtimeCTR   float64  `json:"realtime_ctr,omitempty"`    // 实时 CTR
	ShortTermTags []string `json:"short_term_tags,omitempty"` // 短期兴趣标签
}

// KafkaConfig Kafka 消费者配置.
type KafkaConfig struct {
	Brokers []string `json:"brokers" yaml:"brokers"` // Kafka 地址列表
	Topic   string   `json:"topic" yaml:"topic"`     // 订阅的 topic
	GroupID string   `json:"group_id" yaml:"group_id"` // 消费者组 ID
}
