// feature/provider.go
//
// 特征提供器.
// 封装特征获取逻辑，支持实时特征（Kafka）和离线特征（JSON文件）的降级策略.

package feature

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
)

// Provider 特征提供器.
// 优先从 Kafka 实时缓存获取，未命中时回退到离线 JSON 文件.
type Provider struct {
	consumer      *Consumer
	fallbackPath  string
	fallbackCache map[string]map[string]interface{} // 离线特征缓存
	once          sync.Once
}

// NewProvider 创建特征提供器.
//
// Args:
//     consumer: Kafka 消费者，可为 nil（纯离线模式）
//     fallbackPath: 离线 JSON 文件路径，作为降级使用
//
// Returns:
//     特征提供器实例
func NewProvider(consumer *Consumer, fallbackPath string) *Provider {
	return &Provider{
		consumer:      consumer,
		fallbackPath:  fallbackPath,
		fallbackCache: make(map[string]map[string]interface{}),
	}
}

// GetUserHistory 获取用户历史点击序列.
// 优先查实时缓存，未命中时回退到离线 JSON.
//
// Args:
//     userID: 用户 ID
//
// Returns:
//     物品 ID 列表，如果不存在返回 nil
func (p *Provider) GetUserHistory(userID string) ([]int, error) {
	// 1. 优先查实时缓存
	if p.consumer != nil {
		if features := p.consumer.Get(userID); features != nil {
			return features.ClickHistory, nil
		}
	}

	// 2. 降级到离线 JSON
	return p.getFromFallback(userID)
}

// GetUserFeatures 获取完整用户特征（包含实时特征）.
//
// Args:
//     userID: 用户 ID
//
// Returns:
//     用户特征结构，如果不存在返回 nil
func (p *Provider) GetUserFeatures(userID string) *UserFeatures {
	// 优先查实时缓存
	if p.consumer != nil {
		if features := p.consumer.Get(userID); features != nil {
			return features
		}
	}
	return nil
}

// IsRealtimeAvailable 检查实时特征是否可用.
func (p *Provider) IsRealtimeAvailable() bool {
	return p.consumer != nil && p.consumer.IsHealthy()
}

// GetStats 获取统计信息.
func (p *Provider) GetStats() map[string]interface{} {
	stats := map[string]interface{}{
		"fallback_path": p.fallbackPath,
		"realtime_available": p.IsRealtimeAvailable(),
	}

	if p.consumer != nil {
		stats["cache_size"] = p.consumer.GetCacheSize()
	}

	return stats
}

// getFromFallback 从离线 JSON 获取用户历史.
func (p *Provider) getFromFallback(userID string) ([]int, error) {
	p.once.Do(func() {
		p.loadFallback()
	})

	userData, ok := p.fallbackCache[userID]
	if !ok {
		return nil, fmt.Errorf("user %s not found in fallback", userID)
	}

	historyStr := ""
	if val, ok := userData["click_history"]; ok {
		historyStr = fmt.Sprintf("%v", val)
	}

	return parseHistoryString(historyStr)
}

// loadFallback 加载离线 JSON 文件.
func (p *Provider) loadFallback() {
	data, err := os.ReadFile(p.fallbackPath)
	if err != nil {
		fmt.Printf("[FeatureProvider] Failed to load fallback: %v\n", err)
		return
	}

	if err := json.Unmarshal(data, &p.fallbackCache); err != nil {
		fmt.Printf("[FeatureProvider] Failed to parse fallback: %v\n", err)
		return
	}

	fmt.Printf("[FeatureProvider] Loaded fallback: %d users from %s\n",
		len(p.fallbackCache), p.fallbackPath)
}

// parseHistoryString 解析逗号分隔的历史字符串.
func parseHistoryString(historyStr string) ([]int, error) {
	if historyStr == "" {
		return nil, nil
	}

	parts := strings.Split(historyStr, ",")
	history := make([]int, 0, len(parts))

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		id, err := strconv.Atoi(part)
		if err != nil {
			continue
		}
		history = append(history, id)
	}

	return history, nil
}
