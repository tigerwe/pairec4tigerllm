// recall/trtllm_client.go
//
// TensorRT-LLM 推理服务客户端.
// 用于与生成式召回推理服务通信.

package recall

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"pairec4tigerllm/services/config"
)

// RecommendRequest 推荐请求.
type RecommendRequest struct {
	UserID      string    `json:"user_id"`
	History     [][]int   `json:"history"`      // 语义 ID 序列，每个元素是 [num_quantizers]
	Topk        int       `json:"topk"`
	Temperature float64   `json:"temperature"`
	BeamWidth   int       `json:"beam_width"`
}

// Recommendation 推荐结果.
type Recommendation struct {
	ItemID     int       `json:"item_id"`
	SemanticID []int     `json:"semantic_id"`
	Score      float64   `json:"score"`
}

// RecommendResponse 推荐响应.
type RecommendResponse struct {
	Code             int               `json:"code"`
	UserID           string            `json:"user_id"`
	Recommendations  []Recommendation  `json:"recommendations"`
	InferenceTimeMs  float64           `json:"inference_time_ms"`
	Error            string            `json:"error,omitempty"`
}

// TRTLLMClient TensorRT-LLM 客户端.
type TRTLLMClient struct {
	config     *config.GenerativeRecallConfig
	httpClient *http.Client
}

// NewTRTLLMClient 创建客户端.
func NewTRTLLMClient(cfg *config.GenerativeRecallConfig) (*TRTLLMClient, error) {
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	return &TRTLLMClient{
		config: cfg,
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
		},
	}, nil
}

// Recommend 获取推荐.
func (c *TRTLLMClient) Recommend(req *RecommendRequest) (*RecommendResponse, error) {
	// 设置默认值
	if req.Topk == 0 {
		req.Topk = c.config.TopK
	}
	if req.Temperature == 0 {
		req.Temperature = c.config.Temperature
	}
	if req.BeamWidth == 0 {
		req.BeamWidth = c.config.BeamWidth
	}

	// 序列化请求
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request failed: %w", err)
	}
	
	// 调试：打印请求体
	fmt.Printf("[DEBUG-CLIENT] Request JSON: %s\n", string(jsonData))
	fmt.Printf("[DEBUG-CLIENT] Request length: %d\n", len(jsonData))

	// 构建 HTTP 请求
	url := c.config.ServerURL + "/recommend"
	httpReq, err := http.NewRequestWithContext(
		context.Background(),
		"POST",
		url,
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, fmt.Errorf("create request failed: %w", err)
	}

	// 发送请求（带重试）- 每次重试都需要重新创建请求，因为 Body 只能读一次
	var httpResp *http.Response
	var lastErr error
	
	for attempt := 0; attempt < c.config.MaxRetries; attempt++ {
		// 重新创建请求（关键修复！）
		httpReq, err := http.NewRequestWithContext(
			context.Background(),
			"POST",
			url,
			bytes.NewBuffer(jsonData),  // 每次用新的 Buffer
		)
		if err != nil {
			return nil, fmt.Errorf("create request failed: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")
		
		httpResp, lastErr = c.httpClient.Do(httpReq)
		if lastErr == nil {
			break
		}
		
		if attempt < c.config.MaxRetries-1 {
			time.Sleep(time.Duration(attempt+1) * 100 * time.Millisecond)
		}
	}

	if lastErr != nil {
		return nil, fmt.Errorf("request failed after %d retries: %w", c.config.MaxRetries, lastErr)
	}

	defer httpResp.Body.Close()

	// 读取响应
	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response failed: %w", err)
	}

	// 解析响应
	var resp RecommendResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}

	// 检查错误
	if resp.Code != 200 {
		return nil, fmt.Errorf("service error: %s", resp.Error)
	}

	return &resp, nil
}

// HealthCheck 健康检查.
func (c *TRTLLMClient) HealthCheck() bool {
	url := c.config.ServerURL + "/health"
	
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return false
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	return resp.StatusCode == 200
}

// Close 关闭客户端.
func (c *TRTLLMClient) Close() {
	c.httpClient.CloseIdleConnections()
}
