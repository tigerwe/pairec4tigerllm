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
	UserID              string  `json:"user_id"`
	History             [][]int `json:"history"` // 语义 ID 序列，每个元素是 [num_quantizers]
	Topk                int     `json:"topk"`
	Temperature         float64 `json:"temperature"`
	BeamWidth           int     `json:"beam_width"`
	PayloadPaddingBytes int     `json:"-"` // probe-only brpc payload padding; ignored by HTTP.
}

// TraceInfo 推理服务回传的性能追踪信息.
type TraceInfo struct {
	TotalMs                  float64 `json:"total_ms"`                       // 推理服务总耗时
	PrepareInputMs           float64 `json:"prepare_input_ms"`               // 输入准备耗时
	InferMs                  float64 `json:"infer_ms"`                       // 推理分支总耗时
	ModelForwardMs           float64 `json:"model_forward_ms"`               // 模型前向耗时
	GenerateMs               float64 `json:"generate_ms"`                    // 生成总耗时
	PromptMs                 float64 `json:"prompt_ms"`                      // TRT prompt 构造和 tokenize 耗时
	RunnerGenerateMs         float64 `json:"runner_generate_ms"`             // TRT runner.generate 累计耗时
	ParseComboMs             float64 `json:"parse_combo_ms"`                 // TRT token 组合解析耗时
	OutputPadMs              float64 `json:"output_pad_ms"`                  // TRT 输出补齐耗时
	BackendTotalMs           float64 `json:"backend_total_ms"`               // TRT 后端 generate 总耗时
	MapItemMs                float64 `json:"map_item_ms"`                    // 语义 ID 映射耗时
	KvLookupMs               float64 `json:"kv_lookup_ms"`                   // KV Cache 查询耗时
	KvWriteMs                float64 `json:"kv_write_ms"`                    // KV Cache 写入提交耗时
	KvSource                 string  `json:"kv_source"`                      // KV Cache 来源
	ResultCacheSource        string  `json:"result_cache_source"`            // 推荐结果缓存来源
	ResultCacheLookupMs      float64 `json:"result_cache_lookup_ms"`         // HBM 结果缓存查询耗时
	ResultCacheDSLookupMs    float64 `json:"result_cache_ds_lookup_ms"`      // DataSystem 结果缓存查询耗时
	ResultCacheWriteSubmitMs float64 `json:"result_cache_write_submit_ms"`   // 结果缓存异步写入提交耗时
	Backend                  string  `json:"backend"`                        // 后端类型
	WrapperTotalMs           float64 `json:"wrapper_total_ms"`               // 前置Wrapper处理总耗时
	WrapperBackendRPCMs      float64 `json:"wrapper_backend_rpc_ms"`         // Wrapper到推理后端的BRPC墙钟耗时
	WrapperOverheadMs        float64 `json:"wrapper_overhead_ms"`            // Wrapper本地处理和转发开销
	WrapperHealthAtStart     int64   `json:"wrapper_active_health_at_start"` // 业务路进入时在途Health数
	WrapperMaxActiveHealth   int64   `json:"wrapper_max_active_health"`      // 业务路执行期间Health峰值
	WrapperMaxActiveTotal    int64   `json:"wrapper_max_active_total"`       // 业务路加Health的峰值
	WrapperBackendBRPCMs     float64 `json:"wrapper_backend_brpc_ms"`        // 后端RPC墙钟减后端推理耗时
}

// Recommendation 推荐结果.
type Recommendation struct {
	ItemID     int     `json:"item_id"`
	SemanticID []int   `json:"semantic_id"`
	Score      float64 `json:"score"`
}

// RecommendResponse 推荐响应.
type RecommendResponse struct {
	Code            int              `json:"code"`
	UserID          string           `json:"user_id"`
	Recommendations []Recommendation `json:"recommendations"`
	InferenceTimeMs float64          `json:"inference_time_ms"`
	Error           string           `json:"error,omitempty"`
	Trace           *TraceInfo       `json:"trace,omitempty"` // 推理服务内部追踪信息
}

// TRTLLMClient TensorRT-LLM 客户端.
type TRTLLMClient struct {
	config     *config.GenerativeRecallConfig
	httpClient *http.Client
	brpcClient *BRPCRecommendClient
}

// NewTRTLLMClient 创建客户端.
func NewTRTLLMClient(cfg *config.GenerativeRecallConfig) (*TRTLLMClient, error) {
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	var brpcClient *BRPCRecommendClient
	if cfg.Protocol == "brpc" {
		client, err := NewBRPCRecommendClient(cfg.BRPCEndpoint, cfg.BRPCServiceName, cfg.Timeout, cfg.MaxRetries)
		if err != nil {
			return nil, fmt.Errorf("create brpc client failed: %w", err)
		}
		brpcClient = client
	}

	return &TRTLLMClient{
		config: cfg,
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
		},
		brpcClient: brpcClient,
	}, nil
}

// Recommend 获取推荐.
// traceID 用于端到端追踪，HTTP 通过 Header 透传，brpc 通过 request_id 字段透传.
func (c *TRTLLMClient) Recommend(req *RecommendRequest, traceID string) (*RecommendResponse, error) {
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
	if req.PayloadPaddingBytes == 0 && c.config.Protocol == "brpc" && c.config.BRPCPayloadBytes > 0 {
		req.PayloadPaddingBytes = c.config.BRPCPayloadBytes
	}

	if c.config.Protocol == "brpc" {
		resp, err := c.brpcClient.Recommend(context.Background(), req, traceID)
		if err == nil {
			return resp, nil
		}
		if !c.config.BRPCFallbackToHTTP {
			return nil, fmt.Errorf("brpc request failed: %w", err)
		}

		httpResp, httpErr := c.recommendHTTP(req, traceID)
		if httpErr != nil {
			return nil, fmt.Errorf("brpc request failed: %v; http fallback failed: %w", err, httpErr)
		}
		return httpResp, nil
	}

	return c.recommendHTTP(req, traceID)
}

func (c *TRTLLMClient) recommendHTTP(req *RecommendRequest, traceID string) (*RecommendResponse, error) {
	// 序列化请求
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request failed: %w", err)
	}

	// 构建 HTTP 请求 URL
	url := c.config.ServerURL + "/recommend"

	// 发送请求（带重试）- 每次重试都需要重新创建请求，因为 Body 只能读一次
	var httpResp *http.Response
	var lastErr error

	for attempt := 0; attempt < c.config.MaxRetries; attempt++ {
		// 重新创建请求（关键修复！）
		httpReq, err := http.NewRequestWithContext(
			context.Background(),
			"POST",
			url,
			bytes.NewBuffer(jsonData), // 每次用新的 Buffer
		)
		if err != nil {
			return nil, fmt.Errorf("create request failed: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")
		// 非侵入式追踪：透传 Trace ID
		if traceID != "" {
			httpReq.Header.Set("X-Request-ID", traceID)
		}

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
	if c.config.Protocol == "brpc" && c.brpcClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		resp, err := c.brpcClient.HealthCheck(ctx)
		cancel()
		if err == nil && resp.Code == 200 {
			return true
		}
		if !c.config.BRPCFallbackToHTTP {
			return false
		}
	}

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
