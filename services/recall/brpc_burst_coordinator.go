package recall

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

type BRPCBurstConfig struct {
	Concurrency     int
	PayloadBytes    int
	PressureTimeout time.Duration
}

type brpcBurstSession interface {
	Connect(context.Context) error
	Recommend(context.Context, *RecommendRequest, string) (*RecommendResponse, error)
	HealthCheckWithPayload(context.Context, int) (*brpcHealthResponse, error)
	Close() error
}

type BRPCBurstCoordinator struct {
	sessions        []brpcBurstSession
	concurrency     int
	payloadBytes    int
	pressureTimeout time.Duration
	stride          int
	sequence        uint64
	slot            chan struct{}
	logEvent        func(any)
}

type brpcBurstLaneResult struct {
	index         int
	business      bool
	startOffsetUs int64
	latencyUs     int64
	response      *RecommendResponse
	err           error
}

type brpcBurstStartEvent struct {
	Event             string  `json:"event"`
	RequestID         string  `json:"request_id"`
	Concurrency       int     `json:"concurrency"`
	BusinessLane      int     `json:"business_lane"`
	PayloadBytes      int     `json:"pressure_payload_bytes"`
	ConnectedSessions int     `json:"connected_sessions"`
	PreflightMs       float64 `json:"preflight_ms"`
	Stride            int     `json:"business_lane_stride"`
}

type brpcBurstBusinessEvent struct {
	Event                string  `json:"event"`
	RequestID            string  `json:"request_id"`
	Concurrency          int     `json:"concurrency"`
	BusinessLane         int     `json:"business_lane"`
	Success              bool    `json:"business_success"`
	Error                string  `json:"business_error,omitempty"`
	ClientWallMs         float64 `json:"business_client_wall_ms"`
	InferenceMs          float64 `json:"business_inference_ms"`
	RunnerGenerateMs     float64 `json:"business_runner_generate_ms"`
	FrontBRPCMs          float64 `json:"business_front_brpc_ms"`
	WrapperTotalMs       float64 `json:"wrapper_total_ms"`
	WrapperBackendRPCMs  float64 `json:"wrapper_backend_rpc_ms"`
	WrapperBackendBRPCMs float64 `json:"wrapper_backend_brpc_ms"`
	TraceValid           bool    `json:"trace_valid"`
}

type brpcBurstCompleteEvent struct {
	Event                string  `json:"event"`
	RequestID            string  `json:"request_id"`
	Concurrency          int     `json:"concurrency"`
	BusinessLane         int     `json:"business_lane"`
	ArmedWorkers         int     `json:"armed_workers"`
	MaxActiveWorkers     int64   `json:"max_active_workers"`
	StartSkewUs          int64   `json:"start_skew_us"`
	BurstTotalMs         float64 `json:"burst_total_ms"`
	PressureRequests     int     `json:"pressure_requests"`
	PressureSuccess      int     `json:"pressure_success"`
	PressureErrors       int     `json:"pressure_errors"`
	PressureLatencyAvgMs float64 `json:"pressure_latency_avg_ms"`
	PressureLatencyP95Ms float64 `json:"pressure_latency_p95_ms"`
	BusinessSuccess      bool    `json:"business_success"`
	TraceValid           bool    `json:"trace_valid"`
	BurstValid           bool    `json:"burst_valid"`
}

func NewBRPCBurstCoordinator(client *BRPCRecommendClient, cfg BRPCBurstConfig) (*BRPCBurstCoordinator, error) {
	if client == nil {
		return nil, fmt.Errorf("brpc client is nil")
	}
	sessions := make([]brpcBurstSession, cfg.Concurrency)
	for index := range sessions {
		sessions[index] = client.NewSession()
	}
	coordinator, err := newBRPCBurstCoordinator(sessions, cfg, writeBRPCBurstEvent)
	if err != nil {
		closeBRPCBurstSessions(sessions)
		return nil, err
	}
	return coordinator, nil
}

func newBRPCBurstCoordinator(sessions []brpcBurstSession, cfg BRPCBurstConfig, logger func(any)) (*BRPCBurstCoordinator, error) {
	if cfg.Concurrency < 1 || cfg.Concurrency > 1000 {
		return nil, fmt.Errorf("concurrency must be in [1,1000]")
	}
	if len(sessions) != cfg.Concurrency {
		return nil, fmt.Errorf("session count %d does not match concurrency %d", len(sessions), cfg.Concurrency)
	}
	if cfg.PayloadBytes < 0 || cfg.PayloadBytes > 1<<20 {
		return nil, fmt.Errorf("payload bytes must be in [0,1048576]")
	}
	if cfg.PressureTimeout <= 0 {
		return nil, fmt.Errorf("pressure timeout must be positive")
	}
	if logger == nil {
		logger = writeBRPCBurstEvent
	}

	c := &BRPCBurstCoordinator{
		sessions:        sessions,
		concurrency:     cfg.Concurrency,
		payloadBytes:    cfg.PayloadBytes,
		pressureTimeout: cfg.PressureTimeout,
		stride:          coprimeStride(cfg.Concurrency),
		slot:            make(chan struct{}, 1),
		logEvent:        logger,
	}
	if _, err := c.connectAll(); err != nil {
		return nil, err
	}
	c.slot <- struct{}{}
	c.logEvent(map[string]any{
		"event":                "pairec_brpc_burst_ready",
		"concurrency":          c.concurrency,
		"connected_sessions":   c.concurrency,
		"payload_bytes":        c.payloadBytes,
		"business_lane_stride": c.stride,
	})
	return c, nil
}

func (c *BRPCBurstCoordinator) Recommend(req *RecommendRequest, requestID string) (*RecommendResponse, error) {
	<-c.slot
	preflightStarted := time.Now()
	connected, err := c.connectAll()
	preflightMs := elapsedMilliseconds(preflightStarted)
	if err != nil {
		c.slot <- struct{}{}
		return nil, fmt.Errorf("brpc burst preflight connected %d/%d sessions: %w", connected, c.concurrency, err)
	}

	sequence := atomic.AddUint64(&c.sequence, 1)
	businessLane := int((sequence*uint64(c.stride))%uint64(c.concurrency)) + 1
	results := make(chan brpcBurstLaneResult, c.concurrency)
	businessResult := make(chan brpcBurstLaneResult, 1)
	gate := make(chan struct{})
	var armed sync.WaitGroup
	var complete sync.WaitGroup
	var active int64
	var maxActive int64
	armed.Add(c.concurrency)
	complete.Add(c.concurrency)
	startOffsets := make([]int64, c.concurrency)
	var releasedAt time.Time

	for lane := 1; lane <= c.concurrency; lane++ {
		go func(lane int) {
			defer complete.Done()
			armed.Done()
			<-gate
			callStarted := time.Now()
			result := brpcBurstLaneResult{
				index:         lane,
				business:      lane == businessLane,
				startOffsetUs: callStarted.Sub(releasedAt).Microseconds(),
			}
			current := atomic.AddInt64(&active, 1)
			updateBurstMaximum(&maxActive, current)
			defer atomic.AddInt64(&active, -1)

			if result.business {
				ctx, cancel := context.WithTimeout(context.Background(), c.pressureTimeout)
				result.response, result.err = c.sessions[lane-1].Recommend(ctx, req, requestID)
				cancel()
			} else {
				ctx, cancel := context.WithTimeout(context.Background(), c.pressureTimeout)
				_, result.err = c.sessions[lane-1].HealthCheckWithPayload(ctx, c.payloadBytes)
				cancel()
			}
			result.latencyUs = time.Since(callStarted).Microseconds()
			startOffsets[lane-1] = result.startOffsetUs
			results <- result
			if result.business {
				businessResult <- result
			}
		}(lane)
	}

	armed.Wait()
	c.logEvent(brpcBurstStartEvent{
		Event:             "pairec_brpc_burst_start",
		RequestID:         requestID,
		Concurrency:       c.concurrency,
		BusinessLane:      businessLane,
		PayloadBytes:      c.payloadBytes,
		ConnectedSessions: connected,
		PreflightMs:       preflightMs,
		Stride:            c.stride,
	})
	releasedAt = time.Now()
	close(gate)

	business := <-businessResult
	businessEvent := makeBRPCBurstBusinessEvent(requestID, c.concurrency, businessLane, business)
	c.logEvent(businessEvent)

	go func() {
		complete.Wait()
		close(results)
		c.logEvent(makeBRPCBurstCompleteEvent(
			requestID, c.concurrency, businessLane, releasedAt,
			atomic.LoadInt64(&maxActive), startOffsets, businessEvent, results,
		))
		c.slot <- struct{}{}
	}()

	if business.err != nil {
		return nil, business.err
	}
	return business.response, nil
}

func (c *BRPCBurstCoordinator) connectAll() (int, error) {
	var wg sync.WaitGroup
	errs := make(chan error, len(c.sessions))
	var connected int64
	for _, session := range c.sessions {
		wg.Add(1)
		go func(session brpcBurstSession) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), c.pressureTimeout)
			err := session.Connect(ctx)
			cancel()
			if err != nil {
				errs <- err
				return
			}
			atomic.AddInt64(&connected, 1)
		}(session)
	}
	wg.Wait()
	close(errs)
	if len(errs) > 0 {
		return int(connected), <-errs
	}
	return int(connected), nil
}

func (c *BRPCBurstCoordinator) Close() {
	closeBRPCBurstSessions(c.sessions)
}

func closeBRPCBurstSessions(sessions []brpcBurstSession) {
	for _, session := range sessions {
		_ = session.Close()
	}
}

func makeBRPCBurstBusinessEvent(requestID string, concurrency, businessLane int, result brpcBurstLaneResult) brpcBurstBusinessEvent {
	event := brpcBurstBusinessEvent{
		Event:        "pairec_brpc_burst_business_complete",
		RequestID:    requestID,
		Concurrency:  concurrency,
		BusinessLane: businessLane,
		Success:      result.err == nil && result.response != nil,
		ClientWallMs: float64(result.latencyUs) / 1000,
	}
	if result.err != nil {
		event.Error = result.err.Error()
	}
	if result.response == nil {
		return event
	}
	event.InferenceMs = result.response.InferenceTimeMs
	if result.response.Trace == nil {
		return event
	}
	trace := result.response.Trace
	event.RunnerGenerateMs = trace.RunnerGenerateMs
	event.WrapperTotalMs = trace.WrapperTotalMs
	event.WrapperBackendRPCMs = trace.WrapperBackendRPCMs
	event.WrapperBackendBRPCMs = trace.WrapperBackendBRPCMs
	event.FrontBRPCMs = event.ClientWallMs - trace.WrapperTotalMs
	event.TraceValid = trace.WrapperTotalMs > 0 && trace.WrapperBackendRPCMs > 0 && trace.Backend == "trtllm_cpp"
	return event
}

func makeBRPCBurstCompleteEvent(
	requestID string,
	concurrency int,
	businessLane int,
	releasedAt time.Time,
	maxActive int64,
	startOffsets []int64,
	business brpcBurstBusinessEvent,
	results <-chan brpcBurstLaneResult,
) brpcBurstCompleteEvent {
	pressureLatencies := make([]int64, 0, concurrency-1)
	pressureSuccess := 0
	for result := range results {
		if result.business {
			continue
		}
		if result.err == nil {
			pressureSuccess++
			pressureLatencies = append(pressureLatencies, result.latencyUs)
		}
	}
	pressureRequests := concurrency - 1
	return brpcBurstCompleteEvent{
		Event:                "pairec_brpc_burst_complete",
		RequestID:            requestID,
		Concurrency:          concurrency,
		BusinessLane:         businessLane,
		ArmedWorkers:         concurrency,
		MaxActiveWorkers:     maxActive,
		StartSkewUs:          spreadBurstOffsets(startOffsets),
		BurstTotalMs:         elapsedMilliseconds(releasedAt),
		PressureRequests:     pressureRequests,
		PressureSuccess:      pressureSuccess,
		PressureErrors:       pressureRequests - pressureSuccess,
		PressureLatencyAvgMs: averageBurstLatency(pressureLatencies),
		PressureLatencyP95Ms: percentileBurstLatency(pressureLatencies, 0.95),
		BusinessSuccess:      business.Success,
		TraceValid:           business.TraceValid,
		BurstValid:           business.Success && business.TraceValid && pressureSuccess == pressureRequests,
	}
}

func writeBRPCBurstEvent(event any) {
	encoded, err := json.Marshal(event)
	if err != nil {
		fmt.Printf("{\"event\":\"pairec_brpc_burst_log_error\",\"error\":%q}\n", err.Error())
		return
	}
	fmt.Println(string(encoded))
}

func coprimeStride(concurrency int) int {
	if concurrency <= 1 {
		return 1
	}
	stride := 7919 % concurrency
	if stride == 0 {
		stride = 1
	}
	for greatestCommonDivisor(stride, concurrency) != 1 {
		stride++
		if stride >= concurrency {
			stride = 1
		}
	}
	return stride
}

func greatestCommonDivisor(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func elapsedMilliseconds(start time.Time) float64 {
	return float64(time.Since(start).Microseconds()) / 1000
}

func averageBurstLatency(values []int64) float64 {
	if len(values) == 0 {
		return 0
	}
	var total int64
	for _, value := range values {
		total += value
	}
	return float64(total) / float64(len(values)) / 1000
}

func percentileBurstLatency(values []int64, quantile float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]int64(nil), values...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	index := int(float64(len(sorted)-1) * quantile)
	return float64(sorted[index]) / 1000
}

func updateBurstMaximum(target *int64, value int64) {
	for {
		current := atomic.LoadInt64(target)
		if value <= current || atomic.CompareAndSwapInt64(target, current, value) {
			return
		}
	}
}

func spreadBurstOffsets(values []int64) int64 {
	if len(values) == 0 {
		return 0
	}
	minimum := values[0]
	maximum := values[0]
	for _, value := range values[1:] {
		if value < minimum {
			minimum = value
		}
		if value > maximum {
			maximum = value
		}
	}
	return maximum - minimum
}
