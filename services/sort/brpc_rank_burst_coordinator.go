package ranksort

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"pairec4tigerllm/services/pipelineclient"
	"pairec4tigerllm/services/pipelinepb"
)

type RankBurstConfig struct {
	Concurrency           int
	PoolSize              int
	PressureBytes         int
	BusinessBytes         int
	BusinessTimeout       time.Duration
	PressureTimeout       time.Duration
	EventPrefix           string
	TraceComponent        string
	DedicatedBusinessLane bool
}

type rankBurstSession interface {
	Connect(context.Context) error
	Rank(context.Context, *pipelinepb.RankRequest) (*pipelinepb.RankResponse, error)
	HealthWithPayload(context.Context, string, int) (*pipelinepb.HealthResponse, error)
	Close() error
}

type RankBurstCoordinator struct {
	sessions              []rankBurstSession
	workers               []chan rankBurstJob
	concurrency           int
	poolSize              int
	pressureBytes         int
	businessBytes         int
	businessTimeout       time.Duration
	pressureTimeout       time.Duration
	poolCursor            int
	sequence              uint64
	slot                  chan struct{}
	logEvent              func(any)
	eventPrefix           string
	traceComponent        string
	dedicatedBusinessLane bool
	pressureDispatch      sync.Mutex
	rounds                sync.WaitGroup
}

type rankBurstJob struct {
	request        *pipelinepb.RankRequest
	requestID      string
	lane           int
	business       bool
	gate           <-chan struct{}
	releasedAt     *time.Time
	results        chan<- rankBurstLaneResult
	businessResult chan<- rankBurstLaneResult
	complete       *sync.WaitGroup
	started        chan<- struct{}
	active         *int64
	maxActive      *int64
}

type rankBurstLaneResult struct {
	lane          int
	business      bool
	startOffsetUS int64
	latencyUS     int64
	startEpochNS  int64
	endEpochNS    int64
	response      *pipelinepb.RankResponse
	err           error
}

type rankBurstBusinessEvent struct {
	Event                string  `json:"event"`
	RequestID            string  `json:"request_id"`
	Concurrency          int     `json:"concurrency"`
	BusinessLane         int     `json:"business_lane"`
	BusinessPayloadBytes int     `json:"business_payload_bytes"`
	Success              bool    `json:"business_success"`
	Error                string  `json:"business_error,omitempty"`
	ClientWallMS         float64 `json:"business_client_wall_ms"`
	ServiceTotalMS       float64 `json:"service_total_ms"`
	FrontBRPCEstimateMS  float64 `json:"front_brpc_estimate_ms"`
	StartEpochNS         int64   `json:"rank_business_start_epoch_ns"`
	EndEpochNS           int64   `json:"rank_business_end_epoch_ns"`
	TraceValid           bool    `json:"trace_valid"`
}

type rankBurstCompleteEvent struct {
	Event                       string   `json:"event"`
	RequestID                   string   `json:"request_id"`
	Concurrency                 int      `json:"concurrency"`
	BusinessLane                int      `json:"business_lane"`
	ArmedWorkers                int      `json:"armed_workers"`
	ConnectedSessions           int      `json:"connected_sessions"`
	MaxActiveWorkers            int64    `json:"max_active_workers"`
	StartSkewUS                 int64    `json:"start_skew_us"`
	BurstTotalMS                float64  `json:"burst_total_ms"`
	PressureRequests            int      `json:"pressure_requests"`
	PressureSuccess             int      `json:"pressure_success"`
	PressureErrors              int      `json:"pressure_errors"`
	PressureErrorSamples        []string `json:"pressure_error_samples,omitempty"`
	PressureLatencyAverageMS    float64  `json:"pressure_latency_avg_ms"`
	PressureLatencyP95MS        float64  `json:"pressure_latency_p95_ms"`
	PressureFirstStartEpochNS   int64    `json:"rank_pressure_first_start_epoch_ns"`
	PressureLastEndEpochNS      int64    `json:"rank_pressure_last_end_epoch_ns"`
	PressureOverlapBusiness     int      `json:"pressure_overlap_business"`
	PressureTailAfterBusinessMS float64  `json:"pressure_tail_after_business_ms"`
	BusinessStartEpochNS        int64    `json:"rank_business_start_epoch_ns"`
	BusinessEndEpochNS          int64    `json:"rank_business_end_epoch_ns"`
	BusinessSuccess             bool     `json:"business_success"`
	TraceValid                  bool     `json:"trace_valid"`
	BurstValid                  bool     `json:"burst_valid"`
	PressurePayloadTransport    string   `json:"pressure_payload_transport"`
}

func NewRankBurstCoordinator(client *pipelineclient.RankClient, cfg RankBurstConfig) (*RankBurstCoordinator, error) {
	if client == nil {
		return nil, fmt.Errorf("rank brpc client is nil")
	}
	if cfg.PoolSize == 0 {
		cfg.PoolSize = cfg.Concurrency
	}
	sessions := make([]rankBurstSession, cfg.PoolSize)
	for index := range sessions {
		sessions[index] = client.NewSession()
	}
	coordinator, err := newRankBurstCoordinator(sessions, cfg, writeRankBurstEvent)
	if err != nil {
		for _, session := range sessions {
			_ = session.Close()
		}
		return nil, err
	}
	return coordinator, nil
}

func newRankBurstCoordinator(sessions []rankBurstSession, cfg RankBurstConfig, logger func(any)) (*RankBurstCoordinator, error) {
	if cfg.Concurrency < 1 || cfg.Concurrency > 1000 {
		return nil, fmt.Errorf("rank burst concurrency must be in [1,1000]")
	}
	if cfg.PoolSize == 0 {
		cfg.PoolSize = cfg.Concurrency
	}
	if cfg.PoolSize < cfg.Concurrency || cfg.PoolSize > 1000 {
		return nil, fmt.Errorf("rank burst pool size must be in [concurrency,1000]")
	}
	if cfg.DedicatedBusinessLane && cfg.PoolSize != cfg.Concurrency {
		return nil, fmt.Errorf("dedicated business lane requires pool size to equal concurrency")
	}
	if len(sessions) != cfg.PoolSize {
		return nil, fmt.Errorf("rank session count %d does not match pool size %d", len(sessions), cfg.PoolSize)
	}
	if cfg.PressureBytes < 0 || cfg.PressureBytes > 1<<20 || cfg.BusinessBytes < 0 || cfg.BusinessBytes > 1<<20 {
		return nil, fmt.Errorf("rank payload bytes must be in [0,1048576]")
	}
	if cfg.PressureTimeout <= 0 {
		return nil, fmt.Errorf("rank pressure timeout must be positive")
	}
	if cfg.BusinessTimeout <= 0 {
		return nil, fmt.Errorf("rank business timeout must be positive")
	}
	if logger == nil {
		logger = writeRankBurstEvent
	}
	if cfg.EventPrefix == "" {
		cfg.EventPrefix = "pairec_rank_brpc_burst"
	}
	if cfg.TraceComponent == "" {
		cfg.TraceComponent = "deepfm_rank_adapter"
	}
	c := &RankBurstCoordinator{
		sessions: sessions, concurrency: cfg.Concurrency, poolSize: cfg.PoolSize,
		pressureBytes: cfg.PressureBytes, businessBytes: cfg.BusinessBytes,
		businessTimeout: cfg.BusinessTimeout, pressureTimeout: cfg.PressureTimeout,
		slot: make(chan struct{}, 1), logEvent: logger,
		eventPrefix: cfg.EventPrefix, traceComponent: cfg.TraceComponent,
		dedicatedBusinessLane: cfg.DedicatedBusinessLane,
	}
	c.workers = make([]chan rankBurstJob, cfg.PoolSize)
	for index := range sessions {
		c.workers[index] = make(chan rankBurstJob)
		go c.runWorker(index, c.workers[index])
	}
	if err := c.connectAll(); err != nil {
		for _, worker := range c.workers {
			close(worker)
		}
		return nil, err
	}
	c.slot <- struct{}{}
	c.logEvent(map[string]any{
		"event": c.eventPrefix + "_ready", "concurrency": c.concurrency,
		"connected_sessions": c.poolSize, "pool_size": c.poolSize,
		"pressure_payload_bytes": c.pressureBytes, "business_payload_bytes": c.businessBytes,
	})
	return c, nil
}

func (c *RankBurstCoordinator) Rank(request *pipelinepb.RankRequest, requestID string) (*pipelinepb.RankResponse, error) {
	<-c.slot
	selected := c.selectSessions()
	sequence := atomic.AddUint64(&c.sequence, 1)
	businessLane := int((sequence-1)%uint64(c.concurrency)) + 1
	if c.dedicatedBusinessLane {
		businessLane = 1
	}
	results := make(chan rankBurstLaneResult, c.concurrency)
	businessResult := make(chan rankBurstLaneResult, 1)
	gate := make(chan struct{})
	businessGate := gate
	pressureGate := gate
	var businessStarted chan struct{}
	if c.dedicatedBusinessLane {
		businessGate = make(chan struct{})
		pressureGate = make(chan struct{})
		businessStarted = make(chan struct{}, 1)
	}
	var complete sync.WaitGroup
	var active, maxActive int64
	var releasedAt time.Time
	complete.Add(c.concurrency)
	c.rounds.Add(1)
	makeJob := func(lane, sessionIndex int, laneGate <-chan struct{}, started chan<- struct{}) rankBurstJob {
		return rankBurstJob{
			request: request, requestID: requestID, lane: lane,
			business: lane == businessLane, gate: laneGate, releasedAt: &releasedAt,
			results: results, businessResult: businessResult, complete: &complete,
			started: started, active: &active, maxActive: &maxActive,
		}
	}
	if c.dedicatedBusinessLane {
		c.workers[selected[0]] <- makeJob(1, selected[0], businessGate, businessStarted)
	} else {
		for lane, sessionIndex := range selected {
			c.workers[sessionIndex] <- makeJob(lane+1, sessionIndex, gate, nil)
		}
	}
	c.logEvent(map[string]any{
		"event": c.eventPrefix + "_start", "request_id": requestID,
		"concurrency": c.concurrency, "business_lane": businessLane,
		"armed_workers": c.concurrency, "connected_sessions": c.poolSize,
		"pressure_payload_bytes": c.pressureBytes, "business_payload_bytes": c.businessBytes,
	})
	releasedAt = time.Now()
	if c.dedicatedBusinessLane {
		close(businessGate)
		<-businessStarted
		close(pressureGate)
		go func() {
			c.pressureDispatch.Lock()
			defer c.pressureDispatch.Unlock()
			for lane := 1; lane < len(selected); lane++ {
				sessionIndex := selected[lane]
				c.workers[sessionIndex] <- makeJob(lane+1, sessionIndex, pressureGate, nil)
			}
		}()
	} else {
		close(gate)
	}
	business := <-businessResult
	businessEvent := makeRankBurstBusinessEvent(
		c.eventPrefix, c.traceComponent, requestID, c.concurrency,
		businessLane, c.businessBytes, business)
	c.logEvent(businessEvent)
	go func() {
		complete.Wait()
		close(results)
		c.logEvent(makeRankBurstCompleteEvent(
			c.eventPrefix, requestID, c.concurrency, businessLane, releasedAt,
			atomic.LoadInt64(&maxActive), businessEvent, results, c.poolSize,
			!c.dedicatedBusinessLane,
		))
		c.rounds.Done()
		if !c.dedicatedBusinessLane {
			c.slot <- struct{}{}
		}
	}()
	if c.dedicatedBusinessLane {
		c.slot <- struct{}{}
	}
	if business.err != nil {
		return nil, business.err
	}
	return business.response, nil
}

func (c *RankBurstCoordinator) runWorker(sessionIndex int, jobs <-chan rankBurstJob) {
	for job := range jobs {
		<-job.gate
		started := time.Now()
		result := rankBurstLaneResult{
			lane: job.lane, business: job.business,
			startOffsetUS: started.Sub(*job.releasedAt).Microseconds(),
			startEpochNS:  started.UnixNano(),
		}
		current := atomic.AddInt64(job.active, 1)
		updateRankBurstMaximum(job.maxActive, current)
		timeout := c.pressureTimeout
		if job.business {
			timeout = c.businessTimeout
		}
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		if job.business {
			if job.started != nil {
				job.started <- struct{}{}
			}
			result.response, result.err = c.sessions[sessionIndex].Rank(ctx, job.request)
		} else {
			_, result.err = c.sessions[sessionIndex].HealthWithPayload(ctx, job.requestID, c.pressureBytes)
		}
		cancel()
		result.endEpochNS = time.Now().UnixNano()
		result.latencyUS = (result.endEpochNS - result.startEpochNS) / 1000
		atomic.AddInt64(job.active, -1)
		job.results <- result
		if result.business {
			job.businessResult <- result
		}
		job.complete.Done()
	}
}

func (c *RankBurstCoordinator) selectSessions() []int {
	selected := make([]int, c.concurrency)
	for offset := range selected {
		selected[offset] = (c.poolCursor + offset) % c.poolSize
	}
	c.poolCursor = (c.poolCursor + c.concurrency) % c.poolSize
	return selected
}

func (c *RankBurstCoordinator) connectAll() error {
	var wg sync.WaitGroup
	errs := make(chan error, len(c.sessions))
	limit := make(chan struct{}, 256)
	for _, session := range c.sessions {
		wg.Add(1)
		go func(session rankBurstSession) {
			defer wg.Done()
			limit <- struct{}{}
			defer func() { <-limit }()
			ctx, cancel := context.WithTimeout(context.Background(), c.pressureTimeout)
			err := session.Connect(ctx)
			cancel()
			if err != nil {
				errs <- err
			}
		}(session)
	}
	wg.Wait()
	close(errs)
	if len(errs) > 0 {
		return <-errs
	}
	return nil
}

func (c *RankBurstCoordinator) Close() {
	<-c.slot
	c.rounds.Wait()
	for _, worker := range c.workers {
		close(worker)
	}
	for _, session := range c.sessions {
		_ = session.Close()
	}
}

func makeRankBurstBusinessEvent(eventPrefix, traceComponent, requestID string,
	concurrency, businessLane, payloadBytes int, result rankBurstLaneResult) rankBurstBusinessEvent {
	event := rankBurstBusinessEvent{
		Event: eventPrefix + "_business_complete", RequestID: requestID,
		Concurrency: concurrency, BusinessLane: businessLane, BusinessPayloadBytes: payloadBytes,
		Success:      result.err == nil && result.response != nil,
		ClientWallMS: float64(result.latencyUS) / 1000,
		StartEpochNS: result.startEpochNS, EndEpochNS: result.endEpochNS,
	}
	if result.err != nil {
		event.Error = result.err.Error()
	}
	if result.response == nil || result.response.Trace == nil {
		return event
	}
	event.ServiceTotalMS = float64(pipelinepb.Int64(result.response.Trace.TotalUS)) / 1000
	event.FrontBRPCEstimateMS = event.ClientWallMS - event.ServiceTotalMS
	event.TraceValid = pipelinepb.Int32(result.response.Code) == 200 &&
		pipelinepb.String(result.response.Trace.Component) == traceComponent &&
		pipelinepb.Int64(result.response.Trace.TotalUS) > 0
	return event
}

func makeRankBurstCompleteEvent(eventPrefix, requestID string,
	concurrency, businessLane int, releasedAt time.Time, maxActive int64,
	business rankBurstBusinessEvent, results <-chan rankBurstLaneResult,
	poolSize int, requirePressureOverlap bool) rankBurstCompleteEvent {
	latencies := make([]int64, 0, concurrency-1)
	starts := make([]int64, 0, concurrency)
	pressureSuccess, overlap := 0, 0
	var firstStart, lastEnd int64
	errorSamples := make([]string, 0, 3)
	errorSeen := make(map[string]struct{})
	for result := range results {
		starts = append(starts, result.startOffsetUS)
		if result.business {
			continue
		}
		if firstStart == 0 || result.startEpochNS < firstStart {
			firstStart = result.startEpochNS
		}
		if result.endEpochNS > lastEnd {
			lastEnd = result.endEpochNS
		}
		if result.startEpochNS <= business.EndEpochNS && result.endEpochNS >= business.StartEpochNS {
			overlap++
		}
		if result.err == nil {
			pressureSuccess++
			latencies = append(latencies, result.latencyUS)
		} else if len(errorSamples) < cap(errorSamples) {
			message := result.err.Error()
			if _, exists := errorSeen[message]; !exists {
				errorSeen[message] = struct{}{}
				errorSamples = append(errorSamples, message)
			}
		}
	}
	pressureRequests := concurrency - 1
	tailMS := float64(lastEnd-business.EndEpochNS) / 1e6
	if tailMS < 0 {
		tailMS = 0
	}
	return rankBurstCompleteEvent{
		Event: eventPrefix + "_complete", RequestID: requestID,
		Concurrency: concurrency, BusinessLane: businessLane, ArmedWorkers: concurrency,
		ConnectedSessions: poolSize, MaxActiveWorkers: maxActive,
		StartSkewUS: rankBurstSpread(starts), BurstTotalMS: float64(time.Since(releasedAt).Microseconds()) / 1000,
		PressureRequests: pressureRequests, PressureSuccess: pressureSuccess,
		PressureErrors: pressureRequests - pressureSuccess, PressureErrorSamples: errorSamples,
		PressureLatencyAverageMS: rankBurstAverage(latencies), PressureLatencyP95MS: rankBurstPercentile(latencies, 0.95),
		PressureFirstStartEpochNS: firstStart, PressureLastEndEpochNS: lastEnd,
		PressureOverlapBusiness: overlap, PressureTailAfterBusinessMS: tailMS,
		BusinessStartEpochNS: business.StartEpochNS, BusinessEndEpochNS: business.EndEpochNS,
		BusinessSuccess: business.Success, TraceValid: business.TraceValid,
		BurstValid: business.Success && business.TraceValid && pressureSuccess == pressureRequests &&
			(!requirePressureOverlap || pressureRequests == 0 || overlap > 0),
		PressurePayloadTransport: "protobuf",
	}
}

func updateRankBurstMaximum(maximum *int64, value int64) {
	for observed := atomic.LoadInt64(maximum); observed < value; observed = atomic.LoadInt64(maximum) {
		if atomic.CompareAndSwapInt64(maximum, observed, value) {
			return
		}
	}
}

func rankBurstSpread(values []int64) int64 {
	if len(values) == 0 {
		return 0
	}
	minimum, maximum := values[0], values[0]
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

func rankBurstAverage(values []int64) float64 {
	if len(values) == 0 {
		return 0
	}
	var total int64
	for _, value := range values {
		total += value
	}
	return float64(total) / float64(len(values)) / 1000
}

func rankBurstPercentile(values []int64, percentile float64) float64 {
	if len(values) == 0 {
		return 0
	}
	ordered := append([]int64(nil), values...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	index := int(float64(len(ordered)-1)*percentile + 0.999999)
	if index >= len(ordered) {
		index = len(ordered) - 1
	}
	return float64(ordered[index]) / 1000
}

func writeRankBurstEvent(event any) {
	encoded, err := json.Marshal(event)
	if err != nil {
		fmt.Printf("{\"event\":\"pairec_rank_brpc_burst_log_error\",\"error\":%q}\n", err.Error())
		return
	}
	fmt.Println(string(encoded))
}
