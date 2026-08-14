package recall

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sys/unix"
)

type BRPCBurstConfig struct {
	Concurrency     int
	PoolSize        int
	Active          int
	CPUShards       []int
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
	poolSize        int
	active          int
	cpuShards       []int
	shardSessions   [][]int
	poolCursor      int
	workers         []chan brpcBurstJob
	payloadBytes    int
	pressureTimeout time.Duration
	stride          int
	sequence        uint64
	slot            chan struct{}
	logEvent        func(any)
}

type brpcBurstJob struct {
	req            *RecommendRequest
	requestID      string
	lane           int
	business       bool
	gate           <-chan struct{}
	releasedAt     *time.Time
	results        chan<- brpcBurstLaneResult
	businessResult chan<- brpcBurstLaneResult
	complete       *sync.WaitGroup
	active         *int64
	maxActive      *int64
}

type brpcBurstLaneResult struct {
	index         int
	business      bool
	startOffsetUs int64
	latencyUs     int64
	response      *RecommendResponse
	err           error
	shard         int
}

type brpcBurstStartEvent struct {
	Event             string  `json:"event"`
	RequestID         string  `json:"request_id"`
	Concurrency       int     `json:"concurrency"`
	BusinessLane      int     `json:"business_lane"`
	PayloadBytes      int     `json:"pressure_payload_bytes"`
	ConnectedSessions int     `json:"connected_sessions"`
	SelectedSessions  int     `json:"selected_sessions"`
	PreflightMs       float64 `json:"preflight_ms"`
	Stride            int     `json:"business_lane_stride"`
	PoolSize          int     `json:"pool_size"`
	ActiveConnections int     `json:"active_connections"`
	CPUShards         []int   `json:"cpu_shards"`
	ShardSelected     []int   `json:"shard_selected"`
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
	PoolSize             int     `json:"pool_size"`
	ActiveConnections    int     `json:"active_connections"`
	CPUShards            []int   `json:"cpu_shards"`
	ShardRequests        []int   `json:"shard_requests"`
	ShardSuccess         []int   `json:"shard_success"`
	ShardBytes           []int64 `json:"shard_bytes"`
}

func NewBRPCBurstCoordinator(client *BRPCRecommendClient, cfg BRPCBurstConfig) (*BRPCBurstCoordinator, error) {
	if client == nil {
		return nil, fmt.Errorf("brpc client is nil")
	}
	poolSize := cfg.PoolSize
	if poolSize == 0 {
		poolSize = cfg.Concurrency
	}
	sessions := make([]brpcBurstSession, poolSize)
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
	if cfg.PoolSize == 0 {
		cfg.PoolSize = cfg.Concurrency
	}
	if cfg.Active == 0 {
		cfg.Active = cfg.Concurrency
	}
	if cfg.PoolSize < 1 || cfg.PoolSize > 10000 {
		return nil, fmt.Errorf("pool size must be in [1,10000]")
	}
	if cfg.Active < 1 || cfg.Active > cfg.Concurrency || cfg.Active > cfg.PoolSize {
		return nil, fmt.Errorf("active connections must be in [1,min(concurrency,pool size)]")
	}
	if len(sessions) != cfg.PoolSize {
		return nil, fmt.Errorf("session count %d does not match pool size %d", len(sessions), cfg.PoolSize)
	}
	cpuShards, err := resolveBRPCBurstCPUShards(cfg.CPUShards, cfg.Active)
	if err != nil {
		return nil, err
	}
	if len(cpuShards) > cfg.Active {
		return nil, fmt.Errorf("CPU shard count %d exceeds active connections %d", len(cpuShards), cfg.Active)
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
		concurrency:     cfg.Active,
		poolSize:        cfg.PoolSize,
		active:          cfg.Active,
		cpuShards:       cpuShards,
		payloadBytes:    cfg.PayloadBytes,
		pressureTimeout: cfg.PressureTimeout,
		stride:          coprimeStride(cfg.Concurrency),
		slot:            make(chan struct{}, 1),
		logEvent:        logger,
	}
	c.shardSessions = make([][]int, len(cpuShards))
	c.workers = make([]chan brpcBurstJob, cfg.PoolSize)
	for index := range sessions {
		shard := index % len(cpuShards)
		c.shardSessions[shard] = append(c.shardSessions[shard], index)
		c.workers[index] = make(chan brpcBurstJob)
		go c.runSessionWorker(index, shard, c.workers[index])
	}
	if _, err := c.connectAll(); err != nil {
		for _, worker := range c.workers {
			close(worker)
		}
		return nil, err
	}
	c.slot <- struct{}{}
	c.logEvent(map[string]any{
		"event":                "pairec_brpc_burst_ready",
		"concurrency":          c.concurrency,
		"connected_sessions":   c.poolSize,
		"pool_size":            c.poolSize,
		"active_connections":   c.active,
		"cpu_shards":           c.cpuShards,
		"shard_connections":    shardLengths(c.shardSessions),
		"payload_bytes":        c.payloadBytes,
		"business_lane_stride": c.stride,
	})
	return c, nil
}

func (c *BRPCBurstCoordinator) Recommend(req *RecommendRequest, requestID string) (*RecommendResponse, error) {
	<-c.slot
	selected, shardSelected := c.selectSessions()
	preflightMs := 0.0

	sequence := atomic.AddUint64(&c.sequence, 1)
	businessLane := int((sequence*uint64(c.stride))%uint64(c.active)) + 1
	results := make(chan brpcBurstLaneResult, c.active)
	businessResult := make(chan brpcBurstLaneResult, 1)
	gate := make(chan struct{})
	var complete sync.WaitGroup
	var active int64
	var maxActive int64
	complete.Add(c.active)
	var releasedAt time.Time

	for lane, sessionIndex := range selected {
		c.workers[sessionIndex] <- brpcBurstJob{
			req: req, requestID: requestID, lane: lane + 1,
			business: lane+1 == businessLane, gate: gate, releasedAt: &releasedAt,
			results: results, businessResult: businessResult, complete: &complete,
			active: &active, maxActive: &maxActive,
		}
	}

	c.logEvent(brpcBurstStartEvent{
		Event:             "pairec_brpc_burst_start",
		RequestID:         requestID,
		Concurrency:       c.concurrency,
		BusinessLane:      businessLane,
		PayloadBytes:      c.payloadBytes,
		ConnectedSessions: c.poolSize,
		SelectedSessions:  len(selected),
		PreflightMs:       preflightMs,
		Stride:            c.stride,
		PoolSize:          c.poolSize,
		ActiveConnections: c.active,
		CPUShards:         c.cpuShards,
		ShardSelected:     shardSelected,
	})
	releasedAt = time.Now()
	close(gate)

	business := <-businessResult
	businessEvent := makeBRPCBurstBusinessEvent(requestID, c.active, businessLane, business)
	c.logEvent(businessEvent)

	go func() {
		complete.Wait()
		close(results)
		c.logEvent(makeBRPCBurstCompleteEvent(
			requestID, c.active, businessLane, releasedAt,
			atomic.LoadInt64(&maxActive), businessEvent, results,
			c.poolSize, c.payloadBytes, c.cpuShards,
		))
		c.slot <- struct{}{}
	}()

	if business.err != nil {
		return nil, business.err
	}
	return business.response, nil
}

func (c *BRPCBurstCoordinator) runSessionWorker(sessionIndex, shard int, jobs <-chan brpcBurstJob) {
	for job := range jobs {
		<-job.gate
		callStarted := time.Now()
		result := brpcBurstLaneResult{
			index: job.lane, business: job.business, shard: shard,
			startOffsetUs: callStarted.Sub(*job.releasedAt).Microseconds(),
		}
		current := atomic.AddInt64(job.active, 1)
		updateBurstMaximum(job.maxActive, current)
		ctx, cancel := context.WithTimeout(context.Background(), c.pressureTimeout)
		if job.business {
			result.response, result.err = c.sessions[sessionIndex].Recommend(ctx, job.req, job.requestID)
		} else {
			_, result.err = c.sessions[sessionIndex].HealthCheckWithPayload(ctx, c.payloadBytes)
		}
		cancel()
		atomic.AddInt64(job.active, -1)
		result.latencyUs = time.Since(callStarted).Microseconds()
		job.results <- result
		if result.business {
			job.businessResult <- result
		}
		job.complete.Done()
	}
}

func (c *BRPCBurstCoordinator) selectSessions() ([]int, []int) {
	selected := make([]int, 0, c.active)
	counts := make([]int, len(c.cpuShards))
	for offset := 0; offset < c.active; offset++ {
		session := (c.poolCursor + offset) % c.poolSize
		selected = append(selected, session)
		counts[session%len(c.cpuShards)]++
	}
	c.poolCursor = (c.poolCursor + c.active) % c.poolSize
	return selected, counts
}

func (c *BRPCBurstCoordinator) connectAll() (int, error) {
	indices := make([]int, len(c.sessions))
	for index := range indices {
		indices[index] = index
	}
	return c.connectSessions(indices)
}

func (c *BRPCBurstCoordinator) connectSessions(indices []int) (int, error) {
	var wg sync.WaitGroup
	errs := make(chan error, len(indices))
	limit := make(chan struct{}, 256)
	var connected int64
	for _, index := range indices {
		session := c.sessions[index]
		wg.Add(1)
		go func(session brpcBurstSession) {
			defer wg.Done()
			limit <- struct{}{}
			defer func() { <-limit }()
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
	<-c.slot
	for _, worker := range c.workers {
		close(worker)
	}
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
	business brpcBurstBusinessEvent,
	results <-chan brpcBurstLaneResult,
	poolSize int,
	payloadBytes int,
	cpuShards []int,
) brpcBurstCompleteEvent {
	pressureLatencies := make([]int64, 0, concurrency-1)
	startOffsets := make([]int64, 0, concurrency)
	pressureSuccess := 0
	shardRequests := make([]int, len(cpuShards))
	shardSuccess := make([]int, len(cpuShards))
	shardBytes := make([]int64, len(cpuShards))
	for result := range results {
		startOffsets = append(startOffsets, result.startOffsetUs)
		shardRequests[result.shard]++
		if result.err == nil {
			shardSuccess[result.shard]++
		}
		if result.business {
			continue
		}
		if !result.business {
			shardBytes[result.shard] += int64(payloadBytes)
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
		PoolSize:             poolSize,
		ActiveConnections:    concurrency,
		CPUShards:            append([]int(nil), cpuShards...),
		ShardRequests:        shardRequests,
		ShardSuccess:         shardSuccess,
		ShardBytes:           shardBytes,
	}
}

func resolveBRPCBurstCPUShards(configured []int, active int) ([]int, error) {
	var allowed unix.CPUSet
	if err := unix.SchedGetaffinity(0, &allowed); err != nil {
		return nil, fmt.Errorf("read process CPU affinity: %w", err)
	}
	if len(configured) == 0 {
		for cpu := 0; cpu < 1024; cpu++ {
			if allowed.IsSet(cpu) {
				configured = append(configured, cpu)
				if len(configured) == active {
					break
				}
			}
		}
	}
	if len(configured) == 0 {
		return nil, fmt.Errorf("no CPU shards available")
	}
	seen := make(map[int]struct{}, len(configured))
	for _, cpu := range configured {
		if cpu < 0 || !allowed.IsSet(cpu) {
			return nil, fmt.Errorf("CPU shard %d is outside process affinity", cpu)
		}
		if _, exists := seen[cpu]; exists {
			return nil, fmt.Errorf("duplicate CPU shard %d", cpu)
		}
		seen[cpu] = struct{}{}
	}
	return append([]int(nil), configured...), nil
}

func balancedCounts(total, shards, rotation int) []int {
	counts := make([]int, shards)
	for index := 0; index < total; index++ {
		counts[(index+rotation)%shards]++
	}
	return counts
}

func shardLengths(shards [][]int) []int {
	result := make([]int, len(shards))
	for index := range shards {
		result[index] = len(shards[index])
	}
	return result
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
