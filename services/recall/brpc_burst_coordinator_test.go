package recall

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"
)

type fakeBRPCBurstSession struct {
	connectErr error
	healthErr  error
	healthGate <-chan struct{}
	omitTrace  bool
}

func (s *fakeBRPCBurstSession) Connect(context.Context) error {
	return s.connectErr
}

func (s *fakeBRPCBurstSession) Recommend(context.Context, *RecommendRequest, string) (*RecommendResponse, error) {
	response := &RecommendResponse{
		Code:            200,
		InferenceTimeMs: 100,
	}
	if !s.omitTrace {
		response.Trace = &TraceInfo{
			RunnerGenerateMs:    98,
			Backend:             "trtllm_cpp",
			WrapperTotalMs:      101,
			WrapperBackendRPCMs: 100,
		}
	}
	return response, nil
}

func (s *fakeBRPCBurstSession) HealthCheckWithPayload(context.Context, int) (*brpcHealthResponse, error) {
	if s.healthGate != nil {
		<-s.healthGate
	}
	if s.healthErr != nil {
		return nil, s.healthErr
	}
	return &brpcHealthResponse{Code: 200, Status: "healthy"}, nil
}

func (s *fakeBRPCBurstSession) Close() error { return nil }

type burstEventRecorder struct {
	mu     sync.Mutex
	events []any
	notify chan struct{}
}

func newBurstEventRecorder() *burstEventRecorder {
	return &burstEventRecorder{notify: make(chan struct{}, 64)}
}

func (r *burstEventRecorder) log(event any) {
	r.mu.Lock()
	r.events = append(r.events, event)
	r.mu.Unlock()
	select {
	case r.notify <- struct{}{}:
	default:
	}
}

func (r *burstEventRecorder) waitComplete(t *testing.T) brpcBurstCompleteEvent {
	t.Helper()
	deadline := time.After(time.Second)
	for {
		r.mu.Lock()
		for _, event := range r.events {
			if complete, ok := event.(brpcBurstCompleteEvent); ok {
				r.mu.Unlock()
				return complete
			}
		}
		r.mu.Unlock()
		select {
		case <-r.notify:
		case <-deadline:
			t.Fatal("timed out waiting for burst completion event")
		}
	}
}

func TestBRPCBurstReturnsBusinessBeforePressureCompletes(t *testing.T) {
	healthGate := make(chan struct{})
	sessions := []brpcBurstSession{
		&fakeBRPCBurstSession{healthGate: healthGate},
		&fakeBRPCBurstSession{healthGate: healthGate},
	}
	recorder := newBurstEventRecorder()
	coordinator, err := newBRPCBurstCoordinator(sessions, BRPCBurstConfig{
		Concurrency: 2, PressureTimeout: time.Second,
	}, recorder.log)
	if err != nil {
		t.Fatal(err)
	}

	returned := make(chan error, 1)
	go func() {
		_, callErr := coordinator.Recommend(&RecommendRequest{UserID: "5"}, "request-1")
		returned <- callErr
	}()

	select {
	case callErr := <-returned:
		if callErr != nil {
			t.Fatal(callErr)
		}
	case <-time.After(time.Second):
		t.Fatal("business response waited for pressure lane")
	}

	close(healthGate)
	complete := recorder.waitComplete(t)
	if !complete.BurstValid || complete.PressureSuccess != 1 {
		t.Fatalf("unexpected completion: %+v", complete)
	}
}

func TestBRPCBurstPressureFailureOnlyInvalidatesSample(t *testing.T) {
	sessions := []brpcBurstSession{
		&fakeBRPCBurstSession{healthErr: errors.New("pressure failed")},
		&fakeBRPCBurstSession{healthErr: errors.New("pressure failed")},
	}
	recorder := newBurstEventRecorder()
	coordinator, err := newBRPCBurstCoordinator(sessions, BRPCBurstConfig{
		Concurrency: 2, PressureTimeout: time.Second,
	}, recorder.log)
	if err != nil {
		t.Fatal(err)
	}

	response, err := coordinator.Recommend(&RecommendRequest{UserID: "5"}, "request-2")
	if err != nil || response == nil || response.Code != 200 {
		t.Fatalf("business result changed by pressure failure: response=%+v err=%v", response, err)
	}
	complete := recorder.waitComplete(t)
	if complete.BurstValid || complete.PressureErrors != 1 || !complete.BusinessSuccess {
		t.Fatalf("unexpected completion: %+v", complete)
	}
}

func TestBRPCBurstMissingWrapperTraceOnlyInvalidatesSample(t *testing.T) {
	sessions := []brpcBurstSession{&fakeBRPCBurstSession{omitTrace: true}}
	recorder := newBurstEventRecorder()
	coordinator, err := newBRPCBurstCoordinator(sessions, BRPCBurstConfig{
		Concurrency: 1, PressureTimeout: time.Second,
	}, recorder.log)
	if err != nil {
		t.Fatal(err)
	}

	response, err := coordinator.Recommend(&RecommendRequest{UserID: "5"}, "request-no-trace")
	if err != nil || response == nil || response.Code != 200 {
		t.Fatalf("missing trace changed business result: response=%+v err=%v", response, err)
	}
	complete := recorder.waitComplete(t)
	if complete.TraceValid || complete.BurstValid || !complete.BusinessSuccess {
		t.Fatalf("unexpected completion: %+v", complete)
	}
}

func TestBRPCBurstSerializesConcurrentBusinessRequests(t *testing.T) {
	healthGate := make(chan struct{})
	sessions := []brpcBurstSession{
		&fakeBRPCBurstSession{healthGate: healthGate},
		&fakeBRPCBurstSession{healthGate: healthGate},
	}
	coordinator, err := newBRPCBurstCoordinator(sessions, BRPCBurstConfig{
		Concurrency: 2, PressureTimeout: time.Second,
	}, func(any) {})
	if err != nil {
		t.Fatal(err)
	}

	first := make(chan error, 1)
	go func() {
		_, callErr := coordinator.Recommend(&RecommendRequest{UserID: "first"}, "first")
		first <- callErr
	}()
	select {
	case err := <-first:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("first business request did not return")
	}

	second := make(chan error, 1)
	go func() {
		_, callErr := coordinator.Recommend(&RecommendRequest{UserID: "second"}, "second")
		second <- callErr
	}()
	select {
	case err := <-second:
		t.Fatalf("second request entered before pressure completion: %v", err)
	case <-time.After(20 * time.Millisecond):
	}

	close(healthGate)
	select {
	case err := <-second:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("second request remained blocked after pressure completion")
	}
}

func TestBRPCBurstBusinessLaneCoversEverySession(t *testing.T) {
	const concurrency = 7
	sessions := make([]brpcBurstSession, concurrency)
	for index := range sessions {
		sessions[index] = &fakeBRPCBurstSession{}
	}
	var mu sync.Mutex
	lanes := make(map[int]bool)
	coordinator, err := newBRPCBurstCoordinator(sessions, BRPCBurstConfig{
		Concurrency: concurrency, PressureTimeout: time.Second,
	}, func(event any) {
		if start, ok := event.(brpcBurstStartEvent); ok {
			mu.Lock()
			lanes[start.BusinessLane] = true
			mu.Unlock()
		}
	})
	if err != nil {
		t.Fatal(err)
	}

	for request := 0; request < concurrency; request++ {
		if _, err := coordinator.Recommend(&RecommendRequest{UserID: "5"}, "lane-test"); err != nil {
			t.Fatal(err)
		}
	}
	// The last request returns before its pressure lanes complete.
	<-coordinator.slot
	coordinator.slot <- struct{}{}

	mu.Lock()
	defer mu.Unlock()
	if len(lanes) != concurrency {
		t.Fatalf("business lane coverage=%v, want all %d lanes", lanes, concurrency)
	}
}

func TestBRPCBurstStartupFailsWhenAnySessionCannotConnect(t *testing.T) {
	sessions := []brpcBurstSession{
		&fakeBRPCBurstSession{},
		&fakeBRPCBurstSession{connectErr: errors.New("connect failed")},
	}
	_, err := newBRPCBurstCoordinator(sessions, BRPCBurstConfig{
		Concurrency: 2, PressureTimeout: time.Second,
	}, func(any) {})
	if err == nil {
		t.Fatal("expected strict preconnect failure")
	}
}

func TestBalancedCountsSupportsArbitraryCPUShardCount(t *testing.T) {
	for _, shards := range []int{1, 2, 3, 7, 16, 64} {
		counts := balancedCounts(1000, shards, 5)
		total, minimum, maximum := 0, 1000, 0
		for _, count := range counts {
			total += count
			if count < minimum {
				minimum = count
			}
			if count > maximum {
				maximum = count
			}
		}
		if total != 1000 || maximum-minimum > 1 {
			t.Fatalf("shards=%d counts=%v total=%d spread=%d", shards, counts, total, maximum-minimum)
		}
	}
}

func TestBRPCBurstPoolSelectsActiveSessionsAcrossArbitraryShards(t *testing.T) {
	const poolSize = 10000
	coordinator := &BRPCBurstCoordinator{
		poolSize:      poolSize,
		active:        1000,
		cpuShards:     []int{0, 1, 2, 3, 4, 5, 6},
		shardSessions: make([][]int, 7),
	}
	for index := 0; index < poolSize; index++ {
		shard := index % len(coordinator.cpuShards)
		coordinator.shardSessions[shard] = append(coordinator.shardSessions[shard], index)
	}
	seen := make(map[int]struct{}, poolSize)
	for request := 0; request < 10; request++ {
		selected, shardCounts := coordinator.selectSessions()
		if len(selected) != 1000 {
			t.Fatalf("request=%d selected=%d", request, len(selected))
		}
		minimum, maximum := 1000, 0
		for _, count := range shardCounts {
			if count < minimum {
				minimum = count
			}
			if count > maximum {
				maximum = count
			}
		}
		if maximum-minimum > 1 {
			t.Fatalf("request=%d shardCounts=%v", request, shardCounts)
		}
		for _, session := range selected {
			seen[session] = struct{}{}
		}
	}
	if len(seen) != poolSize {
		t.Fatalf("ten selections covered %d/%d sessions", len(seen), poolSize)
	}
}

func TestBRPCBurstCompletionReportsBalancedShardTraffic(t *testing.T) {
	results := make(chan brpcBurstLaneResult, 10)
	for lane := 0; lane < 10; lane++ {
		results <- brpcBurstLaneResult{
			index: lane + 1, business: lane == 3, shard: lane % 3,
			latencyUs: 1000, startOffsetUs: int64(lane),
		}
	}
	close(results)
	event := makeBRPCBurstCompleteEvent(
		"request-shards", 10, 4, time.Now(), 10,
		brpcBurstBusinessEvent{Success: true, TraceValid: true}, results,
		100, 102400, []int{0, 2, 4},
	)
	if !event.BurstValid || event.PressureSuccess != 9 {
		t.Fatalf("unexpected completion: %+v", event)
	}
	if event.ShardRequests[0] != 4 || event.ShardRequests[1] != 3 || event.ShardRequests[2] != 3 {
		t.Fatalf("unexpected shard requests: %v", event.ShardRequests)
	}
	if event.ShardBytes[0] != 3*102400 || event.ShardBytes[1] != 3*102400 || event.ShardBytes[2] != 3*102400 {
		t.Fatalf("unexpected shard bytes: %v", event.ShardBytes)
	}
}

func TestBRPCBurstCompletionReportsPressureErrorSamples(t *testing.T) {
	results := make(chan brpcBurstLaneResult, 4)
	results <- brpcBurstLaneResult{index: 0, business: true, shard: 0}
	results <- brpcBurstLaneResult{index: 1, shard: 0, err: errors.New("connection reset")}
	results <- brpcBurstLaneResult{index: 2, shard: 0, err: errors.New("connection reset")}
	results <- brpcBurstLaneResult{index: 3, shard: 0, err: errors.New("broken pipe")}
	close(results)

	event := makeBRPCBurstCompleteEvent(
		"request-errors", 4, 0, time.Now(), 4,
		brpcBurstBusinessEvent{Success: true, TraceValid: true}, results,
		4, 102400, []int{0},
	)
	if event.BurstValid || event.PressureErrors != 3 {
		t.Fatalf("unexpected completion: %+v", event)
	}
	want := []string{"connection reset", "broken pipe"}
	if !reflect.DeepEqual(event.PressureErrorSamples, want) {
		t.Fatalf("pressure error samples=%v want=%v", event.PressureErrorSamples, want)
	}
}
