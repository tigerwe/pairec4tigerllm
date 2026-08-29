package ranksort

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	proto "github.com/gogo/protobuf/proto"
	"pairec4tigerllm/services/pipelinepb"
)

type fakeRankBurstSession struct {
	pressureDelay time.Duration
	rankDelay     time.Duration
	connected     atomic.Bool
	rankCalls     atomic.Int64
}

func (s *fakeRankBurstSession) Connect(context.Context) error {
	s.connected.Store(true)
	return nil
}

func (s *fakeRankBurstSession) Rank(_ context.Context, request *pipelinepb.RankRequest) (*pipelinepb.RankResponse, error) {
	s.rankCalls.Add(1)
	if s.rankDelay > 0 {
		time.Sleep(s.rankDelay)
	}
	return &pipelinepb.RankResponse{
		Code: proto.Int32(200),
		Trace: &pipelinepb.ServiceTrace{
			Context:   request.Context,
			Component: proto.String("deepfm_rank_adapter"),
			TotalUS:   proto.Int64(1000),
		},
	}, nil
}

func TestRankBurstCoordinatorDedicatedLaneDoesNotWaitForPreviousPressureTail(t *testing.T) {
	const concurrency = 4
	sessions := make([]rankBurstSession, concurrency)
	fakes := make([]*fakeRankBurstSession, concurrency)
	for index := range sessions {
		fakes[index] = &fakeRankBurstSession{
			pressureDelay: 80 * time.Millisecond,
			rankDelay:     20 * time.Millisecond,
		}
		sessions[index] = fakes[index]
	}
	events := make(chan any, 16)
	coordinator, err := newRankBurstCoordinator(
		sessions,
		RankBurstConfig{
			Concurrency: concurrency, PoolSize: concurrency,
			BusinessBytes: 102400, PressureBytes: 102400,
			BusinessTimeout: time.Second, PressureTimeout: time.Second,
			DedicatedBusinessLane: true,
		},
		func(event any) { events <- event },
	)
	if err != nil {
		t.Fatal(err)
	}
	defer coordinator.Close()
	for index := 0; index < 2; index++ {
		started := time.Now()
		if _, err := coordinator.Rank(testRankBurstRequest(), fmt.Sprintf("dedicated-%d", index)); err != nil {
			t.Fatal(err)
		}
		if elapsed := time.Since(started); elapsed >= 50*time.Millisecond {
			t.Fatalf("business request %d waited for a pressure tail: %s", index, elapsed)
		}
	}
	for index := 0; index < 2; index++ {
		complete := waitRankBurstComplete(t, events)
		if complete.BusinessLane != 1 || !complete.BurstValid || complete.PressureSuccess != concurrency-1 {
			t.Fatalf("unexpected dedicated completion: %#v", complete)
		}
	}
	if got := fakes[0].rankCalls.Load(); got != 2 {
		t.Fatalf("dedicated business session calls=%d, want 2", got)
	}
	for index := 1; index < len(fakes); index++ {
		if got := fakes[index].rankCalls.Load(); got != 0 {
			t.Fatalf("pressure session %d handled %d business calls", index, got)
		}
	}
}

func TestRankBurstCoordinatorWaitsForPressureMarkerAndAddsLocalWindow(t *testing.T) {
	const concurrency = 4
	sessions := make([]rankBurstSession, concurrency)
	for index := range sessions {
		sessions[index] = &fakeRankBurstSession{
			pressureDelay: 80 * time.Millisecond,
			rankDelay:     5 * time.Millisecond,
		}
	}
	events := make(chan any, 16)
	coordinator, err := newRankBurstCoordinator(
		sessions,
		RankBurstConfig{
			Concurrency: concurrency, PoolSize: concurrency,
			BusinessBytes: 102400, PressureBytes: 102400,
			BusinessTimeout: time.Second, PressureTimeout: time.Second,
			DedicatedBusinessLane: true, PressureStartQuorum: concurrency - 1,
			PressureStartTimeout: 100 * time.Millisecond,
			MinimumLocalWindow:   20 * time.Millisecond,
		},
		func(event any) { events <- event },
	)
	if err != nil {
		t.Fatal(err)
	}
	defer coordinator.Close()
	started := time.Now()
	if _, err := coordinator.Rank(testRankBurstRequest(), "marker-window"); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(started); elapsed < 19*time.Millisecond || elapsed >= 60*time.Millisecond {
		t.Fatalf("unexpected marker/window business latency: %s", elapsed)
	}
	business := waitRankBurstBusiness(t, events)
	if business.PressureStartQuorum != concurrency-1 ||
		business.PressureStartedAtBusinessStart < concurrency-1 {
		t.Fatalf("business did not wait for pressure marker: %#v", business)
	}
	if business.LocalBusinessWindowMS < 19.5 || business.BusinessHoldMS < 10 {
		t.Fatalf("local business window was not enforced: %#v", business)
	}
	complete := waitRankBurstComplete(t, events)
	if !complete.BurstValid || complete.PressureSuccess != concurrency-1 {
		t.Fatalf("unexpected marker/window completion: %#v", complete)
	}
}

func (s *fakeRankBurstSession) HealthWithPayload(ctx context.Context, _ string, _ int) (*pipelinepb.HealthResponse, error) {
	select {
	case <-time.After(s.pressureDelay):
		return &pipelinepb.HealthResponse{Code: proto.Int32(200)}, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (s *fakeRankBurstSession) Close() error { return nil }

func testRankBurstRequest() *pipelinepb.RankRequest {
	return &pipelinepb.RankRequest{
		Context: &pipelinepb.TraceContext{
			RequestID:       proto.String("rank-burst-test"),
			ContractVersion: proto.String(pipelinepb.TraceContractVersion),
		},
		PayloadPadding: make([]byte, 102400),
	}
}

func TestRankBurstCoordinatorC1IsValidWithoutPressure(t *testing.T) {
	events := make(chan any, 4)
	coordinator, err := newRankBurstCoordinator(
		[]rankBurstSession{&fakeRankBurstSession{}},
		RankBurstConfig{
			Concurrency: 1, PoolSize: 1, BusinessBytes: 102400,
			BusinessTimeout: time.Second, PressureTimeout: time.Second,
		},
		func(event any) { events <- event },
	)
	if err != nil {
		t.Fatal(err)
	}
	defer coordinator.Close()
	if _, err := coordinator.Rank(testRankBurstRequest(), "rank-burst-test"); err != nil {
		t.Fatal(err)
	}
	complete := waitRankBurstComplete(t, events)
	if !complete.BurstValid || complete.PressureRequests != 0 || complete.PressureSuccess != 0 {
		t.Fatalf("unexpected c1 completion: %#v", complete)
	}
}

func TestRankBurstCoordinatorReturnsBusinessBeforePressureTail(t *testing.T) {
	const concurrency = 4
	sessions := make([]rankBurstSession, concurrency)
	for index := range sessions {
		sessions[index] = &fakeRankBurstSession{pressureDelay: 80 * time.Millisecond}
	}
	events := make(chan any, 8)
	coordinator, err := newRankBurstCoordinator(
		sessions,
		RankBurstConfig{
			Concurrency: concurrency, PoolSize: concurrency,
			BusinessBytes: 102400, PressureBytes: 102400,
			BusinessTimeout: time.Second, PressureTimeout: time.Second,
		},
		func(event any) { events <- event },
	)
	if err != nil {
		t.Fatal(err)
	}
	defer coordinator.Close()
	started := time.Now()
	if _, err := coordinator.Rank(testRankBurstRequest(), "rank-burst-test"); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(started); elapsed >= 50*time.Millisecond {
		t.Fatalf("business waited for pressure tail: %s", elapsed)
	}
	complete := waitRankBurstComplete(t, events)
	if !complete.BurstValid || complete.PressureRequests != concurrency-1 ||
		complete.PressureSuccess != concurrency-1 || complete.PressureOverlapBusiness <= 0 {
		t.Fatalf("unexpected pressure completion: %#v", complete)
	}
	if complete.PressureTailAfterBusinessMS < 50 {
		t.Fatalf("pressure tail was not observed: %#v", complete)
	}
}

func waitRankBurstComplete(t *testing.T, events <-chan any) rankBurstCompleteEvent {
	t.Helper()
	timer := time.NewTimer(time.Second)
	defer timer.Stop()
	for {
		select {
		case event := <-events:
			if complete, ok := event.(rankBurstCompleteEvent); ok {
				return complete
			}
		case <-timer.C:
			t.Fatal("timed out waiting for rank burst completion")
		}
	}
}

func waitRankBurstBusiness(t *testing.T, events <-chan any) rankBurstBusinessEvent {
	t.Helper()
	timer := time.NewTimer(time.Second)
	defer timer.Stop()
	for {
		select {
		case event := <-events:
			if business, ok := event.(rankBurstBusinessEvent); ok {
				return business
			}
		case <-timer.C:
			t.Fatal("timed out waiting for rank burst business event")
		}
	}
}
