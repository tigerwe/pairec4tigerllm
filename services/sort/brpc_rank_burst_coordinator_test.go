package ranksort

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	proto "github.com/gogo/protobuf/proto"
	"pairec4tigerllm/services/pipelinepb"
)

type fakeRankBurstSession struct {
	pressureDelay time.Duration
	connected     atomic.Bool
}

func (s *fakeRankBurstSession) Connect(context.Context) error {
	s.connected.Store(true)
	return nil
}

func (s *fakeRankBurstSession) Rank(_ context.Context, request *pipelinepb.RankRequest) (*pipelinepb.RankResponse, error) {
	return &pipelinepb.RankResponse{
		Code: proto.Int32(200),
		Trace: &pipelinepb.ServiceTrace{
			Context:   request.Context,
			Component: proto.String("deepfm_rank_adapter"),
			TotalUS:   proto.Int64(1000),
		},
	}, nil
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
