package observability

import (
	"testing"
	"time"
)

func TestRecorderClosesAndPreservesParallelDiagnostics(t *testing.T) {
	recorder := NewRecorder("req-1")
	started := time.Now()
	time.Sleep(2 * time.Millisecond)
	recorder.Record("recommend_service", "pairec", "in_process", "", true,
		started, time.Since(started), "ok", nil)
	recorder.Record("vector_recall", "vector_recall", "brpc", "recommend_service", false,
		started, 10*time.Millisecond, "ok", nil)
	trace := recorder.Finalize("ok")
	if !trace.Valid {
		t.Fatalf("trace invalid: %v", trace.InvalidReasons)
	}
	if trace.PaiRecTotalUS != trace.AccountedUS || trace.ClosureErrorUS != 0 {
		t.Fatalf("trace did not close: total=%d accounted=%d error=%d",
			trace.PaiRecTotalUS, trace.AccountedUS, trace.ClosureErrorUS)
	}
}

func TestDataSystemStrictAttributionGate(t *testing.T) {
	t.Setenv("PAIREC_DATASYSTEM_EXPECTED", "1")
	t.Setenv("PAIREC_REQUIRE_DATASYSTEM_ATTRIBUTION", "1")
	trace := NewRecorder("req-ds").Finalize("ok")
	if trace.Valid {
		t.Fatal("strict trace unexpectedly valid without DataSystem attribution")
	}
}
