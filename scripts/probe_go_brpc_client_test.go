package main

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestRunSynchronizedBurstArmsEveryLane(t *testing.T) {
	const lanes = 64
	var active int64
	var maxActive int64
	var calls int64
	var allCallsStarted sync.WaitGroup
	allCallsStarted.Add(lanes)

	results, releasedAt := runSynchronizedBurst(lanes, &active, &maxActive,
		func(index int, releaseTime time.Time) probeResult {
			if releaseTime.IsZero() {
				t.Errorf("lane %d received a zero release time", index)
			}
			atomic.AddInt64(&calls, 1)
			allCallsStarted.Done()
			allCallsStarted.Wait()
			return probeResult{index: index, startOffsetUs: time.Since(releaseTime).Microseconds()}
		})

	if releasedAt.IsZero() {
		t.Fatal("burst release time was not recorded")
	}
	if got := atomic.LoadInt64(&maxActive); got != lanes {
		t.Fatalf("max active lanes = %d, want %d", got, lanes)
	}
	if got := atomic.LoadInt64(&active); got != 0 {
		t.Fatalf("active lanes after completion = %d, want 0", got)
	}
	if got := atomic.LoadInt64(&calls); got != lanes {
		t.Fatalf("executed calls = %d, want %d", got, lanes)
	}
	for index, result := range results {
		if result.index != index+1 {
			t.Fatalf("result[%d].index = %d, want %d", index, result.index, index+1)
		}
	}
}

func TestBurstStatistics(t *testing.T) {
	values := []int64{1000, 2000, 3000, 4000, 5000}
	if got := averageMicroseconds(values); got != 3 {
		t.Fatalf("average = %.3fms, want 3ms", got)
	}
	if got := percentileMicroseconds(values, 0.95); got != 4.8 {
		t.Fatalf("p95 = %.3fms, want 4.8ms", got)
	}
	if got := spreadInt64([]int64{8, 3, 13, 5}); got != 10 {
		t.Fatalf("spread = %dus, want 10us", got)
	}
}
