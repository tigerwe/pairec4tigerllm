package config

import (
	"testing"
	"time"
)

func TestGenerativeRecallConfigDefaultsBurstTimeout(t *testing.T) {
	config := &GenerativeRecallConfig{
		Protocol:             "brpc",
		BRPCEndpoint:         "127.0.0.1:18103",
		BRPCBurstEnabled:     true,
		BRPCBurstConcurrency: 1000,
		BRPCBurstPreconnect:  true,
		HistoryFeatureName:   "click_history",
	}
	if err := config.Validate(); err != nil {
		t.Fatal(err)
	}
	if config.Timeout != 3*time.Second || config.BRPCBurstPressureTimeout != config.Timeout {
		t.Fatalf("timeout=%s burst_timeout=%s", config.Timeout, config.BRPCBurstPressureTimeout)
	}
}

func TestGenerativeRecallConfigRejectsBurstWithoutPreconnect(t *testing.T) {
	config := &GenerativeRecallConfig{
		Protocol:             "brpc",
		BRPCEndpoint:         "127.0.0.1:18103",
		BRPCBurstEnabled:     true,
		BRPCBurstConcurrency: 100,
		HistoryFeatureName:   "click_history",
	}
	if err := config.Validate(); err == nil {
		t.Fatal("expected preconnect validation error")
	}
}

func TestGenerativeRecallConfigAcceptsShardedConnectionPool(t *testing.T) {
	config := &GenerativeRecallConfig{
		Protocol:             "brpc",
		BRPCEndpoint:         "127.0.0.1:18103",
		BRPCBurstEnabled:     true,
		BRPCBurstConcurrency: 1000,
		BRPCBurstPoolSize:    10000,
		BRPCBurstActive:      1000,
		BRPCBurstCPUShards:   []int{0, 2, 4, 6, 8, 10, 12, 14},
		BRPCBurstPreconnect:  true,
		HistoryFeatureName:   "click_history",
	}
	if err := config.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestGenerativeRecallConfigAllowsAutomaticCPUShards(t *testing.T) {
	config := &GenerativeRecallConfig{
		Protocol:             "brpc",
		BRPCEndpoint:         "127.0.0.1:18103",
		BRPCBurstEnabled:     true,
		BRPCBurstConcurrency: 1000,
		BRPCBurstPoolSize:    10000,
		BRPCBurstActive:      1000,
		BRPCBurstPreconnect:  true,
		HistoryFeatureName:   "click_history",
	}
	if err := config.Validate(); err != nil {
		t.Fatal(err)
	}
	if config.BRPCBurstCPUShards != nil {
		t.Fatalf("automatic CPU shards changed during validation: %v", config.BRPCBurstCPUShards)
	}
}

func TestGenerativeRecallConfigRejectsDuplicateCPUShards(t *testing.T) {
	config := &GenerativeRecallConfig{
		Protocol:             "brpc",
		BRPCEndpoint:         "127.0.0.1:18103",
		BRPCBurstEnabled:     true,
		BRPCBurstConcurrency: 1000,
		BRPCBurstPoolSize:    10000,
		BRPCBurstActive:      1000,
		BRPCBurstCPUShards:   []int{1, 1},
		BRPCBurstPreconnect:  true,
		HistoryFeatureName:   "click_history",
	}
	if err := config.Validate(); err == nil {
		t.Fatal("expected duplicate CPU shard validation error")
	}
}
