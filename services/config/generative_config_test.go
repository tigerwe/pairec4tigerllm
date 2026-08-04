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
