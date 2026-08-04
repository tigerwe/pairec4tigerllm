package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	recall "pairec4tigerllm/services/recall"
)

type probeRequest struct {
	userID  string
	history [][]int
}

func main() {
	endpoint := flag.String("endpoint", getenv("BRPC_ENDPOINT", "127.0.0.1:18100"), "brpc endpoint host:port")
	service := flag.String("service", getenv("BRPC_SERVICE", "pairec.inference.RecommendService"), "brpc service name")
	method := flag.String("method", getenv("BRPC_METHOD", "recommend"), "health, recommend, or burst")
	userID := flag.String("user_id", getenv("USER_ID", "go_brpc_probe"), "user id for recommend")
	topk := flag.Int("topk", getenvInt("TOPK", 10), "recommend topk")
	requests := flag.Int("requests", getenvInt("REQUESTS", 1), "number of recommend requests")
	concurrency := flag.Int("concurrency", getenvInt("CONCURRENCY", 1), "number of concurrent in-flight requests")
	qps := flag.Int("qps", getenvInt("QPS", 0), "global request start rate limit; 0 disables limiting")
	timeoutMs := flag.Int("timeout_ms", getenvInt("TIMEOUT_MS", 5000), "request timeout in ms")
	maxRetries := flag.Int("max_retries", getenvInt("MAX_RETRIES", 1), "max retries")
	payloadBytes := flag.Int("payload_bytes", getenvInt("PAYLOAD_BYTES", 0), "extra protobuf payload padding bytes")
	pressurePayloadBytes := flag.Int("pressure_payload_bytes", getenvInt("PRESSURE_PAYLOAD_BYTES", -1), "Health payload bytes in burst mode; defaults to payload_bytes")
	businessPayloadBytes := flag.Int("business_payload_bytes", getenvInt("BUSINESS_PAYLOAD_BYTES", 0), "Recommend payload bytes in burst mode")
	burstConcurrency := flag.Int("burst_concurrency", getenvInt("BURST_CONCURRENCY", 0), "total burst lanes; 1 Recommend plus N-1 Health; defaults to concurrency")
	resultJSON := flag.String("result_json", getenv("RESULT_JSON", ""), "write the burst result as JSON")
	historySource := flag.String("history_source", getenv("HISTORY_SOURCE", "synthetic"), "synthetic or user_features")
	uids := flag.String("uids", getenv("UIDS", ""), "comma-separated user ids for user_features history source")
	userFeaturesPath := flag.String("user_features_path", getenv("USER_FEATURES_PATH", "data/user_features.json"), "user_features.json path")
	semanticMapPath := flag.String("semantic_map_path", getenv("SEMANTIC_MAP_PATH", "data/tenrec/processed/semantic_id_map.json"), "semantic_id_map.json path")
	historyMaxLength := flag.Int("history_max_length", getenvInt("HISTORY_MAX_LENGTH", 20), "max history items from user_features")
	varyUserID := flag.Bool("vary_user_id", getenvBool("VARY_USER_ID", true), "append request index to synthetic user_id")
	quiet := flag.Bool("quiet", getenvBool("QUIET", false), "suppress per-request success output")
	reuseConnections := flag.Bool("reuse_connections", getenvBool("REUSE_CONNECTIONS", false), "reuse one brpc TCP connection per worker")
	preconnect := flag.Bool("preconnect", getenvBool("PRECONNECT", false), "pre-establish one TCP session per Health worker, then synchronously release one measured request per connection")
	preconnectHoldMs := flag.Int("preconnect_hold_ms", getenvInt("PRECONNECT_HOLD_MS", 0), "hold fully established Health sessions before request release for observation; excluded from request latency")
	readyFile := flag.String("ready_file", getenv("READY_FILE", ""), "write worker concurrency evidence after the target is reached")
	statsFile := flag.String("stats_file", getenv("STATS_FILE", ""), "append periodic pressure statistics as JSON lines")
	statsIntervalMs := flag.Int("stats_interval_ms", getenvInt("STATS_INTERVAL_MS", 1000), "pressure statistics interval in milliseconds")
	flag.Parse()

	endpoints := splitEndpoints(*endpoint)
	if len(endpoints) == 0 {
		fmt.Fprintln(os.Stderr, "at least one endpoint is required")
		os.Exit(2)
	}

	switch *method {
	case "health":
		if *requests < 1 {
			fmt.Fprintln(os.Stderr, "requests must be positive")
			os.Exit(2)
		}
		if *preconnect && *requests != *concurrency {
			fmt.Fprintln(os.Stderr, "preconnect requires requests to equal concurrency so every connection carries exactly one measured request")
			os.Exit(2)
		}
		if *preconnect && *qps != 0 {
			fmt.Fprintln(os.Stderr, "preconnect requires qps=0 because the measured requests are synchronously released")
			os.Exit(2)
		}
		if *preconnectHoldMs < 0 {
			fmt.Fprintln(os.Stderr, "preconnect_hold_ms must not be negative")
			os.Exit(2)
		}
		if !*preconnect && *preconnectHoldMs > 0 {
			fmt.Fprintln(os.Stderr, "preconnect_hold_ms requires preconnect=true")
			os.Exit(2)
		}
		okCount := 0
		totalStart := time.Now()
		measurementStart := totalStart
		workerCount := normalizedConcurrency(*requests, *concurrency)
		var active int64
		var maxActive int64
		var readyOnce sync.Once
		var pressure pressureCounters
		var statsStop chan struct{}
		clients := make([]*recall.BRPCRecommendClient, len(endpoints))
		for index, target := range endpoints {
			client, err := recall.NewBRPCRecommendClient(
				target,
				*service,
				time.Duration(*timeoutMs)*time.Millisecond,
				*maxRetries,
			)
			if err != nil {
				fmt.Fprintf(os.Stderr, "create client failed endpoint=%s: %v\n", target, err)
				os.Exit(1)
			}
			clients[index] = client
		}
		useSessions := *reuseConnections || *preconnect
		sessions := make([]*recall.BRPCRecommendSession, workerCount)
		if useSessions {
			for worker := range sessions {
				sessions[worker] = clients[worker%len(clients)].NewSession()
				defer sessions[worker].Close()
			}
		}

		connectedSessions := 0
		if *preconnect {
			connectResults, _ := runSynchronizedBurst(workerCount, nil, nil,
				func(index int, _ time.Time) probeResult {
					worker := index - 1
					result := probeResult{
						index:    index,
						endpoint: endpoints[worker%len(endpoints)],
					}
					ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeoutMs)*time.Millisecond)
					result.err = sessions[worker].Connect(ctx)
					cancel()
					return result
				})
			for _, result := range connectResults {
				if result.err != nil {
					fmt.Fprintf(os.Stderr, "preconnect failed index=%d endpoint=%s error=%v\n",
						result.index, result.endpoint, result.err)
					continue
				}
				connectedSessions++
			}
			fmt.Printf("preconnect summary connected_sessions=%d total_sessions=%d\n",
				connectedSessions, workerCount)
			if connectedSessions != workerCount {
				os.Exit(1)
			}
			if *preconnectHoldMs > 0 {
				fmt.Printf("preconnect observation hold_ms=%d connected_sessions=%d\n", *preconnectHoldMs, connectedSessions)
				time.Sleep(time.Duration(*preconnectHoldMs) * time.Millisecond)
			}
			measurementStart = time.Now()
		}

		if *statsFile != "" {
			if *statsIntervalMs <= 0 {
				fmt.Fprintln(os.Stderr, "stats_interval_ms must be positive")
				os.Exit(2)
			}
			_ = os.Remove(*statsFile)
			statsStop = make(chan struct{})
			go reportPressureStats(*statsFile, time.Duration(*statsIntervalMs)*time.Millisecond,
				measurementStart, *payloadBytes, &pressure, &active, &maxActive, statsStop)
		}

		healthCall := func(worker, index int, releaseTime time.Time) (result probeResult) {
			target := endpoints[worker%len(endpoints)]
			callStarted := time.Now()
			result.index = index
			result.endpoint = target
			if !releaseTime.IsZero() {
				result.startOffsetUs = callStarted.Sub(releaseTime).Microseconds()
			}
			defer func() {
				atomic.AddInt64(&pressure.calls, 1)
				atomic.AddInt64(&pressure.latencyUs, time.Since(callStarted).Microseconds())
				if result.err != nil {
					atomic.AddInt64(&pressure.errors, 1)
				} else {
					atomic.AddInt64(&pressure.success, 1)
					atomic.AddInt64(&pressure.bytes, int64(*payloadBytes))
				}
			}()
			currentActive := atomic.AddInt64(&active, 1)
			updateAtomicMaximum(&maxActive, currentActive)
			if currentActive >= int64(workerCount) && *readyFile != "" {
				readyOnce.Do(func() {
					armed := 0
					if *preconnect {
						armed = workerCount
					}
					content := fmt.Sprintf("workers=%d armed_workers=%d max_active=%d endpoints=%d payload_bytes=%d reuse_connections=%t preconnect=%t synchronized_start=%t\n",
						workerCount, armed, atomic.LoadInt64(&maxActive), len(endpoints), *payloadBytes,
						useSessions, *preconnect, *preconnect)
					if err := os.WriteFile(*readyFile, []byte(content), 0o644); err != nil {
						fmt.Fprintf(os.Stderr, "write ready file failed: %v\n", err)
					}
				})
			}
			defer atomic.AddInt64(&active, -1)
			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeoutMs)*time.Millisecond)
			started := time.Now()
			if useSessions {
				resp, err := sessions[worker].HealthCheckWithPayload(ctx, *payloadBytes)
				cancel()
				result.latencyUs = time.Since(started).Microseconds()
				result.latencyMs = result.latencyUs / 1000
				if err != nil {
					result.err = err
					return result
				}
				result.code = resp.Code
				result.status = resp.Status
				result.backend = resp.Backend
				return result
			}
			resp, err := clients[worker%len(clients)].HealthCheckWithPayload(ctx, *payloadBytes)
			cancel()
			result.latencyUs = time.Since(started).Microseconds()
			result.latencyMs = result.latencyUs / 1000
			if err != nil {
				result.err = err
				return result
			}
			result.code = resp.Code
			result.status = resp.Status
			result.backend = resp.Backend
			return result
		}

		var results []probeResult
		requestTotalMs := 0.0
		requestStartSkewUs := int64(0)
		armedWorkers := 0
		if *preconnect {
			armedWorkers = workerCount
			var releasedAt time.Time
			var releasedActive int64
			var releasedMaxActive int64
			results, releasedAt = runSynchronizedBurst(workerCount, &releasedActive, &releasedMaxActive,
				func(index int, releaseTime time.Time) probeResult {
					return healthCall(index-1, index, releaseTime)
				})
			requestTotalMs = float64(time.Since(releasedAt).Microseconds()) / 1000
			startOffsets := make([]int64, 0, len(results))
			for _, result := range results {
				startOffsets = append(startOffsets, result.startOffsetUs)
			}
			requestStartSkewUs = spreadInt64(startOffsets)
		} else {
			results = runIndexedWithWorker(*requests, *concurrency, *qps,
				func(worker, index int) probeResult {
					return healthCall(worker, index, time.Time{})
				})
			requestTotalMs = float64(time.Since(measurementStart).Microseconds()) / 1000
		}
		if *statsFile != "" {
			close(statsStop)
			writePressureStats(*statsFile, measurementStart, *payloadBytes, &pressure, &active, &maxActive, true)
		}
		for _, result := range results {
			if result.err != nil {
				fmt.Fprintf(os.Stderr, "health failed index=%d endpoint=%s payload_bytes=%d latency_ms=%d error=%v\n",
					result.index, result.endpoint, *payloadBytes, result.latencyMs, result.err)
				continue
			}
			okCount++
			if !*quiet {
				fmt.Printf("health ok index=%d endpoint=%s payload_bytes=%d latency_ms=%d code=%d status=%s backend=%s\n",
					result.index, result.endpoint, *payloadBytes, result.latencyMs, result.code, result.status, result.backend)
			}
		}
		totalElapsed := time.Since(totalStart).Milliseconds()
		fmt.Printf("summary ok=%d total=%d total_ms=%d request_total_ms=%.3f payload_bytes=%d concurrency=%d armed_workers=%d max_active=%d start_skew_us=%d qps=%d reuse_connections=%t preconnect=%t preconnect_hold_ms=%d connected_sessions=%d synchronized_start=%t\n",
			okCount, *requests, totalElapsed, requestTotalMs, *payloadBytes, workerCount, armedWorkers,
			atomic.LoadInt64(&maxActive), requestStartSkewUs, *qps, useSessions, *preconnect, *preconnectHoldMs,
			connectedSessions, *preconnect)
		if okCount != *requests {
			os.Exit(1)
		}
	case "burst":
		if len(endpoints) != 1 {
			fmt.Fprintln(os.Stderr, "burst method requires exactly one endpoint")
			os.Exit(2)
		}
		totalLanes := *burstConcurrency
		if totalLanes == 0 {
			totalLanes = *concurrency
		}
		if totalLanes < 1 {
			fmt.Fprintln(os.Stderr, "burst_concurrency must be positive")
			os.Exit(2)
		}
		pressureBytes := *pressurePayloadBytes
		if pressureBytes < 0 {
			pressureBytes = *payloadBytes
		}
		if pressureBytes < 0 || *businessPayloadBytes < 0 {
			fmt.Fprintln(os.Stderr, "payload byte counts must not be negative")
			os.Exit(2)
		}

		selectedUIDs := *uids
		if *historySource == "user_features" && strings.TrimSpace(selectedUIDs) == "" {
			selectedUIDs = *userID
		}
		requestPlans, err := buildProbeRequests(*historySource, *userID, selectedUIDs,
			*userFeaturesPath, *semanticMapPath, *historyMaxLength, false, 1)
		if err != nil {
			fmt.Fprintf(os.Stderr, "build burst business request failed: %v\n", err)
			os.Exit(1)
		}
		client, err := recall.NewBRPCRecommendClient(
			endpoints[0],
			*service,
			time.Duration(*timeoutMs)*time.Millisecond,
			*maxRetries,
		)
		if err != nil {
			fmt.Fprintf(os.Stderr, "create client failed endpoint=%s: %v\n", endpoints[0], err)
			os.Exit(1)
		}
		sessions := make([]*recall.BRPCRecommendSession, totalLanes)
		connectedSessions := 0
		preconnectMs := 0.0
		if *preconnect {
			for lane := range sessions {
				sessions[lane] = client.NewSession()
				defer sessions[lane].Close()
			}
			connectStarted := time.Now()
			connectResults, _ := runSynchronizedBurst(totalLanes, nil, nil,
				func(index int, _ time.Time) probeResult {
					result := probeResult{index: index, endpoint: endpoints[0]}
					ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeoutMs)*time.Millisecond)
					result.err = sessions[index-1].Connect(ctx)
					cancel()
					return result
				})
			preconnectMs = float64(time.Since(connectStarted).Microseconds()) / 1000
			for _, result := range connectResults {
				if result.err != nil {
					fmt.Fprintf(os.Stderr, "burst preconnect failed index=%d endpoint=%s error=%v\n",
						result.index, result.endpoint, result.err)
					continue
				}
				connectedSessions++
			}
			fmt.Printf("burst preconnect summary connected_sessions=%d total_sessions=%d preconnect_ms=%.3f\n",
				connectedSessions, totalLanes, preconnectMs)
			if connectedSessions != totalLanes {
				os.Exit(1)
			}
		}

		requestID := fmt.Sprintf("go-brpc-burst-%d", time.Now().UnixNano())
		businessPlan := requestPlans[0]
		var active int64
		var maxActive int64
		totalStarted := time.Now()
		results, releasedAt := runSynchronizedBurst(totalLanes, &active, &maxActive,
			func(index int, releaseTime time.Time) probeResult {
				callStarted := time.Now()
				result := probeResult{
					index:         index,
					endpoint:      endpoints[0],
					startOffsetUs: callStarted.Sub(releaseTime).Microseconds(),
				}
				ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeoutMs)*time.Millisecond)
				defer cancel()
				if index == 1 {
					result.role = "business"
					result.requestID = requestID
					req := &recall.RecommendRequest{
						UserID:              businessPlan.userID,
						History:             businessPlan.history,
						Topk:                *topk,
						Temperature:         1.0,
						BeamWidth:           1,
						PayloadPaddingBytes: *businessPayloadBytes,
					}
					var resp *recall.RecommendResponse
					var callErr error
					if *preconnect {
						resp, callErr = sessions[index-1].Recommend(ctx, req, requestID)
					} else {
						resp, callErr = client.Recommend(ctx, req, requestID)
					}
					result.latencyUs = time.Since(callStarted).Microseconds()
					result.latencyMs = result.latencyUs / 1000
					if callErr != nil {
						result.err = callErr
						return result
					}
					result.code = resp.Code
					result.userID = resp.UserID
					result.items = len(resp.Recommendations)
					result.inferenceMs = resp.InferenceTimeMs
					if resp.Trace != nil {
						result.backend = resp.Trace.Backend
						result.runnerGenerateMs = resp.Trace.RunnerGenerateMs
						result.wrapperTotalMs = resp.Trace.WrapperTotalMs
						result.wrapperBackendRPCMs = resp.Trace.WrapperBackendRPCMs
						result.wrapperOverheadMs = resp.Trace.WrapperOverheadMs
						result.wrapperHealthAtStart = resp.Trace.WrapperHealthAtStart
						result.wrapperMaxActiveHealth = resp.Trace.WrapperMaxActiveHealth
						result.wrapperMaxActiveTotal = resp.Trace.WrapperMaxActiveTotal
						result.wrapperBackendBRPCMs = resp.Trace.WrapperBackendBRPCMs
					}
					if result.items == 0 {
						result.err = fmt.Errorf("Recommend returned no items")
					}
					return result
				}

				result.role = "pressure"
				if *preconnect {
					resp, callErr := sessions[index-1].HealthCheckWithPayload(ctx, pressureBytes)
					result.latencyUs = time.Since(callStarted).Microseconds()
					result.latencyMs = result.latencyUs / 1000
					if callErr != nil {
						result.err = callErr
						return result
					}
					result.code = resp.Code
					result.status = resp.Status
					result.backend = resp.Backend
				} else {
					resp, callErr := client.HealthCheckWithPayload(ctx, pressureBytes)
					result.latencyUs = time.Since(callStarted).Microseconds()
					result.latencyMs = result.latencyUs / 1000
					if callErr != nil {
						result.err = callErr
						return result
					}
					result.code = resp.Code
					result.status = resp.Status
					result.backend = resp.Backend
				}
				if result.code != 200 {
					result.err = fmt.Errorf("Health returned code %d", result.code)
				}
				return result
			})

		business := results[0]
		pressureOK := 0
		pressureLatencies := make([]int64, 0, totalLanes-1)
		startOffsets := make([]int64, 0, totalLanes)
		for _, result := range results {
			startOffsets = append(startOffsets, result.startOffsetUs)
			if result.role == "business" {
				if result.err != nil {
					fmt.Fprintf(os.Stderr, "burst business failed endpoint=%s request_id=%s client_wall_ms=%.3f error=%v\n",
						result.endpoint, result.requestID, float64(result.latencyUs)/1000, result.err)
				} else {
					fmt.Printf("burst business ok endpoint=%s request_id=%s client_wall_ms=%.3f inference_ms=%.3f runner_generate_ms=%.3f brpc_delta_ms=%.3f front_brpc_ms=%.3f wrapper_total_ms=%.3f wrapper_backend_rpc_ms=%.3f wrapper_overhead_ms=%.3f wrapper_backend_brpc_ms=%.3f wrapper_health_at_start=%d wrapper_max_active_health=%d wrapper_max_active_total=%d items=%d backend=%s\n",
						result.endpoint, result.requestID, float64(result.latencyUs)/1000, result.inferenceMs,
						result.runnerGenerateMs, float64(result.latencyUs)/1000-result.inferenceMs,
						result.frontBRPCMs(), result.wrapperTotalMs,
						result.wrapperBackendRPCMs, result.wrapperOverheadMs, result.wrapperBackendBRPCMs,
						result.wrapperHealthAtStart, result.wrapperMaxActiveHealth, result.wrapperMaxActiveTotal,
						result.items, result.backend)
				}
				continue
			}
			if result.err != nil {
				fmt.Fprintf(os.Stderr, "burst pressure failed index=%d endpoint=%s payload_bytes=%d latency_ms=%.3f error=%v\n",
					result.index, result.endpoint, pressureBytes, float64(result.latencyUs)/1000, result.err)
				continue
			}
			pressureOK++
			pressureLatencies = append(pressureLatencies, result.latencyUs)
			if !*quiet {
				fmt.Printf("burst pressure ok index=%d endpoint=%s payload_bytes=%d latency_ms=%.3f code=%d status=%s backend=%s\n",
					result.index, result.endpoint, pressureBytes, float64(result.latencyUs)/1000,
					result.code, result.status, result.backend)
			}
		}

		businessError := ""
		if business.err != nil {
			businessError = business.err.Error()
		}
		pressureTotal := totalLanes - 1
		pressureSuccessRate := 1.0
		if pressureTotal > 0 {
			pressureSuccessRate = float64(pressureOK) / float64(pressureTotal)
		}
		summary := burstSummary{
			Event:                "brpc_burst_result",
			Endpoint:             endpoints[0],
			BurstConcurrency:     totalLanes,
			ArmedWorkers:         totalLanes,
			MaxActiveWorkers:     atomic.LoadInt64(&maxActive),
			StartSkewUs:          spreadInt64(startOffsets),
			TotalMs:              float64(time.Since(totalStarted).Microseconds()) / 1000,
			PressurePayloadBytes: pressureBytes,
			PressureRequests:     pressureTotal,
			PressureSuccess:      pressureOK,
			PressureErrors:       pressureTotal - pressureOK,
			PressureSuccessRate:  pressureSuccessRate,
			PressureLatencyAvgMs: averageMicroseconds(pressureLatencies),
			PressureLatencyP95Ms: percentileMicroseconds(pressureLatencies, 0.95),
			ReleasedAtUnixNano:   releasedAt.UnixNano(),
			BusinessRequestID:    business.requestID,
			BusinessSuccess:      business.err == nil,
			BusinessError:        businessError,
			BusinessClientWallMs: float64(business.latencyUs) / 1000,
			BusinessInferenceMs:  business.inferenceMs,
			BusinessRunnerMs:     business.runnerGenerateMs,
			BusinessBRPCDeltaMs:  float64(business.latencyUs)/1000 - business.inferenceMs,
			BusinessItems:        business.items,
			BusinessCode:         business.code,
			BusinessUserID:       business.userID,
			BusinessBackend:      business.backend,
			BusinessFrontBRPCMs:  business.frontBRPCMs(),
			WrapperTotalMs:       business.wrapperTotalMs,
			WrapperBackendRPCMs:  business.wrapperBackendRPCMs,
			WrapperOverheadMs:    business.wrapperOverheadMs,
			WrapperHealthAtStart: business.wrapperHealthAtStart,
			WrapperMaxHealth:     business.wrapperMaxActiveHealth,
			WrapperMaxTotal:      business.wrapperMaxActiveTotal,
			WrapperBackendBRPCMs: business.wrapperBackendBRPCMs,
			Preconnect:           *preconnect,
			ConnectedSessions:    connectedSessions,
			PreconnectMs:         preconnectMs,
		}
		encoded, err := json.Marshal(summary)
		if err != nil {
			fmt.Fprintf(os.Stderr, "marshal burst result failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(string(encoded))
		if *resultJSON != "" {
			if err := writeJSONFile(*resultJSON, encoded); err != nil {
				fmt.Fprintf(os.Stderr, "write burst result failed: %v\n", err)
				os.Exit(1)
			}
		}
		fmt.Printf("burst summary business_ok=%t pressure_ok=%d pressure_total=%d armed_workers=%d max_active=%d start_skew_us=%d preconnect=%t connected_sessions=%d total_ms=%.3f\n",
			summary.BusinessSuccess, pressureOK, pressureTotal, totalLanes, summary.MaxActiveWorkers,
			summary.StartSkewUs, summary.Preconnect, summary.ConnectedSessions, summary.TotalMs)
		if !summary.BusinessSuccess || pressureOK != pressureTotal {
			os.Exit(1)
		}
	case "recommend":
		if len(endpoints) != 1 {
			fmt.Fprintln(os.Stderr, "recommend method requires exactly one endpoint")
			os.Exit(2)
		}
		client, err := recall.NewBRPCRecommendClient(
			endpoints[0],
			*service,
			time.Duration(*timeoutMs)*time.Millisecond,
			*maxRetries,
		)
		if err != nil {
			fmt.Fprintf(os.Stderr, "create client failed endpoint=%s: %v\n", endpoints[0], err)
			os.Exit(1)
		}
		if *requests < 1 {
			fmt.Fprintln(os.Stderr, "requests must be positive")
			os.Exit(2)
		}
		requestPlans, err := buildProbeRequests(*historySource, *userID, *uids, *userFeaturesPath, *semanticMapPath, *historyMaxLength, *varyUserID, *requests)
		if err != nil {
			fmt.Fprintf(os.Stderr, "build probe requests failed: %v\n", err)
			os.Exit(1)
		}
		okCount := 0
		totalStart := time.Now()
		results := runIndexed(*requests, *concurrency, *qps, func(index int) probeResult {
			plan := requestPlans[(index-1)%len(requestPlans)]
			requestID := fmt.Sprintf("go-brpc-probe-%d", index)
			req := &recall.RecommendRequest{
				UserID:              plan.userID,
				History:             plan.history,
				Topk:                *topk,
				Temperature:         1.0,
				BeamWidth:           1,
				PayloadPaddingBytes: *payloadBytes,
			}
			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeoutMs)*time.Millisecond)
			started := time.Now()
			resp, err := client.Recommend(ctx, req, requestID)
			cancel()
			elapsed := time.Since(started).Milliseconds()
			if err != nil {
				return probeResult{index: index, requestID: requestID, latencyMs: elapsed, err: err}
			}
			inferenceMs := resp.InferenceTimeMs
			backend := ""
			if resp.Trace != nil {
				backend = resp.Trace.Backend
			}
			return probeResult{
				index:       index,
				requestID:   requestID,
				latencyMs:   elapsed,
				code:        resp.Code,
				userID:      resp.UserID,
				items:       len(resp.Recommendations),
				inferenceMs: inferenceMs,
				backend:     backend,
			}
		})
		for _, result := range results {
			if result.err != nil {
				fmt.Fprintf(os.Stderr, "recommend failed index=%d endpoint=%s request_id=%s latency_ms=%d error=%v\n",
					result.index, *endpoint, result.requestID, result.latencyMs, result.err)
				continue
			}
			okCount++
			if !*quiet {
				fmt.Printf("recommend ok index=%d endpoint=%s request_id=%s payload_bytes=%d latency_ms=%d code=%d user_id=%s items=%d inference_ms=%.0f backend=%s\n",
					result.index, *endpoint, result.requestID, *payloadBytes, result.latencyMs, result.code, result.userID, result.items, result.inferenceMs, result.backend)
			}
		}
		totalElapsed := time.Since(totalStart).Milliseconds()
		fmt.Printf("summary ok=%d total=%d total_ms=%d payload_bytes=%d concurrency=%d qps=%d\n",
			okCount, *requests, totalElapsed, *payloadBytes, normalizedConcurrency(*requests, *concurrency), *qps)
		if okCount != *requests {
			os.Exit(1)
		}
	default:
		fmt.Fprintf(os.Stderr, "unknown method %q\n", *method)
		os.Exit(2)
	}
}

type probeResult struct {
	index                  int
	endpoint               string
	role                   string
	requestID              string
	latencyMs              int64
	latencyUs              int64
	startOffsetUs          int64
	err                    error
	code                   int
	status                 string
	backend                string
	userID                 string
	items                  int
	inferenceMs            float64
	runnerGenerateMs       float64
	wrapperTotalMs         float64
	wrapperBackendRPCMs    float64
	wrapperOverheadMs      float64
	wrapperHealthAtStart   int64
	wrapperMaxActiveHealth int64
	wrapperMaxActiveTotal  int64
	wrapperBackendBRPCMs   float64
}

func (result probeResult) frontBRPCMs() float64 {
	clientWallMs := float64(result.latencyUs) / 1000
	if result.wrapperTotalMs > 0 {
		return clientWallMs - result.wrapperTotalMs
	}
	return clientWallMs - result.inferenceMs
}

type burstSummary struct {
	Event                string  `json:"event"`
	Endpoint             string  `json:"endpoint"`
	BurstConcurrency     int     `json:"burst_concurrency"`
	ArmedWorkers         int     `json:"armed_workers"`
	MaxActiveWorkers     int64   `json:"max_active_workers"`
	StartSkewUs          int64   `json:"start_skew_us"`
	TotalMs              float64 `json:"total_ms"`
	PressurePayloadBytes int     `json:"pressure_payload_bytes"`
	PressureRequests     int     `json:"pressure_requests"`
	PressureSuccess      int     `json:"pressure_success"`
	PressureErrors       int     `json:"pressure_errors"`
	PressureSuccessRate  float64 `json:"pressure_success_rate"`
	PressureLatencyAvgMs float64 `json:"pressure_latency_avg_ms"`
	PressureLatencyP95Ms float64 `json:"pressure_latency_p95_ms"`
	ReleasedAtUnixNano   int64   `json:"released_at_unix_nano"`
	BusinessRequestID    string  `json:"business_request_id"`
	BusinessSuccess      bool    `json:"business_success"`
	BusinessError        string  `json:"business_error,omitempty"`
	BusinessClientWallMs float64 `json:"business_client_wall_ms"`
	BusinessInferenceMs  float64 `json:"business_inference_ms"`
	BusinessRunnerMs     float64 `json:"business_runner_generate_ms"`
	BusinessBRPCDeltaMs  float64 `json:"business_brpc_delta_ms"`
	BusinessItems        int     `json:"business_items"`
	BusinessCode         int     `json:"business_code"`
	BusinessUserID       string  `json:"business_user_id"`
	BusinessBackend      string  `json:"business_backend"`
	BusinessFrontBRPCMs  float64 `json:"business_front_brpc_ms"`
	WrapperTotalMs       float64 `json:"wrapper_total_ms"`
	WrapperBackendRPCMs  float64 `json:"wrapper_backend_rpc_ms"`
	WrapperOverheadMs    float64 `json:"wrapper_overhead_ms"`
	WrapperHealthAtStart int64   `json:"wrapper_active_health_at_start"`
	WrapperMaxHealth     int64   `json:"wrapper_max_active_health"`
	WrapperMaxTotal      int64   `json:"wrapper_max_active_total"`
	WrapperBackendBRPCMs float64 `json:"wrapper_backend_brpc_ms"`
	Preconnect           bool    `json:"preconnect"`
	ConnectedSessions    int     `json:"connected_sessions"`
	PreconnectMs         float64 `json:"preconnect_ms"`
}

func splitEndpoints(value string) []string {
	var endpoints []string
	for _, endpoint := range strings.Split(value, ",") {
		endpoint = strings.TrimSpace(endpoint)
		if endpoint != "" {
			endpoints = append(endpoints, endpoint)
		}
	}
	return endpoints
}

type pressureCounters struct {
	calls     int64
	success   int64
	errors    int64
	bytes     int64
	latencyUs int64
}

func reportPressureStats(path string, interval time.Duration, started time.Time, payloadBytes int,
	counters *pressureCounters, active, maxActive *int64, stop <-chan struct{}) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			writePressureStats(path, started, payloadBytes, counters, active, maxActive, false)
		case <-stop:
			return
		}
	}
}

func writePressureStats(path string, started time.Time, payloadBytes int, counters *pressureCounters,
	active, maxActive *int64, final bool) {
	elapsed := time.Since(started).Seconds()
	calls := atomic.LoadInt64(&counters.calls)
	success := atomic.LoadInt64(&counters.success)
	errors := atomic.LoadInt64(&counters.errors)
	bytes := atomic.LoadInt64(&counters.bytes)
	latencyUs := atomic.LoadInt64(&counters.latencyUs)
	qps := 0.0
	gbps := 0.0
	avgMs := 0.0
	if elapsed > 0 {
		qps = float64(calls) / elapsed
		gbps = float64(bytes) * 8 / elapsed / 1e9
	}
	if calls > 0 {
		avgMs = float64(latencyUs) / float64(calls) / 1000
	}
	line := fmt.Sprintf("{\"event\":\"brpc_pressure_stats\",\"final\":%t,\"elapsed_s\":%.6f,\"payload_bytes\":%d,\"calls\":%d,\"success\":%d,\"errors\":%d,\"bytes\":%d,\"qps\":%.6f,\"gbps\":%.6f,\"avg_ms\":%.6f,\"active_calls\":%d,\"max_active_calls\":%d}\n",
		final, elapsed, payloadBytes, calls, success, errors, bytes, qps, gbps, avgMs,
		atomic.LoadInt64(active), atomic.LoadInt64(maxActive))
	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open stats file failed: %v\n", err)
		return
	}
	_, _ = file.WriteString(line)
	_ = file.Close()
}

func runIndexed(total int, concurrency int, qps int, fn func(index int) probeResult) []probeResult {
	return runIndexedWithWorker(total, concurrency, qps, func(_ int, index int) probeResult {
		return fn(index)
	})
}

func runIndexedWithWorker(total int, concurrency int, qps int, fn func(worker, index int) probeResult) []probeResult {
	results := make([]probeResult, total)
	concurrency = normalizedConcurrency(total, concurrency)
	jobs := make(chan int)
	var wg sync.WaitGroup
	for worker := 0; worker < concurrency; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for index := range jobs {
				results[index-1] = fn(worker, index)
			}
		}(worker)
	}
	var ticker *time.Ticker
	if qps > 0 {
		interval := time.Second / time.Duration(qps)
		if interval < time.Nanosecond {
			interval = time.Nanosecond
		}
		ticker = time.NewTicker(interval)
		defer ticker.Stop()
	}
	for index := 1; index <= total; index++ {
		if ticker != nil {
			<-ticker.C
		}
		jobs <- index
	}
	close(jobs)
	wg.Wait()
	return results
}

// runSynchronizedBurst stages every lane before releasing any RPC. When
// counters are provided, workers are counted from release until the call returns.
func runSynchronizedBurst(total int, active, maxActive *int64,
	fn func(index int, releasedAt time.Time) probeResult) ([]probeResult, time.Time) {
	results := make([]probeResult, total)
	var armed sync.WaitGroup
	var completed sync.WaitGroup
	armed.Add(total)
	completed.Add(total)
	callGate := make(chan struct{})
	var releasedAt time.Time

	for index := 1; index <= total; index++ {
		go func(index int) {
			defer completed.Done()
			armed.Done()
			<-callGate
			if active != nil {
				currentActive := atomic.AddInt64(active, 1)
				if maxActive != nil {
					updateAtomicMaximum(maxActive, currentActive)
				}
				defer atomic.AddInt64(active, -1)
			}
			results[index-1] = fn(index, releasedAt)
		}(index)
	}

	armed.Wait()
	releasedAt = time.Now()
	close(callGate)
	completed.Wait()
	return results, releasedAt
}

func normalizedConcurrency(total int, concurrency int) int {
	if concurrency < 1 {
		concurrency = 1
	}
	if total > 0 && concurrency > total {
		concurrency = total
	}
	return concurrency
}

func updateAtomicMaximum(target *int64, value int64) {
	for {
		current := atomic.LoadInt64(target)
		if current >= value || atomic.CompareAndSwapInt64(target, current, value) {
			return
		}
	}
}

func averageMicroseconds(values []int64) float64 {
	if len(values) == 0 {
		return 0
	}
	var total int64
	for _, value := range values {
		total += value
	}
	return float64(total) / float64(len(values)) / 1000
}

func percentileMicroseconds(values []int64, quantile float64) float64 {
	if len(values) == 0 {
		return 0
	}
	ordered := append([]int64(nil), values...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	if len(ordered) == 1 {
		return float64(ordered[0]) / 1000
	}
	rank := float64(len(ordered)-1) * quantile
	lower := int(math.Floor(rank))
	upper := int(math.Ceil(rank))
	value := float64(ordered[lower])
	if lower != upper {
		value += (float64(ordered[upper]) - value) * (rank - float64(lower))
	}
	return value / 1000
}

func spreadInt64(values []int64) int64 {
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

func writeJSONFile(path string, encoded []byte) error {
	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, append(encoded, '\n'), 0o644)
}

func buildProbeRequests(historySource, baseUserID, uids, userFeaturesPath, semanticMapPath string, historyMaxLength int, varyUserID bool, requestCount int) ([]probeRequest, error) {
	switch historySource {
	case "synthetic":
		plans := make([]probeRequest, 0, requestCount)
		for index := 1; index <= requestCount; index++ {
			id := baseUserID
			if varyUserID {
				id = fmt.Sprintf("%s_%d", baseUserID, index)
			}
			plans = append(plans, probeRequest{
				userID: id,
				history: [][]int{
					{169, 41, 0, 0},
					{20, 53, 0, 0},
					{80, 201, 0, 0},
				},
			})
		}
		return plans, nil
	case "user_features":
		return buildUserFeatureProbeRequests(uids, userFeaturesPath, semanticMapPath, historyMaxLength)
	default:
		return nil, fmt.Errorf("unknown history_source %q", historySource)
	}
}

func buildUserFeatureProbeRequests(uids, userFeaturesPath, semanticMapPath string, historyMaxLength int) ([]probeRequest, error) {
	data, err := os.ReadFile(userFeaturesPath)
	if err != nil {
		return nil, fmt.Errorf("read user features %s: %w", userFeaturesPath, err)
	}
	var features map[string]map[string]any
	if err := json.Unmarshal(data, &features); err != nil {
		return nil, fmt.Errorf("parse user features %s: %w", userFeaturesPath, err)
	}

	semanticMap, err := loadSemanticMap(semanticMapPath)
	if err != nil {
		return nil, err
	}

	selectedUIDs := splitCSV(uids)
	if len(selectedUIDs) == 0 {
		selectedUIDs = make([]string, 0, len(features))
		for uid := range features {
			selectedUIDs = append(selectedUIDs, uid)
		}
		sort.Strings(selectedUIDs)
	}

	plans := make([]probeRequest, 0, len(selectedUIDs))
	for _, uid := range selectedUIDs {
		record, ok := features[uid]
		if !ok {
			continue
		}
		rawHistory, ok := record["click_history"]
		if !ok {
			continue
		}
		itemIDs := parseItemHistory(rawHistory)
		if historyMaxLength > 0 && len(itemIDs) > historyMaxLength {
			itemIDs = itemIDs[len(itemIDs)-historyMaxLength:]
		}
		history := convertToSemanticIDs(itemIDs, semanticMap)
		if len(history) == 0 {
			continue
		}
		plans = append(plans, probeRequest{userID: uid, history: history})
	}
	if len(plans) == 0 {
		return nil, fmt.Errorf("no usable user histories in %s", userFeaturesPath)
	}
	return plans, nil
}

func loadSemanticMap(path string) (map[int][]int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read semantic map %s: %w", path, err)
	}
	var raw map[string][]int
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse semantic map %s: %w", path, err)
	}
	out := make(map[int][]int, len(raw))
	for key, value := range raw {
		itemID, err := strconv.Atoi(key)
		if err != nil {
			continue
		}
		out[itemID] = value
	}
	return out, nil
}

func splitCSV(value string) []string {
	if value == "" {
		return nil
	}
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func parseItemHistory(raw any) []int {
	switch value := raw.(type) {
	case string:
		return parseHistoryString(value)
	case []any:
		out := make([]int, 0, len(value))
		for _, item := range value {
			switch typed := item.(type) {
			case string:
				if parsed, err := strconv.Atoi(strings.TrimSpace(typed)); err == nil {
					out = append(out, parsed)
				}
			case float64:
				out = append(out, int(typed))
			}
		}
		return out
	default:
		return nil
	}
}

func parseHistoryString(value string) []int {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}
	if strings.HasPrefix(value, "[") {
		var raw []any
		if err := json.Unmarshal([]byte(value), &raw); err == nil {
			return parseItemHistory(raw)
		}
	}
	parts := strings.Split(value, ",")
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(strings.Trim(part, `"'[]`))
		if part == "" {
			continue
		}
		if parsed, err := strconv.Atoi(part); err == nil {
			out = append(out, parsed)
		}
	}
	return out
}

func convertToSemanticIDs(itemIDs []int, semanticMap map[int][]int) [][]int {
	history := make([][]int, 0, len(itemIDs))
	for _, itemID := range itemIDs {
		if values, ok := semanticMap[itemID]; ok {
			history = append(history, values)
			continue
		}
		history = append(history, []int{
			itemID % 256,
			(itemID / 256) % 256,
			(itemID / 65536) % 256,
			(itemID / 16777216) % 256,
		})
	}
	return history
}

func getenv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func getenvInt(key string, fallback int) int {
	var value int
	if _, err := fmt.Sscanf(os.Getenv(key), "%d", &value); err == nil {
		return value
	}
	return fallback
}

func getenvBool(key string, fallback bool) bool {
	value := strings.ToLower(strings.TrimSpace(os.Getenv(key)))
	switch value {
	case "1", "true", "yes", "y", "on":
		return true
	case "0", "false", "no", "n", "off":
		return false
	default:
		return fallback
	}
}
