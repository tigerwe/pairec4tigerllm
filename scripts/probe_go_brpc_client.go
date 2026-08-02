package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
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
	method := flag.String("method", getenv("BRPC_METHOD", "recommend"), "health or recommend")
	userID := flag.String("user_id", getenv("USER_ID", "go_brpc_probe"), "user id for recommend")
	topk := flag.Int("topk", getenvInt("TOPK", 10), "recommend topk")
	requests := flag.Int("requests", getenvInt("REQUESTS", 1), "number of recommend requests")
	concurrency := flag.Int("concurrency", getenvInt("CONCURRENCY", 1), "number of concurrent in-flight requests")
	qps := flag.Int("qps", getenvInt("QPS", 0), "global request start rate limit; 0 disables limiting")
	timeoutMs := flag.Int("timeout_ms", getenvInt("TIMEOUT_MS", 5000), "request timeout in ms")
	maxRetries := flag.Int("max_retries", getenvInt("MAX_RETRIES", 1), "max retries")
	payloadBytes := flag.Int("payload_bytes", getenvInt("PAYLOAD_BYTES", 0), "extra protobuf payload padding bytes")
	historySource := flag.String("history_source", getenv("HISTORY_SOURCE", "synthetic"), "synthetic or user_features")
	uids := flag.String("uids", getenv("UIDS", ""), "comma-separated user ids for user_features history source")
	userFeaturesPath := flag.String("user_features_path", getenv("USER_FEATURES_PATH", "data/user_features.json"), "user_features.json path")
	semanticMapPath := flag.String("semantic_map_path", getenv("SEMANTIC_MAP_PATH", "data/tenrec/processed/semantic_id_map.json"), "semantic_id_map.json path")
	historyMaxLength := flag.Int("history_max_length", getenvInt("HISTORY_MAX_LENGTH", 20), "max history items from user_features")
	varyUserID := flag.Bool("vary_user_id", getenvBool("VARY_USER_ID", true), "append request index to synthetic user_id")
	quiet := flag.Bool("quiet", getenvBool("QUIET", false), "suppress per-request success output")
	reuseConnections := flag.Bool("reuse_connections", getenvBool("REUSE_CONNECTIONS", false), "reuse one brpc TCP connection per worker")
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
		okCount := 0
		totalStart := time.Now()
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
		sessions := make([]*recall.BRPCRecommendSession, workerCount)
		if *reuseConnections {
			for worker := range sessions {
				sessions[worker] = clients[worker%len(clients)].NewSession()
				defer sessions[worker].Close()
			}
		}
		if *statsFile != "" {
			if *statsIntervalMs <= 0 {
				fmt.Fprintln(os.Stderr, "stats_interval_ms must be positive")
				os.Exit(2)
			}
			_ = os.Remove(*statsFile)
			statsStop = make(chan struct{})
			go reportPressureStats(*statsFile, time.Duration(*statsIntervalMs)*time.Millisecond,
				totalStart, *payloadBytes, &pressure, &active, &maxActive, statsStop)
		}
		results := runIndexedWithWorker(*requests, *concurrency, *qps, func(worker, index int) (result probeResult) {
			target := endpoints[worker%len(endpoints)]
			callStarted := time.Now()
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
					content := fmt.Sprintf("workers=%d max_active=%d endpoints=%d payload_bytes=%d reuse_connections=%t\n",
						workerCount, atomic.LoadInt64(&maxActive), len(endpoints), *payloadBytes, *reuseConnections)
					if err := os.WriteFile(*readyFile, []byte(content), 0o644); err != nil {
						fmt.Fprintf(os.Stderr, "write ready file failed: %v\n", err)
					}
				})
			}
			defer atomic.AddInt64(&active, -1)
			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeoutMs)*time.Millisecond)
			started := time.Now()
			if *reuseConnections {
				resp, err := sessions[worker].HealthCheckWithPayload(ctx, *payloadBytes)
				cancel()
				elapsed := time.Since(started).Milliseconds()
				if err != nil {
					return probeResult{index: index, endpoint: target, latencyMs: elapsed, err: err}
				}
				return probeResult{
					index:     index,
					endpoint:  target,
					latencyMs: elapsed,
					code:      resp.Code,
					status:    resp.Status,
					backend:   resp.Backend,
				}
			}
			resp, err := clients[worker%len(clients)].HealthCheckWithPayload(ctx, *payloadBytes)
			cancel()
			elapsed := time.Since(started).Milliseconds()
			if err != nil {
				return probeResult{index: index, endpoint: target, latencyMs: elapsed, err: err}
			}
			return probeResult{
				index:     index,
				endpoint:  target,
				latencyMs: elapsed,
				code:      resp.Code,
				status:    resp.Status,
				backend:   resp.Backend,
			}
		})
		if *statsFile != "" {
			close(statsStop)
			writePressureStats(*statsFile, totalStart, *payloadBytes, &pressure, &active, &maxActive, true)
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
		fmt.Printf("summary ok=%d total=%d total_ms=%d payload_bytes=%d concurrency=%d max_active=%d qps=%d reuse_connections=%t\n",
			okCount, *requests, totalElapsed, *payloadBytes, workerCount, atomic.LoadInt64(&maxActive), *qps, *reuseConnections)
		if okCount != *requests {
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
	index       int
	endpoint    string
	requestID   string
	latencyMs   int64
	err         error
	code        int
	status      string
	backend     string
	userID      string
	items       int
	inferenceMs float64
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
