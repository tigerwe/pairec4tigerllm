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
	timeoutMs := flag.Int("timeout_ms", getenvInt("TIMEOUT_MS", 5000), "request timeout in ms")
	maxRetries := flag.Int("max_retries", getenvInt("MAX_RETRIES", 1), "max retries")
	payloadBytes := flag.Int("payload_bytes", getenvInt("PAYLOAD_BYTES", 0), "extra protobuf payload padding bytes")
	historySource := flag.String("history_source", getenv("HISTORY_SOURCE", "synthetic"), "synthetic or user_features")
	uids := flag.String("uids", getenv("UIDS", ""), "comma-separated user ids for user_features history source")
	userFeaturesPath := flag.String("user_features_path", getenv("USER_FEATURES_PATH", "data/user_features.json"), "user_features.json path")
	semanticMapPath := flag.String("semantic_map_path", getenv("SEMANTIC_MAP_PATH", "data/tenrec/processed/semantic_id_map.json"), "semantic_id_map.json path")
	historyMaxLength := flag.Int("history_max_length", getenvInt("HISTORY_MAX_LENGTH", 20), "max history items from user_features")
	varyUserID := flag.Bool("vary_user_id", getenvBool("VARY_USER_ID", true), "append request index to synthetic user_id")
	flag.Parse()

	client, err := recall.NewBRPCRecommendClient(
		*endpoint,
		*service,
		time.Duration(*timeoutMs)*time.Millisecond,
		*maxRetries,
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "create client failed: %v\n", err)
		os.Exit(1)
	}

	switch *method {
	case "health":
		if *requests < 1 {
			fmt.Fprintln(os.Stderr, "requests must be positive")
			os.Exit(2)
		}
		okCount := 0
		totalStart := time.Now()
		for index := 1; index <= *requests; index++ {
			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeoutMs)*time.Millisecond)
			started := time.Now()
			resp, err := client.HealthCheckWithPayload(ctx, *payloadBytes)
			cancel()
			elapsed := time.Since(started).Milliseconds()
			if err != nil {
				fmt.Fprintf(os.Stderr, "health failed index=%d endpoint=%s payload_bytes=%d latency_ms=%d error=%v\n",
					index, *endpoint, *payloadBytes, elapsed, err)
				continue
			}
			okCount++
			fmt.Printf("health ok index=%d endpoint=%s payload_bytes=%d latency_ms=%d code=%d status=%s backend=%s\n",
				index, *endpoint, *payloadBytes, elapsed, resp.Code, resp.Status, resp.Backend)
		}
		totalElapsed := time.Since(totalStart).Milliseconds()
		fmt.Printf("summary ok=%d total=%d total_ms=%d payload_bytes=%d\n", okCount, *requests, totalElapsed, *payloadBytes)
		if okCount != *requests {
			os.Exit(1)
		}
	case "recommend":
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
		for index := 1; index <= *requests; index++ {
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
				fmt.Fprintf(os.Stderr, "recommend failed index=%d endpoint=%s request_id=%s latency_ms=%d error=%v\n",
					index, *endpoint, requestID, elapsed, err)
				continue
			}
			okCount++
			inferenceMs := resp.InferenceTimeMs
			backend := ""
			if resp.Trace != nil {
				backend = resp.Trace.Backend
			}
			fmt.Printf("recommend ok index=%d endpoint=%s request_id=%s payload_bytes=%d latency_ms=%d code=%d user_id=%s items=%d inference_ms=%.0f backend=%s\n",
				index, *endpoint, requestID, *payloadBytes, elapsed, resp.Code, resp.UserID, len(resp.Recommendations), inferenceMs, backend)
		}
		totalElapsed := time.Since(totalStart).Milliseconds()
		fmt.Printf("summary ok=%d total=%d total_ms=%d payload_bytes=%d\n", okCount, *requests, totalElapsed, *payloadBytes)
		if okCount != *requests {
			os.Exit(1)
		}
	default:
		fmt.Fprintf(os.Stderr, "unknown method %q\n", *method)
		os.Exit(2)
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
