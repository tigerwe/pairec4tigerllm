package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	recall "pairec4tigerllm/services/recall"
)

func main() {
	endpoint := flag.String("endpoint", getenv("BRPC_ENDPOINT", "127.0.0.1:18100"), "brpc endpoint host:port")
	service := flag.String("service", getenv("BRPC_SERVICE", "pairec.inference.RecommendService"), "brpc service name")
	method := flag.String("method", getenv("BRPC_METHOD", "recommend"), "health or recommend")
	userID := flag.String("user_id", getenv("USER_ID", "go_brpc_probe"), "user id for recommend")
	topk := flag.Int("topk", getenvInt("TOPK", 10), "recommend topk")
	timeoutMs := flag.Int("timeout_ms", getenvInt("TIMEOUT_MS", 5000), "request timeout in ms")
	maxRetries := flag.Int("max_retries", getenvInt("MAX_RETRIES", 1), "max retries")
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

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeoutMs)*time.Millisecond)
	defer cancel()

	start := time.Now()
	switch *method {
	case "health":
		resp, err := client.HealthCheck(ctx)
		elapsed := time.Since(start).Milliseconds()
		if err != nil {
			fmt.Fprintf(os.Stderr, "health failed endpoint=%s latency_ms=%d error=%v\n", *endpoint, elapsed, err)
			os.Exit(1)
		}
		fmt.Printf("health ok endpoint=%s latency_ms=%d code=%d status=%s backend=%s\n",
			*endpoint, elapsed, resp.Code, resp.Status, resp.Backend)
	case "recommend":
		req := &recall.RecommendRequest{
			UserID: *userID,
			History: [][]int{
				{169, 41, 0, 0},
				{20, 53, 0, 0},
				{80, 201, 0, 0},
			},
			Topk:        *topk,
			Temperature: 1.0,
			BeamWidth:   1,
		}
		resp, err := client.Recommend(ctx, req, "go-brpc-probe")
		elapsed := time.Since(start).Milliseconds()
		if err != nil {
			fmt.Fprintf(os.Stderr, "recommend failed endpoint=%s latency_ms=%d error=%v\n", *endpoint, elapsed, err)
			os.Exit(1)
		}
		inferenceMs := resp.InferenceTimeMs
		backend := ""
		if resp.Trace != nil {
			backend = resp.Trace.Backend
		}
		fmt.Printf("recommend ok endpoint=%s latency_ms=%d code=%d user_id=%s items=%d inference_ms=%.0f backend=%s\n",
			*endpoint, elapsed, resp.Code, resp.UserID, len(resp.Recommendations), inferenceMs, backend)
	default:
		fmt.Fprintf(os.Stderr, "unknown method %q\n", *method)
		os.Exit(2)
	}
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
