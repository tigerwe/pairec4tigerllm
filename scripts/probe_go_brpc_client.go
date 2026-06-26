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
	requests := flag.Int("requests", getenvInt("REQUESTS", 1), "number of recommend requests")
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

	start := time.Now()
	switch *method {
	case "health":
		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeoutMs)*time.Millisecond)
		resp, err := client.HealthCheck(ctx)
		cancel()
		elapsed := time.Since(start).Milliseconds()
		if err != nil {
			fmt.Fprintf(os.Stderr, "health failed endpoint=%s latency_ms=%d error=%v\n", *endpoint, elapsed, err)
			os.Exit(1)
		}
		fmt.Printf("health ok endpoint=%s latency_ms=%d code=%d status=%s backend=%s\n",
			*endpoint, elapsed, resp.Code, resp.Status, resp.Backend)
	case "recommend":
		if *requests < 1 {
			fmt.Fprintln(os.Stderr, "requests must be positive")
			os.Exit(2)
		}
		okCount := 0
		totalStart := time.Now()
		for index := 1; index <= *requests; index++ {
			requestID := fmt.Sprintf("go-brpc-probe-%d", index)
			req := &recall.RecommendRequest{
				UserID: fmt.Sprintf("%s_%d", *userID, index),
				History: [][]int{
					{169, 41, 0, 0},
					{20, 53, 0, 0},
					{80, 201, 0, 0},
				},
				Topk:        *topk,
				Temperature: 1.0,
				BeamWidth:   1,
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
			fmt.Printf("recommend ok index=%d endpoint=%s request_id=%s latency_ms=%d code=%d user_id=%s items=%d inference_ms=%.0f backend=%s\n",
				index, *endpoint, requestID, elapsed, resp.Code, resp.UserID, len(resp.Recommendations), inferenceMs, backend)
		}
		totalElapsed := time.Since(totalStart).Milliseconds()
		fmt.Printf("summary ok=%d total=%d total_ms=%d\n", okCount, *requests, totalElapsed)
		if okCount != *requests {
			os.Exit(1)
		}
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
