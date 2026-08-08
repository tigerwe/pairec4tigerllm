package pipelineclient

import (
	"context"
	"fmt"
	"time"

	proto "github.com/gogo/protobuf/proto"
	"pairec4tigerllm/services/brpcwire"
	"pairec4tigerllm/services/pipelinepb"
)

const (
	VectorRecallServiceName = "pairec.pipeline.VectorRecallService"
	DeepFMRankServiceName   = "pairec.pipeline.DeepFMRankService"
)

type VectorClient struct{ wire *brpcwire.Client }
type RankClient struct{ wire *brpcwire.Client }

func NewVectorClient(endpoint, service string, timeout time.Duration) (*VectorClient, error) {
	if service == "" {
		service = VectorRecallServiceName
	}
	wire, err := brpcwire.NewClient(endpoint, service, timeout, 1)
	if err != nil {
		return nil, err
	}
	return &VectorClient{wire: wire}, nil
}

func NewRankClient(endpoint, service string, timeout time.Duration) (*RankClient, error) {
	if service == "" {
		service = DeepFMRankServiceName
	}
	wire, err := brpcwire.NewClient(endpoint, service, timeout, 1)
	if err != nil {
		return nil, err
	}
	return &RankClient{wire: wire}, nil
}

func NewTraceContext(requestID, spanID, parentSpanID string, timeout time.Duration) *pipelinepb.TraceContext {
	return &pipelinepb.TraceContext{
		RequestID: proto.String(requestID), SpanID: proto.String(spanID),
		ParentSpanID: proto.String(parentSpanID), Sampled: proto.Bool(true),
		ContractVersion: proto.String(pipelinepb.TraceContractVersion),
		TimeoutBudgetUS: proto.Int64(timeout.Microseconds()),
	}
}

func (c *VectorClient) Recall(ctx context.Context, request *pipelinepb.VectorRecallRequest) (*pipelinepb.VectorRecallResponse, error) {
	var response pipelinepb.VectorRecallResponse
	if err := c.wire.Call(ctx, "Recall", request, &response); err != nil {
		return nil, err
	}
	if err := validateContext(request.Context, response.Trace); err != nil {
		return nil, fmt.Errorf("vector trace: %w", err)
	}
	return &response, nil
}

func (c *VectorClient) Health(ctx context.Context, requestID string) (*pipelinepb.HealthResponse, error) {
	request := &pipelinepb.HealthRequest{Context: NewTraceContext(requestID, "vector-health", "", 0)}
	var response pipelinepb.HealthResponse
	if err := c.wire.Call(ctx, "Health", request, &response); err != nil {
		return nil, err
	}
	return &response, nil
}

func (c *RankClient) Rank(ctx context.Context, request *pipelinepb.RankRequest) (*pipelinepb.RankResponse, error) {
	var response pipelinepb.RankResponse
	if err := c.wire.Call(ctx, "Rank", request, &response); err != nil {
		return nil, err
	}
	if err := validateContext(request.Context, response.Trace); err != nil {
		return nil, fmt.Errorf("rank trace: %w", err)
	}
	return &response, nil
}

func (c *RankClient) Health(ctx context.Context, requestID string) (*pipelinepb.HealthResponse, error) {
	request := &pipelinepb.HealthRequest{Context: NewTraceContext(requestID, "rank-health", "", 0)}
	var response pipelinepb.HealthResponse
	if err := c.wire.Call(ctx, "Health", request, &response); err != nil {
		return nil, err
	}
	return &response, nil
}

func validateContext(request *pipelinepb.TraceContext, trace *pipelinepb.ServiceTrace) error {
	if request == nil || trace == nil || trace.Context == nil {
		return fmt.Errorf("missing request or response trace context")
	}
	if pipelinepb.String(trace.Context.RequestID) != pipelinepb.String(request.RequestID) {
		return fmt.Errorf("request_id mismatch: got=%q want=%q",
			pipelinepb.String(trace.Context.RequestID), pipelinepb.String(request.RequestID))
	}
	if pipelinepb.String(trace.Context.ContractVersion) != pipelinepb.TraceContractVersion {
		return fmt.Errorf("contract version mismatch: %q", pipelinepb.String(trace.Context.ContractVersion))
	}
	if pipelinepb.Int64(trace.TotalUS) <= 0 {
		return fmt.Errorf("total_us must be positive")
	}
	return nil
}
