package pipelineclient

import (
	"testing"
	"time"

	proto "github.com/gogo/protobuf/proto"
	"pairec4tigerllm/services/pipelinepb"
)

func TestValidateContext(t *testing.T) {
	request := NewTraceContext("req", "span", "parent", time.Second)
	trace := &pipelinepb.ServiceTrace{Context: &pipelinepb.TraceContext{
		RequestID: proto.String("req"), ContractVersion: proto.String(pipelinepb.TraceContractVersion),
	}, TotalUS: proto.Int64(1)}
	if err := validateContext(request, trace); err != nil {
		t.Fatal(err)
	}
	trace.Context.RequestID = proto.String("wrong")
	if err := validateContext(request, trace); err == nil {
		t.Fatal("expected request identity mismatch")
	}
}
