package pipelinepb

import (
	"testing"

	proto "github.com/gogo/protobuf/proto"
)

func TestRankResponseRoundTrip(t *testing.T) {
	want := &RankResponse{
		Code: proto.Int32(200), ModelVersion: proto.String("model-v1"),
		Items: []*RankedItem{{ItemID: proto.String("42"), Score: proto.Float64(0.75)}},
		Trace: &ServiceTrace{TotalUS: proto.Int64(321), Context: &TraceContext{
			RequestID: proto.String("req-1"), ContractVersion: proto.String(TraceContractVersion),
		}},
	}
	encoded, err := proto.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	var got RankResponse
	if err := proto.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	if String(got.ModelVersion) != "model-v1" || Int64(got.Trace.TotalUS) != 321 {
		t.Fatalf("unexpected round trip: %s", got.String())
	}
}
