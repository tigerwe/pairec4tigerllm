package web

import (
	"encoding/json"
	"testing"

	"github.com/alibaba/pairec/v2/context"
)

func TestAllowPartialRecommendResults(t *testing.T) {
	t.Setenv("PAIREC_ALLOW_PARTIAL_RESULTS", "")
	if allowPartialRecommendResults(2) {
		t.Fatal("partial results must be disabled by default")
	}
	t.Setenv("PAIREC_ALLOW_PARTIAL_RESULTS", "1")
	if !allowPartialRecommendResults(2) {
		t.Fatal("non-empty partial results should be enabled")
	}
	if allowPartialRecommendResults(0) {
		t.Fatal("empty results must not be reported as success")
	}
}

func TestDeepFMRankFailureResponse(t *testing.T) {
	controller := &RecommendController{}
	controller.RequestId = "request-1"
	controller.context = context.NewRecommendContext()
	if controller.deepFMRankFailureResponse() != nil {
		t.Fatal("rank failure response should be absent without context error")
	}
	controller.context.AddContextParam("deepfm_rank_error", "timeout")
	response := controller.deepFMRankFailureResponse()
	if response == nil || response.Code != SERVER_ERROR_CODE || response.Size != 0 || len(response.Items) != 0 {
		t.Fatalf("unexpected fail-closed response: %#v", response)
	}
	var body map[string]interface{}
	if err := json.Unmarshal([]byte(response.ToString()), &body); err != nil {
		t.Fatal(err)
	}
	if body["msg"] != "deepfm rank failed" || body["request_id"] != "request-1" {
		t.Fatalf("unexpected response body: %#v", body)
	}
}
