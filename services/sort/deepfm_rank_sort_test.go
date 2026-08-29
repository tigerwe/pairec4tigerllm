package ranksort

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	paireccontext "github.com/alibaba/pairec/v2/context"
	"github.com/alibaba/pairec/v2/module"
	pairecsort "github.com/alibaba/pairec/v2/sort"
)

func TestRankBurstTransportTimeoutDoesNotClipEitherLane(t *testing.T) {
	tests := []struct {
		name               string
		business, pressure time.Duration
		want               time.Duration
	}{
		{name: "pressure longer", business: 100 * time.Millisecond, pressure: 5 * time.Second, want: 5 * time.Second},
		{name: "business longer", business: 2 * time.Second, pressure: 500 * time.Millisecond, want: 2 * time.Second},
		{name: "equal", business: time.Second, pressure: time.Second, want: time.Second},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := rankBurstTransportTimeout(test.business, test.pressure); got != test.want {
				t.Fatalf("transport timeout=%s want=%s", got, test.want)
			}
		})
	}
}

func TestOrderedCandidateSHA256CoversOrderAndBoundaries(t *testing.T) {
	base := orderedCandidateSHA256([]string{"ab", "c"})
	if base == orderedCandidateSHA256([]string{"a", "bc"}) {
		t.Fatal("candidate hash must preserve item boundaries")
	}
	if base == orderedCandidateSHA256([]string{"c", "ab"}) {
		t.Fatal("candidate hash must preserve rank order")
	}
	if base != orderedCandidateSHA256([]string{"ab", "c"}) {
		t.Fatal("candidate hash must be deterministic")
	}
}

func TestPostRankHopConfigIsStrict(t *testing.T) {
	_, err := NewDeepFMRankSort(Config{
		Name: "test", Protocol: "brpc", BRPCEndpoint: "127.0.0.1:18211",
		TimeoutMS: 1000, ExpectedCandidates: 50, RequiredModelRole: "engineering",
		PostRankHopsEnabled: true, PostRankHop1Endpoint: "127.0.0.1:18311",
		PostRankTimeoutMS: 1500, PostRankBurstConcurrency: 999,
		PostRankBurstPoolSize: 1000, PostRankPayloadBytes: 102400,
		PostRankPressureTimeoutMS: 5000,
	})
	if err == nil {
		t.Fatal("post-rank hops must reject non-c1000 configuration")
	}
}

func testSortData() *pairecsort.SortData {
	ctx := paireccontext.NewRecommendContext()
	ctx.RecommendId = "rank-request-1"
	return &pairecsort.SortData{
		Data: []*module.Item{
			module.NewItem("a"),
			module.NewItem("b"),
			module.NewItem("c"),
		},
		Context: ctx,
		User:    module.NewUser("user-1"),
	}
}

func TestDeepFMRankSortStableSuccess(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var payload rankRequest
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		if payload.RequestID != "rank-request-1" || len(payload.Items) != 3 {
			t.Fatalf("unexpected request: %#v", payload)
		}
		json.NewEncoder(writer).Encode(rankResponse{
			Code: 200, RequestID: payload.RequestID, ModelVersion: "test-v1",
			ModelRole: "engineering", Trace: responseTrace{ScoreUniqueCount: 2},
			Items: []responseItem{
				{ItemID: "a", Score: 0.5},
				{ItemID: "b", Score: 0.9},
				{ItemID: "c", Score: 0.5},
			},
		})
	}))
	defer server.Close()
	ranker, err := NewDeepFMRankSort(Config{
		Name: "test", ServerURL: server.URL, TimeoutMS: 1000, ExpectedCandidates: 3,
		RequiredModelRole: "engineering",
	})
	if err != nil {
		t.Fatal(err)
	}
	data := testSortData()
	if err := ranker.Sort(data); err != nil {
		t.Fatal(err)
	}
	items := data.Data.([]*module.Item)
	got := []string{string(items[0].Id), string(items[1].Id), string(items[2].Id)}
	want := []string{"b", "a", "c"}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("stable rank mismatch got=%v want=%v", got, want)
		}
	}
	if data.Context.GetContextParam(RankErrorContextKey) != nil {
		t.Fatal("unexpected rank error context")
	}
	if data.Context.GetContextParam(RankTraceContextKey) == nil {
		t.Fatal("missing rank trace")
	}
}

func TestDeepFMRankSortFailsClosedOnResponseMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		json.NewEncoder(writer).Encode(rankResponse{
			Code: 200, RequestID: "wrong", ModelVersion: "test-v1",
			ModelRole: "engineering", Trace: responseTrace{ScoreUniqueCount: 1},
			Items: []responseItem{{ItemID: "a", Score: 0.5}},
		})
	}))
	defer server.Close()
	ranker, err := NewDeepFMRankSort(Config{
		Name: "test", ServerURL: server.URL, TimeoutMS: 1000, ExpectedCandidates: 3,
		RequiredModelRole: "engineering",
	})
	if err != nil {
		t.Fatal(err)
	}
	data := testSortData()
	if err := ranker.Sort(data); err == nil {
		t.Fatal("expected strict response validation failure")
	}
	if len(data.Data.([]*module.Item)) != 0 {
		t.Fatal("rank failure must clear candidate output")
	}
	if data.Context.GetContextParam(RankErrorContextKey) == nil {
		t.Fatal("missing rank error context")
	}
}

func TestDeepFMRankSortFailsClosedOnModelRoleMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var payload rankRequest
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		json.NewEncoder(writer).Encode(rankResponse{
			Code: 200, RequestID: payload.RequestID, ModelVersion: "test-v1",
			ModelRole: "engineering", Trace: responseTrace{ScoreUniqueCount: 3},
			Items: []responseItem{
				{ItemID: "a", Score: 0.5},
				{ItemID: "b", Score: 0.9},
				{ItemID: "c", Score: 0.4},
			},
		})
	}))
	defer server.Close()
	ranker, err := NewDeepFMRankSort(Config{
		Name: "test", ServerURL: server.URL, TimeoutMS: 1000, ExpectedCandidates: 3,
		RequiredModelRole: "production_candidate",
	})
	if err != nil {
		t.Fatal(err)
	}
	data := testSortData()
	if err := ranker.Sort(data); err == nil {
		t.Fatal("expected model role mismatch")
	}
	if len(data.Data.([]*module.Item)) != 0 {
		t.Fatal("model role mismatch must clear candidate output")
	}
}
