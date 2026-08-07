package ranksort

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	paireccontext "github.com/alibaba/pairec/v2/context"
	"github.com/alibaba/pairec/v2/module"
	pairecsort "github.com/alibaba/pairec/v2/sort"
)

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
			Items: []responseItem{{ItemID: "a", Score: 0.5}},
		})
	}))
	defer server.Close()
	ranker, err := NewDeepFMRankSort(Config{
		Name: "test", ServerURL: server.URL, TimeoutMS: 1000, ExpectedCandidates: 3,
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
