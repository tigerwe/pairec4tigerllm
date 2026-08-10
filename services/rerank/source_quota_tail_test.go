package rerank

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/alibaba/pairec/v2/module"
)

func testReranker(t *testing.T) *SourceQuotaTail {
	t.Helper()
	r, err := NewSourceQuotaTail(Config{
		Name: PolicyName, Enabled: true, GenerativeSource: "generative_recall",
		VectorSource: "milvus_recall", ExpectedCandidates: 50,
		MinimumGenerative: 1, MaximumGenerative: 2, Placement: "tail", FailClosed: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func candidates(generative int) []*module.Item {
	items := make([]*module.Item, 0, 50)
	for index := 0; index < generative; index++ {
		item := module.NewItem(fmt.Sprintf("g%d", index+1))
		item.RetrieveId = "generative_recall"
		item.Score = 1.0 - float64(index)*0.01
		items = append(items, item)
	}
	for index := len(items); index < 50; index++ {
		item := module.NewItem(fmt.Sprintf("v%d", index+1))
		item.RetrieveId = "milvus_recall"
		item.Score = 0.9 - float64(index)*0.001
		items = append(items, item)
	}
	return items
}

func ids(items []*module.Item) []string {
	result := make([]string, len(items))
	for index, item := range items {
		result[index] = string(item.Id)
	}
	return result
}

func TestSourceQuotaTailKeepsTwoGenerativeAtTail(t *testing.T) {
	items := candidates(2)
	originalScores := map[string]float64{}
	for _, item := range items {
		originalScores[string(item.Id)] = item.Score
	}
	output, result, err := testReranker(t).Apply(items, 10)
	if err != nil {
		t.Fatal(err)
	}
	if result.GenerativeSelected != 2 || result.VectorSelected != 8 {
		t.Fatalf("unexpected result: %#v", result)
	}
	if got := ids(output[8:]); !reflect.DeepEqual(got, []string{"g1", "g2"}) {
		t.Fatalf("unexpected tail: %v", got)
	}
	for _, item := range output {
		if item.Score != originalScores[string(item.Id)] {
			t.Fatalf("score changed for %s", item.Id)
		}
	}
	if output[0].GetProperty("rerank_position") != 1 ||
		output[9].GetProperty("rerank_position") != 10 ||
		output[9].GetProperty("rerank_reason") != PolicyName {
		t.Fatalf("missing rerank properties: first=%v last=%v",
			output[0].GetProperties(), output[9].GetProperties())
	}
}

func TestSourceQuotaTailPreservesOrderWithinEachSource(t *testing.T) {
	items := candidates(2)
	reordered := append([]*module.Item{}, items[2:12]...)
	reordered = append(reordered, items[:2]...)
	reordered = append(reordered, items[12:]...)
	items = reordered
	output, _, err := testReranker(t).Apply(items, 10)
	if err != nil {
		t.Fatal(err)
	}
	want := append(ids(items[:8]), "g1", "g2")
	if got := ids(output); !reflect.DeepEqual(got, want) {
		t.Fatalf("source-relative order changed: got=%v want=%v", got, want)
	}
}

func TestSourceQuotaTailSupportsOneAndArbitrarySize(t *testing.T) {
	output, result, err := testReranker(t).Apply(candidates(1), 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(output) != 5 || output[4].RetrieveId != "generative_recall" ||
		result.GenerativeSelected != 1 || result.VectorSelected != 4 {
		t.Fatalf("unexpected output=%v result=%#v", ids(output), result)
	}
	output, result, err = testReranker(t).Apply(candidates(2), 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(output) != 1 || string(output[0].Id) != "g1" || result.GenerativeSelected != 1 {
		t.Fatalf("unexpected size-one output=%v result=%#v", ids(output), result)
	}
}

func TestSourceQuotaTailRejectsMissingGenerative(t *testing.T) {
	_, _, err := testReranker(t).Apply(candidates(0), 10)
	if err == nil || !strings.Contains(err.Error(), "missing generative") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestSourceQuotaTailRejectsUnknownDuplicateAndInsufficientInputs(t *testing.T) {
	tests := []struct {
		name string
		edit func([]*module.Item) []*module.Item
		want string
	}{
		{"unknown", func(items []*module.Item) []*module.Item {
			items[3].RetrieveId = "other"
			return items
		}, "unknown retrieve source"},
		{"duplicate", func(items []*module.Item) []*module.Item {
			items[3].Id = items[2].Id
			return items
		}, "duplicate candidate"},
		{"short", func(items []*module.Item) []*module.Item { return items[:49] }, "expected 50 candidates"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, err := testReranker(t).Apply(test.edit(candidates(2)), 10)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestSourceQuotaTailRejectsInvalidConfig(t *testing.T) {
	_, err := NewSourceQuotaTail(Config{Name: PolicyName, Enabled: true})
	if err == nil {
		t.Fatal("invalid config unexpectedly accepted")
	}
}
