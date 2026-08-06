package web

import "testing"

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
