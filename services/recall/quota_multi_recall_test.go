package recall

import (
	"fmt"
	"testing"

	"github.com/alibaba/pairec/v2/module"
)

func recallItems(source string, ids ...int) []*module.Item {
	items := make([]*module.Item, 0, len(ids))
	for _, id := range ids {
		item := module.NewItem(fmt.Sprint(id))
		item.RetrieveId = source
		items = append(items, item)
	}
	return items
}

func TestMergeQuotaRecallItemsUsesTwoPrimaryAndFillsFromSecondary(t *testing.T) {
	primary := recallItems("generative_recall", 1, 2, 3, 4)
	secondary := recallItems("milvus_recall", 2, 5, 6, 7, 8)

	got, stats := mergeQuotaRecallItems(primary, secondary, 2, 5)
	if len(got) != 5 {
		t.Fatalf("len=%d, want 5", len(got))
	}
	want := []string{"1", "2", "5", "6", "7"}
	for i, item := range got {
		if string(item.Id) != want[i] {
			t.Fatalf("item[%d]=%s, want %s", i, item.Id, want[i])
		}
	}
	if stats.primarySelected != 2 || stats.secondarySelected != 3 || stats.duplicateCount != 1 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}

func TestMergeQuotaRecallItemsBackfillsFromPrimary(t *testing.T) {
	primary := recallItems("generative_recall", 1, 2, 3, 4, 5)
	secondary := recallItems("milvus_recall", 6)

	got, stats := mergeQuotaRecallItems(primary, secondary, 2, 5)
	if len(got) != 5 {
		t.Fatalf("len=%d, want 5", len(got))
	}
	if stats.primarySelected != 4 || stats.secondarySelected != 1 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}
