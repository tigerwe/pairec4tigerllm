package recall

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/alibaba/pairec/v2/context"
	"github.com/alibaba/pairec/v2/log"
	"github.com/alibaba/pairec/v2/module"
	"github.com/alibaba/pairec/v2/recconf"
	baserecall "github.com/alibaba/pairec/v2/service/recall"
	"github.com/alibaba/pairec/v2/utils"
)

type quotaMultiRecallConfigJSON struct {
	PrimaryRecallName   string `json:"primary_recall_name"`
	SecondaryRecallName string `json:"secondary_recall_name"`
	PrimaryQuota        int    `json:"primary_quota"`
	TotalLimit          int    `json:"total_limit"`
}

// QuotaMultiRecall runs two registered recalls concurrently and combines them
// with a deterministic source quota. Child items keep their original source.
type QuotaMultiRecall struct {
	*baserecall.BaseRecall
	modelName           string
	primaryRecallName   string
	secondaryRecallName string
	primaryQuota        int
	totalLimit          int
}

func NewQuotaMultiRecall(conf recconf.RecallConfig) *QuotaMultiRecall {
	algoConf := quotaMultiRecallConfigJSON{
		PrimaryQuota: 2,
		TotalLimit:   conf.RecallCount,
	}
	if algoConf.TotalLimit <= 0 {
		algoConf.TotalLimit = 50
	}
	if err := json.Unmarshal([]byte(conf.RecallAlgo), &algoConf); err != nil {
		panic(fmt.Sprintf("invalid QuotaMultiRecall config %s: %v", conf.Name, err))
	}
	if algoConf.PrimaryRecallName == "" || algoConf.SecondaryRecallName == "" {
		panic(fmt.Sprintf("QuotaMultiRecall %s requires both child recall names", conf.Name))
	}
	if algoConf.PrimaryRecallName == conf.Name || algoConf.SecondaryRecallName == conf.Name {
		panic(fmt.Sprintf("QuotaMultiRecall %s cannot reference itself", conf.Name))
	}
	if algoConf.PrimaryQuota < 0 || algoConf.TotalLimit <= 0 || algoConf.PrimaryQuota > algoConf.TotalLimit {
		panic(fmt.Sprintf("QuotaMultiRecall %s has invalid quota=%d limit=%d",
			conf.Name, algoConf.PrimaryQuota, algoConf.TotalLimit))
	}

	r := &QuotaMultiRecall{
		BaseRecall:          baserecall.NewBaseRecall(conf),
		modelName:           conf.Name,
		primaryRecallName:   algoConf.PrimaryRecallName,
		secondaryRecallName: algoConf.SecondaryRecallName,
		primaryQuota:        algoConf.PrimaryQuota,
		totalLimit:          algoConf.TotalLimit,
	}
	fmt.Printf("[QuotaMultiRecall] init name=%s primary=%s quota=%d secondary=%s limit=%d\n",
		r.modelName, r.primaryRecallName, r.primaryQuota, r.secondaryRecallName, r.totalLimit)
	return r
}

func (r *QuotaMultiRecall) GetCandidateItems(user *module.User, ctx *context.RecommendContext) []*module.Item {
	start := time.Now()
	primary, primaryErr := baserecall.GetRecall(r.primaryRecallName)
	secondary, secondaryErr := baserecall.GetRecall(r.secondaryRecallName)
	if primaryErr != nil || secondaryErr != nil {
		log.Error(fmt.Sprintf(
			"requestId=%s\tmodule=QuotaMultiRecall\tname=%s\tprimary_error=%v\tsecondary_error=%v",
			ctx.RecommendId, r.modelName, primaryErr, secondaryErr))
	}

	var primaryItems, secondaryItems []*module.Item
	var wg sync.WaitGroup
	if primary != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			primaryItems = r.runChildRecall(primary, r.primaryRecallName, user, ctx)
		}()
	}
	if secondary != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			secondaryItems = r.runChildRecall(secondary, r.secondaryRecallName, user, ctx)
		}()
	}
	wg.Wait()

	items, stats := mergeQuotaRecallItems(primaryItems, secondaryItems, r.primaryQuota, r.totalLimit)
	degraded := len(primaryItems) == 0 || len(secondaryItems) == 0 || len(items) < r.totalLimit
	log.Info(fmt.Sprintf(
		"requestId=%s\tmodule=QuotaMultiRecall\tname=%s\tprimary=%s\tsecondary=%s"+
			"\tprimary_input=%d\tsecondary_input=%d\tprimary_selected=%d\tsecondary_selected=%d"+
			"\tduplicate_count=%d\tfinal_count=%d\tdegraded=%t\tcost=%d",
		ctx.RecommendId, r.modelName, r.primaryRecallName, r.secondaryRecallName,
		len(primaryItems), len(secondaryItems), stats.primarySelected, stats.secondarySelected,
		stats.duplicateCount, len(items), degraded, utils.CostTime(start)))
	writeTraceStdout(
		"requestId=%s request_id=%s module=QuotaMultiRecall name=%s primary=%s secondary=%s"+
			" primary_input=%d secondary_input=%d primary_selected=%d secondary_selected=%d"+
			" duplicate_count=%d final_count=%d degraded=%t cost=%d",
		ctx.RecommendId, ctx.RecommendId, r.modelName, r.primaryRecallName, r.secondaryRecallName,
		len(primaryItems), len(secondaryItems), stats.primarySelected, stats.secondarySelected,
		stats.duplicateCount, len(items), degraded, utils.CostTime(start),
	)
	return items
}

func (r *QuotaMultiRecall) runChildRecall(child baserecall.Recall, childName string, user *module.User,
	ctx *context.RecommendContext) (items []*module.Item) {
	defer func() {
		if recovered := recover(); recovered != nil {
			log.Error(fmt.Sprintf(
				"requestId=%s\tmodule=QuotaMultiRecall\tname=%s\tchild=%s\tpanic=%v",
				ctx.RecommendId, r.modelName, childName, recovered))
			items = nil
		}
	}()
	return child.GetCandidateItems(user, ctx)
}

type quotaMergeStats struct {
	primarySelected   int
	secondarySelected int
	duplicateCount    int
}

func mergeQuotaRecallItems(primary, secondary []*module.Item, primaryQuota, totalLimit int) ([]*module.Item, quotaMergeStats) {
	result := make([]*module.Item, 0, totalLimit)
	seen := make(map[module.ItemId]struct{}, totalLimit)
	stats := quotaMergeStats{}

	appendUnique := func(items []*module.Item, limit int, selected *int) {
		for _, item := range items {
			if item == nil || len(result) >= totalLimit || *selected >= limit {
				continue
			}
			if _, exists := seen[item.Id]; exists {
				stats.duplicateCount++
				continue
			}
			seen[item.Id] = struct{}{}
			result = append(result, item)
			*selected++
		}
	}

	appendUnique(primary, primaryQuota, &stats.primarySelected)
	appendUnique(secondary, totalLimit, &stats.secondarySelected)
	if len(result) < totalLimit {
		appendUnique(primary, totalLimit, &stats.primarySelected)
	}
	return result, stats
}
