// recall/generative_recall.go
//
// 生成式召回服务实现.
// 基于 pairec 框架的召回接口，集成 TensorRT-LLM 推理服务.

package recall

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/alibaba/pairec/v2/context"
	"github.com/alibaba/pairec/v2/log"
	"github.com/alibaba/pairec/v2/module"
	"github.com/alibaba/pairec/v2/recconf"
	"github.com/alibaba/pairec/v2/service/recall"
	"github.com/alibaba/pairec/v2/utils"

	"pairec4tigerllm/services/config"
)

// semanticIDMap 全局语义 ID 映射缓存
var (
	semanticIDMap     map[int][]int
	semanticIDMapOnce sync.Once
	semanticIDMapErr  error
)

// GenerativeRecall 生成式召回.
// 基于 TensorRT-LLM 的生成式推荐模型.
type GenerativeRecall struct {
	*recall.BaseRecall
	client             *TRTLLMClient
	historyFrom        string
	historyFeatureName string
	historyDelimiter   string
	historyMaxLength   int
	topK               int
	temperature        float64
	beamWidth          int
	cachePrefix        string
	cacheTime          int
	semanticIDMapPath  string
}

// NewGenerativeRecall 创建生成式召回实例.
func NewGenerativeRecall(conf recconf.RecallConfig) *GenerativeRecall {
	// 从 TigerRecallConf 解析配置
	// 注意：TigerName 在这里被用作 ServerURL
	serverURL := conf.TigerRecallConf.TigerName
	if serverURL == "" {
		serverURL = "http://localhost:8000"
	}

	// 创建配置
	genConfig := &config.GenerativeRecallConfig{
		ServerURL:          serverURL,
		Timeout:            500 * time.Millisecond, // 默认 500ms
		MaxRetries:         3,                      // 默认 3 次重试
		TopK:               conf.TigerRecallConf.TopK,
		Temperature:        conf.TigerRecallConf.Temperature,
		BeamWidth:          conf.TigerRecallConf.BeamWidth,
		HistoryFrom:        conf.TigerRecallConf.HistoryFrom,
		HistoryFeatureName: conf.TigerRecallConf.HistoryFeatureName,
		HistoryDelimiter:   conf.TigerRecallConf.HistoryDelimiter,
		HistoryMaxLength:   conf.TigerRecallConf.HistoryMaxLength,
		CacheEnable:        conf.CacheAdapter != "",
		CacheTime:          conf.CacheTime,
		CachePrefix:        conf.CachePrefix,
	}

	// 与默认配置合并
	genConfig.MergeWithDefault()

	// 如果没有设置 topK，使用 RecallCount
	if genConfig.TopK == 0 {
		genConfig.TopK = conf.RecallCount
	}
	if genConfig.TopK == 0 {
		genConfig.TopK = 50
	}

	// 创建客户端
	client, err := NewTRTLLMClient(genConfig)
	if err != nil {
		// 客户端创建失败时 panic，在服务启动时就能发现问题
		panic(fmt.Sprintf("failed to create TRTLLMClient: %v", err))
	}

	// 加载语义 ID 映射
	semanticIDMapPath := "./data/tenrec/processed/semantic_id_map.json"
	loadSemanticIDMap(semanticIDMapPath)

	// 创建召回实例
	recallInstance := &GenerativeRecall{
		BaseRecall:         recall.NewBaseRecall(conf),
		client:             client,
		historyFrom:        genConfig.HistoryFrom,
		historyFeatureName: genConfig.HistoryFeatureName,
		historyDelimiter:   genConfig.HistoryDelimiter,
		historyMaxLength:   genConfig.HistoryMaxLength,
		topK:               genConfig.TopK,
		temperature:        genConfig.Temperature,
		beamWidth:          genConfig.BeamWidth,
		cachePrefix:        genConfig.CachePrefix,
		cacheTime:          genConfig.CacheTime,
		semanticIDMapPath:  semanticIDMapPath,
	}

	return recallInstance
}

// loadSemanticIDMap 加载语义 ID 映射.
func loadSemanticIDMap(path string) {
	semanticIDMapOnce.Do(func() {
		semanticIDMap = make(map[int][]int)

		data, err := os.ReadFile(path)
		if err != nil {
			semanticIDMapErr = fmt.Errorf("failed to read semantic ID map: %w", err)
			log.Error(fmt.Sprintf("module=GenerativeRecall\terror=load_semantic_map_failed:%v", err))
			return
		}

		var rawMap map[string][]int
		if err := json.Unmarshal(data, &rawMap); err != nil {
			semanticIDMapErr = fmt.Errorf("failed to unmarshal semantic ID map: %w", err)
			log.Error(fmt.Sprintf("module=GenerativeRecall\terror=parse_semantic_map_failed:%v", err))
			return
		}

		// 转换为 int key
		for k, v := range rawMap {
			itemID, err := strconv.Atoi(k)
			if err != nil {
				continue
			}
			semanticIDMap[itemID] = v
		}

		log.Info(fmt.Sprintf("module=GenerativeRecall\tmsg=loaded_semantic_id_map\tcount=%d", len(semanticIDMap)))
	})
}

// GetCandidateItems 获取候选物品.
// 这是召回服务的核心方法，由 pairec 框架调用.
func (r *GenerativeRecall) GetCandidateItems(user *module.User, ctx *context.RecommendContext) []*module.Item {
	start := time.Now()

	// 尝试从缓存获取
	if r.cache != nil {
		key := r.cachePrefix + string(user.Id)
		cacheRet := r.cache.Get(key)
		if items := r.parseCacheResult(cacheRet); len(items) > 0 {
			log.Info(fmt.Sprintf("requestId=%s\tmodule=GenerativeRecall\tfrom=cache\tname=%s\tcount=%d\tcost=%d",
				ctx.RecommendId, r.modelName, len(items), utils.CostTime(start)))
			return items
		}
	}

	// 获取用户历史行为
	history, err := r.getUserHistory(user, ctx)
	if err != nil {
		log.Error(fmt.Sprintf("requestId=%s\tmodule=GenerativeRecall\tname=%s\terr=get_user_history:%v",
			ctx.RecommendId, r.modelName, err))
		return nil
	}

	if len(history) == 0 {
		log.Info(fmt.Sprintf("requestId=%s\tmodule=GenerativeRecall\tname=%s\tmsg=empty_history",
			ctx.RecommendId, r.modelName))
		return nil
	}

	// 转换历史为语义 ID 格式
	semanticHistory := r.convertToSemanticIDs(history)
	if len(semanticHistory) == 0 {
		log.Error(fmt.Sprintf("requestId=%s\tmodule=GenerativeRecall\tname=%s\terr=convert_history_failed",
			ctx.RecommendId, r.modelName))
		return nil
	}

	// 调用生成式推理服务
	request := &RecommendRequest{
		UserID:      string(user.Id),
		History:     semanticHistory,
		Topk:        r.topK,
		Temperature: r.temperature,
		BeamWidth:   r.beamWidth,
	}

	response, err := r.client.Recommend(request)
	if err != nil {
		log.Error(fmt.Sprintf("requestId=%s\tmodule=GenerativeRecall\tname=%s\terr=generative_recommend:%v",
			ctx.RecommendId, r.modelName, err))
		return nil
	}

	// 转换结果为 pairec Item
	items := r.convertToItems(response)

	// 写入缓存
	if r.cache != nil && len(items) > 0 {
		go r.writeCache(user.Id, items)
	}

	log.Info(fmt.Sprintf("requestId=%s\tmodule=GenerativeRecall\tname=%s\tcount=%d\tinference_time=%.2fms\tcost=%d",
		ctx.RecommendId, r.modelName, len(items), response.InferenceTimeMs, utils.CostTime(start)))

	return items
}

// getUserHistory 获取用户历史行为.
func (r *GenerativeRecall) getUserHistory(user *module.User, ctx *context.RecommendContext) ([]int, error) {
	switch r.historyFrom {
	case "user_feature", "":
		return r.getHistoryFromUserFeature(user)
	case "context":
		return r.getHistoryFromContext(ctx)
	default:
		return r.getHistoryFromUserFeature(user)
	}
}

// getHistoryFromUserFeature 从用户特征获取历史.
func (r *GenerativeRecall) getHistoryFromUserFeature(user *module.User) ([]int, error) {
	if r.historyFeatureName == "" {
		return nil, fmt.Errorf("history_feature_name is empty")
	}

	val := user.GetProperty(r.historyFeatureName)
	if val == nil {
		return nil, fmt.Errorf("feature %s not found", r.historyFeatureName)
	}

	historyStr := utils.ToString(val, "")
	if historyStr == "" {
		return nil, nil
	}

	return r.parseHistoryString(historyStr)
}

// getHistoryFromContext 从上下文获取历史.
func (r *GenerativeRecall) getHistoryFromContext(ctx *context.RecommendContext) ([]int, error) {
	if r.historyFeatureName == "" {
		return nil, fmt.Errorf("history_feature_name is empty")
	}

	val := ctx.GetParameterByPath("features." + r.historyFeatureName)
	if val == nil {
		val = ctx.GetParameter(r.historyFeatureName)
	}
	if val == nil {
		return nil, fmt.Errorf("context param %s not found", r.historyFeatureName)
	}

	historyStr := utils.ToString(val, "")
	if historyStr == "" {
		return nil, nil
	}

	return r.parseHistoryString(historyStr)
}

// parseHistoryString 解析历史行为字符串.
func (r *GenerativeRecall) parseHistoryString(historyStr string) ([]int, error) {
	delimiter := r.historyDelimiter
	if delimiter == "" {
		delimiter = ","
	}

	parts := strings.Split(historyStr, delimiter)
	history := make([]int, 0, len(parts))

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		id, err := strconv.Atoi(part)
		if err != nil {
			continue
		}
		history = append(history, id)
	}

	// 限制历史长度
	if len(history) > r.historyMaxLength {
		history = history[len(history)-r.historyMaxLength:]
	}

	return history, nil
}

// convertToSemanticIDs 将物品 ID 序列转换为语义 ID 序列.
func (r *GenerativeRecall) convertToSemanticIDs(history []int) [][]int {
	semanticHistory := make([][]int, 0, len(history))

	for _, itemID := range history {
		// 优先从映射表查询
		if semanticIDMap != nil {
			if semIDs, ok := semanticIDMap[itemID]; ok {
				semanticHistory = append(semanticHistory, semIDs)
				continue
			}
		}

		// 回退到哈希方式（临时方案）
		semIDs := []int{
			itemID % 256,
			(itemID / 256) % 256,
			(itemID / 65536) % 256,
			(itemID / 16777216) % 256,
		}
		semanticHistory = append(semanticHistory, semIDs)
	}

	return semanticHistory
}

// convertToItems 将推理响应转换为 pairec Item.
func (r *GenerativeRecall) convertToItems(response *RecommendResponse) []*module.Item {
	items := make([]*module.Item, 0, len(response.Recommendations))

	for _, rec := range response.Recommendations {
		itemID := strconv.Itoa(rec.ItemID)
		item := module.NewItem(itemID)
		item.Score = rec.Score
		item.RetrieveId = r.modelName
		item.ItemType = r.itemType

		// 添加额外属性
		item.AddProperty("generative_rank", len(items)+1)
		item.AddProperty("semantic_id", rec.SemanticID)

		items = append(items, item)
	}

	return items
}

// parseCacheResult 解析缓存结果.
func (r *GenerativeRecall) parseCacheResult(cacheRet interface{}) []*module.Item {
	if cacheRet == nil {
		return nil
	}

	itemStr, ok := cacheRet.([]uint8)
	if !ok {
		return nil
	}

	items := make([]*module.Item, 0)
	parts := strings.Split(string(itemStr), ",")

	for _, part := range parts {
		if part == "" {
			continue
		}

		if strings.Contains(part, ":") {
			vars := strings.Split(part, ":")
			if len(vars) >= 2 {
				item := module.NewItem(vars[0])
				if score, err := strconv.ParseFloat(vars[1], 64); err == nil {
					item.Score = score
				}
				item.RetrieveId = r.modelName
				item.ItemType = r.itemType
				items = append(items, item)
			}
		} else {
			item := module.NewItem(part)
			item.RetrieveId = r.modelName
			item.ItemType = r.itemType
			items = append(items, item)
		}
	}

	return items
}

// writeCache 写入缓存.
func (r *GenerativeRecall) writeCache(userID module.UID, items []*module.Item) {
	key := r.cachePrefix + string(userID)
	var itemIDs string
	for _, item := range items {
		itemIDs += fmt.Sprintf("%s:%v", string(item.Id), item.Score) + ","
	}
	if len(itemIDs) > 0 {
		itemIDs = itemIDs[:len(itemIDs)-1]
	}

	if err := r.cache.Put(key, itemIDs, time.Duration(r.cacheTime)*time.Second); err != nil {
		log.Error(fmt.Sprintf("module=GenerativeRecall\terror=cache_write_failed:%v", err))
	}
}
