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

	"github.com/alibaba/pairec/v2/persist/cache"
	"github.com/alibaba/pairec/v2/context"
	"github.com/alibaba/pairec/v2/log"
	"github.com/alibaba/pairec/v2/module"
	"github.com/alibaba/pairec/v2/recconf"
	"github.com/alibaba/pairec/v2/service/recall"
	"github.com/alibaba/pairec/v2/utils"

	"pairec4tigerllm/services/config"
)

// 调试日志函数
func writeDebugLog(format string, args ...interface{}) {
	f, _ := os.OpenFile("/tmp/recall_debug.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if f != nil {
		fmt.Fprintf(f, format+"\n", args...)
		f.Close()
	}
}

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
	modelName          string        // 自己保存 modelName
	itemType           string        // 自己保存 itemType
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
	cache              cache.Cache // 自己管理缓存
}

// recallConfigJSON 用于从 RecallAlgo 字段解析配置
type recallConfigJSON struct {
	ServerURL          string  `json:"server_url"`
	TopK               int     `json:"topk"`
	Temperature        float64 `json:"temperature"`
	BeamWidth          int     `json:"beam_width"`
	HistoryFrom        string  `json:"history_from"`
	HistoryFeatureName string  `json:"history_feature_name"`
	HistoryDelimiter   string  `json:"history_delimiter"`
	HistoryMaxLength   int     `json:"history_max_length"`
}

// NewGenerativeRecall 创建生成式召回实例.
func NewGenerativeRecall(conf recconf.RecallConfig) *GenerativeRecall {
	writeDebugLog(" NewGenerativeRecall called, name=%s, RecallAlgo=%s\n", 
		conf.Name, conf.RecallAlgo)
	
	// 从 RecallAlgo 字段解析 JSON 配置
	var algoConf recallConfigJSON
	var genConfig *config.GenerativeRecallConfig
	
	if conf.RecallAlgo != "" {
		if err := json.Unmarshal([]byte(conf.RecallAlgo), &algoConf); err == nil {
			writeDebugLog(" Parsed RecallAlgo: server_url=%s, history_feature_name=%s\n",
				algoConf.ServerURL, algoConf.HistoryFeatureName)
			genConfig = &config.GenerativeRecallConfig{
				ServerURL:          algoConf.ServerURL,
				TopK:               algoConf.TopK,
				Temperature:        algoConf.Temperature,
				BeamWidth:          algoConf.BeamWidth,
				HistoryFrom:        algoConf.HistoryFrom,
				HistoryFeatureName: algoConf.HistoryFeatureName,
				HistoryDelimiter:   algoConf.HistoryDelimiter,
				HistoryMaxLength:   algoConf.HistoryMaxLength,
				CacheEnable:        conf.CacheAdapter != "",
				CacheTime:          conf.CacheTime,
				CachePrefix:        conf.CachePrefix,
			}
		}
	}
	
	// 如果解析失败，使用默认配置
	if genConfig == nil {
		writeDebugLog(" Using default config\n")
		genConfig = config.DefaultGenerativeRecallConfig()
		genConfig.CacheEnable = conf.CacheAdapter != ""
		genConfig.CacheTime = conf.CacheTime
		genConfig.CachePrefix = conf.CachePrefix
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
	writeDebugLog(" Creating TRTLLMClient with server_url=%s\n", genConfig.ServerURL)
	client, err := NewTRTLLMClient(genConfig)
	if err != nil {
		// 客户端创建失败时 panic，在服务启动时就能发现问题
		panic(fmt.Sprintf("failed to create TRTLLMClient: %v", err))
	}
	writeDebugLog(" TRTLLMClient created successfully\n")

	// 加载语义 ID 映射
	semanticIDMapPath := "../data/tenrec/processed/semantic_id_map.json"
	loadSemanticIDMap(semanticIDMapPath)

	// 初始化缓存
	var c cache.Cache
	if conf.CacheAdapter != "" {
		c, _ = cache.NewCache(conf.CacheAdapter, conf.CacheConfig)
	}

	// 创建召回实例
	recallInstance := &GenerativeRecall{
		BaseRecall:         recall.NewBaseRecall(conf),
		modelName:          conf.Name,
		itemType:           conf.ItemType,
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
		cache:              c,
	}
	
	writeDebugLog(" GenerativeRecall instance created: name=%s, historyFeatureName=%s, topK=%d\n",
		recallInstance.modelName, recallInstance.historyFeatureName, recallInstance.topK)

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
	
	// 调试日志 - 输出到文件
	f, _ := os.OpenFile("/tmp/recall_debug.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if f != nil {
		fmt.Fprintf(f, "[DEBUG] GetCandidateItems called, user=%s, modelName=%s\n", user.Id, r.modelName)
		f.Close()
	}
	writeDebugLog(" GetCandidateItems called, user=%s\n", user.Id)

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
	
	writeDebugLog("[DEBUG] Cache miss or empty, getting user history")

	// 获取用户历史行为
	history, err := r.getUserHistory(user, ctx)
	if err != nil {
		writeDebugLog(" getUserHistory error: %v\n", err)
		log.Error(fmt.Sprintf("requestId=%s\tmodule=GenerativeRecall\tname=%s\terr=get_user_history:%v",
			ctx.RecommendId, r.modelName, err))
		return nil
	}
	
	writeDebugLog(" Got history, len=%d\n", len(history))
	
	// 限制历史长度，避免超过模型 max_seq_len (50)
	// 留 1 个位置给特殊 token，所以最多 49 条
	if len(history) > 49 {
		history = history[len(history)-49:]
		writeDebugLog("[DEBUG] History truncated to len=%d", len(history))
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
	
	writeDebugLog(" Calling inference service, history=%v\n", semanticHistory)

	response, err := r.client.Recommend(request)
	if err != nil {
		writeDebugLog(" Inference error: %v\n", err)
		log.Error(fmt.Sprintf("requestId=%s\tmodule=GenerativeRecall\tname=%s\terr=generative_recommend:%v",
			ctx.RecommendId, r.modelName, err))
		return nil
	}
	
	writeDebugLog(" Got response, recommendations=%d\n", len(response.Recommendations))

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

// userFeaturesCache 缓存用户特征
var userFeaturesCache map[string]map[string]interface{}
var userFeaturesCacheOnce sync.Once

// loadUserFeatures 加载用户特征文件
func loadUserFeatures() {
	userFeaturesCacheOnce.Do(func() {
		userFeaturesCache = make(map[string]map[string]interface{})
		data, err := os.ReadFile("../data/user_features.json")
		if err != nil {
			log.Error(fmt.Sprintf("Failed to load user features: %v", err))
			return
		}
		if err := json.Unmarshal(data, &userFeaturesCache); err != nil {
			log.Error(fmt.Sprintf("Failed to parse user features: %v", err))
			return
		}
		log.Info(fmt.Sprintf("Loaded user features: %d users", len(userFeaturesCache)))
	})
}

// getHistoryFromUserFeature 从用户特征获取历史.
func (r *GenerativeRecall) getHistoryFromUserFeature(user *module.User) ([]int, error) {
	loadUserFeatures()
	
	if r.historyFeatureName == "" {
		return nil, fmt.Errorf("history_feature_name is empty")
	}

	// 从文件缓存获取
	userData, ok := userFeaturesCache[string(user.Id)]
	if !ok {
		return nil, fmt.Errorf("user %s not found in features", user.Id)
	}
	
	historyStr := ""
	if val, ok := userData[r.historyFeatureName]; ok {
		historyStr = fmt.Sprintf("%v", val)
	}
	
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
		var semIDs []int
		
		// 优先从映射表查询
		if semanticIDMap != nil {
			if fullIDs, ok := semanticIDMap[itemID]; ok {
				semanticHistory = append(semanticHistory, fullIDs)
				continue
			}
		}

		// 回退到哈希方式（生成 4 层，值限制在 0-255）
		semIDs = []int{
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
