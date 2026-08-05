// recall/milvus_recall.go
//
// Milvus 向量召回服务实现.
// 调用 DSSM 召回 HTTP 服务 (inference/dssm_recall_server.py):
//   POST /recall {"user_id": "...", "topk": N}
//   -> {"code":200, "items":[{"item_id":"...","score":0.9}]}
//
// user 特征查询与向量检索都在服务端完成, 这里只做 HTTP 调用和结果转换.

package recall

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/alibaba/pairec/v2/context"
	"github.com/alibaba/pairec/v2/log"
	"github.com/alibaba/pairec/v2/module"
	"github.com/alibaba/pairec/v2/recconf"
	"github.com/alibaba/pairec/v2/service/recall"
	"github.com/alibaba/pairec/v2/utils"
)

// MilvusRecall Milvus 向量召回.
type MilvusRecall struct {
	*recall.BaseRecall
	modelName  string
	itemType   string
	serverURL  string
	timeoutMs  int
	topK       int
	httpClient *http.Client
}

// milvusRecallConfigJSON 从 RecallAlgo 字段解析配置
type milvusRecallConfigJSON struct {
	ServerURL string `json:"server_url"`
	TimeoutMs int    `json:"timeout_ms"`
	TopK      int    `json:"topk"`
}

// milvusRecallRequest 召回请求
type milvusRecallRequest struct {
	UserID string `json:"user_id"`
	TopK   int    `json:"topk"`
}

// milvusRecallItem 召回结果 item
type milvusRecallItem struct {
	ItemID string  `json:"item_id"`
	Score  float64 `json:"score"`
}

// milvusRecallResponse 召回响应
type milvusRecallResponse struct {
	Code      int                `json:"code"`
	Msg       string             `json:"msg"`
	Items     []milvusRecallItem `json:"items"`
	Source    string             `json:"source"`
	LatencyMs float64            `json:"latency_ms"`
}

// NewMilvusRecall 创建 Milvus 向量召回实例.
func NewMilvusRecall(conf recconf.RecallConfig) *MilvusRecall {
	algoConf := milvusRecallConfigJSON{
		ServerURL: "http://localhost:18200",
		TimeoutMs: 3000,
		TopK:      20,
	}
	if conf.RecallAlgo != "" {
		if err := json.Unmarshal([]byte(conf.RecallAlgo), &algoConf); err != nil {
			log.Error(fmt.Sprintf("module=MilvusRecall\tname=%s\terr=parse_recall_algo:%v",
				conf.Name, err))
		}
	}

	r := &MilvusRecall{
		BaseRecall: recall.NewBaseRecall(conf),
		modelName:  conf.Name,
		itemType:   conf.ItemType,
		serverURL:  algoConf.ServerURL,
		timeoutMs:  algoConf.TimeoutMs,
		topK:       algoConf.TopK,
		httpClient: &http.Client{
			Timeout: time.Duration(algoConf.TimeoutMs) * time.Millisecond,
		},
	}
	if r.topK <= 0 {
		r.topK = 20
	}
	fmt.Printf("[MilvusRecall] init name=%s server_url=%s topk=%d timeout_ms=%d\n",
		conf.Name, r.serverURL, r.topK, r.timeoutMs)
	return r
}

// GetCandidateItems 实现 pairec Recall 接口.
func (r *MilvusRecall) GetCandidateItems(user *module.User, ctx *context.RecommendContext) []*module.Item {
	stageStart := time.Now()

	reqBody, err := json.Marshal(milvusRecallRequest{
		UserID: string(user.Id),
		TopK:   r.topK,
	})
	if err != nil {
		log.Error(fmt.Sprintf("requestId=%s\tmodule=MilvusRecall\tname=%s\terr=marshal:%v",
			ctx.RecommendId, r.modelName, err))
		return nil
	}

	url := r.serverURL + "/recall"
	resp, err := r.httpClient.Post(url, "application/json", bytes.NewReader(reqBody))
	if err != nil {
		// fallback: Milvus/user tower 不可用时返回空, 不影响其他召回路
		log.Error(fmt.Sprintf("requestId=%s\tmodule=MilvusRecall\tname=%s\terr=http:%v",
			ctx.RecommendId, r.modelName, err))
		return nil
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Error(fmt.Sprintf("requestId=%s\tmodule=MilvusRecall\tname=%s\terr=read_body:%v",
			ctx.RecommendId, r.modelName, err))
		return nil
	}

	var recallResp milvusRecallResponse
	if err := json.Unmarshal(body, &recallResp); err != nil {
		log.Error(fmt.Sprintf("requestId=%s\tmodule=MilvusRecall\tname=%s\terr=unmarshal:%v",
			ctx.RecommendId, r.modelName, err))
		return nil
	}

	if recallResp.Code != 200 {
		log.Info(fmt.Sprintf("requestId=%s\tmodule=MilvusRecall\tname=%s\tuser=%s\tcode=%d\tmsg=%s",
			ctx.RecommendId, r.modelName, user.Id, recallResp.Code, recallResp.Msg))
		return nil
	}

	items := make([]*module.Item, 0, len(recallResp.Items))
	for _, rec := range recallResp.Items {
		item := module.NewItem(rec.ItemID)
		item.Score = rec.Score
		item.RetrieveId = r.modelName
		item.ItemType = r.itemType
		item.AddProperty("milvus_rank", len(items)+1)
		items = append(items, item)
	}

	log.Info(fmt.Sprintf("requestId=%s\tmodule=MilvusRecall\tname=%s\tuser=%s\tcount=%d\tsource=%s\tservice_ms=%.3f\tcost=%d",
		ctx.RecommendId, r.modelName, user.Id, len(items), recallResp.Source,
		recallResp.LatencyMs, utils.CostTime(stageStart)))
	return items
}
