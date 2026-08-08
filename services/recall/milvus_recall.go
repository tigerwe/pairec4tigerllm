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
	stdcontext "context"
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
	proto "github.com/gogo/protobuf/proto"
	"pairec4tigerllm/services/observability"
	"pairec4tigerllm/services/pipelineclient"
	"pairec4tigerllm/services/pipelinepb"
)

// MilvusRecall Milvus 向量召回.
type MilvusRecall struct {
	*recall.BaseRecall
	modelName  string
	itemType   string
	serverURL  string
	protocol   string
	brpcClient *pipelineclient.VectorClient
	timeoutMs  int
	topK       int
	httpClient *http.Client
}

// milvusRecallConfigJSON 从 RecallAlgo 字段解析配置
type milvusRecallConfigJSON struct {
	Protocol        string `json:"protocol"`
	ServerURL       string `json:"server_url"`
	BRPCEndpoint    string `json:"brpc_endpoint"`
	BRPCServiceName string `json:"brpc_service_name"`
	TimeoutMs       int    `json:"timeout_ms"`
	TopK            int    `json:"topk"`
}

// milvusRecallRequest 召回请求
type milvusRecallRequest struct {
	RequestID string                 `json:"request_id"`
	Context   map[string]interface{} `json:"context"`
	UserID    string                 `json:"user_id"`
	TopK      int                    `json:"topk"`
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
	Trace     struct {
		FeatureUS      int64 `json:"feature_us"`
		ComputeUS      int64 `json:"compute_us"`
		BackendTotalUS int64 `json:"backend_total_us"`
		TotalUS        int64 `json:"total_us"`
	} `json:"trace"`
}

// NewMilvusRecall 创建 Milvus 向量召回实例.
func NewMilvusRecall(conf recconf.RecallConfig) *MilvusRecall {
	algoConf := milvusRecallConfigJSON{
		Protocol:  "http",
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
		protocol:   algoConf.Protocol,
		timeoutMs:  algoConf.TimeoutMs,
		topK:       algoConf.TopK,
		httpClient: &http.Client{
			Timeout: time.Duration(algoConf.TimeoutMs) * time.Millisecond,
		},
	}
	if r.protocol == "" {
		r.protocol = "http"
	}
	if r.protocol == "brpc" {
		client, err := pipelineclient.NewVectorClient(
			algoConf.BRPCEndpoint, algoConf.BRPCServiceName,
			time.Duration(algoConf.TimeoutMs)*time.Millisecond)
		if err != nil {
			log.Error(fmt.Sprintf("module=MilvusRecall\tname=%s\terr=brpc_config:%v", conf.Name, err))
		} else {
			r.brpcClient = client
		}
	} else if r.protocol != "http" {
		log.Error(fmt.Sprintf("module=MilvusRecall\tname=%s\terr=unsupported_protocol:%s", conf.Name, r.protocol))
	}
	if r.topK <= 0 {
		r.topK = 20
	}
	fmt.Printf("[MilvusRecall] init name=%s protocol=%s server_url=%s brpc_endpoint=%s topk=%d timeout_ms=%d\n",
		conf.Name, r.protocol, r.serverURL, algoConf.BRPCEndpoint, r.topK, r.timeoutMs)
	return r
}

// GetCandidateItems 实现 pairec Recall 接口.
func (r *MilvusRecall) GetCandidateItems(user *module.User, ctx *context.RecommendContext) []*module.Item {
	stageStart := time.Now()
	recallResp, err := r.call(user, ctx)
	if err != nil {
		log.Error(fmt.Sprintf("requestId=%s\tmodule=MilvusRecall\tname=%s\terr=%s:%v",
			ctx.RecommendId, r.modelName, r.protocol, err))
		observability.RecordDuration(ctx, "vector_recall", "vector_recall", r.protocol,
			"recall", false, stageStart, "error", map[string]interface{}{"error": err.Error()})
		return nil
	}

	if recallResp.Code != 200 {
		log.Info(fmt.Sprintf("requestId=%s\tmodule=MilvusRecall\tname=%s\tuser=%s\tcode=%d\tmsg=%s",
			ctx.RecommendId, r.modelName, user.Id, recallResp.Code, recallResp.Msg))
		observability.RecordDuration(ctx, "vector_recall", "vector_recall", r.protocol,
			"recall", false, stageStart, "error", map[string]interface{}{"code": recallResp.Code})
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

	serviceUS := recallResp.Trace.TotalUS
	if serviceUS <= 0 {
		serviceUS = int64(recallResp.LatencyMs * 1000)
	}
	observability.RecordDuration(ctx, "vector_recall", "vector_recall", r.protocol,
		"recall", false, stageStart, "ok", map[string]interface{}{
			"item_count": len(items), "source": recallResp.Source,
			"service_total_us": serviceUS, "feature_us": recallResp.Trace.FeatureUS,
			"compute_us": recallResp.Trace.ComputeUS,
		})

	log.Info(fmt.Sprintf("requestId=%s\tmodule=MilvusRecall\tname=%s\tuser=%s\tcount=%d\tsource=%s\tservice_ms=%.3f\tcost=%d",
		ctx.RecommendId, r.modelName, user.Id, len(items), recallResp.Source,
		float64(serviceUS)/1000, utils.CostTime(stageStart)))
	writeTraceStdout(
		"requestId=%s request_id=%s module=MilvusRecall from=service name=%s user=%s protocol=%s count=%d source=%s service_us=%d cost=%d",
		ctx.RecommendId, ctx.RecommendId, r.modelName, user.Id, r.protocol, len(items), recallResp.Source,
		serviceUS, utils.CostTime(stageStart),
	)
	return items
}

func (r *MilvusRecall) call(user *module.User, ctx *context.RecommendContext) (milvusRecallResponse, error) {
	if r.protocol == "brpc" {
		return r.callBRPC(user, ctx)
	}
	if r.protocol != "http" {
		return milvusRecallResponse{}, fmt.Errorf("unsupported protocol %q", r.protocol)
	}
	return r.callHTTP(user, ctx)
}

func (r *MilvusRecall) callHTTP(user *module.User, ctx *context.RecommendContext) (milvusRecallResponse, error) {
	traceContext := map[string]interface{}{
		"request_id": ctx.RecommendId, "span_id": "vector-recall", "parent_span_id": "recall",
		"sampled": true, "contract_version": pipelinepb.TraceContractVersion,
	}

	reqBody, err := json.Marshal(milvusRecallRequest{
		RequestID: ctx.RecommendId,
		Context:   traceContext,
		UserID:    string(user.Id),
		TopK:      r.topK,
	})
	if err != nil {
		return milvusRecallResponse{}, fmt.Errorf("marshal request: %w", err)
	}

	url := r.serverURL + "/recall"
	resp, err := r.httpClient.Post(url, "application/json", bytes.NewReader(reqBody))
	if err != nil {
		return milvusRecallResponse{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return milvusRecallResponse{}, fmt.Errorf("vector recall service HTTP status=%d", resp.StatusCode)
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return milvusRecallResponse{}, fmt.Errorf("read body: %w", err)
	}

	var recallResp milvusRecallResponse
	if err := json.Unmarshal(body, &recallResp); err != nil {
		return milvusRecallResponse{}, fmt.Errorf("decode response: %w", err)
	}
	return recallResp, nil
}

func (r *MilvusRecall) callBRPC(user *module.User, ctx *context.RecommendContext) (milvusRecallResponse, error) {
	if r.brpcClient == nil {
		return milvusRecallResponse{}, fmt.Errorf("BRPC client is not configured")
	}
	timeout := time.Duration(r.timeoutMs) * time.Millisecond
	callCtx, cancel := stdcontext.WithTimeout(stdcontext.Background(), timeout)
	defer cancel()
	response, err := r.brpcClient.Recall(callCtx, &pipelinepb.VectorRecallRequest{
		Context: pipelineclient.NewTraceContext(ctx.RecommendId, "vector-recall", "recall", timeout),
		UserID:  proto.String(string(user.Id)), TopK: proto.Int32(int32(r.topK)),
	})
	if err != nil {
		return milvusRecallResponse{}, err
	}
	mapped := milvusRecallResponse{Code: int(pipelinepb.Int32(response.Code)), Msg: pipelinepb.String(response.Message), Source: pipelinepb.String(response.Source)}
	for _, item := range response.Items {
		mapped.Items = append(mapped.Items, milvusRecallItem{ItemID: pipelinepb.String(item.ItemID), Score: pipelinepb.Float64(item.Score)})
	}
	if response.Trace != nil {
		mapped.Trace.FeatureUS = pipelinepb.Int64(response.Trace.FeatureUS)
		mapped.Trace.ComputeUS = pipelinepb.Int64(response.Trace.ComputeUS)
		mapped.Trace.BackendTotalUS = pipelinepb.Int64(response.Trace.BackendTotalUS)
		mapped.Trace.TotalUS = pipelinepb.Int64(response.Trace.TotalUS)
	}
	return mapped, nil
}
