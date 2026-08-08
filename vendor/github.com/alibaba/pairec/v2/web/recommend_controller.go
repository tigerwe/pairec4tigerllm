package web

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/alibaba/pairec/v2/abtest"
	"github.com/alibaba/pairec/v2/context"
	"github.com/alibaba/pairec/v2/log"
	"github.com/alibaba/pairec/v2/recconf"
	"github.com/alibaba/pairec/v2/service"
	"github.com/alibaba/pairec/v2/utils"
	"github.com/aliyun/aliyun-pairec-config-go-sdk/v2/model"
	"pairec4tigerllm/services/observability"
)

const (
	Default_Size int = 10
)

type RecommendParam struct {
	SceneId  string                 `json:"scene_id"`
	Category string                 `json:"category"`
	Uid      string                 `json:"uid"`  // user id
	Size     int                    `json:"size"` // get recommend items size
	Debug    bool                   `json:"debug"`
	Features map[string]interface{} `json:"features"`
}

func (r *RecommendParam) GetParameter(name string) interface{} {
	if name == "uid" {
		return r.Uid
	} else if name == "scene" {
		return r.SceneId
	} else if name == "category" {
		if r.Category != "" {
			return r.Category
		}
		return "default"
	} else if name == "features" {
		return r.Features
	}

	return nil
}

type RecommendResponse struct {
	Response
	Size  int         `json:"size"`
	Items []*ItemData `json:"items"`
}
type ItemData struct {
	ItemId     string  `json:"item_id"`
	ItemType   string  `json:"item_type"`
	RetrieveId string  `json:"retrieve_id"`
	Score      float64 `json:"score"`
}

func (r *RecommendResponse) ToString() string {
	j, _ := json.Marshal(r)
	return string(j)
}

type RecommendController struct {
	Controller
	param   RecommendParam
	context *context.RecommendContext
}

func (c *RecommendController) Process(w http.ResponseWriter, r *http.Request) {
	c.Start = time.Now()
	var err error
	c.RequestBody, err = io.ReadAll(r.Body)
	if err != nil {
		c.SendError(w, ERROR_PARAMETER_CODE, "read parammeter error")
		return
	}
	if len(c.RequestBody) == 0 {
		c.SendError(w, ERROR_PARAMETER_CODE, "request body empty")
		return
	}
	c.RequestId = utils.UUID()
	c.LogRequestBegin(r)
	if err := c.CheckParameter(); err != nil {
		c.SendError(w, ERROR_PARAMETER_CODE, err.Error())
		return
	}
	c.doProcess(w, r)
	c.End = time.Now()
	c.LogRequestEnd(r)
}
func (r *RecommendController) CheckParameter() error {
	if err := json.Unmarshal(r.RequestBody, &r.param); err != nil {
		return err
	}

	if len(r.param.Uid) == 0 {
		return errors.New("uid not empty")
	}
	if r.param.Size <= 0 {
		r.param.Size = Default_Size
	}
	if r.param.SceneId == "" {
		r.param.SceneId = "default_scene"
	}
	if r.param.Category == "" {
		r.param.Category = "default"
	}

	return nil
}
func (c *RecommendController) doProcess(w http.ResponseWriter, r *http.Request) {
	c.makeRecommendContext()
	recorder := observability.NewRecorder(c.RequestId)
	observability.Attach(c.context, recorder)
	traceStatus := "ok"
	defer func() {
		if c.context.GetContextParam("deepfm_rank_error") != nil {
			traceStatus = "error"
		}
		observability.FinalizeContext(c.context, traceStatus)
	}()
	userRecommendService := service.NewUserRecommendService()
	recommendStarted := time.Now()
	items := userRecommendService.Recommend(c.context)
	observability.RecordDuration(c.context, "recommend_service", "pairec", "in_process", "", true,
		recommendStarted, "ok", map[string]interface{}{"item_count": len(items)})
	if c.context.GetContextParam("deepfm_rank_error") != nil {
		traceStatus = "error"
	}
	responseStarted := time.Now()
	defer func() {
		observability.RecordDuration(c.context, "response_build", "pairec", "http", "", true,
			responseStarted, traceStatus, nil)
	}()
	if response := c.deepFMRankFailureResponse(); response != nil {
		io.WriteString(w, response.ToString())
		return
	}
	data := make([]*ItemData, 0)
	for _, item := range items {
		if c.param.Debug {
			fmt.Println(item)
		}

		idata := &ItemData{
			ItemId:     string(item.Id),
			ItemType:   item.ItemType,
			RetrieveId: item.RetrieveId,
			Score:      item.Score,
		}

		data = append(data, idata)
	}

	if len(data) < c.param.Size && !allowPartialRecommendResults(len(data)) {
		response := RecommendResponse{
			Size:  len(data),
			Items: data,
			Response: Response{
				RequestId: c.RequestId,
				Code:      299,
				Message:   "items size not enough",
			},
		}
		io.WriteString(w, response.ToString())
		return
	}
	message := "success"
	if len(data) < c.param.Size {
		message = "partial success"
	}

	response := RecommendResponse{
		Size:  len(data),
		Items: data,
		Response: Response{
			RequestId: c.RequestId,
			Code:      200,
			Message:   message,
		},
	}
	io.WriteString(w, response.ToString())
}

func (c *RecommendController) deepFMRankFailureResponse() *RecommendResponse {
	if c.context == nil || c.context.GetContextParam("deepfm_rank_error") == nil {
		return nil
	}
	return &RecommendResponse{
		Size:  0,
		Items: []*ItemData{},
		Response: Response{
			RequestId: c.RequestId,
			Code:      SERVER_ERROR_CODE,
			Message:   "deepfm rank failed",
		},
	}
}

func allowPartialRecommendResults(itemCount int) bool {
	return itemCount > 0 && os.Getenv("PAIREC_ALLOW_PARTIAL_RESULTS") == "1"
}
func (c *RecommendController) makeRecommendContext() {
	c.context = context.NewRecommendContext()
	c.context.Size = c.param.Size
	c.context.Debug = c.param.Debug
	c.context.Param = &c.param
	c.context.RecommendId = c.RequestId
	c.context.Config = recconf.Config

	abcontext := model.ExperimentContext{
		Uid:          c.param.Uid,
		RequestId:    c.RequestId,
		FilterParams: map[string]interface{}{},
	}

	if abtest.GetExperimentClient() != nil {
		c.context.ExperimentResult = abtest.GetExperimentClient().MatchExperiment(c.param.SceneId, &abcontext)
		log.Info(c.context.ExperimentResult.Info())
	}
}
