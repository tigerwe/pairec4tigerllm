package ranksort

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	gosort "sort"
	"time"

	"github.com/alibaba/pairec/v2/module"
	"github.com/alibaba/pairec/v2/recconf"
	pairecsort "github.com/alibaba/pairec/v2/sort"
)

const (
	RankErrorContextKey = "deepfm_rank_error"
	RankTraceContextKey = "deepfm_rank_trace"
)

type Config struct {
	Name               string `json:"name"`
	ServerURL          string `json:"server_url"`
	TimeoutMS          int    `json:"timeout_ms"`
	ExpectedCandidates int    `json:"expected_candidates"`
	RequiredModelRole  string `json:"required_model_role"`
}

type userDefineConfig struct {
	DeepFMRankSorts []Config `json:"DeepFMRankSorts"`
}

type requestItem struct {
	ItemID string `json:"item_id"`
}

type rankRequest struct {
	RequestID string        `json:"request_id"`
	UserID    string        `json:"user_id"`
	Items     []requestItem `json:"items"`
}

type responseItem struct {
	ItemID string  `json:"item_id"`
	Score  float64 `json:"score"`
}

type responseTrace struct {
	FeatureMS         float64 `json:"feature_ms"`
	ForwardMS         float64 `json:"forward_ms"`
	TotalMS           float64 `json:"total_ms"`
	ProfileMissing    bool    `json:"profile_missing"`
	UserOOV           bool    `json:"user_oov"`
	ItemOOVCount      int     `json:"item_oov_count"`
	CategoryOOVCount  int     `json:"category_oov_count"`
	GenderOOV         bool    `json:"gender_oov"`
	AgeOOV            bool    `json:"age_oov"`
	HistoryValidCount int     `json:"history_valid_count"`
	HistoryOOVCount   int     `json:"history_oov_count"`
	ScoreUniqueCount  int     `json:"score_unique_count"`
	ScoreMin          float64 `json:"score_min"`
	ScoreMax          float64 `json:"score_max"`
}

type rankResponse struct {
	Code         int            `json:"code"`
	Message      string         `json:"msg"`
	RequestID    string         `json:"request_id"`
	ModelVersion string         `json:"model_version"`
	ModelRole    string         `json:"model_role"`
	Items        []responseItem `json:"items"`
	Trace        responseTrace  `json:"trace"`
}

type DeepFMRankSort struct {
	config Config
	client *http.Client
}

func NewDeepFMRankSort(config Config) (*DeepFMRankSort, error) {
	if config.Name == "" {
		return nil, errors.New("DeepFM rank sort name is required")
	}
	if config.ServerURL == "" {
		return nil, errors.New("DeepFM rank server_url is required")
	}
	if config.TimeoutMS <= 0 {
		return nil, errors.New("DeepFM rank timeout_ms must be positive")
	}
	if config.ExpectedCandidates <= 0 {
		return nil, errors.New("DeepFM expected_candidates must be positive")
	}
	if config.RequiredModelRole == "" {
		return nil, errors.New("DeepFM required_model_role is required")
	}
	transport := &http.Transport{
		Proxy: nil,
		DialContext: (&net.Dialer{
			Timeout: time.Duration(config.TimeoutMS) * time.Millisecond,
		}).DialContext,
	}
	return &DeepFMRankSort{
		config: config,
		client: &http.Client{
			Transport: transport,
			Timeout:   time.Duration(config.TimeoutMS) * time.Millisecond,
		},
	}, nil
}

func RegisterFromConfig() error {
	if recconf.Config == nil || len(recconf.Config.UserDefineConfs) == 0 {
		return nil
	}
	var userConfig userDefineConfig
	if err := json.Unmarshal(recconf.Config.UserDefineConfs, &userConfig); err != nil {
		return fmt.Errorf("parse DeepFM rank config: %w", err)
	}
	for _, config := range userConfig.DeepFMRankSorts {
		instance, err := NewDeepFMRankSort(config)
		if err != nil {
			return err
		}
		pairecsort.RegisterSort(config.Name, instance)
		fmt.Printf("Registering DeepFMRankSort: %s endpoint=%s timeout_ms=%d candidates=%d model_role=%s\n",
			config.Name, config.ServerURL, config.TimeoutMS, config.ExpectedCandidates,
			config.RequiredModelRole)
	}
	return nil
}

func (s *DeepFMRankSort) Sort(sortData *pairecsort.SortData) error {
	items, ok := sortData.Data.([]*module.Item)
	if !ok {
		return s.fail(sortData, errors.New("sort data is not []*module.Item"))
	}
	if sortData.Context == nil || sortData.User == nil {
		return s.fail(sortData, errors.New("rank context and user are required"))
	}
	if len(items) != s.config.ExpectedCandidates {
		return s.fail(sortData, fmt.Errorf("expected %d candidates, got %d",
			s.config.ExpectedCandidates, len(items)))
	}

	requestItems := make([]requestItem, 0, len(items))
	originalOrder := make([]string, 0, len(items))
	inputSet := make(map[string]struct{}, len(items))
	for _, item := range items {
		if item == nil || item.Id == "" {
			return s.fail(sortData, errors.New("candidate has empty item_id"))
		}
		itemID := string(item.Id)
		if _, duplicate := inputSet[itemID]; duplicate {
			return s.fail(sortData, fmt.Errorf("duplicate candidate item_id=%s", itemID))
		}
		inputSet[itemID] = struct{}{}
		originalOrder = append(originalOrder, itemID)
		requestItems = append(requestItems, requestItem{ItemID: itemID})
	}

	payload := rankRequest{
		RequestID: sortData.Context.RecommendId,
		UserID:    string(sortData.User.Id),
		Items:     requestItems,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return s.fail(sortData, fmt.Errorf("encode rank request: %w", err))
	}
	started := time.Now()
	req, err := http.NewRequest(http.MethodPost, s.config.ServerURL+"/rank", bytes.NewReader(body))
	if err != nil {
		return s.fail(sortData, fmt.Errorf("create rank request: %w", err))
	}
	req.Header.Set("Content-Type", "application/json")
	response, err := s.client.Do(req)
	if err != nil {
		return s.fail(sortData, fmt.Errorf("call rank service: %w", err))
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return s.fail(sortData, fmt.Errorf("rank service HTTP status=%d", response.StatusCode))
	}
	responseBody, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return s.fail(sortData, fmt.Errorf("read rank response: %w", err))
	}
	var ranked rankResponse
	if err := json.Unmarshal(responseBody, &ranked); err != nil {
		return s.fail(sortData, fmt.Errorf("decode rank response: %w", err))
	}
	if err := validateResponse(ranked, payload, inputSet, s.config.RequiredModelRole); err != nil {
		return s.fail(sortData, err)
	}

	scores := make(map[string]float64, len(ranked.Items))
	for _, rankedItem := range ranked.Items {
		scores[rankedItem.ItemID] = rankedItem.Score
	}
	for _, item := range items {
		item.Score = scores[string(item.Id)]
		item.AddAlgoScore("deepfm", item.Score)
		item.AddProperty("deepfm_score", item.Score)
	}
	gosort.SliceStable(items, func(left, right int) bool {
		return items[left].Score > items[right].Score
	})
	reordered := false
	for index, item := range items {
		if string(item.Id) != originalOrder[index] {
			reordered = true
			break
		}
	}
	sortData.Data = items
	clientTotalMS := float64(time.Since(started).Microseconds()) / 1000
	trace := map[string]interface{}{
		"model_version":       ranked.ModelVersion,
		"model_role":          ranked.ModelRole,
		"candidate_count":     len(items),
		"service_feature_ms":  ranked.Trace.FeatureMS,
		"service_forward_ms":  ranked.Trace.ForwardMS,
		"service_total_ms":    ranked.Trace.TotalMS,
		"client_total_ms":     clientTotalMS,
		"reordered":           reordered,
		"profile_missing":     ranked.Trace.ProfileMissing,
		"user_oov":            ranked.Trace.UserOOV,
		"item_oov_count":      ranked.Trace.ItemOOVCount,
		"category_oov_count":  ranked.Trace.CategoryOOVCount,
		"gender_oov":          ranked.Trace.GenderOOV,
		"age_oov":             ranked.Trace.AgeOOV,
		"history_valid_count": ranked.Trace.HistoryValidCount,
		"history_oov_count":   ranked.Trace.HistoryOOVCount,
		"score_unique_count":  ranked.Trace.ScoreUniqueCount,
		"score_min":           ranked.Trace.ScoreMin,
		"score_max":           ranked.Trace.ScoreMax,
	}
	sortData.Context.AddContextParam(RankTraceContextKey, trace)
	if os.Getenv("PAIREC_TRACE_STDOUT") == "1" {
		encoded, _ := json.Marshal(map[string]interface{}{
			"event":               "deepfm_rank_complete",
			"request_id":          payload.RequestID,
			"model_version":       ranked.ModelVersion,
			"model_role":          ranked.ModelRole,
			"candidate_count":     len(items),
			"service_feature_ms":  ranked.Trace.FeatureMS,
			"service_forward_ms":  ranked.Trace.ForwardMS,
			"service_total_ms":    ranked.Trace.TotalMS,
			"client_total_ms":     clientTotalMS,
			"reordered":           reordered,
			"profile_missing":     ranked.Trace.ProfileMissing,
			"user_oov":            ranked.Trace.UserOOV,
			"item_oov_count":      ranked.Trace.ItemOOVCount,
			"category_oov_count":  ranked.Trace.CategoryOOVCount,
			"gender_oov":          ranked.Trace.GenderOOV,
			"age_oov":             ranked.Trace.AgeOOV,
			"history_valid_count": ranked.Trace.HistoryValidCount,
			"history_oov_count":   ranked.Trace.HistoryOOVCount,
			"score_unique_count":  ranked.Trace.ScoreUniqueCount,
			"score_min":           ranked.Trace.ScoreMin,
			"score_max":           ranked.Trace.ScoreMax,
		})
		fmt.Println(string(encoded))
	}
	return nil
}

func validateResponse(response rankResponse, request rankRequest,
	inputSet map[string]struct{}, requiredModelRole string) error {
	if response.Code != http.StatusOK {
		return fmt.Errorf("rank service code=%d msg=%s", response.Code, response.Message)
	}
	if response.RequestID != request.RequestID {
		return fmt.Errorf("rank request_id mismatch: got %q", response.RequestID)
	}
	if response.ModelVersion == "" {
		return errors.New("rank model_version is empty")
	}
	if response.ModelRole != requiredModelRole {
		return fmt.Errorf("rank model_role mismatch: got %q want %q",
			response.ModelRole, requiredModelRole)
	}
	if response.Trace.ScoreUniqueCount <= 0 {
		return errors.New("rank score_unique_count must be positive")
	}
	if len(response.Items) != len(request.Items) {
		return fmt.Errorf("rank item count mismatch: got %d want %d",
			len(response.Items), len(request.Items))
	}
	seen := make(map[string]struct{}, len(response.Items))
	for _, item := range response.Items {
		if _, expected := inputSet[item.ItemID]; !expected {
			return fmt.Errorf("rank response has unknown item_id=%s", item.ItemID)
		}
		if _, duplicate := seen[item.ItemID]; duplicate {
			return fmt.Errorf("rank response has duplicate item_id=%s", item.ItemID)
		}
		if math.IsNaN(item.Score) || math.IsInf(item.Score, 0) {
			return fmt.Errorf("rank response has non-finite score item_id=%s", item.ItemID)
		}
		seen[item.ItemID] = struct{}{}
	}
	return nil
}

func (s *DeepFMRankSort) fail(sortData *pairecsort.SortData, err error) error {
	if sortData.Context != nil {
		sortData.Context.AddContextParam(RankErrorContextKey, err.Error())
		sortData.Context.LogError("module=DeepFMRankSort\terr=" + err.Error())
		if os.Getenv("PAIREC_TRACE_STDOUT") == "1" {
			encoded, _ := json.Marshal(map[string]interface{}{
				"event":      "deepfm_rank_error",
				"request_id": sortData.Context.RecommendId,
				"error":      err.Error(),
			})
			fmt.Println(string(encoded))
		}
	}
	sortData.Data = []*module.Item{}
	return err
}

var _ pairecsort.ISort = (*DeepFMRankSort)(nil)
