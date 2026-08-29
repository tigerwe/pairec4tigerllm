package ranksort

import (
	"bytes"
	stdcontext "context"
	"crypto/sha256"
	"encoding/binary"
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
	proto "github.com/gogo/protobuf/proto"
	"pairec4tigerllm/services/observability"
	"pairec4tigerllm/services/pipelineclient"
	"pairec4tigerllm/services/pipelinepb"
)

const (
	RankErrorContextKey = "deepfm_rank_error"
	RankTraceContextKey = "deepfm_rank_trace"
)

type Config struct {
	Name                           string `json:"name"`
	Protocol                       string `json:"protocol"`
	ServerURL                      string `json:"server_url"`
	BRPCEndpoint                   string `json:"brpc_endpoint"`
	BRPCServiceName                string `json:"brpc_service_name"`
	TimeoutMS                      int    `json:"timeout_ms"`
	ExpectedCandidates             int    `json:"expected_candidates"`
	RequiredModelRole              string `json:"required_model_role"`
	BRPCPayloadBytes               int    `json:"brpc_payload_bytes"`
	BRPCBurstEnabled               bool   `json:"brpc_burst_enabled"`
	BRPCBurstConcurrency           int    `json:"brpc_burst_concurrency"`
	BRPCBurstPoolSize              int    `json:"brpc_burst_pool_size"`
	BRPCBurstPayloadBytes          int    `json:"brpc_burst_payload_bytes"`
	BRPCBurstPreconnect            bool   `json:"brpc_burst_preconnect"`
	BRPCBurstPressureTimeoutMS     int    `json:"brpc_burst_pressure_timeout_ms"`
	PostRankHopsEnabled            bool   `json:"post_rank_hops_enabled"`
	PostRankHop1Endpoint           string `json:"post_rank_hop1_endpoint"`
	PostRankServiceName            string `json:"post_rank_service_name"`
	PostRankTimeoutMS              int    `json:"post_rank_timeout_ms"`
	PostRankBurstConcurrency       int    `json:"post_rank_burst_concurrency"`
	PostRankBurstPoolSize          int    `json:"post_rank_burst_pool_size"`
	PostRankPayloadBytes           int    `json:"post_rank_payload_bytes"`
	PostRankPressureTimeoutMS      int    `json:"post_rank_pressure_timeout_ms"`
	PostRankPressureStartQuorum    int    `json:"post_rank_pressure_start_quorum"`
	PostRankPressureStartTimeoutMS int    `json:"post_rank_pressure_start_timeout_ms"`
}

type userDefineConfig struct {
	DeepFMRankSorts []Config `json:"DeepFMRankSorts"`
}

type requestItem struct {
	ItemID string `json:"item_id"`
}

type rankRequest struct {
	RequestID string                 `json:"request_id"`
	Context   map[string]interface{} `json:"context,omitempty"`
	UserID    string                 `json:"user_id"`
	Items     []requestItem          `json:"items"`
}

type responseItem struct {
	ItemID string  `json:"item_id"`
	Score  float64 `json:"score"`
}

type responseTrace struct {
	FeatureMS            float64 `json:"feature_ms"`
	ForwardMS            float64 `json:"forward_ms"`
	TotalMS              float64 `json:"total_ms"`
	ProfileMissing       bool    `json:"profile_missing"`
	UserOOV              bool    `json:"user_oov"`
	ItemOOVCount         int     `json:"item_oov_count"`
	CategoryOOVCount     int     `json:"category_oov_count"`
	GenderOOV            bool    `json:"gender_oov"`
	AgeOOV               bool    `json:"age_oov"`
	HistoryValidCount    int     `json:"history_valid_count"`
	HistoryOOVCount      int     `json:"history_oov_count"`
	ScoreUniqueCount     int     `json:"score_unique_count"`
	ScoreMin             float64 `json:"score_min"`
	ScoreMax             float64 `json:"score_max"`
	FeatureUS            int64   `json:"feature_us"`
	ComputeUS            int64   `json:"compute_us"`
	BackendRPCUS         int64   `json:"backend_rpc_us"`
	BackendTotalUS       int64   `json:"backend_total_us"`
	TotalUS              int64   `json:"total_us"`
	ClientStartEpochNS   int64   `json:"client_start_epoch_ns"`
	ClientEndEpochNS     int64   `json:"client_end_epoch_ns"`
	BusinessPayloadBytes int     `json:"business_payload_bytes"`
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
	config     Config
	client     *http.Client
	brpcClient *pipelineclient.RankClient
	brpcBurst  *RankBurstCoordinator
	postRank   *RankBurstCoordinator
}

func NewDeepFMRankSort(config Config) (*DeepFMRankSort, error) {
	if config.Name == "" {
		return nil, errors.New("DeepFM rank sort name is required")
	}
	if config.Protocol == "" {
		config.Protocol = "http"
	}
	if config.Protocol != "http" && config.Protocol != "brpc" {
		return nil, fmt.Errorf("DeepFM rank protocol must be http or brpc, got %q", config.Protocol)
	}
	if config.Protocol == "http" && config.ServerURL == "" {
		return nil, errors.New("DeepFM rank server_url is required for HTTP")
	}
	if config.Protocol == "brpc" && config.BRPCEndpoint == "" {
		return nil, errors.New("DeepFM rank brpc_endpoint is required for BRPC")
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
	if config.BRPCPayloadBytes < 0 || config.BRPCPayloadBytes > 1<<20 {
		return nil, errors.New("DeepFM rank brpc_payload_bytes must be in [0,1048576]")
	}
	if config.BRPCBurstEnabled {
		if config.BRPCBurstConcurrency < 1 || config.BRPCBurstConcurrency > 1000 {
			return nil, errors.New("DeepFM rank burst concurrency must be in [1,1000]")
		}
		if config.BRPCBurstPoolSize == 0 {
			config.BRPCBurstPoolSize = config.BRPCBurstConcurrency
		}
		if config.BRPCBurstPayloadBytes < 0 || config.BRPCBurstPayloadBytes > 1<<20 {
			return nil, errors.New("DeepFM rank burst payload bytes must be in [0,1048576]")
		}
		if !config.BRPCBurstPreconnect {
			return nil, errors.New("DeepFM rank burst requires strict preconnect")
		}
		if config.BRPCBurstPressureTimeoutMS <= 0 {
			return nil, errors.New("DeepFM rank burst pressure timeout must be positive")
		}
	}
	if config.PostRankHopsEnabled {
		if config.Protocol != "brpc" {
			return nil, errors.New("post-rank hops require BRPC rank protocol")
		}
		if config.PostRankHop1Endpoint == "" {
			return nil, errors.New("post-rank hop1 endpoint is required")
		}
		if config.PostRankTimeoutMS <= 0 || config.PostRankTimeoutMS > 1500 {
			return nil, errors.New("post-rank timeout must be in [1,1500] ms")
		}
		if config.PostRankBurstConcurrency != 1000 || config.PostRankBurstPoolSize != 1000 {
			return nil, errors.New("post-rank burst requires concurrency=1000 and pool_size=1000")
		}
		if config.PostRankPayloadBytes != 102400 {
			return nil, errors.New("post-rank payload must be exactly 102400 bytes")
		}
		if config.PostRankPressureTimeoutMS != 5000 {
			return nil, errors.New("post-rank pressure timeout must be exactly 5000 ms")
		}
		if config.PostRankPressureStartQuorum < 1 ||
			config.PostRankPressureStartQuorum > config.PostRankBurstConcurrency-1 {
			return nil, errors.New("post-rank pressure start quorum must be in [1,999]")
		}
		if config.PostRankPressureStartTimeoutMS < 1 || config.PostRankPressureStartTimeoutMS > 1000 {
			return nil, errors.New("post-rank pressure start timeout must be in [1,1000] ms")
		}
	}
	transport := &http.Transport{
		Proxy: nil,
		DialContext: (&net.Dialer{
			Timeout: time.Duration(config.TimeoutMS) * time.Millisecond,
		}).DialContext,
	}
	ranker := &DeepFMRankSort{
		config: config,
		client: &http.Client{
			Transport: transport,
			Timeout:   time.Duration(config.TimeoutMS) * time.Millisecond,
		},
	}
	if config.Protocol == "brpc" {
		businessTimeout := time.Duration(config.TimeoutMS) * time.Millisecond
		client, err := pipelineclient.NewRankClient(
			config.BRPCEndpoint, config.BRPCServiceName,
			businessTimeout)
		if err != nil {
			return nil, err
		}
		ranker.brpcClient = client
		if config.BRPCBurstEnabled {
			pressureTimeout := time.Duration(config.BRPCBurstPressureTimeoutMS) * time.Millisecond
			// Burst sessions carry both business and pressure lanes. Their transport
			// deadline must not clip the longer pressure context; the outer per-lane
			// contexts still enforce the shorter business timeout.
			burstClient, err := pipelineclient.NewRankClient(
				config.BRPCEndpoint, config.BRPCServiceName,
				rankBurstTransportTimeout(businessTimeout, pressureTimeout))
			if err != nil {
				return nil, err
			}
			burst, err := NewRankBurstCoordinator(burstClient, RankBurstConfig{
				Concurrency:     config.BRPCBurstConcurrency,
				PoolSize:        config.BRPCBurstPoolSize,
				PressureBytes:   config.BRPCBurstPayloadBytes,
				BusinessBytes:   config.BRPCPayloadBytes,
				BusinessTimeout: businessTimeout,
				PressureTimeout: pressureTimeout,
			})
			if err != nil {
				return nil, fmt.Errorf("initialize DeepFM rank burst: %w", err)
			}
			ranker.brpcBurst = burst
		}
		if config.PostRankHopsEnabled {
			postTimeout := time.Duration(config.PostRankTimeoutMS) * time.Millisecond
			pressureTimeout := time.Duration(config.PostRankPressureTimeoutMS) * time.Millisecond
			postClient, err := pipelineclient.NewRankClient(
				config.PostRankHop1Endpoint, config.PostRankServiceName,
				rankBurstTransportTimeout(postTimeout, pressureTimeout))
			if err != nil {
				return nil, fmt.Errorf("create post-rank hop1 client: %w", err)
			}
			postRank, err := NewRankBurstCoordinator(postClient, RankBurstConfig{
				Concurrency: config.PostRankBurstConcurrency, PoolSize: config.PostRankBurstPoolSize,
				PressureBytes: config.PostRankPayloadBytes, BusinessBytes: config.PostRankPayloadBytes,
				BusinessTimeout: postTimeout, PressureTimeout: pressureTimeout,
				EventPrefix:           "pairec_post_rank_hop1_brpc_burst",
				TraceComponent:        "post_rank_hop1",
				DedicatedBusinessLane: true,
				PressureStartQuorum:   config.PostRankPressureStartQuorum,
				PressureStartTimeout:  time.Duration(config.PostRankPressureStartTimeoutMS) * time.Millisecond,
			})
			if err != nil {
				return nil, fmt.Errorf("initialize post-rank hop1 burst: %w", err)
			}
			ranker.postRank = postRank
		}
	}
	return ranker, nil
}

func rankBurstTransportTimeout(businessTimeout, pressureTimeout time.Duration) time.Duration {
	if pressureTimeout > businessTimeout {
		return pressureTimeout
	}
	return businessTimeout
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
		endpoint := config.ServerURL
		if config.Protocol == "brpc" {
			endpoint = config.BRPCEndpoint
		}
		fmt.Printf("Registering DeepFMRankSort: %s protocol=%s endpoint=%s timeout_ms=%d candidates=%d model_role=%s\n",
			config.Name, config.Protocol, endpoint, config.TimeoutMS, config.ExpectedCandidates,
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
		Context: map[string]interface{}{
			"request_id":       sortData.Context.RecommendId,
			"span_id":          "deepfm-rank",
			"parent_span_id":   "sort",
			"sampled":          true,
			"contract_version": pipelinepb.TraceContractVersion,
		},
		UserID: string(sortData.User.Id),
		Items:  requestItems,
	}
	started := time.Now()
	ranked, err := s.call(payload)
	if err != nil {
		observability.RecordDuration(sortData.Context, "deepfm_rank", "deepfm_rank", s.config.Protocol,
			"sort", false, started, "error", map[string]interface{}{"error": err.Error()})
		return s.fail(sortData, err)
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
	observability.RecordDuration(sortData.Context, "deepfm_rank", "deepfm_rank", s.config.Protocol,
		"sort", false, started, "ok", map[string]interface{}{
			"candidate_count": len(items), "model_version": ranked.ModelVersion,
			"service_total_us": ranked.Trace.TotalUS, "feature_us": ranked.Trace.FeatureUS,
			"compute_us": ranked.Trace.ComputeUS, "backend_rpc_us": ranked.Trace.BackendRPCUS,
		})
	if s.postRank != nil {
		postStarted := time.Now()
		postTrace, err := s.callPostRankHops(sortData, items)
		if err != nil {
			observability.RecordDuration(sortData.Context, "post_rank_two_hop", "post_rank_hops", "brpc",
				"sort", false, postStarted, "error", map[string]interface{}{"error": err.Error()})
			return s.fail(sortData, err)
		}
		observability.RecordDuration(sortData.Context, "post_rank_two_hop", "post_rank_hops", "brpc",
			"sort", false, postStarted, "ok", postTrace)
	}
	trace := map[string]interface{}{
		"protocol":               s.config.Protocol,
		"model_version":          ranked.ModelVersion,
		"model_role":             ranked.ModelRole,
		"candidate_count":        len(items),
		"service_feature_ms":     ranked.Trace.FeatureMS,
		"service_forward_ms":     ranked.Trace.ForwardMS,
		"service_total_ms":       ranked.Trace.TotalMS,
		"service_total_us":       ranked.Trace.TotalUS,
		"service_feature_us":     ranked.Trace.FeatureUS,
		"service_compute_us":     ranked.Trace.ComputeUS,
		"adapter_backend_rpc_us": ranked.Trace.BackendRPCUS,
		"client_start_epoch_ns":  ranked.Trace.ClientStartEpochNS,
		"client_end_epoch_ns":    ranked.Trace.ClientEndEpochNS,
		"business_payload_bytes": ranked.Trace.BusinessPayloadBytes,
		"client_total_ms":        clientTotalMS,
		"reordered":              reordered,
		"profile_missing":        ranked.Trace.ProfileMissing,
		"user_oov":               ranked.Trace.UserOOV,
		"item_oov_count":         ranked.Trace.ItemOOVCount,
		"category_oov_count":     ranked.Trace.CategoryOOVCount,
		"gender_oov":             ranked.Trace.GenderOOV,
		"age_oov":                ranked.Trace.AgeOOV,
		"history_valid_count":    ranked.Trace.HistoryValidCount,
		"history_oov_count":      ranked.Trace.HistoryOOVCount,
		"score_unique_count":     ranked.Trace.ScoreUniqueCount,
		"score_min":              ranked.Trace.ScoreMin,
		"score_max":              ranked.Trace.ScoreMax,
	}
	sortData.Context.AddContextParam(RankTraceContextKey, trace)
	if os.Getenv("PAIREC_TRACE_STDOUT") == "1" {
		encoded, _ := json.Marshal(map[string]interface{}{
			"event":                  "deepfm_rank_complete",
			"request_id":             payload.RequestID,
			"model_version":          ranked.ModelVersion,
			"model_role":             ranked.ModelRole,
			"candidate_count":        len(items),
			"service_feature_ms":     ranked.Trace.FeatureMS,
			"service_forward_ms":     ranked.Trace.ForwardMS,
			"service_total_ms":       ranked.Trace.TotalMS,
			"client_total_ms":        clientTotalMS,
			"client_start_epoch_ns":  ranked.Trace.ClientStartEpochNS,
			"client_end_epoch_ns":    ranked.Trace.ClientEndEpochNS,
			"business_payload_bytes": ranked.Trace.BusinessPayloadBytes,
			"reordered":              reordered,
			"profile_missing":        ranked.Trace.ProfileMissing,
			"user_oov":               ranked.Trace.UserOOV,
			"item_oov_count":         ranked.Trace.ItemOOVCount,
			"category_oov_count":     ranked.Trace.CategoryOOVCount,
			"gender_oov":             ranked.Trace.GenderOOV,
			"age_oov":                ranked.Trace.AgeOOV,
			"history_valid_count":    ranked.Trace.HistoryValidCount,
			"history_oov_count":      ranked.Trace.HistoryOOVCount,
			"score_unique_count":     ranked.Trace.ScoreUniqueCount,
			"score_min":              ranked.Trace.ScoreMin,
			"score_max":              ranked.Trace.ScoreMax,
		})
		fmt.Println(string(encoded))
	}
	return nil
}

func (s *DeepFMRankSort) callPostRankHops(sortData *pairecsort.SortData,
	items []*module.Item) (map[string]interface{}, error) {
	requestID := sortData.Context.RecommendId
	timeout := time.Duration(s.config.PostRankTimeoutMS) * time.Millisecond
	request := &pipelinepb.RankRequest{
		Context: pipelineclient.NewTraceContext(requestID, "post-rank-hop1", "sort", timeout),
		UserID:  proto.String(string(sortData.User.Id)), Items: make([]*pipelinepb.RankCandidate, 0, len(items)),
		PayloadPadding: make([]byte, s.config.PostRankPayloadBytes),
	}
	orderedIDs := make([]string, 0, len(items))
	for _, item := range items {
		itemID := string(item.Id)
		orderedIDs = append(orderedIDs, itemID)
		request.Items = append(request.Items, &pipelinepb.RankCandidate{ItemID: proto.String(itemID)})
	}
	inputHash := orderedCandidateSHA256(orderedIDs)
	started := time.Now()
	response, err := s.postRank.Rank(request, requestID)
	ended := time.Now()
	if err != nil {
		return nil, fmt.Errorf("post-rank two-hop BRPC failed: %w", err)
	}
	if pipelinepb.Int32(response.Code) != http.StatusOK {
		return nil, fmt.Errorf("post-rank two-hop code=%d msg=%s",
			pipelinepb.Int32(response.Code), pipelinepb.String(response.Message))
	}
	if len(response.Items) != len(orderedIDs) {
		return nil, fmt.Errorf("post-rank item count mismatch: got %d want %d",
			len(response.Items), len(orderedIDs))
	}
	outputIDs := make([]string, 0, len(response.Items))
	for index, rankedItem := range response.Items {
		itemID := pipelinepb.String(rankedItem.ItemID)
		if itemID != orderedIDs[index] {
			return nil, fmt.Errorf("post-rank order mismatch at index=%d got=%q want=%q",
				index, itemID, orderedIDs[index])
		}
		outputIDs = append(outputIDs, itemID)
	}
	outputHash := orderedCandidateSHA256(outputIDs)
	if outputHash != inputHash {
		return nil, fmt.Errorf("post-rank candidate sha mismatch: got=%s want=%s", outputHash, inputHash)
	}
	serviceUS := int64(0)
	if response.Trace != nil {
		serviceUS = pipelinepb.Int64(response.Trace.TotalUS)
	}
	event := map[string]interface{}{
		"event": "post_rank_two_hop_complete", "request_id": requestID,
		"candidate_count": len(orderedIDs), "candidate_sha256": inputHash,
		"client_total_ms":  float64(ended.Sub(started).Microseconds()) / 1000,
		"service_total_ms": float64(serviceUS) / 1000,
		"front_brpc_ms":    float64(ended.Sub(started).Microseconds()-serviceUS) / 1000,
		"payload_bytes":    len(request.PayloadPadding), "strict_order_valid": true,
	}
	if os.Getenv("PAIREC_TRACE_STDOUT") == "1" {
		encoded, _ := json.Marshal(event)
		fmt.Println(string(encoded))
	}
	return event, nil
}

func orderedCandidateSHA256(itemIDs []string) string {
	hash := sha256.New()
	var length [4]byte
	for _, itemID := range itemIDs {
		binary.BigEndian.PutUint32(length[:], uint32(len(itemID)))
		_, _ = hash.Write(length[:])
		_, _ = hash.Write([]byte(itemID))
	}
	return fmt.Sprintf("%x", hash.Sum(nil))
}

func (s *DeepFMRankSort) call(payload rankRequest) (rankResponse, error) {
	if s.config.Protocol == "brpc" {
		return s.callBRPC(payload)
	}
	return s.callHTTP(payload)
}

func (s *DeepFMRankSort) callHTTP(payload rankRequest) (rankResponse, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return rankResponse{}, fmt.Errorf("encode rank request: %w", err)
	}
	req, err := http.NewRequest(http.MethodPost, s.config.ServerURL+"/rank", bytes.NewReader(body))
	if err != nil {
		return rankResponse{}, fmt.Errorf("create rank request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	response, err := s.client.Do(req)
	if err != nil {
		return rankResponse{}, fmt.Errorf("call rank service: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return rankResponse{}, fmt.Errorf("rank service HTTP status=%d", response.StatusCode)
	}
	responseBody, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return rankResponse{}, fmt.Errorf("read rank response: %w", err)
	}
	var ranked rankResponse
	if err := json.Unmarshal(responseBody, &ranked); err != nil {
		return rankResponse{}, fmt.Errorf("decode rank response: %w", err)
	}
	return ranked, nil
}

func (s *DeepFMRankSort) callBRPC(payload rankRequest) (rankResponse, error) {
	if s.brpcClient == nil {
		return rankResponse{}, errors.New("BRPC rank client is not configured")
	}
	timeout := time.Duration(s.config.TimeoutMS) * time.Millisecond
	request := &pipelinepb.RankRequest{
		Context:        pipelineclient.NewTraceContext(payload.RequestID, "deepfm-rank", "sort", timeout),
		UserID:         proto.String(payload.UserID),
		Items:          make([]*pipelinepb.RankCandidate, 0, len(payload.Items)),
		PayloadPadding: make([]byte, s.config.BRPCPayloadBytes),
	}
	for _, item := range payload.Items {
		request.Items = append(request.Items, &pipelinepb.RankCandidate{ItemID: proto.String(item.ItemID)})
	}
	callStarted := time.Now()
	var response *pipelinepb.RankResponse
	var err error
	if s.brpcBurst != nil {
		response, err = s.brpcBurst.Rank(request, payload.RequestID)
	} else {
		callContext, cancel := stdcontext.WithTimeout(stdcontext.Background(), timeout)
		response, err = s.brpcClient.Rank(callContext, request)
		cancel()
	}
	callEnded := time.Now()
	if err != nil {
		return rankResponse{}, fmt.Errorf("call BRPC rank service: %w", err)
	}
	ranked := rankResponse{
		Code:         int(pipelinepb.Int32(response.Code)),
		Message:      pipelinepb.String(response.Message),
		RequestID:    payload.RequestID,
		ModelVersion: pipelinepb.String(response.ModelVersion),
		ModelRole:    pipelinepb.String(response.ModelRole),
	}
	ranked.Trace.ClientStartEpochNS = callStarted.UnixNano()
	ranked.Trace.ClientEndEpochNS = callEnded.UnixNano()
	ranked.Trace.BusinessPayloadBytes = len(request.PayloadPadding)
	for _, item := range response.Items {
		ranked.Items = append(ranked.Items, responseItem{
			ItemID: pipelinepb.String(item.ItemID), Score: pipelinepb.Float64(item.Score),
		})
	}
	if coverage := response.Coverage; coverage != nil {
		ranked.Trace.ProfileMissing = pipelinepb.Bool(coverage.ProfileMissing)
		ranked.Trace.UserOOV = pipelinepb.Bool(coverage.UserOOV)
		ranked.Trace.ItemOOVCount = int(pipelinepb.Int32(coverage.ItemOOVCount))
		ranked.Trace.CategoryOOVCount = int(pipelinepb.Int32(coverage.CategoryOOVCount))
		ranked.Trace.GenderOOV = pipelinepb.Bool(coverage.GenderOOV)
		ranked.Trace.AgeOOV = pipelinepb.Bool(coverage.AgeOOV)
		ranked.Trace.HistoryValidCount = int(pipelinepb.Int32(coverage.HistoryValidCount))
		ranked.Trace.HistoryOOVCount = int(pipelinepb.Int32(coverage.HistoryOOVCount))
		ranked.Trace.ScoreUniqueCount = int(pipelinepb.Int32(coverage.ScoreUniqueCount))
		ranked.Trace.ScoreMin = pipelinepb.Float64(coverage.ScoreMin)
		ranked.Trace.ScoreMax = pipelinepb.Float64(coverage.ScoreMax)
	}
	if trace := response.Trace; trace != nil {
		ranked.Trace.FeatureUS = pipelinepb.Int64(trace.FeatureUS)
		ranked.Trace.ComputeUS = pipelinepb.Int64(trace.ComputeUS)
		ranked.Trace.BackendRPCUS = pipelinepb.Int64(trace.BackendRPCUS)
		ranked.Trace.BackendTotalUS = pipelinepb.Int64(trace.BackendTotalUS)
		ranked.Trace.TotalUS = pipelinepb.Int64(trace.TotalUS)
		ranked.Trace.FeatureMS = float64(ranked.Trace.FeatureUS) / 1000
		ranked.Trace.ForwardMS = float64(ranked.Trace.ComputeUS) / 1000
		ranked.Trace.TotalMS = float64(ranked.Trace.TotalUS) / 1000
	}
	return ranked, nil
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
