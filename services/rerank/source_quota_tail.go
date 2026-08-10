package rerank

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

	paireccontext "github.com/alibaba/pairec/v2/context"
	"github.com/alibaba/pairec/v2/module"
	"github.com/alibaba/pairec/v2/recconf"
	"github.com/prometheus/client_golang/prometheus"
	"pairec4tigerllm/services/observability"
)

const (
	ErrorContextKey = "rerank_error"
	PolicyName      = "source_quota_tail"
)

type Config struct {
	Name               string `json:"name"`
	Enabled            bool   `json:"enabled"`
	GenerativeSource   string `json:"generative_source"`
	VectorSource       string `json:"vector_source"`
	ExpectedCandidates int    `json:"expected_candidates"`
	MinimumGenerative  int    `json:"minimum_generative"`
	MaximumGenerative  int    `json:"max_generative"`
	Placement          string `json:"placement"`
	FailClosed         bool   `json:"fail_closed"`
}

type userDefineConfig struct {
	RerankConfs []Config `json:"RerankConfs"`
}

type Result struct {
	Policy             string
	InputCount         int
	OutputCount        int
	GenerativeInput    int
	VectorInput        int
	GenerativeSelected int
	VectorSelected     int
	MovedCount         int
}

type SourceQuotaTail struct {
	config Config
}

var (
	configuredMu sync.RWMutex
	configured   *SourceQuotaTail

	rerankDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "pairec", Subsystem: "rerank", Name: "duration_seconds",
		Help:    "In-process rerank duration by bounded policy and status labels.",
		Buckets: []float64{.00001, .000025, .00005, .0001, .00025, .0005, .001, .002, .005},
	}, []string{"policy", "status"})
	rerankTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "pairec", Subsystem: "rerank", Name: "requests_total",
		Help: "In-process rerank requests by bounded policy and status labels.",
	}, []string{"policy", "status"})
)

func init() {
	for _, collector := range []prometheus.Collector{rerankDuration, rerankTotal} {
		if err := prometheus.Register(collector); err != nil {
			if _, ok := err.(prometheus.AlreadyRegisteredError); !ok {
				panic(err)
			}
		}
	}
}

func NewSourceQuotaTail(config Config) (*SourceQuotaTail, error) {
	if !config.Enabled {
		return nil, errors.New("source quota rerank must be enabled")
	}
	if config.Name == "" {
		return nil, errors.New("rerank name is required")
	}
	if config.Name != PolicyName {
		return nil, fmt.Errorf("unsupported rerank policy %q", config.Name)
	}
	if config.GenerativeSource == "" || config.VectorSource == "" {
		return nil, errors.New("rerank generative_source and vector_source are required")
	}
	if config.GenerativeSource == config.VectorSource {
		return nil, errors.New("rerank sources must be distinct")
	}
	if config.ExpectedCandidates <= 0 {
		return nil, errors.New("rerank expected_candidates must be positive")
	}
	if config.MinimumGenerative <= 0 || config.MaximumGenerative < config.MinimumGenerative {
		return nil, errors.New("rerank generative quota is invalid")
	}
	if config.Placement != "tail" {
		return nil, fmt.Errorf("unsupported rerank placement %q", config.Placement)
	}
	if !config.FailClosed {
		return nil, errors.New("source quota rerank must fail closed")
	}
	return &SourceQuotaTail{config: config}, nil
}

func RegisterFromConfig() error {
	configuredMu.Lock()
	defer configuredMu.Unlock()
	configured = nil
	if recconf.Config == nil || len(recconf.Config.UserDefineConfs) == 0 {
		return nil
	}
	var userConfig userDefineConfig
	if err := json.Unmarshal(recconf.Config.UserDefineConfs, &userConfig); err != nil {
		return fmt.Errorf("parse rerank config: %w", err)
	}
	for _, config := range userConfig.RerankConfs {
		if !config.Enabled {
			continue
		}
		if configured != nil {
			return errors.New("only one rerank policy may be enabled")
		}
		instance, err := NewSourceQuotaTail(config)
		if err != nil {
			return err
		}
		configured = instance
		fmt.Printf("Registering rerank: %s sources=%s,%s quota=%d..%d placement=%s fail_closed=true\n",
			config.Name, config.VectorSource, config.GenerativeSource,
			config.MinimumGenerative, config.MaximumGenerative, config.Placement)
	}
	return nil
}

func ApplyConfigured(ctx *paireccontext.RecommendContext, items []*module.Item, size int) ([]*module.Item, bool, error) {
	configuredMu.RLock()
	instance := configured
	configuredMu.RUnlock()
	if instance == nil {
		return items, false, nil
	}
	started := time.Now()
	resultItems, result, err := instance.Apply(items, size)
	status := "ok"
	attributes := result.attributes(instance.config)
	if err != nil {
		status = "error"
		attributes["error"] = err.Error()
		if ctx != nil {
			ctx.AddContextParam(ErrorContextKey, err.Error())
			ctx.LogError("module=SourceQuotaRerank\terr=" + err.Error())
		}
		resultItems = []*module.Item{}
	}
	duration := time.Since(started)
	observability.RecordDuration(ctx, "rerank", "pairec", "in_process", "recommend_service", false,
		started, status, attributes)
	rerankDuration.WithLabelValues(PolicyName, status).Observe(duration.Seconds())
	rerankTotal.WithLabelValues(PolicyName, status).Inc()
	writeTrace(ctx, result, status, duration, err)
	return resultItems, true, err
}

func (r *SourceQuotaTail) Apply(items []*module.Item, size int) ([]*module.Item, Result, error) {
	result := Result{Policy: PolicyName, InputCount: len(items)}
	if len(items) != r.config.ExpectedCandidates {
		return nil, result, fmt.Errorf("expected %d candidates, got %d",
			r.config.ExpectedCandidates, len(items))
	}
	if size <= 0 || size > len(items) {
		return nil, result, fmt.Errorf("invalid output size %d for %d candidates", size, len(items))
	}

	generative := make([]*module.Item, 0, r.config.MaximumGenerative)
	vectors := make([]*module.Item, 0, len(items))
	seen := make(map[module.ItemId]struct{}, len(items))
	for _, item := range items {
		if item == nil || item.Id == "" {
			return nil, result, errors.New("candidate has empty item_id")
		}
		if _, duplicate := seen[item.Id]; duplicate {
			return nil, result, fmt.Errorf("duplicate candidate item_id=%s", item.Id)
		}
		seen[item.Id] = struct{}{}
		switch item.RetrieveId {
		case r.config.GenerativeSource:
			generative = append(generative, item)
		case r.config.VectorSource:
			vectors = append(vectors, item)
		default:
			return nil, result, fmt.Errorf("unknown retrieve source %q for item_id=%s",
				item.RetrieveId, item.Id)
		}
	}
	result.GenerativeInput = len(generative)
	result.VectorInput = len(vectors)
	if len(generative) < r.config.MinimumGenerative {
		return nil, result, fmt.Errorf("missing generative candidates: got %d want at least %d",
			len(generative), r.config.MinimumGenerative)
	}

	generativeCount := minInt(r.config.MaximumGenerative, len(generative), size)
	if generativeCount < r.config.MinimumGenerative {
		return nil, result, fmt.Errorf("output size %d cannot satisfy minimum generative quota %d",
			size, r.config.MinimumGenerative)
	}
	vectorCount := size - generativeCount
	if len(vectors) < vectorCount {
		return nil, result, fmt.Errorf("insufficient vector candidates: got %d want %d",
			len(vectors), vectorCount)
	}

	output := make([]*module.Item, 0, size)
	output = append(output, vectors[:vectorCount]...)
	output = append(output, generative[:generativeCount]...)
	result.OutputCount = len(output)
	result.GenerativeSelected = generativeCount
	result.VectorSelected = vectorCount
	for index, item := range output {
		if items[index].Id != item.Id {
			result.MovedCount++
		}
		item.AddProperty("rerank_position", index+1)
		item.AddProperty("rerank_reason", PolicyName)
	}
	return output, result, nil
}

func (r Result) attributes(config Config) map[string]interface{} {
	return map[string]interface{}{
		"policy": r.Policy, "placement": config.Placement,
		"input_count": r.InputCount, "output_count": r.OutputCount,
		"generative_input": r.GenerativeInput, "vector_input": r.VectorInput,
		"generative_selected": r.GenerativeSelected, "vector_selected": r.VectorSelected,
		"minimum_generative": config.MinimumGenerative,
		"maximum_generative": config.MaximumGenerative, "moved_count": r.MovedCount,
	}
}

func writeTrace(ctx *paireccontext.RecommendContext, result Result, status string,
	duration time.Duration, err error) {
	if os.Getenv("PAIREC_TRACE_STDOUT") != "1" {
		return
	}
	requestID := ""
	if ctx != nil {
		requestID = ctx.RecommendId
	}
	event := map[string]interface{}{
		"event": "source_quota_rerank_complete", "request_id": requestID,
		"policy": PolicyName, "status": status, "duration_us": duration.Microseconds(),
		"input_count": result.InputCount, "output_count": result.OutputCount,
		"generative_input": result.GenerativeInput, "vector_input": result.VectorInput,
		"generative_selected": result.GenerativeSelected, "vector_selected": result.VectorSelected,
		"moved_count": result.MovedCount,
	}
	if err != nil {
		event["error"] = err.Error()
	}
	encoded, _ := json.Marshal(event)
	fmt.Println(string(encoded))
}

func minInt(values ...int) int {
	result := values[0]
	for _, value := range values[1:] {
		if value < result {
			result = value
		}
	}
	return result
}
