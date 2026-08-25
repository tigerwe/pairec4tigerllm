package observability

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"sync"
	"time"

	paireccontext "github.com/alibaba/pairec/v2/context"
	"github.com/prometheus/client_golang/prometheus"
)

const (
	ContextKey      = "pairec_pipeline_trace_recorder"
	ContractVersion = "pairec.pipeline_trace.v1"
)

type Span struct {
	ID            string                 `json:"span_id"`
	ParentID      string                 `json:"parent_span_id,omitempty"`
	Name          string                 `json:"name"`
	Component     string                 `json:"component"`
	Protocol      string                 `json:"protocol,omitempty"`
	Status        string                 `json:"status"`
	Enabled       bool                   `json:"enabled"`
	Accounted     bool                   `json:"accounted"`
	StartOffsetUS int64                  `json:"start_offset_us"`
	DurationUS    int64                  `json:"duration_us"`
	Attributes    map[string]interface{} `json:"attributes,omitempty"`
}

type DataSystemAttribution struct {
	Expected            bool   `json:"expected"`
	Complete            bool   `json:"attribution_complete"`
	SynchronousGetCount int    `json:"synchronous_get_count"`
	SynchronousSetCount int    `json:"synchronous_set_count"`
	SynchronousGetUS    int64  `json:"synchronous_get_us"`
	SynchronousSetUS    int64  `json:"synchronous_set_us"`
	AsynchronousCount   int    `json:"asynchronous_count"`
	AsynchronousUS      int64  `json:"asynchronous_us"`
	Reason              string `json:"reason,omitempty"`
}

type Trace struct {
	Event              string                `json:"event"`
	ContractVersion    string                `json:"contract_version"`
	RequestID          string                `json:"request_id"`
	Status             string                `json:"status"`
	Sampled            bool                  `json:"sampled"`
	Valid              bool                  `json:"valid"`
	InvalidReasons     []string              `json:"invalid_reasons,omitempty"`
	StartEpochNS       int64                 `json:"start_epoch_ns"`
	EndEpochNS         int64                 `json:"end_epoch_ns"`
	PaiRecTotalUS      int64                 `json:"pairec_total_us"`
	AccountedUS        int64                 `json:"accounted_us"`
	ClosureErrorUS     int64                 `json:"closure_error_us"`
	ClosureThresholdUS int64                 `json:"closure_threshold_us"`
	Spans              []Span                `json:"spans"`
	DataSystem         DataSystemAttribution `json:"datasystem"`
}

type Recorder struct {
	mu         sync.Mutex
	requestID  string
	started    time.Time
	sampled    bool
	closed     bool
	nextSpanID int64
	spans      []Span
	invalid    []string
	ds         DataSystemAttribution
}

var (
	traceDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "pairec", Subsystem: "pipeline", Name: "span_duration_seconds",
		Help:    "Request pipeline span duration with bounded labels.",
		Buckets: []float64{.001, .002, .003, .005, .008, .01, .02, .05, .1, .2, .5, 1, 5},
	}, []string{"component", "protocol", "status"})
	traceTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "pairec", Subsystem: "pipeline", Name: "trace_total",
		Help: "Completed request traces by validity and status.",
	}, []string{"valid", "status"})
	traceClosureError = prometheus.NewHistogram(prometheus.HistogramOpts{
		Namespace: "pairec", Subsystem: "pipeline", Name: "closure_error_seconds",
		Help:    "Absolute request trace closure error.",
		Buckets: []float64{.0001, .0005, .001, .003, .005, .01, .02, .05},
	})
	servicePhaseDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "pairec", Subsystem: "pipeline", Name: "service_phase_duration_seconds",
		Help:    "Service-reported phase duration with bounded component and phase labels.",
		Buckets: []float64{.0001, .0005, .001, .002, .003, .005, .008, .01, .02, .05, .1, .2, .5, 1, 5},
	}, []string{"component", "phase"})
)

func init() {
	for _, collector := range []prometheus.Collector{
		traceDuration, traceTotal, traceClosureError, servicePhaseDuration,
	} {
		if err := prometheus.Register(collector); err != nil {
			if _, ok := err.(prometheus.AlreadyRegisteredError); !ok {
				panic(err)
			}
		}
	}
}

func NewRecorder(requestID string) *Recorder {
	return &Recorder{
		requestID: requestID,
		started:   time.Now(),
		sampled:   shouldSample(requestID),
		ds: DataSystemAttribution{
			Expected: os.Getenv("PAIREC_DATASYSTEM_EXPECTED") == "1",
			Complete: false,
			Reason:   "native_trt_kvc_request_identity_not_propagated",
		},
	}
}

func Attach(ctx *paireccontext.RecommendContext, recorder *Recorder) {
	if ctx != nil && recorder != nil {
		ctx.AddContextParam(ContextKey, recorder)
	}
}

func FromContext(ctx *paireccontext.RecommendContext) *Recorder {
	if ctx == nil {
		return nil
	}
	recorder, _ := ctx.GetContextParam(ContextKey).(*Recorder)
	return recorder
}

func RecordDuration(ctx *paireccontext.RecommendContext, name, component, protocol, parent string,
	accounted bool, started time.Time, status string, attributes map[string]interface{}) {
	if recorder := FromContext(ctx); recorder != nil {
		recorder.Record(name, component, protocol, parent, accounted, started, time.Since(started), status, attributes)
	}
}

func (r *Recorder) Record(name, component, protocol, parent string, accounted bool,
	started time.Time, duration time.Duration, status string, attributes map[string]interface{}) {
	if duration < 0 {
		r.Invalidate("negative_duration:" + name)
		return
	}
	if status == "" {
		status = "ok"
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return
	}
	r.nextSpanID++
	span := Span{
		ID: fmt.Sprintf("%s-%d", r.requestID, r.nextSpanID), ParentID: parent,
		Name: name, Component: component, Protocol: protocol, Status: status,
		Enabled: true, Accounted: accounted,
		StartOffsetUS: started.Sub(r.started).Microseconds(), DurationUS: duration.Microseconds(),
		Attributes: cloneAttributes(attributes),
	}
	if span.StartOffsetUS < 0 {
		span.StartOffsetUS = 0
	}
	r.spans = append(r.spans, span)
	traceDuration.WithLabelValues(normalizeLabel(component), normalizeLabel(protocol), normalizeLabel(status)).Observe(duration.Seconds())
	for _, phase := range []string{
		"service_total_us", "feature_us", "compute_us", "backend_rpc_us",
		"inference_total_us", "runner_generate_us", "rpc_us", "history_us", "convert_us",
	} {
		if valueUS, ok := numericMicroseconds(attributes[phase]); ok && valueUS >= 0 {
			servicePhaseDuration.WithLabelValues(normalizeLabel(component), phase).
				Observe(float64(valueUS) / 1e6)
		}
	}
}

func (r *Recorder) RecordDisabled(name, component, parent string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return
	}
	r.nextSpanID++
	r.spans = append(r.spans, Span{
		ID: fmt.Sprintf("%s-%d", r.requestID, r.nextSpanID), ParentID: parent,
		Name: name, Component: component, Status: "disabled", Enabled: false,
	})
}

func (r *Recorder) Invalidate(reason string) {
	if reason == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, existing := range r.invalid {
		if existing == reason {
			return
		}
	}
	r.invalid = append(r.invalid, reason)
}

func (r *Recorder) SetDataSystemAttribution(attribution DataSystemAttribution) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ds = attribution
}

func (r *Recorder) Finalize(status string) Trace {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return Trace{Event: "pipeline_trace_complete", RequestID: r.requestID, Status: "duplicate_finalize", Valid: false}
	}
	r.closed = true
	ended := time.Now()
	totalUS := ended.Sub(r.started).Microseconds()
	accountedUS := int64(0)
	for _, span := range r.spans {
		if span.Enabled && span.Accounted {
			accountedUS += span.DurationUS
		}
	}
	if residual := totalUS - accountedUS; residual >= 0 {
		r.nextSpanID++
		r.spans = append(r.spans, Span{
			ID: fmt.Sprintf("%s-%d", r.requestID, r.nextSpanID), Name: "controller_overhead",
			Component: "pairec", Protocol: "in_process", Status: "ok", Enabled: true,
			Accounted: true, StartOffsetUS: accountedUS, DurationUS: residual,
		})
		accountedUS += residual
	} else {
		r.invalid = append(r.invalid, "accounted_duration_exceeds_pairec_total")
	}
	closureErrorUS := abs64(totalUS - accountedUS)
	thresholdUS := int64(math.Max(3000, float64(totalUS)*0.05))
	if closureErrorUS > thresholdUS {
		r.invalid = append(r.invalid, "pairec_closure_error_exceeded")
	}
	if r.ds.Expected && os.Getenv("PAIREC_REQUIRE_DATASYSTEM_ATTRIBUTION") == "1" && !r.ds.Complete {
		r.invalid = append(r.invalid, "datasystem_attribution_incomplete")
	}
	sort.SliceStable(r.spans, func(i, j int) bool {
		if r.spans[i].StartOffsetUS == r.spans[j].StartOffsetUS {
			return r.spans[i].ID < r.spans[j].ID
		}
		return r.spans[i].StartOffsetUS < r.spans[j].StartOffsetUS
	})
	trace := Trace{
		Event: "pipeline_trace_complete", ContractVersion: ContractVersion,
		RequestID: r.requestID, Status: status, Sampled: r.sampled,
		Valid: len(r.invalid) == 0, InvalidReasons: append([]string(nil), r.invalid...),
		StartEpochNS: r.started.UnixNano(), EndEpochNS: ended.UnixNano(),
		PaiRecTotalUS: totalUS, AccountedUS: accountedUS, ClosureErrorUS: closureErrorUS,
		ClosureThresholdUS: thresholdUS, Spans: append([]Span(nil), r.spans...), DataSystem: r.ds,
	}
	traceTotal.WithLabelValues(strconv.FormatBool(trace.Valid), normalizeLabel(status)).Inc()
	traceClosureError.Observe(float64(closureErrorUS) / 1e6)
	return trace
}

func Emit(trace Trace) {
	if !trace.Sampled && trace.Valid && trace.Status == "ok" {
		return
	}
	encoded, err := json.Marshal(trace)
	if err != nil {
		fmt.Printf("{\"event\":\"pipeline_trace_encode_error\",\"request_id\":%q}\n", trace.RequestID)
		return
	}
	fmt.Println(string(encoded))
}

func FinalizeContext(ctx *paireccontext.RecommendContext, status string) Trace {
	recorder := FromContext(ctx)
	if recorder == nil {
		return Trace{Event: "pipeline_trace_complete", Status: "missing_recorder", Valid: false}
	}
	trace := recorder.Finalize(status)
	Emit(trace)
	return trace
}

func shouldSample(requestID string) bool {
	rate := 1.0
	if raw := os.Getenv("PAIREC_PIPELINE_TRACE_SAMPLE_RATE"); raw != "" {
		if parsed, err := strconv.ParseFloat(raw, 64); err == nil {
			rate = math.Max(0, math.Min(1, parsed))
		}
	}
	if rate >= 1 {
		return true
	}
	if rate <= 0 {
		return false
	}
	hash := uint32(2166136261)
	for index := 0; index < len(requestID); index++ {
		hash = (hash ^ uint32(requestID[index])) * 16777619
	}
	return float64(hash)/float64(math.MaxUint32) < rate
}

func normalizeLabel(value string) string {
	if value == "" {
		return "none"
	}
	if len(value) > 64 {
		return value[:64]
	}
	return value
}

func cloneAttributes(attributes map[string]interface{}) map[string]interface{} {
	if len(attributes) == 0 {
		return nil
	}
	cloned := make(map[string]interface{}, len(attributes))
	for key, value := range attributes {
		cloned[key] = value
	}
	return cloned
}

func numericMicroseconds(value interface{}) (int64, bool) {
	switch typed := value.(type) {
	case int:
		return int64(typed), true
	case int32:
		return int64(typed), true
	case int64:
		return typed, true
	case float64:
		return int64(typed), true
	default:
		return 0, false
	}
}

func abs64(value int64) int64 {
	if value < 0 {
		return -value
	}
	return value
}
