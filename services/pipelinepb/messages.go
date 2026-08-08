// Package pipelinepb contains the vendored-build-compatible Go wire types for
// proto/pipeline_service.proto. C++ code is generated from the canonical proto;
// Go keeps explicit structs to avoid requiring protoc plugins in offline builds.
package pipelinepb

import proto "github.com/gogo/protobuf/proto"

const TraceContractVersion = "pairec.pipeline_trace.v1"

type TraceContext struct {
	RequestID       *string `protobuf:"bytes,1,opt,name=request_id,json=requestId" json:"request_id,omitempty"`
	SpanID          *string `protobuf:"bytes,2,opt,name=span_id,json=spanId" json:"span_id,omitempty"`
	ParentSpanID    *string `protobuf:"bytes,3,opt,name=parent_span_id,json=parentSpanId" json:"parent_span_id,omitempty"`
	Sampled         *bool   `protobuf:"varint,4,opt,name=sampled,def=1" json:"sampled,omitempty"`
	ContractVersion *string `protobuf:"bytes,5,opt,name=contract_version,json=contractVersion,def=pairec.pipeline_trace.v1" json:"contract_version,omitempty"`
	TimeoutBudgetUS *int64  `protobuf:"varint,6,opt,name=timeout_budget_us,json=timeoutBudgetUs" json:"timeout_budget_us,omitempty"`
}

func (m *TraceContext) Reset()         { *m = TraceContext{} }
func (m *TraceContext) String() string { return proto.CompactTextString(m) }
func (*TraceContext) ProtoMessage()    {}

type ServiceTrace struct {
	Context             *TraceContext `protobuf:"bytes,1,opt,name=context" json:"context,omitempty"`
	Component           *string       `protobuf:"bytes,2,opt,name=component" json:"component,omitempty"`
	Protocol            *string       `protobuf:"bytes,3,opt,name=protocol" json:"protocol,omitempty"`
	Status              *string       `protobuf:"bytes,4,opt,name=status" json:"status,omitempty"`
	Backend             *string       `protobuf:"bytes,5,opt,name=backend" json:"backend,omitempty"`
	ModelVersion        *string       `protobuf:"bytes,6,opt,name=model_version,json=modelVersion" json:"model_version,omitempty"`
	QueueUS             *int64        `protobuf:"varint,10,opt,name=queue_us,json=queueUs" json:"queue_us,omitempty"`
	DecodeUS            *int64        `protobuf:"varint,11,opt,name=decode_us,json=decodeUs" json:"decode_us,omitempty"`
	FeatureUS           *int64        `protobuf:"varint,12,opt,name=feature_us,json=featureUs" json:"feature_us,omitempty"`
	ComputeUS           *int64        `protobuf:"varint,13,opt,name=compute_us,json=computeUs" json:"compute_us,omitempty"`
	BackendRPCUS        *int64        `protobuf:"varint,14,opt,name=backend_rpc_us,json=backendRpcUs" json:"backend_rpc_us,omitempty"`
	EncodeUS            *int64        `protobuf:"varint,15,opt,name=encode_us,json=encodeUs" json:"encode_us,omitempty"`
	BackendTotalUS      *int64        `protobuf:"varint,16,opt,name=backend_total_us,json=backendTotalUs" json:"backend_total_us,omitempty"`
	TotalUS             *int64        `protobuf:"varint,17,opt,name=total_us,json=totalUs" json:"total_us,omitempty"`
	AttributionComplete *bool         `protobuf:"varint,20,opt,name=attribution_complete,json=attributionComplete,def=1" json:"attribution_complete,omitempty"`
	Asynchronous        *bool         `protobuf:"varint,21,opt,name=asynchronous" json:"asynchronous,omitempty"`
	AttributionReason   *string       `protobuf:"bytes,22,opt,name=attribution_reason,json=attributionReason" json:"attribution_reason,omitempty"`
}

func (m *ServiceTrace) Reset()         { *m = ServiceTrace{} }
func (m *ServiceTrace) String() string { return proto.CompactTextString(m) }
func (*ServiceTrace) ProtoMessage()    {}

type HealthRequest struct {
	Context *TraceContext `protobuf:"bytes,1,opt,name=context" json:"context,omitempty"`
}

func (m *HealthRequest) Reset()         { *m = HealthRequest{} }
func (m *HealthRequest) String() string { return proto.CompactTextString(m) }
func (*HealthRequest) ProtoMessage()    {}

type HealthResponse struct {
	Code    *int32        `protobuf:"varint,1,opt,name=code" json:"code,omitempty"`
	Status  *string       `protobuf:"bytes,2,opt,name=status" json:"status,omitempty"`
	Backend *string       `protobuf:"bytes,3,opt,name=backend" json:"backend,omitempty"`
	Trace   *ServiceTrace `protobuf:"bytes,4,opt,name=trace" json:"trace,omitempty"`
}

func (m *HealthResponse) Reset()         { *m = HealthResponse{} }
func (m *HealthResponse) String() string { return proto.CompactTextString(m) }
func (*HealthResponse) ProtoMessage()    {}

type VectorRecallRequest struct {
	Context *TraceContext `protobuf:"bytes,1,opt,name=context" json:"context,omitempty"`
	UserID  *string       `protobuf:"bytes,2,opt,name=user_id,json=userId" json:"user_id,omitempty"`
	TopK    *int32        `protobuf:"varint,3,opt,name=topk,def=20" json:"topk,omitempty"`
}

func (m *VectorRecallRequest) Reset()         { *m = VectorRecallRequest{} }
func (m *VectorRecallRequest) String() string { return proto.CompactTextString(m) }
func (*VectorRecallRequest) ProtoMessage()    {}

type VectorRecallItem struct {
	ItemID *string  `protobuf:"bytes,1,opt,name=item_id,json=itemId" json:"item_id,omitempty"`
	Score  *float64 `protobuf:"fixed64,2,opt,name=score" json:"score,omitempty"`
}

func (m *VectorRecallItem) Reset()         { *m = VectorRecallItem{} }
func (m *VectorRecallItem) String() string { return proto.CompactTextString(m) }
func (*VectorRecallItem) ProtoMessage()    {}

type VectorRecallResponse struct {
	Code    *int32              `protobuf:"varint,1,opt,name=code" json:"code,omitempty"`
	Message *string             `protobuf:"bytes,2,opt,name=message" json:"message,omitempty"`
	Items   []*VectorRecallItem `protobuf:"bytes,3,rep,name=items" json:"items,omitempty"`
	Source  *string             `protobuf:"bytes,4,opt,name=source" json:"source,omitempty"`
	Trace   *ServiceTrace       `protobuf:"bytes,5,opt,name=trace" json:"trace,omitempty"`
}

func (m *VectorRecallResponse) Reset()         { *m = VectorRecallResponse{} }
func (m *VectorRecallResponse) String() string { return proto.CompactTextString(m) }
func (*VectorRecallResponse) ProtoMessage()    {}

type RankCandidate struct {
	ItemID *string `protobuf:"bytes,1,opt,name=item_id,json=itemId" json:"item_id,omitempty"`
}

func (m *RankCandidate) Reset()         { *m = RankCandidate{} }
func (m *RankCandidate) String() string { return proto.CompactTextString(m) }
func (*RankCandidate) ProtoMessage()    {}

type RankRequest struct {
	Context *TraceContext    `protobuf:"bytes,1,opt,name=context" json:"context,omitempty"`
	UserID  *string          `protobuf:"bytes,2,opt,name=user_id,json=userId" json:"user_id,omitempty"`
	Items   []*RankCandidate `protobuf:"bytes,3,rep,name=items" json:"items,omitempty"`
}

func (m *RankRequest) Reset()         { *m = RankRequest{} }
func (m *RankRequest) String() string { return proto.CompactTextString(m) }
func (*RankRequest) ProtoMessage()    {}

type RankedItem struct {
	ItemID *string  `protobuf:"bytes,1,opt,name=item_id,json=itemId" json:"item_id,omitempty"`
	Score  *float64 `protobuf:"fixed64,2,opt,name=score" json:"score,omitempty"`
}

func (m *RankedItem) Reset()         { *m = RankedItem{} }
func (m *RankedItem) String() string { return proto.CompactTextString(m) }
func (*RankedItem) ProtoMessage()    {}

type FeatureCoverage struct {
	ProfileMissing    *bool    `protobuf:"varint,1,opt,name=profile_missing,json=profileMissing" json:"profile_missing,omitempty"`
	UserOOV           *bool    `protobuf:"varint,2,opt,name=user_oov,json=userOov" json:"user_oov,omitempty"`
	ItemOOVCount      *int32   `protobuf:"varint,3,opt,name=item_oov_count,json=itemOovCount" json:"item_oov_count,omitempty"`
	CategoryOOVCount  *int32   `protobuf:"varint,4,opt,name=category_oov_count,json=categoryOovCount" json:"category_oov_count,omitempty"`
	GenderOOV         *bool    `protobuf:"varint,5,opt,name=gender_oov,json=genderOov" json:"gender_oov,omitempty"`
	AgeOOV            *bool    `protobuf:"varint,6,opt,name=age_oov,json=ageOov" json:"age_oov,omitempty"`
	HistoryValidCount *int32   `protobuf:"varint,7,opt,name=history_valid_count,json=historyValidCount" json:"history_valid_count,omitempty"`
	HistoryOOVCount   *int32   `protobuf:"varint,8,opt,name=history_oov_count,json=historyOovCount" json:"history_oov_count,omitempty"`
	ScoreUniqueCount  *int32   `protobuf:"varint,9,opt,name=score_unique_count,json=scoreUniqueCount" json:"score_unique_count,omitempty"`
	ScoreMin          *float64 `protobuf:"fixed64,10,opt,name=score_min,json=scoreMin" json:"score_min,omitempty"`
	ScoreMax          *float64 `protobuf:"fixed64,11,opt,name=score_max,json=scoreMax" json:"score_max,omitempty"`
}

func (m *FeatureCoverage) Reset()         { *m = FeatureCoverage{} }
func (m *FeatureCoverage) String() string { return proto.CompactTextString(m) }
func (*FeatureCoverage) ProtoMessage()    {}

type RankResponse struct {
	Code         *int32           `protobuf:"varint,1,opt,name=code" json:"code,omitempty"`
	Message      *string          `protobuf:"bytes,2,opt,name=message" json:"message,omitempty"`
	Items        []*RankedItem    `protobuf:"bytes,3,rep,name=items" json:"items,omitempty"`
	ModelVersion *string          `protobuf:"bytes,4,opt,name=model_version,json=modelVersion" json:"model_version,omitempty"`
	ModelRole    *string          `protobuf:"bytes,5,opt,name=model_role,json=modelRole" json:"model_role,omitempty"`
	Coverage     *FeatureCoverage `protobuf:"bytes,6,opt,name=coverage" json:"coverage,omitempty"`
	Trace        *ServiceTrace    `protobuf:"bytes,7,opt,name=trace" json:"trace,omitempty"`
}

func (m *RankResponse) Reset()         { *m = RankResponse{} }
func (m *RankResponse) String() string { return proto.CompactTextString(m) }
func (*RankResponse) ProtoMessage()    {}

func String(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func Int32(value *int32) int32 {
	if value == nil {
		return 0
	}
	return *value
}

func Int64(value *int64) int64 {
	if value == nil {
		return 0
	}
	return *value
}

func Float64(value *float64) float64 {
	if value == nil {
		return 0
	}
	return *value
}

func Bool(value *bool) bool {
	return value != nil && *value
}
