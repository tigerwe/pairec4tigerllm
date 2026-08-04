package recall

import proto "github.com/gogo/protobuf/proto"

const brpcDefaultServiceName = "pairec.inference.RecommendService"

type brpcRPCMeta struct {
	Request        *brpcRequestMeta  `protobuf:"bytes,1,opt,name=request" json:"request,omitempty"`
	Response       *brpcResponseMeta `protobuf:"bytes,2,opt,name=response" json:"response,omitempty"`
	CompressType   *int32            `protobuf:"varint,3,opt,name=compress_type,json=compressType,def=0" json:"compress_type,omitempty"`
	CorrelationID  *int64            `protobuf:"varint,4,opt,name=correlation_id,json=correlationId" json:"correlation_id,omitempty"`
	AttachmentSize *int32            `protobuf:"varint,5,opt,name=attachment_size,json=attachmentSize" json:"attachment_size,omitempty"`
	ContentType    *int32            `protobuf:"varint,10,opt,name=content_type,json=contentType,def=0" json:"content_type,omitempty"`
	ChecksumType   *int32            `protobuf:"varint,11,opt,name=checksum_type,json=checksumType,def=0" json:"checksum_type,omitempty"`
}

func (m *brpcRPCMeta) Reset()         { *m = brpcRPCMeta{} }
func (m *brpcRPCMeta) String() string { return proto.CompactTextString(m) }
func (*brpcRPCMeta) ProtoMessage()    {}

type brpcRequestMeta struct {
	ServiceName *string `protobuf:"bytes,1,req,name=service_name,json=serviceName" json:"service_name,omitempty"`
	MethodName  *string `protobuf:"bytes,2,req,name=method_name,json=methodName" json:"method_name,omitempty"`
	LogID       *int64  `protobuf:"varint,3,opt,name=log_id,json=logId" json:"log_id,omitempty"`
}

func (m *brpcRequestMeta) Reset()         { *m = brpcRequestMeta{} }
func (m *brpcRequestMeta) String() string { return proto.CompactTextString(m) }
func (*brpcRequestMeta) ProtoMessage()    {}

type brpcResponseMeta struct {
	ErrorCode *int32  `protobuf:"varint,1,req,name=error_code,json=errorCode" json:"error_code,omitempty"`
	ErrorText *string `protobuf:"bytes,2,opt,name=error_text,json=errorText" json:"error_text,omitempty"`
}

func (m *brpcResponseMeta) Reset()         { *m = brpcResponseMeta{} }
func (m *brpcResponseMeta) String() string { return proto.CompactTextString(m) }
func (*brpcResponseMeta) ProtoMessage()    {}

type semanticIDPB struct {
	Value []int32 `protobuf:"varint,1,rep,name=value" json:"value,omitempty"`
}

func (m *semanticIDPB) Reset()         { *m = semanticIDPB{} }
func (m *semanticIDPB) String() string { return proto.CompactTextString(m) }
func (*semanticIDPB) ProtoMessage()    {}

type recommendRequestPB struct {
	UserID         *string         `protobuf:"bytes,1,opt,name=user_id,json=userId" json:"user_id,omitempty"`
	History        []*semanticIDPB `protobuf:"bytes,2,rep,name=history" json:"history,omitempty"`
	Topk           *int32          `protobuf:"varint,3,opt,name=topk,def=10" json:"topk,omitempty"`
	Temperature    *float64        `protobuf:"fixed64,4,opt,name=temperature,def=1" json:"temperature,omitempty"`
	BeamWidth      *int32          `protobuf:"varint,5,opt,name=beam_width,json=beamWidth,def=1" json:"beam_width,omitempty"`
	RequestID      *string         `protobuf:"bytes,6,opt,name=request_id,json=requestId" json:"request_id,omitempty"`
	PayloadPadding []byte          `protobuf:"bytes,101,opt,name=payload_padding,json=payloadPadding" json:"payload_padding,omitempty"`
}

func (m *recommendRequestPB) Reset()         { *m = recommendRequestPB{} }
func (m *recommendRequestPB) String() string { return proto.CompactTextString(m) }
func (*recommendRequestPB) ProtoMessage()    {}

type traceInfoPB struct {
	TotalMs                  *float64 `protobuf:"fixed64,1,opt,name=total_ms,json=totalMs" json:"total_ms,omitempty"`
	PrepareInputMs           *float64 `protobuf:"fixed64,2,opt,name=prepare_input_ms,json=prepareInputMs" json:"prepare_input_ms,omitempty"`
	InferMs                  *float64 `protobuf:"fixed64,3,opt,name=infer_ms,json=inferMs" json:"infer_ms,omitempty"`
	ModelForwardMs           *float64 `protobuf:"fixed64,4,opt,name=model_forward_ms,json=modelForwardMs" json:"model_forward_ms,omitempty"`
	GenerateMs               *float64 `protobuf:"fixed64,5,opt,name=generate_ms,json=generateMs" json:"generate_ms,omitempty"`
	PromptMs                 *float64 `protobuf:"fixed64,6,opt,name=prompt_ms,json=promptMs" json:"prompt_ms,omitempty"`
	RunnerGenerateMs         *float64 `protobuf:"fixed64,7,opt,name=runner_generate_ms,json=runnerGenerateMs" json:"runner_generate_ms,omitempty"`
	RunnerCalls              *int32   `protobuf:"varint,8,opt,name=runner_calls,json=runnerCalls" json:"runner_calls,omitempty"`
	RunnerAvgMs              *float64 `protobuf:"fixed64,9,opt,name=runner_avg_ms,json=runnerAvgMs" json:"runner_avg_ms,omitempty"`
	RunnerMaxMs              *float64 `protobuf:"fixed64,10,opt,name=runner_max_ms,json=runnerMaxMs" json:"runner_max_ms,omitempty"`
	ParseComboMs             *float64 `protobuf:"fixed64,11,opt,name=parse_combo_ms,json=parseComboMs" json:"parse_combo_ms,omitempty"`
	OutputPadMs              *float64 `protobuf:"fixed64,12,opt,name=output_pad_ms,json=outputPadMs" json:"output_pad_ms,omitempty"`
	BackendTotalMs           *float64 `protobuf:"fixed64,13,opt,name=backend_total_ms,json=backendTotalMs" json:"backend_total_ms,omitempty"`
	MapItemMs                *float64 `protobuf:"fixed64,14,opt,name=map_item_ms,json=mapItemMs" json:"map_item_ms,omitempty"`
	KvLookupMs               *float64 `protobuf:"fixed64,15,opt,name=kv_lookup_ms,json=kvLookupMs" json:"kv_lookup_ms,omitempty"`
	KvWriteMs                *float64 `protobuf:"fixed64,16,opt,name=kv_write_ms,json=kvWriteMs" json:"kv_write_ms,omitempty"`
	KvSource                 *string  `protobuf:"bytes,17,opt,name=kv_source,json=kvSource" json:"kv_source,omitempty"`
	ResultCacheSource        *string  `protobuf:"bytes,18,opt,name=result_cache_source,json=resultCacheSource" json:"result_cache_source,omitempty"`
	ResultCacheLookupMs      *float64 `protobuf:"fixed64,19,opt,name=result_cache_lookup_ms,json=resultCacheLookupMs" json:"result_cache_lookup_ms,omitempty"`
	ResultCacheDSLookupMs    *float64 `protobuf:"fixed64,20,opt,name=result_cache_ds_lookup_ms,json=resultCacheDsLookupMs" json:"result_cache_ds_lookup_ms,omitempty"`
	ResultCacheWriteSubmitMs *float64 `protobuf:"fixed64,21,opt,name=result_cache_write_submit_ms,json=resultCacheWriteSubmitMs" json:"result_cache_write_submit_ms,omitempty"`
	Backend                  *string  `protobuf:"bytes,22,opt,name=backend" json:"backend,omitempty"`
	WrapperTotalMs           *float64 `protobuf:"fixed64,23,opt,name=wrapper_total_ms,json=wrapperTotalMs" json:"wrapper_total_ms,omitempty"`
	WrapperBackendRPCMs      *float64 `protobuf:"fixed64,24,opt,name=wrapper_backend_rpc_ms,json=wrapperBackendRpcMs" json:"wrapper_backend_rpc_ms,omitempty"`
	WrapperOverheadMs        *float64 `protobuf:"fixed64,25,opt,name=wrapper_overhead_ms,json=wrapperOverheadMs" json:"wrapper_overhead_ms,omitempty"`
	WrapperHealthAtStart     *int64   `protobuf:"varint,26,opt,name=wrapper_active_health_at_start,json=wrapperActiveHealthAtStart" json:"wrapper_active_health_at_start,omitempty"`
	WrapperMaxActiveHealth   *int64   `protobuf:"varint,27,opt,name=wrapper_max_active_health,json=wrapperMaxActiveHealth" json:"wrapper_max_active_health,omitempty"`
	WrapperMaxActiveTotal    *int64   `protobuf:"varint,28,opt,name=wrapper_max_active_total,json=wrapperMaxActiveTotal" json:"wrapper_max_active_total,omitempty"`
	WrapperBackendBRPCMs     *float64 `protobuf:"fixed64,29,opt,name=wrapper_backend_brpc_ms,json=wrapperBackendBrpcMs" json:"wrapper_backend_brpc_ms,omitempty"`
}

func (m *traceInfoPB) Reset()         { *m = traceInfoPB{} }
func (m *traceInfoPB) String() string { return proto.CompactTextString(m) }
func (*traceInfoPB) ProtoMessage()    {}

type recommendationPB struct {
	ItemID     *int32   `protobuf:"varint,1,opt,name=item_id,json=itemId" json:"item_id,omitempty"`
	SemanticID []int32  `protobuf:"varint,2,rep,name=semantic_id,json=semanticId" json:"semantic_id,omitempty"`
	Score      *float64 `protobuf:"fixed64,3,opt,name=score" json:"score,omitempty"`
}

func (m *recommendationPB) Reset()         { *m = recommendationPB{} }
func (m *recommendationPB) String() string { return proto.CompactTextString(m) }
func (*recommendationPB) ProtoMessage()    {}

type recommendResponsePB struct {
	Code            *int32              `protobuf:"varint,1,opt,name=code" json:"code,omitempty"`
	UserID          *string             `protobuf:"bytes,2,opt,name=user_id,json=userId" json:"user_id,omitempty"`
	Recommendations []*recommendationPB `protobuf:"bytes,3,rep,name=recommendations" json:"recommendations,omitempty"`
	InferenceTimeMs *float64            `protobuf:"fixed64,4,opt,name=inference_time_ms,json=inferenceTimeMs" json:"inference_time_ms,omitempty"`
	Error           *string             `protobuf:"bytes,5,opt,name=error" json:"error,omitempty"`
	Trace           *traceInfoPB        `protobuf:"bytes,6,opt,name=trace" json:"trace,omitempty"`
	RawJSON         *string             `protobuf:"bytes,100,opt,name=raw_json,json=rawJson" json:"raw_json,omitempty"`
}

func (m *recommendResponsePB) Reset()         { *m = recommendResponsePB{} }
func (m *recommendResponsePB) String() string { return proto.CompactTextString(m) }
func (*recommendResponsePB) ProtoMessage()    {}

type healthRequestPB struct {
	PayloadPadding []byte `protobuf:"bytes,101,opt,name=payload_padding,json=payloadPadding" json:"payload_padding,omitempty"`
}

func (m *healthRequestPB) Reset()         { *m = healthRequestPB{} }
func (m *healthRequestPB) String() string { return proto.CompactTextString(m) }
func (*healthRequestPB) ProtoMessage()    {}

type healthResponsePB struct {
	Code    *int32  `protobuf:"varint,1,opt,name=code" json:"code,omitempty"`
	Status  *string `protobuf:"bytes,2,opt,name=status" json:"status,omitempty"`
	Backend *string `protobuf:"bytes,3,opt,name=backend" json:"backend,omitempty"`
	RawJSON *string `protobuf:"bytes,100,opt,name=raw_json,json=rawJson" json:"raw_json,omitempty"`
}

func (m *healthResponsePB) Reset()         { *m = healthResponsePB{} }
func (m *healthResponsePB) String() string { return proto.CompactTextString(m) }
func (*healthResponsePB) ProtoMessage()    {}

type brpcHealthResponse struct {
	Code    int
	Status  string
	Backend string
}

func recommendRequestToProto(req *RecommendRequest, requestID string) *recommendRequestPB {
	history := make([]*semanticIDPB, 0, len(req.History))
	for _, values := range req.History {
		semantic := make([]int32, 0, len(values))
		for _, value := range values {
			semantic = append(semantic, int32(value))
		}
		history = append(history, &semanticIDPB{Value: semantic})
	}

	pb := &recommendRequestPB{
		UserID:      proto.String(req.UserID),
		History:     history,
		Topk:        proto.Int32(int32(req.Topk)),
		Temperature: proto.Float64(req.Temperature),
		BeamWidth:   proto.Int32(int32(req.BeamWidth)),
	}
	if requestID != "" {
		pb.RequestID = proto.String(requestID)
	}
	if req.PayloadPaddingBytes > 0 {
		pb.PayloadPadding = makePayloadPadding(req.PayloadPaddingBytes)
	}
	return pb
}

func makePayloadPadding(size int) []byte {
	if size <= 0 {
		return nil
	}
	return make([]byte, size)
}

func recommendResponseFromProto(pb *recommendResponsePB) *RecommendResponse {
	resp := &RecommendResponse{
		Code:            int(int32Value(pb.Code)),
		UserID:          stringValue(pb.UserID),
		InferenceTimeMs: float64Value(pb.InferenceTimeMs),
		Error:           stringValue(pb.Error),
	}
	resp.Recommendations = make([]Recommendation, 0, len(pb.Recommendations))
	for _, recPB := range pb.Recommendations {
		rec := Recommendation{
			ItemID: int(int32Value(recPB.ItemID)),
			Score:  float64Value(recPB.Score),
		}
		rec.SemanticID = make([]int, 0, len(recPB.SemanticID))
		for _, value := range recPB.SemanticID {
			rec.SemanticID = append(rec.SemanticID, int(value))
		}
		resp.Recommendations = append(resp.Recommendations, rec)
	}
	if pb.Trace != nil {
		resp.Trace = traceInfoFromProto(pb.Trace)
	}
	return resp
}

func traceInfoFromProto(pb *traceInfoPB) *TraceInfo {
	return &TraceInfo{
		TotalMs:                  float64Value(pb.TotalMs),
		PrepareInputMs:           float64Value(pb.PrepareInputMs),
		InferMs:                  float64Value(pb.InferMs),
		ModelForwardMs:           float64Value(pb.ModelForwardMs),
		GenerateMs:               float64Value(pb.GenerateMs),
		PromptMs:                 float64Value(pb.PromptMs),
		RunnerGenerateMs:         float64Value(pb.RunnerGenerateMs),
		ParseComboMs:             float64Value(pb.ParseComboMs),
		OutputPadMs:              float64Value(pb.OutputPadMs),
		BackendTotalMs:           float64Value(pb.BackendTotalMs),
		MapItemMs:                float64Value(pb.MapItemMs),
		KvLookupMs:               float64Value(pb.KvLookupMs),
		KvWriteMs:                float64Value(pb.KvWriteMs),
		KvSource:                 stringValue(pb.KvSource),
		ResultCacheSource:        stringValue(pb.ResultCacheSource),
		ResultCacheLookupMs:      float64Value(pb.ResultCacheLookupMs),
		ResultCacheDSLookupMs:    float64Value(pb.ResultCacheDSLookupMs),
		ResultCacheWriteSubmitMs: float64Value(pb.ResultCacheWriteSubmitMs),
		Backend:                  stringValue(pb.Backend),
		WrapperTotalMs:           float64Value(pb.WrapperTotalMs),
		WrapperBackendRPCMs:      float64Value(pb.WrapperBackendRPCMs),
		WrapperOverheadMs:        float64Value(pb.WrapperOverheadMs),
		WrapperHealthAtStart:     int64Value(pb.WrapperHealthAtStart),
		WrapperMaxActiveHealth:   int64Value(pb.WrapperMaxActiveHealth),
		WrapperMaxActiveTotal:    int64Value(pb.WrapperMaxActiveTotal),
		WrapperBackendBRPCMs:     float64Value(pb.WrapperBackendBRPCMs),
	}
}

func healthResponseFromProto(pb *healthResponsePB) *brpcHealthResponse {
	return &brpcHealthResponse{
		Code:    int(int32Value(pb.Code)),
		Status:  stringValue(pb.Status),
		Backend: stringValue(pb.Backend),
	}
}

func int32Value(value *int32) int32 {
	if value == nil {
		return 0
	}
	return *value
}

func int64Value(value *int64) int64 {
	if value == nil {
		return 0
	}
	return *value
}

func float64Value(value *float64) float64 {
	if value == nil {
		return 0
	}
	return *value
}

func stringValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}
