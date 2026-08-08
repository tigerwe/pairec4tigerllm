package brpcwire

import proto "github.com/gogo/protobuf/proto"

type rpcMeta struct {
	Request        *requestMeta  `protobuf:"bytes,1,opt,name=request"`
	Response       *responseMeta `protobuf:"bytes,2,opt,name=response"`
	CompressType   *int32        `protobuf:"varint,3,opt,name=compress_type,json=compressType,def=0"`
	CorrelationID  *int64        `protobuf:"varint,4,opt,name=correlation_id,json=correlationId"`
	AttachmentSize *int32        `protobuf:"varint,5,opt,name=attachment_size,json=attachmentSize"`
	ContentType    *int32        `protobuf:"varint,10,opt,name=content_type,json=contentType,def=0"`
	ChecksumType   *int32        `protobuf:"varint,11,opt,name=checksum_type,json=checksumType,def=0"`
}

func (m *rpcMeta) Reset()         { *m = rpcMeta{} }
func (m *rpcMeta) String() string { return proto.CompactTextString(m) }
func (*rpcMeta) ProtoMessage()    {}

type requestMeta struct {
	ServiceName *string `protobuf:"bytes,1,req,name=service_name,json=serviceName"`
	MethodName  *string `protobuf:"bytes,2,req,name=method_name,json=methodName"`
	LogID       *int64  `protobuf:"varint,3,opt,name=log_id,json=logId"`
}

func (m *requestMeta) Reset()         { *m = requestMeta{} }
func (m *requestMeta) String() string { return proto.CompactTextString(m) }
func (*requestMeta) ProtoMessage()    {}

type responseMeta struct {
	ErrorCode *int32  `protobuf:"varint,1,req,name=error_code,json=errorCode"`
	ErrorText *string `protobuf:"bytes,2,opt,name=error_text,json=errorText"`
}

func (m *responseMeta) Reset()         { *m = responseMeta{} }
func (m *responseMeta) String() string { return proto.CompactTextString(m) }
func (*responseMeta) ProtoMessage()    {}
