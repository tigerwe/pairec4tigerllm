package recall

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"net"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	proto "github.com/gogo/protobuf/proto"
)

func TestBRPCSessionReusesConnection(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer listener.Close()

	var accepted int32
	serverDone := make(chan error, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			serverDone <- err
			return
		}
		atomic.AddInt32(&accepted, 1)
		defer conn.Close()
		for i := 0; i < 2; i++ {
			header := make([]byte, brpcHeaderSize)
			if _, err := io.ReadFull(conn, header); err != nil {
				serverDone <- err
				return
			}
			bodySize := binary.BigEndian.Uint32(header[4:8])
			metaSize := binary.BigEndian.Uint32(header[8:12])
			body := make([]byte, bodySize)
			if _, err := io.ReadFull(conn, body); err != nil {
				serverDone <- err
				return
			}
			var requestMeta brpcRPCMeta
			if err := proto.Unmarshal(body[:metaSize], &requestMeta); err != nil {
				serverDone <- err
				return
			}
			responsePayload, err := proto.Marshal(&healthResponsePB{
				Code: proto.Int32(200), Status: proto.String("healthy"), Backend: proto.String("test"),
			})
			if err != nil {
				serverDone <- err
				return
			}
			responseMeta, err := proto.Marshal(&brpcRPCMeta{
				CompressType:   proto.Int32(brpcNoCompression),
				CorrelationID:  requestMeta.CorrelationID,
				AttachmentSize: proto.Int32(0),
				ContentType:    proto.Int32(brpcContentTypePB),
				ChecksumType:   proto.Int32(brpcChecksumNone),
				Response:       &brpcResponseMeta{ErrorCode: proto.Int32(0)},
			})
			if err != nil {
				serverDone <- err
				return
			}
			frame, err := buildBRPCFrame(responseMeta, responsePayload)
			if err != nil {
				serverDone <- err
				return
			}
			if _, err := conn.Write(frame); err != nil {
				serverDone <- err
				return
			}
		}
		serverDone <- nil
	}()

	client, err := NewBRPCRecommendClient(listener.Addr().String(), brpcDefaultServiceName, time.Second, 1)
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	session := client.NewSession()
	defer session.Close()
	for i := 0; i < 2; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		response, err := session.HealthCheckWithPayload(ctx, 102400)
		cancel()
		if err != nil {
			t.Fatalf("health %d: %v", i, err)
		}
		if response.Code != 200 || response.Status != "healthy" {
			t.Fatalf("health %d response: %+v", i, response)
		}
	}
	if err := <-serverDone; err != nil {
		t.Fatalf("server: %v", err)
	}
	if got := atomic.LoadInt32(&accepted); got != 1 {
		t.Fatalf("accepted connections=%d, want 1", got)
	}
}

func TestRecommendProtoRoundTrip(t *testing.T) {
	req := &RecommendRequest{
		UserID:              "u1",
		History:             [][]int{{169, 41, 0, 0}, {20, 53, 0, 0}},
		Topk:                5,
		Temperature:         0.7,
		BeamWidth:           1,
		PayloadPaddingBytes: 1024,
	}

	requestPB := recommendRequestToProto(req, "trace-1")
	data, err := proto.Marshal(requestPB)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	var decoded recommendRequestPB
	if err := proto.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	if got := stringValue(decoded.UserID); got != "u1" {
		t.Fatalf("user_id=%q", got)
	}
	if got := len(decoded.History); got != 2 {
		t.Fatalf("history len=%d", got)
	}
	if got := decoded.History[0].Value[0]; got != 169 {
		t.Fatalf("history[0][0]=%d", got)
	}
	if got := stringValue(decoded.RequestID); got != "trace-1" {
		t.Fatalf("request_id=%q", got)
	}
	if got := len(decoded.PayloadPadding); got != 1024 {
		t.Fatalf("payload_padding bytes=%d", got)
	}

	responsePB := &recommendResponsePB{
		Code:            proto.Int32(200),
		UserID:          proto.String("u1"),
		InferenceTimeMs: proto.Float64(123.5),
		Recommendations: []*recommendationPB{
			{
				ItemID:     proto.Int32(311),
				SemanticID: []int32{55, 1, 0, 0},
				Score:      proto.Float64(1.0),
			},
		},
		Trace: &traceInfoPB{
			Backend:          proto.String("trtllm_cpp"),
			RunnerGenerateMs: proto.Float64(99.9),
		},
	}

	resp := recommendResponseFromProto(responsePB)
	if resp.Code != 200 || resp.UserID != "u1" || len(resp.Recommendations) != 1 {
		t.Fatalf("unexpected response: %+v", resp)
	}
	if resp.Recommendations[0].ItemID != 311 || resp.Recommendations[0].SemanticID[0] != 55 {
		t.Fatalf("unexpected recommendation: %+v", resp.Recommendations[0])
	}
	if resp.Trace == nil || resp.Trace.Backend != "trtllm_cpp" || resp.Trace.RunnerGenerateMs != 99.9 {
		t.Fatalf("unexpected trace: %+v", resp.Trace)
	}
}

func TestBRPCFrameRoundTrip(t *testing.T) {
	meta := &brpcRPCMeta{
		CompressType:   proto.Int32(brpcNoCompression),
		CorrelationID:  proto.Int64(42),
		AttachmentSize: proto.Int32(0),
		ContentType:    proto.Int32(brpcContentTypePB),
		ChecksumType:   proto.Int32(brpcChecksumNone),
		Response:       &brpcResponseMeta{ErrorCode: proto.Int32(0)},
	}
	metaBytes, err := proto.Marshal(meta)
	if err != nil {
		t.Fatalf("marshal meta: %v", err)
	}
	payload := []byte{1, 2, 3, 4}

	frame, err := buildBRPCFrame(metaBytes, payload)
	if err != nil {
		t.Fatalf("build frame: %v", err)
	}
	decoded, err := readBRPCResponse(bytes.NewReader(frame), 42)
	if err != nil {
		t.Fatalf("read response: %v", err)
	}
	if !bytes.Equal(decoded, payload) {
		t.Fatalf("payload=%v", decoded)
	}
}

func TestBRPCFrameServerError(t *testing.T) {
	meta := &brpcRPCMeta{
		CompressType:   proto.Int32(brpcNoCompression),
		CorrelationID:  proto.Int64(42),
		AttachmentSize: proto.Int32(0),
		ContentType:    proto.Int32(brpcContentTypePB),
		ChecksumType:   proto.Int32(brpcChecksumNone),
		Response: &brpcResponseMeta{
			ErrorCode: proto.Int32(1001),
			ErrorText: proto.String("method not found"),
		},
	}
	metaBytes, err := proto.Marshal(meta)
	if err != nil {
		t.Fatalf("marshal meta: %v", err)
	}
	frame, err := buildBRPCFrame(metaBytes, nil)
	if err != nil {
		t.Fatalf("build frame: %v", err)
	}
	_, err = readBRPCResponse(bytes.NewReader(frame), 42)
	if err == nil || !strings.Contains(err.Error(), "method not found") {
		t.Fatalf("expected server error, got %v", err)
	}
}
