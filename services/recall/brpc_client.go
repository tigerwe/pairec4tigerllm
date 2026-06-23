package recall

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"strings"
	"sync/atomic"
	"time"

	proto "github.com/gogo/protobuf/proto"
)

const (
	brpcMagic          = "PRPC"
	brpcHeaderSize     = 12
	brpcNoCompression  = int32(0)
	brpcContentTypePB  = int32(0)
	brpcChecksumNone   = int32(0)
	brpcMaxBodySize    = 32 << 20
	brpcRetrySleepBase = 50 * time.Millisecond
)

// BRPCRecommendClient implements the subset of brpc baidu_std/TCP needed by
// pairec.inference.RecommendService. It intentionally avoids adding an online
// Go SDK dependency because this repository is built with vendored modules.
type BRPCRecommendClient struct {
	endpoint   string
	service    string
	timeout    time.Duration
	maxRetries int
	nextID     int64
}

func NewBRPCRecommendClient(endpoint, service string, timeout time.Duration, maxRetries int) (*BRPCRecommendClient, error) {
	endpoint = strings.TrimSpace(endpoint)
	if endpoint == "" {
		return nil, fmt.Errorf("brpc endpoint is empty")
	}
	service = strings.TrimSpace(service)
	if service == "" {
		service = brpcDefaultServiceName
	}
	if timeout <= 0 {
		timeout = 3 * time.Second
	}
	if maxRetries <= 0 {
		maxRetries = 1
	}

	return &BRPCRecommendClient{
		endpoint:   endpoint,
		service:    service,
		timeout:    timeout,
		maxRetries: maxRetries,
	}, nil
}

func (c *BRPCRecommendClient) Recommend(ctx context.Context, req *RecommendRequest, requestID string) (*RecommendResponse, error) {
	requestPB := recommendRequestToProto(req, requestID)
	var responsePB recommendResponsePB
	if err := c.call(ctx, "Recommend", requestPB, &responsePB); err != nil {
		return nil, err
	}
	resp := recommendResponseFromProto(&responsePB)
	if resp.Code != 200 {
		return nil, fmt.Errorf("service error: %s", resp.Error)
	}
	return resp, nil
}

func (c *BRPCRecommendClient) HealthCheck(ctx context.Context) (*brpcHealthResponse, error) {
	var responsePB healthResponsePB
	if err := c.call(ctx, "Health", &healthRequestPB{}, &responsePB); err != nil {
		return nil, err
	}
	return healthResponseFromProto(&responsePB), nil
}

func (c *BRPCRecommendClient) call(ctx context.Context, method string, request proto.Message, response proto.Message) error {
	var lastErr error
	for attempt := 0; attempt < c.maxRetries; attempt++ {
		attemptCtx, cancel := context.WithTimeout(ctx, c.timeout)
		lastErr = c.callOnce(attemptCtx, method, request, response)
		cancel()
		if lastErr == nil {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if attempt < c.maxRetries-1 {
			sleep := time.Duration(attempt+1) * brpcRetrySleepBase
			select {
			case <-time.After(sleep):
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}
	return fmt.Errorf("brpc %s failed after %d retries: %w", method, c.maxRetries, lastErr)
}

func (c *BRPCRecommendClient) callOnce(ctx context.Context, method string, request proto.Message, response proto.Message) error {
	payload, err := proto.Marshal(request)
	if err != nil {
		return fmt.Errorf("marshal %s request failed: %w", method, err)
	}

	correlationID := atomic.AddInt64(&c.nextID, 1)
	meta := &brpcRPCMeta{
		CompressType:   proto.Int32(brpcNoCompression),
		CorrelationID:  proto.Int64(correlationID),
		AttachmentSize: proto.Int32(0),
		ContentType:    proto.Int32(brpcContentTypePB),
		ChecksumType:   proto.Int32(brpcChecksumNone),
		Request: &brpcRequestMeta{
			ServiceName: proto.String(c.service),
			MethodName:  proto.String(method),
			LogID:       proto.Int64(correlationID),
		},
	}
	metaBytes, err := proto.Marshal(meta)
	if err != nil {
		return fmt.Errorf("marshal brpc meta failed: %w", err)
	}

	frame, err := buildBRPCFrame(metaBytes, payload)
	if err != nil {
		return err
	}

	var dialer net.Dialer
	conn, err := dialer.DialContext(ctx, "tcp", c.endpoint)
	if err != nil {
		return fmt.Errorf("dial %s failed: %w", c.endpoint, err)
	}
	defer conn.Close()

	if deadline, ok := ctx.Deadline(); ok {
		if err := conn.SetDeadline(deadline); err != nil {
			return fmt.Errorf("set deadline failed: %w", err)
		}
	}

	if _, err := conn.Write(frame); err != nil {
		return fmt.Errorf("write brpc frame failed: %w", err)
	}

	responsePayload, err := readBRPCResponse(conn, correlationID)
	if err != nil {
		return err
	}
	if err := proto.Unmarshal(responsePayload, response); err != nil {
		return fmt.Errorf("unmarshal %s response failed: %w", method, err)
	}
	return nil
}

func buildBRPCFrame(metaBytes, payload []byte) ([]byte, error) {
	bodySize := len(metaBytes) + len(payload)
	if bodySize > brpcMaxBodySize {
		return nil, fmt.Errorf("brpc request body too large: %d", bodySize)
	}
	frame := make([]byte, brpcHeaderSize+bodySize)
	copy(frame[:4], brpcMagic)
	binary.BigEndian.PutUint32(frame[4:8], uint32(bodySize))
	binary.BigEndian.PutUint32(frame[8:12], uint32(len(metaBytes)))
	copy(frame[brpcHeaderSize:], metaBytes)
	copy(frame[brpcHeaderSize+len(metaBytes):], payload)
	return frame, nil
}

func readBRPCResponse(reader io.Reader, expectedCorrelationID int64) ([]byte, error) {
	header := make([]byte, brpcHeaderSize)
	if _, err := io.ReadFull(reader, header); err != nil {
		return nil, fmt.Errorf("read brpc header failed: %w", err)
	}
	if string(header[:4]) != brpcMagic {
		return nil, fmt.Errorf("invalid brpc magic: %q", string(header[:4]))
	}

	bodySize := binary.BigEndian.Uint32(header[4:8])
	metaSize := binary.BigEndian.Uint32(header[8:12])
	if bodySize > brpcMaxBodySize {
		return nil, fmt.Errorf("brpc response body too large: %d", bodySize)
	}
	if metaSize > bodySize {
		return nil, fmt.Errorf("invalid brpc meta size: meta=%d body=%d", metaSize, bodySize)
	}

	body := make([]byte, int(bodySize))
	if _, err := io.ReadFull(reader, body); err != nil {
		return nil, fmt.Errorf("read brpc body failed: %w", err)
	}

	var meta brpcRPCMeta
	if err := proto.Unmarshal(body[:metaSize], &meta); err != nil {
		return nil, fmt.Errorf("unmarshal brpc response meta failed: %w", err)
	}
	if correlationID := int64Value(meta.CorrelationID); correlationID != 0 && correlationID != expectedCorrelationID {
		return nil, fmt.Errorf("brpc correlation mismatch: got=%d want=%d", correlationID, expectedCorrelationID)
	}
	if compressType := int32Value(meta.CompressType); compressType != brpcNoCompression {
		return nil, fmt.Errorf("unsupported brpc response compression type: %d", compressType)
	}
	if meta.Response != nil && int32Value(meta.Response.ErrorCode) != 0 {
		return nil, fmt.Errorf("brpc server error %d: %s", int32Value(meta.Response.ErrorCode), stringValue(meta.Response.ErrorText))
	}

	attachmentSize := int64(int32Value(meta.AttachmentSize))
	if attachmentSize < 0 || uint64(attachmentSize) > uint64(bodySize-metaSize) {
		return nil, fmt.Errorf("invalid brpc attachment size: attachment=%d body=%d meta=%d", attachmentSize, bodySize, metaSize)
	}
	payloadEnd := int(bodySize) - int(attachmentSize)
	return body[metaSize:payloadEnd], nil
}
