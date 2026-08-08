package brpcwire

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
	magic         = "PRPC"
	headerSize    = 12
	maxBodySize   = 32 << 20
	noCompression = int32(0)
)

// Client implements protobuf calls over brpc baidu_std/TCP. Attempts is the
// total number of attempts, so Attempts=1 means zero retries.
type Client struct {
	endpoint string
	service  string
	timeout  time.Duration
	attempts int
	nextID   int64
}

func NewClient(endpoint, service string, timeout time.Duration, attempts int) (*Client, error) {
	endpoint = strings.TrimSpace(endpoint)
	service = strings.TrimSpace(service)
	if endpoint == "" || service == "" {
		return nil, fmt.Errorf("brpc endpoint and service are required")
	}
	if timeout <= 0 {
		return nil, fmt.Errorf("brpc timeout must be positive")
	}
	if attempts <= 0 {
		return nil, fmt.Errorf("brpc attempts must be positive")
	}
	return &Client{endpoint: endpoint, service: service, timeout: timeout, attempts: attempts}, nil
}

func (c *Client) Call(ctx context.Context, method string, request, response proto.Message) error {
	if strings.TrimSpace(method) == "" || request == nil || response == nil {
		return fmt.Errorf("brpc method, request, and response are required")
	}
	var lastErr error
	for attempt := 0; attempt < c.attempts; attempt++ {
		attemptCtx, cancel := context.WithTimeout(ctx, c.timeout)
		lastErr = c.callOnce(attemptCtx, method, request, response)
		cancel()
		if lastErr == nil {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
	}
	return fmt.Errorf("brpc %s failed after %d attempt(s): %w", method, c.attempts, lastErr)
}

func (c *Client) callOnce(ctx context.Context, method string, request, response proto.Message) error {
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

	payload, err := proto.Marshal(request)
	if err != nil {
		return fmt.Errorf("marshal %s request failed: %w", method, err)
	}
	id := atomic.AddInt64(&c.nextID, 1)
	meta := &rpcMeta{
		CompressType: proto.Int32(0), CorrelationID: proto.Int64(id),
		AttachmentSize: proto.Int32(0), ContentType: proto.Int32(0),
		ChecksumType: proto.Int32(0), Request: &requestMeta{
			ServiceName: proto.String(c.service), MethodName: proto.String(method), LogID: proto.Int64(id),
		},
	}
	metaBytes, err := proto.Marshal(meta)
	if err != nil {
		return fmt.Errorf("marshal brpc meta failed: %w", err)
	}
	frame, err := buildFrame(metaBytes, payload)
	if err != nil {
		return err
	}
	if _, err := conn.Write(frame); err != nil {
		return fmt.Errorf("write brpc frame failed: %w", err)
	}
	responsePayload, err := readResponse(conn, id)
	if err != nil {
		return err
	}
	if err := proto.Unmarshal(responsePayload, response); err != nil {
		return fmt.Errorf("unmarshal %s response failed: %w", method, err)
	}
	return nil
}

func buildFrame(meta, payload []byte) ([]byte, error) {
	bodySize := len(meta) + len(payload)
	if bodySize > maxBodySize {
		return nil, fmt.Errorf("brpc request body too large: %d", bodySize)
	}
	frame := make([]byte, headerSize+bodySize)
	copy(frame[:4], magic)
	binary.BigEndian.PutUint32(frame[4:8], uint32(bodySize))
	binary.BigEndian.PutUint32(frame[8:12], uint32(len(meta)))
	copy(frame[headerSize:], meta)
	copy(frame[headerSize+len(meta):], payload)
	return frame, nil
}

func readResponse(reader io.Reader, expectedID int64) ([]byte, error) {
	header := make([]byte, headerSize)
	if _, err := io.ReadFull(reader, header); err != nil {
		return nil, fmt.Errorf("read brpc header failed: %w", err)
	}
	if string(header[:4]) != magic {
		return nil, fmt.Errorf("invalid brpc magic: %q", string(header[:4]))
	}
	bodySize := binary.BigEndian.Uint32(header[4:8])
	metaSize := binary.BigEndian.Uint32(header[8:12])
	if bodySize > maxBodySize || metaSize > bodySize {
		return nil, fmt.Errorf("invalid brpc sizes: body=%d meta=%d", bodySize, metaSize)
	}
	body := make([]byte, int(bodySize))
	if _, err := io.ReadFull(reader, body); err != nil {
		return nil, fmt.Errorf("read brpc body failed: %w", err)
	}
	var meta rpcMeta
	if err := proto.Unmarshal(body[:metaSize], &meta); err != nil {
		return nil, fmt.Errorf("unmarshal brpc response meta failed: %w", err)
	}
	if meta.CorrelationID != nil && *meta.CorrelationID != 0 && *meta.CorrelationID != expectedID {
		return nil, fmt.Errorf("brpc correlation mismatch: got=%d want=%d", *meta.CorrelationID, expectedID)
	}
	if meta.CompressType != nil && *meta.CompressType != noCompression {
		return nil, fmt.Errorf("unsupported brpc compression type: %d", *meta.CompressType)
	}
	if meta.Response != nil && meta.Response.ErrorCode != nil && *meta.Response.ErrorCode != 0 {
		text := ""
		if meta.Response.ErrorText != nil {
			text = *meta.Response.ErrorText
		}
		return nil, fmt.Errorf("brpc server error %d: %s", *meta.Response.ErrorCode, text)
	}
	attachment := int32(0)
	if meta.AttachmentSize != nil {
		attachment = *meta.AttachmentSize
	}
	if attachment < 0 || uint32(attachment) > bodySize-metaSize {
		return nil, fmt.Errorf("invalid brpc attachment size: %d", attachment)
	}
	return body[metaSize : bodySize-uint32(attachment)], nil
}
