package brpcwire

import (
	"bytes"
	"testing"

	proto "github.com/gogo/protobuf/proto"
)

func TestBuildFrame(t *testing.T) {
	frame, err := buildFrame([]byte{1, 2}, []byte{3, 4, 5})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(frame[:4]), magic; got != want {
		t.Fatalf("magic=%q want=%q", got, want)
	}
	if !bytes.Equal(frame[headerSize:], []byte{1, 2, 3, 4, 5}) {
		t.Fatalf("unexpected body: %v", frame[headerSize:])
	}
}

func TestNewClientRejectsRetryAmbiguity(t *testing.T) {
	if _, err := NewClient("127.0.0.1:1", "service", 1, 0); err == nil {
		t.Fatal("expected attempts validation error")
	}
}

var _ proto.Message = (*rpcMeta)(nil)
