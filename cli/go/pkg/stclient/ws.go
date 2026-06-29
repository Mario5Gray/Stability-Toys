package stclient

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
)

// progressBuffer bounds how many job:progress frames are retained for a caller
// that reads the returned channel after Generate resolves. Excess frames are
// dropped (see Generate) rather than blocking the read loop.
const progressBuffer = 16

// wsURL derives the WebSocket endpoint from the base URL (http->ws, https->wss).
func (c *Client) wsURL() string {
	return strings.Replace(c.baseURL, "http", "ws", 1) + "/v1/ws"
}

// corrID returns a short random correlation id for a job:submit.
func corrID() string {
	var b [6]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// Generate dials /v1/ws, submits a generate job, and blocks until the job
// resolves. It returns a Result on job:complete (carrying the /storage/<key>
// the caller fetches via FetchStorage) or an error on job:error.
//
// The returned channel holds up to progressBuffer job:progress frames seen
// before completion; it is already closed on return, so a caller may drain it
// or ignore it. The channel is never read while the job runs, so progress sends
// are non-blocking: once the buffer is full, further progress is dropped. This
// keeps Generate from deadlocking when the backend streams many progress frames
// (server/ws_routes.py emits job:progress on every job mutation).
func (c *Client) Generate(ctx context.Context, p GenParams) (<-chan Progress, *Result, error) {
	conn, _, err := websocket.Dial(ctx, c.wsURL(), &websocket.DialOptions{HTTPClient: c.http})
	if err != nil {
		return nil, nil, err
	}
	prog := make(chan Progress, progressBuffer)

	if err := wsjson.Write(ctx, conn, newSubmitFrame(corrID(), p)); err != nil {
		close(prog)
		conn.Close(websocket.StatusInternalError, "submit failed")
		return nil, nil, err
	}

	for {
		var f inFrame
		if err := wsjson.Read(ctx, conn, &f); err != nil {
			close(prog)
			conn.Close(websocket.StatusInternalError, "read failed")
			return nil, nil, err
		}
		switch f.Type {
		case "job:ack":
			continue
		case "job:progress":
			// Non-blocking: drop if no buffer space (no live reader exists yet).
			select {
			case prog <- Progress{Delta: f.Delta}:
			default:
			}
		case "job:error":
			close(prog)
			conn.Close(websocket.StatusNormalClosure, "")
			return prog, nil, fmt.Errorf("job error: %s", f.Error)
		case "job:complete":
			close(prog)
			conn.Close(websocket.StatusNormalClosure, "")
			res := &Result{Meta: f.Meta, CNArtifacts: f.CNArts}
			if len(f.Outputs) > 0 {
				res.StorageKey = f.Outputs[0].Key
				res.StorageURL = f.Outputs[0].URL
			}
			if s, ok := f.Meta["seed"].(float64); ok {
				res.Seed = int64(s)
			}
			return prog, res, nil
		}
	}
}

// controlFrame dials a fresh /v1/ws connection, sends one control frame (with a
// correlation id the server echoes), and waits for the matching *:ack. Cancel
// and priority act on the global worker pool, so they need not share the
// generate connection.
func (c *Client) controlFrame(ctx context.Context, send map[string]any, wantAck string) error {
	send["id"] = corrID()
	conn, _, err := websocket.Dial(ctx, c.wsURL(), &websocket.DialOptions{HTTPClient: c.http})
	if err != nil {
		return err
	}
	defer conn.Close(websocket.StatusNormalClosure, "")
	if err := wsjson.Write(ctx, conn, send); err != nil {
		return err
	}
	for {
		var f inFrame
		if err := wsjson.Read(ctx, conn, &f); err != nil {
			return err
		}
		switch f.Type {
		case wantAck:
			return nil
		case "job:error":
			return fmt.Errorf("job error: %s", f.Error)
		}
	}
}

// Cancel requests cancellation of a running job (job:cancel -> job:cancel:ack).
func (c *Client) Cancel(ctx context.Context, jobID string) error {
	return c.controlFrame(ctx, map[string]any{"type": "job:cancel", "jobId": jobID}, "job:cancel:ack")
}

// SetPriority requests a priority change for a job (job:priority ->
// job:priority:ack). The backend currently acks this as a no-op
// (server/ws_routes.py: "priority not yet implemented").
func (c *Client) SetPriority(ctx context.Context, jobID string, level int) error {
	return c.controlFrame(ctx, map[string]any{"type": "job:priority", "jobId": jobID, "priority": level}, "job:priority:ack")
}
