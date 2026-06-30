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
// resolves. It returns the jobID assigned by the server (from job:ack), a
// Result on job:complete, or an error on job:error.
//
// onAck is called once with the jobID immediately after the server
// acknowledges the submission. onProgress is called synchronously for each
// job:progress frame in the order received. Either callback may be nil.
func (c *Client) Generate(ctx context.Context, p GenParams, onAck func(jobID string), onProgress func(delta string)) (string, *Result, error) {
	conn, _, err := websocket.Dial(ctx, c.wsURL(), &websocket.DialOptions{HTTPClient: c.http})
	if err != nil {
		return "", nil, err
	}
	if err := wsjson.Write(ctx, conn, newSubmitFrame(corrID(), p)); err != nil {
		conn.Close(websocket.StatusInternalError, "submit failed")
		return "", nil, err
	}
	var jobID string
	for {
		var f inFrame
		if err := wsjson.Read(ctx, conn, &f); err != nil {
			conn.Close(websocket.StatusInternalError, "read failed")
			return "", nil, err
		}
		switch f.Type {
		case "job:ack":
			jobID = f.JobID
			if onAck != nil {
				onAck(jobID)
			}
		case "job:progress":
			if onProgress != nil && f.Delta != "" {
				onProgress(f.Delta)
			}
		case "job:error":
			conn.Close(websocket.StatusNormalClosure, "")
			return "", nil, fmt.Errorf("job error: %s", f.Error)
		case "job:complete":
			conn.Close(websocket.StatusNormalClosure, "")
			res := &Result{Meta: f.Meta, CNArtifacts: f.CNArts}
			if len(f.Outputs) > 0 {
				res.StorageKey = f.Outputs[0].Key
				res.StorageURL = f.Outputs[0].URL
			}
			if s, ok := f.Meta["seed"].(float64); ok {
				res.Seed = int64(s)
			}
			return jobID, res, nil
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
