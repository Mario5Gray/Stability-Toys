package stclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// APIError is a typed non-2xx server error carrying the operator-facing
// analysis_* (or other) error code from the {"error":{code,message}} body.
type APIError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *APIError) Error() string { return e.Code + ": " + e.Message }

// Describe POSTs a validated DescribeRequest to /v1/describe and decodes
// the typed DescribeResponse. Invalid requests never leave the client.
// Non-2xx responses with a typed error body return *APIError; anything
// else returns a plain error.
func (c *Client) Describe(ctx context.Context, req DescribeRequest) (*DescribeResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/describe", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		var envelope struct {
			Error APIError `json:"error"`
		}
		if json.Unmarshal(raw, &envelope) == nil && envelope.Error.Code != "" {
			return nil, &envelope.Error
		}
		return nil, fmt.Errorf("POST /v1/describe -> %s: %s", resp.Status, bytes.TrimSpace(raw))
	}
	var out DescribeResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}
