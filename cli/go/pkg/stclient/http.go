package stclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
)

// Mode is one server-side mode. The backend keys modes by name in the
// GET /api/modes response; richer per-mode fields are added by later tasks.
type Mode struct {
	Name string `json:"name"`
}

// ModelsStatus is the untyped GET /api/models/status payload (backend, vram,
// current_mode, capabilities, ...). Callers index the keys they need.
type ModelsStatus map[string]any

// getJSON performs a GET and decodes a 2xx JSON body into out.
func (c *Client) getJSON(ctx context.Context, path string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+path, nil)
	if err != nil {
		return err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("GET %s -> %s", path, resp.Status)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// Modes lists available mode names. The live GET /api/modes returns
// {"modes": {<name>: {...}}, ...}; names are returned sorted for determinism.
func (c *Client) Modes(ctx context.Context) ([]Mode, error) {
	var body struct {
		Modes map[string]json.RawMessage `json:"modes"`
	}
	if err := c.getJSON(ctx, "/api/modes", &body); err != nil {
		return nil, err
	}
	names := make([]string, 0, len(body.Modes))
	for name := range body.Modes {
		names = append(names, name)
	}
	sort.Strings(names)
	modes := make([]Mode, len(names))
	for i, name := range names {
		modes[i] = Mode{Name: name}
	}
	return modes, nil
}

// CurrentMode returns the backend's active mode. The current mode is reported by
// GET /api/models/status (current_mode), not GET /api/modes. An empty string
// means the backend has no mode loaded.
func (c *Client) CurrentMode(ctx context.Context) (string, error) {
	var body struct {
		CurrentMode string `json:"current_mode"`
	}
	if err := c.getJSON(ctx, "/api/models/status", &body); err != nil {
		return "", err
	}
	return body.CurrentMode, nil
}

// Models returns the raw GET /api/models/status payload.
func (c *Client) Models(ctx context.Context) (ModelsStatus, error) {
	var m ModelsStatus
	if err := c.getJSON(ctx, "/api/models/status", &m); err != nil {
		return nil, err
	}
	return m, nil
}

// SwitchMode requests a switch to the named mode via POST /api/modes/switch
// with a JSON body {"mode": name} (ModeSwitchRequest). The switch is queued
// server-side behind any pending jobs.
func (c *Client) SwitchMode(ctx context.Context, name string) error {
	payload, err := json.Marshal(map[string]string{"mode": name})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/modes/switch", bytes.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("switch mode %q -> %s", name, resp.Status)
	}
	return nil
}
