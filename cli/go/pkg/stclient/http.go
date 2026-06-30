package stclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"sort"
	"strconv"
)

// Mode is one server-side mode, decoded from GET /api/modes.
// IsDefault is true when the mode matches the top-level default_mode field.
// ControlNetEnabled is extracted from controlnet_policy.enabled.
type Mode struct {
	Name               string  `json:"name"`
	IsDefault          bool    `json:"is_default,omitempty"`
	Model              string  `json:"model"`
	DefaultSize        string  `json:"default_size"`
	DefaultSteps       int     `json:"default_steps"`
	DefaultGuidance    float64 `json:"default_guidance"`
	DefaultSchedulerID string  `json:"default_scheduler_id,omitempty"`
	ControlNetEnabled  bool    `json:"controlnet_enabled"`
	ChatEnabled        bool    `json:"chat_enabled"`
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

// Modes returns all server-side modes sorted by name. The live GET /api/modes
// response keys modes by name with per-mode config; IsDefault marks the
// top-level default_mode. ControlNetEnabled is extracted from
// controlnet_policy.enabled.
func (c *Client) Modes(ctx context.Context) ([]Mode, error) {
	var body struct {
		DefaultMode string                     `json:"default_mode"`
		Modes       map[string]json.RawMessage `json:"modes"`
	}
	if err := c.getJSON(ctx, "/api/modes", &body); err != nil {
		return nil, err
	}
	names := make([]string, 0, len(body.Modes))
	for name := range body.Modes {
		names = append(names, name)
	}
	sort.Strings(names)
	modes := make([]Mode, 0, len(names))
	for _, name := range names {
		var cfg struct {
			Model              string  `json:"model"`
			DefaultSize        string  `json:"default_size"`
			DefaultSteps       int     `json:"default_steps"`
			DefaultGuidance    float64 `json:"default_guidance"`
			DefaultSchedulerID string  `json:"default_scheduler_id"`
			ControlNetPolicy   struct {
				Enabled bool `json:"enabled"`
			} `json:"controlnet_policy"`
			ChatEnabled bool `json:"chat_enabled"`
		}
		_ = json.Unmarshal(body.Modes[name], &cfg)
		modes = append(modes, Mode{
			Name:               name,
			IsDefault:          name == body.DefaultMode,
			Model:              cfg.Model,
			DefaultSize:        cfg.DefaultSize,
			DefaultSteps:       cfg.DefaultSteps,
			DefaultGuidance:    cfg.DefaultGuidance,
			DefaultSchedulerID: cfg.DefaultSchedulerID,
			ControlNetEnabled:  cfg.ControlNetPolicy.Enabled,
			ChatEnabled:        cfg.ChatEnabled,
		})
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

// ReloadModes requests the server to hot-reload modes.yaml from disk via
// POST /api/modes/reload. The reload is applied after any pending jobs complete.
func (c *Client) ReloadModes(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/modes/reload", nil)
	if err != nil {
		return err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("modes reload -> %s", resp.Status)
	}
	return nil
}

// multipartFile builds a multipart body with a single "file" part plus the
// given extra form fields, returning the body and its Content-Type.
func multipartFile(filename string, data []byte, fields map[string]string) (*bytes.Buffer, string, error) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	fw, err := mw.CreateFormFile("file", filename)
	if err != nil {
		return nil, "", err
	}
	if _, err := fw.Write(data); err != nil {
		return nil, "", err
	}
	for k, v := range fields {
		if err := mw.WriteField(k, v); err != nil {
			return nil, "", err
		}
	}
	if err := mw.Close(); err != nil {
		return nil, "", err
	}
	return &buf, mw.FormDataContentType(), nil
}

// Upload posts data as a multipart "file" to POST /v1/upload and returns the
// fileRef the backend assigns. bucket is an optional intent label (e.g.
// "image", "canny") sent as a "type" form field; an empty bucket adds no
// extra field. The backend may use the type field for routing; v1.x treats
// it as client-side intent only.
func (c *Client) Upload(ctx context.Context, filename string, data []byte, bucket string) (string, error) {
	var fields map[string]string
	if bucket != "" {
		fields = map[string]string{"type": bucket}
	}
	buf, contentType, err := multipartFile(filename, data, fields)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/upload", buf)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", contentType)
	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return "", fmt.Errorf("upload -> %s", resp.Status)
	}
	var body struct {
		FileRef string `json:"fileRef"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return "", err
	}
	return body.FileRef, nil
}

// SuperRes posts data to POST /superres with the given magnitude (1..3) and
// returns the upscaled image bytes. out_format/quality are left to the server
// defaults (png / 92).
func (c *Client) SuperRes(ctx context.Context, data []byte, magnitude int) ([]byte, error) {
	buf, contentType, err := multipartFile("input.png", data, map[string]string{
		"magnitude": strconv.Itoa(magnitude),
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/superres", buf)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", contentType)
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("superres -> %s", resp.Status)
	}
	return io.ReadAll(resp.Body)
}

// FetchStorage GETs the result image bytes for a storage key
// (the key from a job:complete output URL "/storage/<key>").
func (c *Client) FetchStorage(ctx context.Context, key string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/storage/"+key, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("storage/%s -> %s", key, resp.Status)
	}
	return io.ReadAll(resp.Body)
}
