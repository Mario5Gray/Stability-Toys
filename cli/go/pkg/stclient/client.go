// Package stclient is the single operation surface for the backend's HTTP and
// WebSocket APIs. cmd/st (and a future MCP server) are thin adapters over it;
// no HTTP/WS calls live anywhere else.
package stclient

import (
	"net/http"
	"strings"
	"time"
)

// Client talks to one backend base URL. It is safe for sequential use by a CLI.
type Client struct {
	baseURL string
	http    *http.Client
}

// Option customizes a Client at construction.
type Option func(*Client)

// WithHTTPClient overrides the default *http.Client (useful in tests).
func WithHTTPClient(h *http.Client) Option { return func(c *Client) { c.http = h } }

// New returns a Client for baseURL (trailing slash trimmed).
func New(baseURL string, opts ...Option) *Client {
	c := &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: 120 * time.Second},
	}
	for _, o := range opts {
		o(c)
	}
	return c
}
