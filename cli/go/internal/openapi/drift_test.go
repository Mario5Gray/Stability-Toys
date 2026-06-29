package openapi

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
)

// canonicalize normalizes a JSON document for comparison: encoding/json.Marshal
// sorts object keys alphabetically and drops insignificant whitespace, so two
// specs that differ only in key order or formatting compare equal.
func canonicalize(raw []byte) (string, error) {
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return "", err
	}
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func TestCanonicalizeIgnoresKeyOrderAndWhitespace(t *testing.T) {
	a := []byte(`{"b":1,  "a":2}`)
	b := []byte(`{"a":2,"b":1}`)
	ca, err := canonicalize(a)
	if err != nil {
		t.Fatal(err)
	}
	cb, err := canonicalize(b)
	if err != nil {
		t.Fatal(err)
	}
	if ca != cb {
		t.Fatalf("key order/whitespace must not matter: %q vs %q", ca, cb)
	}
}

func TestCanonicalizeDetectsRealDiff(t *testing.T) {
	a, _ := canonicalize([]byte(`{"a":1}`))
	b, _ := canonicalize([]byte(`{"a":2}`))
	if a == b {
		t.Fatal("a real value difference must not canonicalize equal")
	}
}

// TestOpenAPISnapshotMatchesLive guards against backend contract drift. It is
// skipped unless ST_SERVER points at a live backend; the committed snapshot is
// verbatim 3.1.0 (same as the live FastAPI spec), so this is a 3.1-vs-3.1 diff.
func TestOpenAPISnapshotMatchesLive(t *testing.T) {
	base := os.Getenv("ST_SERVER")
	if base == "" {
		t.Skip("set ST_SERVER to run the OpenAPI drift check")
	}
	url := strings.TrimRight(base, "/") + "/openapi.json"
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("fetch %s: %v", url, err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	live, err := canonicalize(body)
	if err != nil {
		t.Fatalf("live spec not JSON: %v", err)
	}

	snap, err := os.ReadFile("../../openapi.snapshot.json")
	if err != nil {
		t.Fatal(err)
	}
	want, err := canonicalize(snap)
	if err != nil {
		t.Fatalf("snapshot not JSON: %v", err)
	}

	if live != want {
		t.Fatalf("OpenAPI drift: live /openapi.json differs from openapi.snapshot.json — " +
			"refresh the snapshot from the backend and run `make gen`")
	}
}
