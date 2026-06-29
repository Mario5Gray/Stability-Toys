// Command downspec converts an OpenAPI 3.1 document into a 3.0.x-compatible
// document that oapi-codegen (which does not yet support 3.1) can consume.
//
// The committed openapi.snapshot.json is kept verbatim so the gated drift guard
// (Task 16) can diff it against the live FastAPI backend, which serves 3.1.0.
// `make gen` runs this converter to produce a throwaway 3.0 intermediate and
// generates Go types from that.
//
// The only 3.1 construct in the backend snapshot that oapi-codegen rejects is
// FastAPI's nullable form `anyOf: [{<schema>}, {"type": "null"}]`. This tool:
//   - strips {"type":"null"} members from anyOf/oneOf, marking the parent
//     nullable: true (the 3.0 spelling);
//   - collapses a resulting single inline member into the parent;
//   - wraps a resulting single $ref member in allOf (a bare $ref sibling of
//     nullable is ignored by strict 3.0 parsers);
//   - leaves genuine multi-member unions untouched;
//   - pins the top-level openapi version to 3.0.3.
//
// Usage: downspec <input.json> <output.json>
package main

import (
	"encoding/json"
	"fmt"
	"os"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: downspec <input.json> <output.json>")
		os.Exit(2)
	}
	in, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "read:", err)
		os.Exit(1)
	}
	out, err := downgradeSpec(in)
	if err != nil {
		fmt.Fprintln(os.Stderr, "downgrade:", err)
		os.Exit(1)
	}
	if err := os.WriteFile(os.Args[2], out, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write:", err)
		os.Exit(1)
	}
}

// downgradeSpec parses an OpenAPI 3.1 JSON document, rewrites nullable anyOf/oneOf
// constructs into their 3.0 form, pins the version to 3.0.3, and returns indented JSON.
func downgradeSpec(in []byte) ([]byte, error) {
	var doc any
	if err := json.Unmarshal(in, &doc); err != nil {
		return nil, err
	}
	doc = walk(doc)
	if m, ok := doc.(map[string]any); ok {
		if _, has := m["openapi"]; has {
			m["openapi"] = "3.0.3"
		}
	}
	return json.MarshalIndent(doc, "", "  ")
}

// walk recursively rewrites nullable anyOf/oneOf nodes in place.
func walk(v any) any {
	switch n := v.(type) {
	case map[string]any:
		for k, child := range n {
			n[k] = walk(child)
		}
		for _, key := range []string{"anyOf", "oneOf"} {
			arr, ok := n[key].([]any)
			if !ok {
				continue
			}
			kept := make([]any, 0, len(arr))
			nullable := false
			for _, m := range arr {
				if isNullSchema(m) {
					nullable = true
					continue
				}
				kept = append(kept, m)
			}
			if !nullable {
				// No null member: a genuine union; leave it untouched.
				continue
			}
			n["nullable"] = true
			switch len(kept) {
			case 0:
				delete(n, key)
			case 1:
				delete(n, key)
				member := kept[0].(map[string]any)
				if _, isRef := member["$ref"]; isRef {
					// A $ref cannot carry sibling keywords in 3.0; wrap in allOf.
					n["allOf"] = []any{member}
				} else {
					// Inline member; existing parent keywords win on conflict.
					for mk, mv := range member {
						if _, exists := n[mk]; !exists {
							n[mk] = mv
						}
					}
				}
			default:
				n[key] = kept
			}
		}
		return n
	case []any:
		for i, child := range n {
			n[i] = walk(child)
		}
		return n
	default:
		return v
	}
}

// isNullSchema reports whether v is exactly {"type": "null"}.
func isNullSchema(v any) bool {
	m, ok := v.(map[string]any)
	if !ok {
		return false
	}
	if len(m) != 1 {
		return false
	}
	return m["type"] == "null"
}
