package main

import (
	"encoding/json"
	"testing"
)

// parse is a small helper that downgrades the input and unmarshals the result.
func parse(t *testing.T, in string) map[string]any {
	t.Helper()
	out, err := downgradeSpec([]byte(in))
	if err != nil {
		t.Fatalf("downgradeSpec: %v", err)
	}
	var doc map[string]any
	if err := json.Unmarshal(out, &doc); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	return doc
}

func TestSetsOpenAPIVersionTo30(t *testing.T) {
	doc := parse(t, `{"openapi":"3.1.0","paths":{}}`)
	if doc["openapi"] != "3.0.3" {
		t.Fatalf("openapi = %v, want 3.0.3", doc["openapi"])
	}
}

func TestCollapsesNullableScalarAnyOf(t *testing.T) {
	doc := parse(t, `{"openapi":"3.1.0","x":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Np","description":"d"}}`)
	x := doc["x"].(map[string]any)
	if _, ok := x["anyOf"]; ok {
		t.Fatalf("anyOf should be removed, got %v", x)
	}
	if x["type"] != "string" {
		t.Fatalf("type = %v, want string", x["type"])
	}
	if x["nullable"] != true {
		t.Fatalf("nullable = %v, want true", x["nullable"])
	}
	if x["title"] != "Np" || x["description"] != "d" {
		t.Fatalf("sibling keywords dropped: %v", x)
	}
}

func TestWrapsNullableRefInAllOf(t *testing.T) {
	doc := parse(t, `{"openapi":"3.1.0","x":{"anyOf":[{"$ref":"#/components/schemas/Y"},{"type":"null"}]}}`)
	x := doc["x"].(map[string]any)
	if _, ok := x["anyOf"]; ok {
		t.Fatalf("anyOf should be removed, got %v", x)
	}
	if x["nullable"] != true {
		t.Fatalf("nullable = %v, want true", x["nullable"])
	}
	allOf, ok := x["allOf"].([]any)
	if !ok || len(allOf) != 1 {
		t.Fatalf("allOf = %v, want single-member slice", x["allOf"])
	}
	ref := allOf[0].(map[string]any)
	if ref["$ref"] != "#/components/schemas/Y" {
		t.Fatalf("ref preserved incorrectly: %v", ref)
	}
	// A bare $ref must NOT be hoisted as a sibling of nullable (ignored in 3.0).
	if _, ok := x["$ref"]; ok {
		t.Fatalf("$ref must be wrapped in allOf, not left as sibling: %v", x)
	}
}

func TestLeavesGenuineUnionUntouched(t *testing.T) {
	doc := parse(t, `{"openapi":"3.1.0","x":{"anyOf":[{"type":"string"},{"type":"integer"}]}}`)
	x := doc["x"].(map[string]any)
	anyOf, ok := x["anyOf"].([]any)
	if !ok || len(anyOf) != 2 {
		t.Fatalf("genuine union should be preserved, got %v", x)
	}
	if _, ok := x["nullable"]; ok {
		t.Fatalf("nullable must not be set for a null-free union: %v", x)
	}
}
