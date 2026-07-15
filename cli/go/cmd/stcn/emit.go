package main

import (
	"encoding/json"

	"github.com/darkbit/stability-toys/cli/st/internal/openapi"
)

// marshalCompact renders the attachment as compact JSON (no indentation, no
// spaces). Combined with shell-safe field validation, the result is exactly
// one argv token under an unquoted $(stcn ...).
func marshalCompact(a openapi.ControlNetAttachment) ([]byte, error) {
	return json.Marshal(a)
}
