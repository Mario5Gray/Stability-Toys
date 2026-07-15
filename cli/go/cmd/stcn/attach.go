package main

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/darkbit/stability-toys/cli/st/internal/openapi"
)

type attachOpts struct {
	controlType string
	mapAssetRef string
	model       string
	id          string
	strength    float64
	start       float64
	end         float64
	strengthSet bool
	startSet    bool
	endSet      bool
}

// shellSafe matches the fields stcn will emit. It excludes whitespace and
// shell-active characters so the compact JSON is exactly one argv token under
// an unquoted $(stcn ...), while allowing ref-bearing punctuation (: / . -).
var shellSafe = regexp.MustCompile(`^[A-Za-z0-9._:/-]+$`)

// parseHead splits "control_type:map_asset_ref" on the FIRST colon, so refs
// that themselves contain a colon (e.g. fileref:MAP1) are preserved.
func parseHead(arg string) (string, string, error) {
	ct, ref, ok := strings.Cut(arg, ":")
	if !ok {
		return "", "", fmt.Errorf("positional must be <control_type>:<map_asset_ref>, got %q", arg)
	}
	return ct, ref, nil
}

func requireSafe(field, value string) error {
	if value == "" {
		return fmt.Errorf("%s must not be empty", field)
	}
	if !shellSafe.MatchString(value) {
		return fmt.Errorf("%s %q contains whitespace or a shell metacharacter (allowed: A-Z a-z 0-9 . _ : / -)", field, value)
	}
	return nil
}

func buildAttachment(o attachOpts) (openapi.ControlNetAttachment, error) {
	var a openapi.ControlNetAttachment

	if err := requireSafe("control_type", o.controlType); err != nil {
		return a, err
	}
	if err := requireSafe("map_asset_ref", o.mapAssetRef); err != nil {
		return a, err
	}

	id := o.id
	if id == "" {
		id = o.controlType // default attachment_id to the control type
	}
	if err := requireSafe("id", id); err != nil {
		return a, err
	}

	a.ControlType = o.controlType
	a.AttachmentId = id
	ref := o.mapAssetRef
	a.MapAssetRef = &ref

	if o.model != "" {
		if err := requireSafe("model", o.model); err != nil {
			return a, err
		}
		m := o.model
		a.ModelId = &m
	}

	if o.strengthSet {
		if o.strength < 0.0 || o.strength > 2.0 {
			return a, fmt.Errorf("strength %g out of range [0.0, 2.0]", o.strength)
		}
		s := float32(o.strength)
		a.Strength = &s
	}
	if o.startSet {
		if o.start < 0.0 || o.start > 1.0 {
			return a, fmt.Errorf("start %g out of range [0.0, 1.0]", o.start)
		}
		s := float32(o.start)
		a.StartPercent = &s
	}
	if o.endSet {
		if o.end < 0.0 || o.end > 1.0 {
			return a, fmt.Errorf("end %g out of range [0.0, 1.0]", o.end)
		}
		e := float32(o.end)
		a.EndPercent = &e
	}
	if o.startSet && o.endSet && o.start > o.end {
		return a, fmt.Errorf("start %g must be <= end %g", o.start, o.end)
	}

	return a, nil
}
