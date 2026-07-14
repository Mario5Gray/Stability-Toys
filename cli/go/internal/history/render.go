package history

import (
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
)

func RenderArgv(argv []string) string {
	parts := make([]string, 0, len(argv))
	for _, arg := range argv {
		if !isShellSafeArg(arg) {
			parts = append(parts, "'"+strings.ReplaceAll(arg, "'", "'\\''")+"'")
			continue
		}
		parts = append(parts, arg)
	}
	return strings.Join(parts, " ")
}

func isShellSafeArg(arg string) bool {
	if arg == "" {
		return false
	}
	for _, char := range arg {
		if (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') || (char >= '0' && char <= '9') {
			continue
		}
		if strings.ContainsRune("_@%+=:,./-", char) {
			continue
		}
		return false
	}
	return true
}

func CanonicalGenArgv(params map[string]any) []string {
	argv := []string{"st", "gen"}
	appendStr := func(flag, key string) {
		if value, ok := params[key].(string); ok && value != "" {
			argv = append(argv, flag, value)
		}
	}
	appendNum := func(flag, key string) {
		switch value := params[key].(type) {
		case int:
			argv = append(argv, flag, strconv.Itoa(value))
		case int64:
			argv = append(argv, flag, strconv.FormatInt(value, 10))
		case float64:
			argv = append(argv, flag, strconv.FormatFloat(value, 'f', -1, 64))
		}
	}

	appendStr("--prompt", "prompt")
	appendStr("--negative", "negative_prompt")
	appendStr("--size", "size")
	appendNum("--steps", "num_inference_steps")
	appendNum("--skip-step", "skip_step")
	appendNum("--cfg", "guidance_scale")
	appendNum("--seed", "seed")
	appendStr("--seed", "seed")
	appendStr("--scheduler", "scheduler_id")
	appendStr("--mode", "mode")
	argv = appendCanonicalControlnets(argv, params["controlnets"])
	return argv
}

func CanonicalGenDisplay(params map[string]any) string {
	return RenderArgv(CanonicalGenArgv(params))
}

func CloneParams(params map[string]any) map[string]any {
	out := make(map[string]any, len(params))
	for key, value := range params {
		out[key] = value
	}
	return out
}

func NormalizeExitCodes(codes []int) []int {
	seen := make(map[int]struct{}, len(codes))
	out := make([]int, 0, len(codes))
	for _, code := range codes {
		if _, ok := seen[code]; ok {
			continue
		}
		seen[code] = struct{}{}
		out = append(out, code)
	}
	slices.Sort(out)
	return out
}

func SnapshotSelector(selector Selector) *PolicySnapshot {
	switch selector.Kind {
	case SelectorHistory:
		return &PolicySnapshot{Selector: "history", HistoryID: selector.HistoryID}
	case SelectorRecent:
		return &PolicySnapshot{Selector: fmt.Sprintf("recent:%s", selector.Family)}
	default:
		return nil
	}
}

func appendCanonicalControlnets(argv []string, raw any) []string {
	controlnets, ok := normalizeControlnets(raw)
	if !ok || len(controlnets) == 0 {
		return argv
	}

	refs, strength, ok := canonicalControlRefs(controlnets)
	if ok {
		for _, ref := range refs {
			argv = append(argv, "--control-ref", ref)
		}
		if strength != nil {
			argv = append(argv, "--control-strength", strconv.FormatFloat(*strength, 'f', -1, 64))
		}
		return argv
	}

	for _, controlnet := range controlnets {
		body, err := json.Marshal(controlnet)
		if err != nil {
			continue
		}
		argv = append(argv, "--controlnet", string(body))
	}
	return argv
}

func normalizeControlnets(raw any) ([]map[string]any, bool) {
	switch value := raw.(type) {
	case []any:
		out := make([]map[string]any, 0, len(value))
		for _, item := range value {
			controlnet, ok := item.(map[string]any)
			if !ok {
				return nil, false
			}
			out = append(out, controlnet)
		}
		return out, true
	case []map[string]any:
		out := make([]map[string]any, 0, len(value))
		for _, item := range value {
			out = append(out, item)
		}
		return out, true
	default:
		return nil, false
	}
}

func canonicalControlRefs(controlnets []map[string]any) ([]string, *float64, bool) {
	refs := make([]string, 0, len(controlnets))
	var sharedStrength *float64
	for _, controlnet := range controlnets {
		ref, strength, ok := canonicalControlRef(controlnet)
		if !ok {
			return nil, nil, false
		}
		refs = append(refs, ref)
		switch {
		case sharedStrength == nil && strength == nil:
		case sharedStrength == nil && strength != nil:
			value := *strength
			sharedStrength = &value
		case sharedStrength != nil && strength != nil && *sharedStrength == *strength:
		default:
			return nil, nil, false
		}
	}
	return refs, sharedStrength, true
}

func canonicalControlRef(controlnet map[string]any) (string, *float64, bool) {
	controlType, _ := controlnet["control_type"].(string)
	mapAssetRef, _ := controlnet["map_asset_ref"].(string)
	if controlType == "" || mapAssetRef == "" {
		return "", nil, false
	}

	var strength *float64
	for key, value := range controlnet {
		switch key {
		case "attachment_id":
			continue
		case "control_type", "map_asset_ref":
			continue
		case "strength":
			if value == nil {
				continue
			}
			parsed, ok := numericValue(value)
			if !ok {
				return "", nil, false
			}
			strength = &parsed
		case "start_percent":
			parsed, ok := numericValue(value)
			if !ok || parsed != 0 {
				return "", nil, false
			}
		case "end_percent":
			parsed, ok := numericValue(value)
			if !ok || parsed != 1 {
				return "", nil, false
			}
		default:
			if value != nil {
				return "", nil, false
			}
		}
	}
	return controlType + ":" + mapAssetRef, strength, true
}

func numericValue(value any) (float64, bool) {
	switch number := value.(type) {
	case float64:
		return number, true
	case float32:
		return float64(number), true
	case int:
		return float64(number), true
	case int64:
		return float64(number), true
	case json.Number:
		parsed, err := number.Float64()
		return parsed, err == nil
	default:
		return 0, false
	}
}
