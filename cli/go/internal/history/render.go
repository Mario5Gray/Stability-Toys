package history

import (
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
