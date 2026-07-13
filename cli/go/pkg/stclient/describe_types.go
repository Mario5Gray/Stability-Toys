package stclient

import (
	"encoding/json"
	"fmt"
)

// Describe contract (spec: docs/superpowers/specs/2026-07-11-describe-analysis-
// interface-design.md). Closed enums; exactly one typed params block per task;
// server-side policy (mode -> analysis_profile -> delegates) is never expressed
// in the request.

type DescribeTaskKind string

const (
	TaskKindCaption DescribeTaskKind = "caption"
	TaskKindDetect  DescribeTaskKind = "detect"
	TaskKindOcr     DescribeTaskKind = "ocr"
	TaskKindPose    DescribeTaskKind = "pose"
	TaskKindEmbed   DescribeTaskKind = "embed"
)

// RolePrimary is the only target role with defined v1 semantics; omitted or
// empty Role means primary. Other role strings pass through as opaque labels.
const RolePrimary = "primary"

type DescribeRequest struct {
	Mode    *string          `json:"mode,omitempty"`
	Targets []DescribeTarget `json:"targets"`
	Tasks   []DescribeTask   `json:"tasks"`
}

type DescribeTarget struct {
	ID       string  `json:"id"`
	AssetRef *string `json:"asset_ref,omitempty"`
	URL      *string `json:"url,omitempty"`
	Role     string  `json:"role,omitempty"`
}

func (t DescribeTarget) effectiveRole() string {
	if t.Role == "" {
		return RolePrimary
	}
	return t.Role
}

type DescribeTask struct {
	ID        string           `json:"id"`
	Kind      DescribeTaskKind `json:"kind"`
	TargetIDs []string         `json:"target_ids,omitempty"`
	Caption   *CaptionParams   `json:"caption,omitempty"`
	Detect    *DetectParams    `json:"detect,omitempty"`
	Ocr       *OcrParams       `json:"ocr,omitempty"`
	Pose      *PoseParams      `json:"pose,omitempty"`
	Embed     *EmbedParams     `json:"embed,omitempty"`
}

// v1-minimal params; fields are additive later.
type CaptionParams struct {
	Prompt *string `json:"prompt,omitempty"`
}

type DetectParams struct {
	Labels        []string `json:"labels,omitempty"`
	MinConfidence *float64 `json:"min_confidence,omitempty"`
}

type OcrParams struct{}
type PoseParams struct{}
type EmbedParams struct{}

type DescribeStatus string

const (
	StatusOK      DescribeStatus = "ok"
	StatusPartial DescribeStatus = "partial"
	StatusFailed  DescribeStatus = "failed"
)

type DescribeResponse struct {
	Status       DescribeStatus        `json:"status"`
	Summary      *string               `json:"summary,omitempty"`
	Observations []DescribeObservation `json:"observations"`
	Artifacts    []DescribeArtifact    `json:"artifacts,omitempty"`
	Runs         []DescribeRun         `json:"runs"`
}

type ObservationKind string

const (
	ObsKindText      ObservationKind = "text"
	ObsKindDetection ObservationKind = "detection"
	ObsKindAttribute ObservationKind = "attribute"
	ObsKindKeypoints ObservationKind = "keypoints"
)

type DescribeObservation struct {
	TaskID   string          `json:"task_id"`
	TargetID string          `json:"target_id"`
	Kind     ObservationKind `json:"kind"`

	Text      *TextObservation      `json:"text,omitempty"`
	Detection *DetectionObservation `json:"detection,omitempty"`
	Attribute *AttributeObservation `json:"attribute,omitempty"`
	Keypoints *KeypointsObservation `json:"keypoints,omitempty"`
}

type TextObservation struct {
	Content string `json:"content"`
}

type DetectionObservation struct {
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"` // [0,1]
	Box        Box     `json:"box"`
}

// Box coordinates are normalized to [0,1] relative to the target image;
// x,y is the top-left corner.
type Box struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	W float64 `json:"w"`
	H float64 `json:"h"`
}

type AttributeObservation struct {
	Name       string   `json:"name"`
	Value      string   `json:"value"`
	Confidence *float64 `json:"confidence,omitempty"` // [0,1]
	Box        *Box     `json:"box,omitempty"`
}

type KeypointsObservation struct {
	Skeleton string     `json:"skeleton,omitempty"`
	Points   []Keypoint `json:"points"`
}

type Keypoint struct {
	Name       string   `json:"name,omitempty"`
	X          float64  `json:"x"` // normalized [0,1]
	Y          float64  `json:"y"` // normalized [0,1]
	Confidence *float64 `json:"confidence,omitempty"`
}

type ArtifactKind string

const ArtifactKindEmbeddingRef ArtifactKind = "embedding_ref"

type DescribeArtifact struct {
	TaskID   string       `json:"task_id"`
	TargetID string       `json:"target_id"`
	Kind     ArtifactKind `json:"kind"`
	Ref      string       `json:"ref"`
	Dims     *int         `json:"dims,omitempty"`
}

type RunStatus string

const (
	RunSucceeded RunStatus = "succeeded"
	RunFailed    RunStatus = "failed"
	RunSkipped   RunStatus = "skipped"
)

type RunError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type DescribeRun struct {
	TaskID   string    `json:"task_id"`
	TargetID string    `json:"target_id"`
	Delegate string    `json:"delegate"`
	Status   RunStatus `json:"status"`
	Error    *RunError `json:"error,omitempty"`
	// RawOutput is the opaque provider payload; the contract deliberately
	// does not type it (spec: raw provider outputs preserved verbatim).
	RawOutput json.RawMessage `json:"raw_output,omitempty"`
}

func validationErr(code, format string, args ...any) error {
	return fmt.Errorf("%s: %s", code, fmt.Sprintf(format, args...))
}

// Validate applies the client-boundary contract rules. The server enforces
// the same rules; this exists so misuse fails before a request is sent.
func (r *DescribeRequest) Validate() error {
	if len(r.Targets) == 0 || len(r.Tasks) == 0 {
		return validationErr("analysis_invalid_request", "targets and tasks must be non-empty")
	}
	targetRoles := make(map[string]string, len(r.Targets))
	primaryCount := 0
	for _, tg := range r.Targets {
		if tg.ID == "" {
			return validationErr("analysis_invalid_request", "target id must be set")
		}
		if _, dup := targetRoles[tg.ID]; dup {
			return validationErr("analysis_invalid_request", "duplicate target id %q", tg.ID)
		}
		hasRef := tg.AssetRef != nil && *tg.AssetRef != ""
		hasURL := tg.URL != nil && *tg.URL != ""
		if hasRef == hasURL {
			return validationErr("analysis_invalid_request",
				"target %q must set exactly one of asset_ref or url", tg.ID)
		}
		targetRoles[tg.ID] = tg.effectiveRole()
		if tg.effectiveRole() == RolePrimary {
			primaryCount++
		}
	}
	taskIDs := make(map[string]bool, len(r.Tasks))
	for _, tk := range r.Tasks {
		if tk.ID == "" {
			return validationErr("analysis_invalid_request", "task id must be set")
		}
		if taskIDs[tk.ID] {
			return validationErr("analysis_invalid_request", "duplicate task id %q", tk.ID)
		}
		taskIDs[tk.ID] = true
		if err := tk.validateParams(); err != nil {
			return err
		}
		for _, id := range tk.TargetIDs {
			if _, ok := targetRoles[id]; !ok {
				return validationErr("analysis_target_binding_invalid",
					"task %q references unknown target %q", tk.ID, id)
			}
		}
		// Zero-run binding is a validation error: omitted target_ids requires
		// at least one effective-primary target.
		if len(tk.TargetIDs) == 0 && primaryCount == 0 {
			return validationErr("analysis_target_binding_invalid",
				"task %q binds to zero targets: no primary targets declared", tk.ID)
		}
	}
	return nil
}

func (t DescribeTask) validateParams() error {
	blocks := 0
	var matched bool
	if t.Caption != nil {
		blocks++
		matched = matched || t.Kind == TaskKindCaption
	}
	if t.Detect != nil {
		blocks++
		matched = matched || t.Kind == TaskKindDetect
	}
	if t.Ocr != nil {
		blocks++
		matched = matched || t.Kind == TaskKindOcr
	}
	if t.Pose != nil {
		blocks++
		matched = matched || t.Kind == TaskKindPose
	}
	if t.Embed != nil {
		blocks++
		matched = matched || t.Kind == TaskKindEmbed
	}
	switch t.Kind {
	case TaskKindCaption, TaskKindDetect, TaskKindOcr, TaskKindPose, TaskKindEmbed:
	default:
		return validationErr("analysis_invalid_request", "task %q has unknown kind %q", t.ID, t.Kind)
	}
	if blocks != 1 || !matched {
		return validationErr("analysis_invalid_request",
			"task %q must set exactly one params block matching kind %q", t.ID, t.Kind)
	}
	return nil
}
