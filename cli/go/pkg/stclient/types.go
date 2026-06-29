package stclient

import "encoding/json"

// WS protocol (verified against server/ws_routes.py):
//   client -> {"type":"job:submit","id":<corr>,"jobType":"generate","params":{...}}
//   server -> {"type":"job:ack","id":<corr>,"jobId":<jobId>}
//          -> {"type":"job:progress","jobId":<jobId>,"delta":<str>}   (zero or more)
//          -> {"type":"job:complete","jobId":<jobId>,"outputs":[{"url":"/storage/<key>","key":<key>}],
//                "meta":{"seed":<int>,"backend":<str>,"sr":<bool>},"controlnet_artifacts":[...]?}
//          -> {"type":"job:error","jobId":<jobId>,"error":<str>}
// The submit `id` is echoed only on the ack; progress/complete/error carry just
// `jobId`, so the client learns jobId from the ack and matches by it thereafter.

// GenParams is the WS `params` payload: GenerateRequest fields plus the WS-only
// `init_image_ref`. Built by the precedence resolver (Task 8).
type GenParams map[string]any

type submitFrame struct {
	Type    string    `json:"type"`
	ID      string    `json:"id"`
	JobType string    `json:"jobType"`
	Params  GenParams `json:"params"`
}

func newSubmitFrame(corrID string, p GenParams) submitFrame {
	return submitFrame{Type: "job:submit", ID: corrID, JobType: "generate", Params: p}
}

// output is one job:complete result (a storage URL + its key).
type output struct {
	URL string `json:"url"`
	Key string `json:"key"`
}

// inFrame decodes any inbound server frame; Type selects which fields are set.
type inFrame struct {
	Type    string            `json:"type"`
	ID      string            `json:"id"`
	JobID   string            `json:"jobId"`
	Delta   string            `json:"delta"`
	Error   string            `json:"error"`
	Outputs []output          `json:"outputs"`
	Meta    map[string]any    `json:"meta"`
	CNArts  []json.RawMessage `json:"controlnet_artifacts"`
}

// Progress is a streamed generation update.
type Progress struct{ Delta string }

// Result is the resolved outcome of a completed generate job.
type Result struct {
	StorageKey  string
	StorageURL  string
	Seed        int64
	Meta        map[string]any
	CNArtifacts []json.RawMessage
}
