// Package pngmeta does minimal, dependency-free PNG chunk surgery so the CLI can
// round-trip generation metadata. The backend stamps an `lcm` tEXt chunk holding
// a JSON blob of the parameters used; ReadLCM/BakedParams recover it (feeding
// precedence layer 2), and WriteText stamps client-side metadata onto outputs.
package pngmeta

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
)

// pngSig is the 8-byte PNG signature that prefixes every PNG file.
var pngSig = []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}

// chunk is one parsed PNG chunk: a 4-byte type and its data payload (the length
// and CRC are derived, not stored).
type chunk struct {
	typ  string
	data []byte
}

// parseChunks validates the signature and walks length|type|data|crc records.
func parseChunks(b []byte) ([]chunk, error) {
	if len(b) < len(pngSig) || !bytes.Equal(b[:len(pngSig)], pngSig) {
		return nil, fmt.Errorf("not a PNG (bad signature)")
	}
	var chunks []chunk
	off := len(pngSig)
	for off+8 <= len(b) {
		length := int(binary.BigEndian.Uint32(b[off : off+4]))
		typ := string(b[off+4 : off+8])
		dataStart := off + 8
		dataEnd := dataStart + length
		if dataEnd+4 > len(b) {
			return nil, fmt.Errorf("truncated chunk %q", typ)
		}
		chunks = append(chunks, chunk{typ: typ, data: b[dataStart:dataEnd]})
		off = dataEnd + 4 // skip CRC
		if typ == "IEND" {
			break
		}
	}
	return chunks, nil
}

// encodeChunk serializes one chunk as length|type|data|crc (CRC32-IEEE over
// type+data).
func encodeChunk(c chunk) []byte {
	out := make([]byte, 0, 12+len(c.data))
	var lenBuf [4]byte
	binary.BigEndian.PutUint32(lenBuf[:], uint32(len(c.data)))
	out = append(out, lenBuf[:]...)
	out = append(out, c.typ...)
	out = append(out, c.data...)

	crc := crc32.NewIEEE()
	crc.Write([]byte(c.typ))
	crc.Write(c.data)
	var crcBuf [4]byte
	binary.BigEndian.PutUint32(crcBuf[:], crc.Sum32())
	out = append(out, crcBuf[:]...)
	return out
}

// Chunks is a PNG parsed once so its tEXt chunks can be queried by keyword
// without re-walking the file for each lookup.
type Chunks struct {
	chunks []chunk
}

// Parse walks pngBytes once into a queryable Chunks value.
func Parse(pngBytes []byte) (Chunks, error) {
	cs, err := parseChunks(pngBytes)
	if err != nil {
		return Chunks{}, err
	}
	return Chunks{chunks: cs}, nil
}

// text returns the raw tEXt payload for keyword, or ok=false if absent. Absence
// is not an error; a malformed-JSON chunk's error surfaces from the decoding
// Find* method instead.
func (c Chunks) text(keyword string) ([]byte, bool) {
	for _, ch := range c.chunks {
		if ch.typ != "tEXt" {
			continue
		}
		i := bytes.IndexByte(ch.data, 0x00)
		if i < 0 {
			continue
		}
		if string(ch.data[:i]) != keyword {
			continue
		}
		return ch.data[i+1:], true
	}
	return nil, false
}

// FindLCM returns the lcm chunk's decoded payload, if present.
func (c Chunks) FindLCM() (map[string]any, bool, error) {
	text, ok := c.text("lcm")
	if !ok {
		return nil, false, nil
	}
	var m map[string]any
	if err := json.Unmarshal(text, &m); err != nil {
		return nil, true, fmt.Errorf("lcm chunk not JSON: %w", err)
	}
	return m, true, nil
}

// FindControlNetMap returns the controlnet_map chunk's decoded payload (a flat
// dict), if present. Written onto standalone control-map PNGs by scripts/cn_metadata.py.
func (c Chunks) FindControlNetMap() (map[string]any, bool, error) {
	text, ok := c.text("controlnet_map")
	if !ok {
		return nil, false, nil
	}
	var m map[string]any
	if err := json.Unmarshal(text, &m); err != nil {
		return nil, true, fmt.Errorf("controlnet_map chunk not JSON: %w", err)
	}
	return m, true, nil
}

// FindControlNet returns the controlnet chunk's decoded payload (a list of
// per-attachment provenance entries), if present. Written onto generation-output
// PNGs alongside lcm whenever the generation used a ControlNet binding.
func (c Chunks) FindControlNet() ([]any, bool, error) {
	text, ok := c.text("controlnet")
	if !ok {
		return nil, false, nil
	}
	var list []any
	if err := json.Unmarshal(text, &list); err != nil {
		return nil, true, fmt.Errorf("controlnet chunk not JSON: %w", err)
	}
	return list, true, nil
}

// WriteText inserts a tEXt chunk (keyword\x00text) immediately before IEND.
func WriteText(pngBytes []byte, keyword, text string) ([]byte, error) {
	chunks, err := parseChunks(pngBytes)
	if err != nil {
		return nil, err
	}
	data := append([]byte(keyword), 0x00)
	data = append(data, text...)
	textChunk := chunk{typ: "tEXt", data: data}

	out := make([]byte, 0, len(pngBytes)+12+len(data))
	out = append(out, pngSig...)
	for _, c := range chunks {
		if c.typ == "IEND" {
			out = append(out, encodeChunk(textChunk)...)
		}
		out = append(out, encodeChunk(c)...)
	}
	return out, nil
}

// ReadLCM finds the `lcm` tEXt chunk and unmarshals its JSON text into a map.
func ReadLCM(pngBytes []byte) (map[string]any, error) {
	chunks, err := Parse(pngBytes)
	if err != nil {
		return nil, err
	}
	m, ok, err := chunks.FindLCM()
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("no lcm tEXt chunk")
	}
	return m, nil
}

// BakedParams maps the lcm metadata keys onto GenerateRequest field names so the
// precedence resolver can apply them as layer 2. Unmapped keys are dropped.
func BakedParams(pngBytes []byte) (map[string]any, error) {
	m, err := ReadLCM(pngBytes)
	if err != nil {
		return nil, err
	}
	out := map[string]any{}
	move := func(from, to string) {
		if v, ok := m[from]; ok {
			out[to] = v
		}
	}
	move("prompt", "prompt")
	move("negative_prompt", "negative_prompt")
	move("seed", "seed")
	move("cfg", "guidance_scale")
	move("steps", "num_inference_steps")
	move("size", "size")
	move("scheduler_id", "scheduler_id")
	return out, nil
}
