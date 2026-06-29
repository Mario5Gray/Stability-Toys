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
	chunks, err := parseChunks(pngBytes)
	if err != nil {
		return nil, err
	}
	for _, c := range chunks {
		if c.typ != "tEXt" {
			continue
		}
		i := bytes.IndexByte(c.data, 0x00)
		if i < 0 {
			continue
		}
		if string(c.data[:i]) != "lcm" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal(c.data[i+1:], &m); err != nil {
			return nil, fmt.Errorf("lcm chunk not JSON: %w", err)
		}
		return m, nil
	}
	return nil, fmt.Errorf("no lcm tEXt chunk")
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
