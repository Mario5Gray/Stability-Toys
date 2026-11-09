// Package output decides where a generated image is written. It supports an
// auto-incrementing out-####.<ext> scheme within the configured output
// directory, plus an explicit --outfile override.
package output

import (
	"fmt"
	"os"
	"path/filepath"
)

// NextPath returns the first free out-####.<format> slot in dir (creating dir).
func NextPath(dir, format string) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	for i := 1; i < 100000; i++ {
		p := filepath.Join(dir, fmt.Sprintf("out-%04d.%s", i, format))
		if _, err := os.Stat(p); os.IsNotExist(err) {
			return p, nil
		}
	}
	return "", fmt.Errorf("no free out-#### slot in %s", dir)
}

// Resolve picks the output path: an explicit outfile (extension appended if
// absent, joined under dir if relative) or the next auto-incremented slot.
func Resolve(outfile, dir, format string) (string, error) {
	if outfile == "" {
		return NextPath(dir, format)
	}
	if filepath.Ext(outfile) == "" {
		outfile += "." + format
	}
	if !filepath.IsAbs(outfile) {
		outfile = filepath.Join(dir, outfile)
	}
	return outfile, nil
}

// Write writes data to path.
func Write(path string, data []byte) error {
	return os.WriteFile(path, data, 0o644)
}

// Save resolves the destination and writes data in one step. Callers that
// need the chosen path (to report it) get it back; the resolve-then-write
// sequence used to be duplicated at every call site.
func Save(outfile, dir, format string, data []byte) (string, error) {
	path, err := Resolve(outfile, dir, format)
	if err != nil {
		return "", err
	}
	if err := Write(path, data); err != nil {
		return "", err
	}
	return path, nil
}
