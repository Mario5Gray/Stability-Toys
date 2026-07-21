package history

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
)

type FSStore struct {
	root string
}

func NewFSStore(root string) *FSStore {
	return &FSStore{root: root}
}

func ResolveStateRoot() (string, error) {
	base := os.Getenv("XDG_STATE_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		base = filepath.Join(home, ".local", "state")
	}

	root := filepath.Join(base, "st")
	if err := os.MkdirAll(root, 0o700); err != nil {
		return "", err
	}
	if err := os.Chmod(root, 0o700); err != nil {
		return "", err
	}
	return root, nil
}

func (s *FSStore) historyPath() string {
	return filepath.Join(s.root, "history.jsonl")
}

func (s *FSStore) nextIDPath() string {
	return filepath.Join(s.root, "next-id")
}

func (s *FSStore) policyPath() string {
	return filepath.Join(s.root, "conflate-policy.json")
}

func (s *FSStore) lockPath() string {
	return filepath.Join(s.root, "state.lock")
}

func (s *FSStore) withLock(fn func() error) error {
	if err := os.MkdirAll(s.root, 0o700); err != nil {
		return fmt.Errorf("initialize state directory %s: %w", s.root, err)
	}
	if err := os.Chmod(s.root, 0o700); err != nil {
		return fmt.Errorf("protect state directory %s: %w", s.root, err)
	}

	lock, err := os.OpenFile(s.lockPath(), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return fmt.Errorf("open state lock %s: %w", s.lockPath(), err)
	}
	defer lock.Close()
	if err := lock.Chmod(0o600); err != nil {
		return fmt.Errorf("protect state lock %s: %w", s.lockPath(), err)
	}
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX); err != nil {
		return fmt.Errorf("lock state %s: %w", s.lockPath(), err)
	}
	defer syscall.Flock(int(lock.Fd()), syscall.LOCK_UN)

	return fn()
}

func (s *FSStore) ReserveID(_ context.Context) (int64, error) {
	var reserved int64
	err := s.withLock(func() error {
		next := int64(1)
		data, err := os.ReadFile(s.nextIDPath())
		switch {
		case err == nil && strings.TrimSpace(string(data)) != "":
			next, err = strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
			if err != nil || next < 1 {
				return fmt.Errorf("%w: %s", ErrCorruptState, s.nextIDPath())
			}
		case err != nil && !os.IsNotExist(err):
			return fmt.Errorf("read %s: %w", s.nextIDPath(), err)
		}

		reserved = next
		return atomicReplace(s.nextIDPath(), []byte(strconv.FormatInt(next+1, 10)+"\n"))
	})
	return reserved, err
}

func (s *FSStore) Append(_ context.Context, entry Entry) error {
	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}

	return s.withLock(func() error {
		file, err := os.OpenFile(s.historyPath(), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
		if err != nil {
			return fmt.Errorf("open %s: %w", s.historyPath(), err)
		}
		defer file.Close()
		if err := file.Chmod(0o600); err != nil {
			return fmt.Errorf("protect %s: %w", s.historyPath(), err)
		}
		if _, err := file.Write(append(data, '\n')); err != nil {
			return fmt.Errorf("append %s: %w", s.historyPath(), err)
		}
		if err := file.Sync(); err != nil {
			return fmt.Errorf("fsync %s: %w", s.historyPath(), err)
		}
		return nil
	})
}

func (s *FSStore) Get(_ context.Context, id int64) (Entry, error) {
	entries, err := s.readAll()
	if err != nil {
		return Entry{}, err
	}
	for _, entry := range entries {
		if entry.ID == id {
			return entry, nil
		}
	}
	return Entry{}, ErrNotFound
}

func (s *FSStore) Latest(_ context.Context, filter Filter) (Entry, error) {
	entries, err := s.readAll()
	if err != nil {
		return Entry{}, err
	}

	var best *Entry
	for i := range entries {
		entry := entries[i]
		if filter.Family != "" && entry.Family != filter.Family {
			continue
		}
		if filter.RequireEffective && entry.Effective == nil {
			continue
		}
		if len(filter.ExitCodes) > 0 && !containsExit(filter.ExitCodes, entry.ExitCode) {
			continue
		}
		if best == nil || entry.ID > best.ID {
			best = &entry
		}
	}
	if best == nil {
		return Entry{}, ErrNoEligibleEntry
	}
	return *best, nil
}

func (s *FSStore) List(_ context.Context) ([]Entry, error) {
	entries, err := s.readAll()
	if err != nil {
		return nil, err
	}
	return entries, nil
}

func (s *FSStore) LoadPolicy(_ context.Context) (Policy, error) {
	data, err := os.ReadFile(s.policyPath())
	if os.IsNotExist(err) {
		return DefaultPolicy(), nil
	}
	if err != nil {
		return Policy{}, err
	}

	var policy Policy
	if err := json.Unmarshal(data, &policy); err != nil {
		return Policy{}, fmt.Errorf("%w: conflate-policy.json: %v", ErrCorruptState, err)
	}
	if err := ValidatePolicy(policy); err != nil {
		return Policy{}, fmt.Errorf("%w: %s: %v", ErrCorruptState, s.policyPath(), err)
	}
	return policy, nil
}

func (s *FSStore) SavePolicy(_ context.Context, policy Policy) error {
	if err := ValidatePolicy(policy); err != nil {
		return err
	}
	data, err := json.MarshalIndent(policy, "", "  ")
	if err != nil {
		return err
	}

	return s.withLock(func() error {
		return atomicReplace(s.policyPath(), append(data, '\n'))
	})
}

func (s *FSStore) readAll() ([]Entry, error) {
	data, err := os.ReadFile(s.historyPath())
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	var out []Entry
	lines := bytes.Split(data, []byte{'\n'})
	completeFinalLine := bytes.HasSuffix(data, []byte{'\n'})
	for i, raw := range lines {
		line := bytes.TrimSpace(raw)
		if len(line) == 0 {
			continue
		}

		var entry Entry
		if err := json.Unmarshal(line, &entry); err != nil {
			if i == len(lines)-1 && !completeFinalLine {
				break
			}
			return nil, fmt.Errorf("%w: history.jsonl: %v", ErrCorruptState, err)
		}
		if entry.SchemaVersion != 1 || entry.ID < 1 {
			return nil, fmt.Errorf("%w: history.jsonl entry has schema_version=%d id=%d", ErrCorruptState, entry.SchemaVersion, entry.ID)
		}
		out = append(out, entry)
	}
	return out, nil
}

func atomicReplace(path string, data []byte) error {
	dir := filepath.Dir(path)
	file, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmp := file.Name()
	defer os.Remove(tmp)

	if err := file.Chmod(0o600); err != nil {
		file.Close()
		return err
	}
	if _, err := io.Copy(file, bytes.NewReader(data)); err != nil {
		file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		return err
	}

	dirFile, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer dirFile.Close()
	return dirFile.Sync()
}

func containsExit(codes []int, code int) bool {
	for _, candidate := range codes {
		if candidate == code {
			return true
		}
	}
	return false
}
