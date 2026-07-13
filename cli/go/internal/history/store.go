package history

import (
	"context"
	"errors"
)

var (
	ErrNotFound        = errors.New("history entry not found")
	ErrNoEligibleEntry = errors.New("no eligible history entry")
	ErrCorruptState    = errors.New("corrupt history state")
)

type Filter struct {
	Family           Family
	ExitCodes        []int
	RequireEffective bool
}

type HistoryStore interface {
	ReserveID(context.Context) (int64, error)
	Append(context.Context, Entry) error
	Get(context.Context, int64) (Entry, error)
	Latest(context.Context, Filter) (Entry, error)
}

type PolicyStore interface {
	LoadPolicy(context.Context) (Policy, error)
	SavePolicy(context.Context, Policy) error
}

type Store interface {
	HistoryStore
	PolicyStore
}
