// Package pvp implements the combat-PvP training mode described in
// mc-agent's docs/plans/COMBAT_COMPETITION_PLAN.md Phase 1 and this
// repository's docs/plans/19-combat-pvp-training.md - independent learners
// (one cRL-go ppo.Trainer per managed combatant) training continuously
// against whatever their live opponent(s) currently do, with every
// participant in one match sharing a synchronized episode boundary.
package pvp

import (
	"context"
	"sync"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// MatchReset performs the actual match-wide world reset exactly once per
// round: healing/re-equipping/repositioning every named participant in the
// match, managed and unmanaged alike (see
// docs/plans/19-combat-pvp-training.md's "Unmanaged participants" note).
// Implementations wrap mc-agent's Phase 0 RCON helpers (SeedMatchReset).
type MatchReset interface {
	ResetMatch(ctx context.Context) error
}

// MatchResetFunc adapts a plain function to MatchReset.
type MatchResetFunc func(ctx context.Context) error

func (f MatchResetFunc) ResetMatch(ctx context.Context) error { return f(ctx) }

// matchBarrier synchronizes episode boundaries across every managed
// participant in one match - see docs/plans/19-combat-pvp-training.md's
// "The one genuinely new primitive" section for the full design rationale.
// It is cyclic: safe to reuse round after round. The zero value is not
// usable; construct with newMatchBarrier.
type matchBarrier struct {
	size  int
	reset MatchReset

	mu       sync.Mutex
	done     bool
	waiting  int
	release  chan struct{}
	resetErr error
}

func newMatchBarrier(size int, reset MatchReset) *matchBarrier {
	return &matchBarrier{size: size, reset: reset, release: make(chan struct{})}
}

// markDone records that the match's shared episode has ended. Idempotent -
// whichever sibling's Step notices first wins; later calls are no-ops.
func (b *matchBarrier) markDone() {
	b.mu.Lock()
	b.done = true
	b.mu.Unlock()
}

// isDone reports whether any sibling has already ended this round's
// episode, so a caller's own Step result can be forced to match.
func (b *matchBarrier) isDone() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.done
}

// arriveAtReset blocks until every one of the match's `size` participants
// has also called arriveAtReset for this round, then returns once the
// match-wide world reset - performed exactly once, by whichever caller
// arrives last - has completed. Every caller receives the same error, if
// the reset failed.
//
// Known limitation: if a sibling never calls arriveAtReset again (e.g. its
// own session errored out and its Trainer loop exited), every other
// sibling blocks here until ctx is canceled. This is accepted for v1: a
// match's sessions share one context from the runner, so a fatal error in
// one session is expected to cancel that same context for the whole match
// rather than being handled by this barrier itself. A caller whose own ctx
// cancels while waiting leaves the barrier's waiting count permanently
// short for this round (not decremented on the way out), which is fine
// precisely because nothing should call arriveAtReset again after
// cancellation.
func (b *matchBarrier) arriveAtReset(ctx context.Context) error {
	b.mu.Lock()
	b.waiting++
	last := b.waiting == b.size
	myRelease := b.release
	if last {
		b.mu.Unlock()
		err := b.reset.ResetMatch(ctx)
		b.mu.Lock()
		b.resetErr = err
		b.done = false
		b.waiting = 0
		b.release = make(chan struct{})
		b.mu.Unlock()
		close(myRelease)
		return err
	}
	b.mu.Unlock()

	select {
	case <-myRelease:
		b.mu.Lock()
		err := b.resetErr
		b.mu.Unlock()
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// matchEnvironment wraps one managed participant's underlying
// rl.Environment (in practice an mc-agent rlenv.Environment), synchronizing
// episode boundaries with every other participant in the same match via a
// shared matchBarrier. It implements rl.Environment unchanged, so
// cRL-go/pkg/ppo.Trainer has no awareness that combat synchronization
// exists at all - see docs/plans/19-combat-pvp-training.md.
//
// It also records the most recent Observation it has seen
// (LastObservation), independent of training/inference concerns - used by
// pkg/pvp/outcome.go to read a managed participant's own health at round
// end for per-game win/loss determination (docs/plans/20-combat-pvp-competition.md),
// without mc-agent needing any new surface for that (COMBAT_COMPETITION_PLAN.md
// finding 3).
type matchEnvironment struct {
	inner   rl.Environment
	barrier *matchBarrier

	mu   sync.Mutex
	last rl.Observation
}

func newMatchEnvironment(inner rl.Environment, barrier *matchBarrier) *matchEnvironment {
	return &matchEnvironment{inner: inner, barrier: barrier}
}

func (m *matchEnvironment) ObservationSize() int { return m.inner.ObservationSize() }
func (m *matchEnvironment) ActionSpace() int     { return m.inner.ActionSpace() }

// LastObservation returns the Observation from the most recent Step or
// Reset call. Zero-valued if neither has been called yet.
func (m *matchEnvironment) LastObservation() rl.Observation {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.last
}

func (m *matchEnvironment) setLastObservation(obs rl.Observation) {
	m.mu.Lock()
	m.last = obs
	m.mu.Unlock()
}

// Step always takes the real action against the underlying environment -
// skipping it to "fast forward" to Done would desync mc-agent's own
// tick-driven world state. If a sibling already ended this round's episode
// (barrier.isDone), this call's own result is forced Done=true even if the
// underlying environment hasn't naturally reached that state yet, so this
// participant's Trainer proceeds to Reset at most one live tick later than
// the sibling that actually ended it - an approximation of "ends together,"
// not literal same-tick synchrony (see docs/plans/19-combat-pvp-training.md).
func (m *matchEnvironment) Step(ctx context.Context, action rl.Action) (rl.StepResult, error) {
	result, err := m.inner.Step(ctx, action)
	if err != nil {
		return result, err
	}
	m.setLastObservation(result.Observation)
	if result.Done {
		m.barrier.markDone()
	} else if m.barrier.isDone() {
		result.Done = true
	}
	return result, nil
}

// Reset blocks until every sibling in the match has also called Reset, then
// returns the underlying environment's own fresh Reset observation once the
// shared match-wide world reset (heal/re-equip/reposition every named
// participant) has completed exactly once for this round.
func (m *matchEnvironment) Reset(ctx context.Context) (rl.Observation, error) {
	if err := m.barrier.arriveAtReset(ctx); err != nil {
		return rl.Observation{}, err
	}
	obs, err := m.inner.Reset(ctx)
	if err == nil {
		m.setLastObservation(obs)
	}
	return obs, err
}
