// Package tournament implements mc-agent's COMBAT_COMPETITION_PLAN.md
// Phase 2 ("competition mode") bracket orchestrator, built on pkg/pvp's
// Phase 1 primitives (match barrier, shared-arena connection/reset, team
// formation, containment) - see mc-rsi-trainer's own
// docs/plans/20-combat-pvp-competition.md for the full design.
package tournament

import "fmt"

// Pairing is one bracket match between two competitors (teams), identified
// by ID. Both competitor IDs are empty until an earlier round resolves
// into this one (or, for round 0, until NewBracket places them directly).
type Pairing struct {
	Round                int
	Slot                 int
	CompetitorA          string
	CompetitorB          string
	WinnerID             string
	GamesWonA, GamesWonB int

	// aliveCount is the number of real (non-padding) competitors
	// anywhere in this pairing's whole subtree, computed once at
	// construction from the competitor-count structure alone - never
	// re-derived from how many of this pairing's own two slots happen
	// to be filled right now. This is what lets IsBye() stay correct
	// across the whole bracket's lifetime: a later-round pairing with
	// exactly one slot filled because its *other* semifinal simply
	// hasn't been played yet (aliveCount >= 2) must never be mistaken
	// for a genuine construction-time bye (aliveCount <= 1) - that
	// conflation was a real bug caught by this package's own tests.
	aliveCount int
}

// IsBye reports whether this pairing auto-advances without a game ever
// being played for it - true exactly when its whole subtree contains at
// most one real competitor.
func (p *Pairing) IsBye() bool {
	return p.aliveCount <= 1
}

// IsReady reports whether both competitor slots are filled and no result
// has been recorded for it yet - i.e. this is an actual pairing to run.
// Never true for a bye (see IsBye): a bye's WinnerID is always set by the
// time either of its slots could be read as "filled" in the first place.
func (p *Pairing) IsReady() bool {
	return p.CompetitorA != "" && p.CompetitorB != "" && p.WinnerID == ""
}

// Bracket is a complete single-elimination structure: Rounds[0] is the
// first round (built directly from the configured competitor order),
// each later round's pairings are filled in as the previous round
// resolves. The final round (Rounds[len(Rounds)-1]) has exactly one
// pairing; its WinnerID is the tournament champion once set.
type Bracket struct {
	Rounds [][]*Pairing
}

// NewBracket builds a single-elimination bracket for competitorIDs, in
// the given order (bracket seeding is the caller's own concern - this
// just lays out whatever order it's given). A non-power-of-two count is
// padded with empty ("bye") slots to the next power of two. Every
// pairing whose whole subtree contains at most one real competitor
// (aliveCount <= 1) is resolved immediately, with that propagation
// cascading through as many further rounds as it takes.
func NewBracket(competitorIDs []string) (*Bracket, error) {
	if len(competitorIDs) < 2 {
		return nil, fmt.Errorf("tournament: at least 2 competitors required, got %d", len(competitorIDs))
	}
	size := nextPowerOfTwo(len(competitorIDs))
	slots := make([]string, size)
	copy(slots, competitorIDs)

	leafAlive := make([]int, size)
	for i, s := range slots {
		if s != "" {
			leafAlive[i] = 1
		}
	}

	firstRound := make([]*Pairing, size/2)
	roundAlive := make([]int, size/2)
	for i := range firstRound {
		a, b := slots[2*i], slots[2*i+1]
		alive := leafAlive[2*i] + leafAlive[2*i+1]
		p := &Pairing{Round: 0, Slot: i, CompetitorA: a, CompetitorB: b, aliveCount: alive}
		if p.IsBye() {
			p.WinnerID = firstNonEmpty(a, b)
		}
		firstRound[i] = p
		roundAlive[i] = alive
	}

	rounds := [][]*Pairing{firstRound}
	currentAlive := roundAlive
	for len(currentAlive) > 1 {
		nextSize := len(currentAlive) / 2
		round := make([]*Pairing, nextSize)
		nextAlive := make([]int, nextSize)
		for i := range round {
			alive := currentAlive[2*i] + currentAlive[2*i+1]
			round[i] = &Pairing{Round: len(rounds), Slot: i, aliveCount: alive}
			nextAlive[i] = alive
		}
		rounds = append(rounds, round)
		currentAlive = nextAlive
	}

	b := &Bracket{Rounds: rounds}
	b.propagateInitialByes()
	return b, nil
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// propagateInitialByes pushes every already-resolved (bye) pairing's
// winner forward into subsequent rounds, in round order so a chain of
// byes cascades correctly.
func (b *Bracket) propagateInitialByes() {
	for round := 0; round < len(b.Rounds)-1; round++ {
		for _, p := range b.Rounds[round] {
			if p.WinnerID != "" {
				b.placeWinner(round, p.Slot, p.WinnerID)
			}
		}
	}
}

// placeWinner writes winnerID into round+1's appropriate competitor
// slot, and resolves that next pairing immediately if it is itself a
// bye (aliveCount <= 1 - a static property computed at construction, not
// re-derived here, so this can never fire just because the *other* slot
// happens to not be filled yet).
func (b *Bracket) placeWinner(round, slot int, winnerID string) {
	nextRound := round + 1
	if nextRound >= len(b.Rounds) {
		return
	}
	next := b.Rounds[nextRound][slot/2]
	if slot%2 == 0 {
		next.CompetitorA = winnerID
	} else {
		next.CompetitorB = winnerID
	}
	if next.WinnerID == "" && next.IsBye() {
		next.WinnerID = firstNonEmpty(next.CompetitorA, next.CompetitorB)
		b.placeWinner(nextRound, next.Slot, next.WinnerID)
	}
}

// RecordResult sets the winner of the pairing at (round, slot) and
// propagates it into the next round. Returns an error if the pairing is
// not ready (already resolved, a bye, or missing a competitor), or if
// winnerID is neither of its two competitors.
func (b *Bracket) RecordResult(round, slot int, winnerID string) error {
	if round < 0 || round >= len(b.Rounds) || slot < 0 || slot >= len(b.Rounds[round]) {
		return fmt.Errorf("tournament: no such pairing (round %d, slot %d)", round, slot)
	}
	p := b.Rounds[round][slot]
	if !p.IsReady() {
		return fmt.Errorf("tournament: pairing (round %d, slot %d) is not ready to record a result", round, slot)
	}
	if winnerID != p.CompetitorA && winnerID != p.CompetitorB {
		return fmt.Errorf("tournament: winner %q is not one of this pairing's competitors (%q, %q)", winnerID, p.CompetitorA, p.CompetitorB)
	}
	p.WinnerID = winnerID
	b.placeWinner(round, slot, winnerID)
	return nil
}

// NextReadyPairing returns the first pairing (in round, then slot, order)
// that is ready to be played - both competitors known, no result
// recorded yet - or ok=false if none remain (either the tournament is
// complete, or every currently fillable pairing is waiting on an earlier
// round that hasn't been played yet).
func (b *Bracket) NextReadyPairing() (pairing *Pairing, ok bool) {
	for _, round := range b.Rounds {
		for _, p := range round {
			if p.IsReady() {
				return p, true
			}
		}
	}
	return nil, false
}

// Champion returns the tournament winner and true once the final round's
// single pairing has resolved.
func (b *Bracket) Champion() (string, bool) {
	final := b.Rounds[len(b.Rounds)-1]
	if len(final) != 1 || final[0].WinnerID == "" {
		return "", false
	}
	return final[0].WinnerID, true
}

func nextPowerOfTwo(n int) int {
	size := 1
	for size < n {
		size *= 2
	}
	return size
}
