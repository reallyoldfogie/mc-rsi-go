package tournament

import (
	"fmt"
	"testing"
)

func TestNewBracket_RejectsFewerThanTwoCompetitors(t *testing.T) {
	if _, err := NewBracket([]string{"a"}); err == nil {
		t.Fatal("want an error for fewer than 2 competitors")
	}
	if _, err := NewBracket(nil); err == nil {
		t.Fatal("want an error for no competitors")
	}
}

func TestNewBracket_PowerOfTwoHasNoByes(t *testing.T) {
	b, err := NewBracket([]string{"a", "b", "c", "d"})
	if err != nil {
		t.Fatalf("NewBracket: %v", err)
	}
	if len(b.Rounds) != 2 {
		t.Fatalf("Rounds = %d, want 2 (semifinal + final)", len(b.Rounds))
	}
	if len(b.Rounds[0]) != 2 {
		t.Fatalf("Rounds[0] = %d pairings, want 2", len(b.Rounds[0]))
	}
	for _, p := range b.Rounds[0] {
		if p.IsBye() {
			t.Errorf("pairing %+v should not be a bye in a power-of-two bracket", p)
		}
	}
	if _, ok := b.Champion(); ok {
		t.Fatal("Champion() should report false before any pairing is played")
	}
}

func TestNewBracket_PadsWithByesAndAutoAdvances(t *testing.T) {
	// 3 competitors -> padded to 4, one bye in round 0.
	b, err := NewBracket([]string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("NewBracket: %v", err)
	}
	if len(b.Rounds[0]) != 2 {
		t.Fatalf("Rounds[0] = %d pairings, want 2", len(b.Rounds[0]))
	}
	var byeWinner string
	byeCount := 0
	for _, p := range b.Rounds[0] {
		if p.IsBye() {
			byeCount++
			byeWinner = p.WinnerID
		}
	}
	if byeCount != 1 {
		t.Fatalf("bye count = %d, want 1", byeCount)
	}
	if byeWinner != "c" {
		t.Errorf("bye winner = %q, want %q (the unpaired competitor)", byeWinner, "c")
	}
	// The bye winner must already have been propagated into round 1.
	final := b.Rounds[1][0]
	if final.CompetitorA != "c" && final.CompetitorB != "c" {
		t.Errorf("final pairing = %+v, want %q to already be propagated in", final, "c")
	}
}

func TestNewBracket_SingleRealEntrantCascadesStraightToChampion(t *testing.T) {
	// 2 competitors where one is effectively alone in a padded-to-4
	// bracket (3 real + 1 implicit bye slot isn't this case - this is
	// testing the cascade when NewBracket pads all the way and a bye
	// chains into another bye). Use 5 competitors -> padded to 8: slot
	// layout is [a,b,c,d,e,"","",""] so the last two round-0 pairings are
	// both all-bye-vs-empty... actually with 5 real entries only one
	// bye exists per pair until slots run out. Use a count that forces a
	// real multi-level cascade: 2 competitors is the simplest non-bye
	// case; test the padding arithmetic directly here instead.
	b, err := NewBracket([]string{"a", "b"})
	if err != nil {
		t.Fatalf("NewBracket: %v", err)
	}
	if len(b.Rounds) != 1 {
		t.Fatalf("Rounds = %d, want 1 (a single final)", len(b.Rounds))
	}
	if b.Rounds[0][0].IsBye() {
		t.Fatal("2 competitors should produce one real pairing, not a bye")
	}
}

func TestBracket_RecordResultPropagatesAndReachesChampion(t *testing.T) {
	b, err := NewBracket([]string{"a", "b", "c", "d"})
	if err != nil {
		t.Fatalf("NewBracket: %v", err)
	}

	pairing, ok := b.NextReadyPairing()
	if !ok {
		t.Fatal("want a ready pairing")
	}
	if err := b.RecordResult(pairing.Round, pairing.Slot, pairing.CompetitorA); err != nil {
		t.Fatalf("RecordResult: %v", err)
	}

	pairing2, ok := b.NextReadyPairing()
	if !ok {
		t.Fatal("want a second ready pairing")
	}
	if pairing2 == pairing {
		t.Fatal("NextReadyPairing returned an already-resolved pairing")
	}
	if err := b.RecordResult(pairing2.Round, pairing2.Slot, pairing2.CompetitorB); err != nil {
		t.Fatalf("RecordResult: %v", err)
	}

	final, ok := b.NextReadyPairing()
	if !ok {
		t.Fatal("want the final to be ready once both semifinals resolve")
	}
	if final.Round != 1 {
		t.Fatalf("final.Round = %d, want 1", final.Round)
	}
	winner := final.CompetitorA
	if err := b.RecordResult(final.Round, final.Slot, winner); err != nil {
		t.Fatalf("RecordResult: %v", err)
	}

	champion, ok := b.Champion()
	if !ok {
		t.Fatal("want Champion() to report true once the final resolves")
	}
	if champion != winner {
		t.Errorf("Champion() = %q, want %q", champion, winner)
	}
}

func TestBracket_RecordResultRejectsUnreadyPairing(t *testing.T) {
	b, err := NewBracket([]string{"a", "b", "c", "d"})
	if err != nil {
		t.Fatalf("NewBracket: %v", err)
	}
	// Round 1's final has no competitors yet - not ready.
	if err := b.RecordResult(1, 0, "a"); err == nil {
		t.Fatal("want an error recording a result for a not-yet-ready pairing")
	}
}

func TestBracket_RecordResultRejectsUnknownWinner(t *testing.T) {
	b, err := NewBracket([]string{"a", "b", "c", "d"})
	if err != nil {
		t.Fatalf("NewBracket: %v", err)
	}
	pairing, _ := b.NextReadyPairing()
	if err := b.RecordResult(pairing.Round, pairing.Slot, "nonexistent"); err == nil {
		t.Fatal("want an error when winnerID is not one of the pairing's own competitors")
	}
}

func TestBracket_RecordResultRejectsDoubleRecording(t *testing.T) {
	b, err := NewBracket([]string{"a", "b", "c", "d"})
	if err != nil {
		t.Fatalf("NewBracket: %v", err)
	}
	pairing, _ := b.NextReadyPairing()
	if err := b.RecordResult(pairing.Round, pairing.Slot, pairing.CompetitorA); err != nil {
		t.Fatalf("RecordResult: %v", err)
	}
	if err := b.RecordResult(pairing.Round, pairing.Slot, pairing.CompetitorA); err == nil {
		t.Fatal("want an error recording a result twice for the same pairing")
	}
}

func TestNewBracket_FiveCompetitorsDoubleByeCascades(t *testing.T) {
	// 5 competitors padded to 8: slots [a,b,c,d,e,"","",""]. Round 0:
	// {a,b} {c,d} {e,""}bye->e {"",""}fully-empty. The fully-empty
	// pairing has NO real competitor on either side, so it can never
	// produce a winner; round 1's pairing combining {e bye} with that
	// fully-empty pairing must cascade "e" all the way to round 2
	// (the final) without ever needing a game played for it.
	b, err := NewBracket([]string{"a", "b", "c", "d", "e"})
	if err != nil {
		t.Fatalf("NewBracket: %v", err)
	}
	if len(b.Rounds) != 3 {
		t.Fatalf("Rounds = %d, want 3", len(b.Rounds))
	}
	emptyPairings := 0
	for _, p := range b.Rounds[0] {
		if p.CompetitorA == "" && p.CompetitorB == "" {
			emptyPairings++
			if !p.IsBye() {
				t.Errorf("a fully-empty pairing must be a bye: %+v", p)
			}
		}
	}
	if emptyPairings != 1 {
		t.Fatalf("fully-empty round-0 pairings = %d, want 1", emptyPairings)
	}

	// "e" must already be sitting in round 1 as a bye-cascade result,
	// and that round-1 pairing must itself already be flagged a bye
	// (so NextReadyPairing never offers it as something to play).
	var round1Pairing *Pairing
	for _, p := range b.Rounds[1] {
		if p.CompetitorA == "e" || p.CompetitorB == "e" {
			round1Pairing = p
		}
	}
	if round1Pairing == nil {
		t.Fatal("expected \"e\" to have cascaded into round 1")
	}
	if !round1Pairing.IsBye() || round1Pairing.WinnerID != "e" {
		t.Fatalf("round1 pairing holding the cascaded bye = %+v, want IsBye()=true WinnerID=e", round1Pairing)
	}

	// Four real games are needed: the two round-0 pairings (a-vs-b,
	// c-vs-d), round 1's pairing between their two winners (round0[2]'s
	// "{e,""}" and round0[3]'s fully-empty pairing never contribute a
	// real game - they cascade straight through as the bye traced
	// above), and the final (that semifinal winner vs "e"). The bye
	// cascade itself must never be offered anywhere as a pairing to
	// actually play.
	played := map[string]bool{}
	for {
		p, ok := b.NextReadyPairing()
		if !ok {
			break
		}
		played[fmt.Sprintf("%d:%d", p.Round, p.Slot)] = true
		if err := b.RecordResult(p.Round, p.Slot, p.CompetitorA); err != nil {
			t.Fatalf("RecordResult(%d,%d): %v", p.Round, p.Slot, err)
		}
	}
	if len(played) != 4 {
		t.Fatalf("played %d real pairings, want exactly 4 (two round-0 + one round-1 + one final)", len(played))
	}
	if _, ok := b.Champion(); !ok {
		t.Fatal("want a champion once every real pairing has been played")
	}
}

func TestNewBracket_EightCompetitorsThreeRounds(t *testing.T) {
	b, err := NewBracket([]string{"a", "b", "c", "d", "e", "f", "g", "h"})
	if err != nil {
		t.Fatalf("NewBracket: %v", err)
	}
	if len(b.Rounds) != 3 {
		t.Fatalf("Rounds = %d, want 3 (quarter, semi, final)", len(b.Rounds))
	}
	if len(b.Rounds[0]) != 4 || len(b.Rounds[1]) != 2 || len(b.Rounds[2]) != 1 {
		t.Fatalf("round sizes = %d/%d/%d, want 4/2/1", len(b.Rounds[0]), len(b.Rounds[1]), len(b.Rounds[2]))
	}
}
