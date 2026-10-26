package tournament

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/pvp"
)

func TestNewMetrics_RejectsEmptyRunID(t *testing.T) {
	if _, err := NewMetrics(""); err == nil {
		t.Fatal("want an error for an empty run ID")
	}
}

func TestMetrics_SetBracketRound(t *testing.T) {
	m, err := NewMetrics("run1")
	if err != nil {
		t.Fatalf("NewMetrics: %v", err)
	}
	m.SetBracketRound(2)
	if got := testutil.ToFloat64(m.bracketRound); got != 2 {
		t.Errorf("bracketRound = %v, want 2", got)
	}
}

func TestMetrics_SetChampionDecided(t *testing.T) {
	m, err := NewMetrics("run1")
	if err != nil {
		t.Fatalf("NewMetrics: %v", err)
	}
	m.SetChampionDecided(false)
	if got := testutil.ToFloat64(m.championDecided); got != 0 {
		t.Errorf("championDecided = %v, want 0", got)
	}
	m.SetChampionDecided(true)
	if got := testutil.ToFloat64(m.championDecided); got != 1 {
		t.Errorf("championDecided = %v, want 1", got)
	}
}

func TestMetrics_RecordPairing(t *testing.T) {
	m, err := NewMetrics("run1")
	if err != nil {
		t.Fatalf("NewMetrics: %v", err)
	}
	m.RecordPairing("red", "blue")
	m.RecordPairing("red", "green")

	if got := testutil.ToFloat64(m.pairingsPlayedTotal.WithLabelValues("run1", "red", matchPhaseLabel)); got != 2 {
		t.Errorf("red pairingsPlayedTotal = %v, want 2", got)
	}
	if got := testutil.ToFloat64(m.pairingsPlayedTotal.WithLabelValues("run1", "blue", matchPhaseLabel)); got != 1 {
		t.Errorf("blue pairingsPlayedTotal = %v, want 1", got)
	}
}

func TestMetrics_RecordGame_WinnerOnly(t *testing.T) {
	m, err := NewMetrics("run1")
	if err != nil {
		t.Fatalf("NewMetrics: %v", err)
	}
	outcome := pvp.GameOutcome{
		Winner: "red",
		DamageDealt: map[pvp.Team]float64{
			"red":  12.5,
			"blue": 4,
		},
	}
	m.RecordGame("red", "blue", outcome)

	if got := testutil.ToFloat64(m.gamesPlayedTotal.WithLabelValues("run1", "red", matchPhaseLabel)); got != 1 {
		t.Errorf("red gamesPlayedTotal = %v, want 1", got)
	}
	if got := testutil.ToFloat64(m.gamesPlayedTotal.WithLabelValues("run1", "blue", matchPhaseLabel)); got != 1 {
		t.Errorf("blue gamesPlayedTotal = %v, want 1", got)
	}
	if got := testutil.ToFloat64(m.gamesWonTotal.WithLabelValues("run1", "red", matchPhaseLabel)); got != 1 {
		t.Errorf("red gamesWonTotal = %v, want 1", got)
	}
	if got := testutil.ToFloat64(m.gamesWonTotal.WithLabelValues("run1", "blue", matchPhaseLabel)); got != 0 {
		t.Errorf("blue gamesWonTotal = %v, want 0", got)
	}
	if got := testutil.ToFloat64(m.damageDealt.WithLabelValues("run1", "red", matchPhaseLabel)); got != 12.5 {
		t.Errorf("red damageDealt = %v, want 12.5", got)
	}
	if got := testutil.ToFloat64(m.damageDealt.WithLabelValues("run1", "blue", matchPhaseLabel)); got != 4 {
		t.Errorf("blue damageDealt = %v, want 4", got)
	}
}

func TestMetrics_RecordGame_DrawIncrementsNeitherWinCount(t *testing.T) {
	m, err := NewMetrics("run1")
	if err != nil {
		t.Fatalf("NewMetrics: %v", err)
	}
	m.RecordGame("red", "blue", pvp.GameOutcome{Draw: true})

	if got := testutil.ToFloat64(m.gamesWonTotal.WithLabelValues("run1", "red", matchPhaseLabel)); got != 0 {
		t.Errorf("red gamesWonTotal = %v, want 0", got)
	}
	if got := testutil.ToFloat64(m.gamesWonTotal.WithLabelValues("run1", "blue", matchPhaseLabel)); got != 0 {
		t.Errorf("blue gamesWonTotal = %v, want 0", got)
	}
}

func TestMetrics_DistinctCompetitorsHaveIndependentSeries(t *testing.T) {
	m, err := NewMetrics("run1")
	if err != nil {
		t.Fatalf("NewMetrics: %v", err)
	}
	m.RecordGame("red", "blue", pvp.GameOutcome{Winner: "red"})
	m.RecordGame("green", "yellow", pvp.GameOutcome{Winner: "yellow"})

	if got := testutil.ToFloat64(m.gamesWonTotal.WithLabelValues("run1", "red", matchPhaseLabel)); got != 1 {
		t.Errorf("red gamesWonTotal = %v, want 1", got)
	}
	if got := testutil.ToFloat64(m.gamesWonTotal.WithLabelValues("run1", "yellow", matchPhaseLabel)); got != 1 {
		t.Errorf("yellow gamesWonTotal = %v, want 1", got)
	}
	if got := testutil.ToFloat64(m.gamesWonTotal.WithLabelValues("run1", "green", matchPhaseLabel)); got != 0 {
		t.Errorf("green gamesWonTotal = %v, want 0", got)
	}
}
