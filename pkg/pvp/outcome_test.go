package pvp

import (
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

func TestComputeGameOutcome_DeadTeamLosesOutright(t *testing.T) {
	outcome, err := ComputeGameOutcome([]ParticipantHealth{
		{Team: "Red", EndHealth: 0},
		{Team: "Blue", EndHealth: 15},
	})
	if err != nil {
		t.Fatalf("ComputeGameOutcome: %v", err)
	}
	if outcome.Winner != "Blue" || outcome.Draw {
		t.Fatalf("outcome = %+v, want Blue to win outright", outcome)
	}
}

func TestComputeGameOutcome_DeadTeamLosesEvenWithLessDamageDealt(t *testing.T) {
	// Red dealt more net damage (Blue is at 2) but Red itself died - death
	// takes priority over damage differential.
	outcome, err := ComputeGameOutcome([]ParticipantHealth{
		{Team: "Red", EndHealth: 0},
		{Team: "Blue", EndHealth: 2},
	})
	if err != nil {
		t.Fatalf("ComputeGameOutcome: %v", err)
	}
	if outcome.Winner != "Blue" || outcome.Draw {
		t.Fatalf("outcome = %+v, want Blue to win despite having taken more damage", outcome)
	}
}

func TestComputeGameOutcome_TimeoutUsesDamageDifferential(t *testing.T) {
	outcome, err := ComputeGameOutcome([]ParticipantHealth{
		{Team: "Red", EndHealth: 18},  // took 2 damage -> Blue dealt 2
		{Team: "Blue", EndHealth: 10}, // took 10 damage -> Red dealt 10
	})
	if err != nil {
		t.Fatalf("ComputeGameOutcome: %v", err)
	}
	if outcome.Winner != "Red" || outcome.Draw {
		t.Fatalf("outcome = %+v, want Red to win on damage differential", outcome)
	}
	if outcome.DamageDealt["Red"] != 10 || outcome.DamageDealt["Blue"] != 2 {
		t.Errorf("DamageDealt = %+v, want Red=10 Blue=2", outcome.DamageDealt)
	}
}

func TestComputeGameOutcome_ExactTieIsADraw(t *testing.T) {
	outcome, err := ComputeGameOutcome([]ParticipantHealth{
		{Team: "Red", EndHealth: 10},
		{Team: "Blue", EndHealth: 10},
	})
	if err != nil {
		t.Fatalf("ComputeGameOutcome: %v", err)
	}
	if !outcome.Draw {
		t.Fatalf("outcome = %+v, want a draw on an exact tie", outcome)
	}
}

func TestComputeGameOutcome_MutualKillFallsBackToDamageDifferential(t *testing.T) {
	// Both teams have a dead member, but Red's surviving member also took
	// more damage than Blue's surviving member - damage differential
	// breaks the tie rather than treating a mutual kill as automatically
	// a draw.
	outcome, err := ComputeGameOutcome([]ParticipantHealth{
		{Team: "Red", EndHealth: 0},
		{Team: "Red", EndHealth: 15},
		{Team: "Blue", EndHealth: 0},
		{Team: "Blue", EndHealth: 18},
	})
	if err != nil {
		t.Fatalf("ComputeGameOutcome: %v", err)
	}
	// Red's total health lost = 20 + 5 = 25 (Blue dealt 25).
	// Blue's total health lost = 20 + 2 = 22 (Red dealt 22).
	if outcome.Winner != "Blue" || outcome.Draw {
		t.Fatalf("outcome = %+v, want Blue to win the damage-differential tiebreak", outcome)
	}
}

func TestComputeGameOutcome_AggregatesMultiMemberTeams(t *testing.T) {
	outcome, err := ComputeGameOutcome([]ParticipantHealth{
		{Team: "Red", EndHealth: 20},
		{Team: "Red", EndHealth: 20},
		{Team: "Blue", EndHealth: 10},
	})
	if err != nil {
		t.Fatalf("ComputeGameOutcome: %v", err)
	}
	// Red took no damage at all (Blue dealt 0); Blue took 10 (Red dealt 10).
	if outcome.DamageDealt["Red"] != 10 {
		t.Errorf("DamageDealt[Red] = %v, want 10", outcome.DamageDealt["Red"])
	}
	if outcome.DamageDealt["Blue"] != 0 {
		t.Errorf("DamageDealt[Blue] = %v, want 0", outcome.DamageDealt["Blue"])
	}
	if outcome.Winner != "Red" || outcome.Draw {
		t.Fatalf("outcome = %+v, want Red to win", outcome)
	}
}

func TestComputeGameOutcome_RejectsEmptyInput(t *testing.T) {
	if _, err := ComputeGameOutcome(nil); err == nil {
		t.Fatal("want an error for no readings at all")
	}
}

func TestComputeGameOutcome_RejectsSingleTeam(t *testing.T) {
	_, err := ComputeGameOutcome([]ParticipantHealth{{Team: "Red", EndHealth: 20}})
	if err == nil {
		t.Fatal("want an error when fewer than two teams are represented")
	}
}

func TestHealthFromObservation(t *testing.T) {
	values := make([]float32, 10)
	values[combatHealthObservationIndex] = 14.5
	obs := rl.Observation{Values: values}
	if got := HealthFromObservation(obs); got != 14.5 {
		t.Errorf("HealthFromObservation = %v, want 14.5", got)
	}
}

func TestHealthFromObservation_TooShortReturnsZero(t *testing.T) {
	obs := rl.Observation{Values: []float32{1, 2}}
	if got := HealthFromObservation(obs); got != 0 {
		t.Errorf("HealthFromObservation = %v, want 0 for a too-short observation", got)
	}
}
