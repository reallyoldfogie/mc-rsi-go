package main

import (
	"math/rand/v2"
	"testing"
)

func TestMaskedProbabilitiesNilMask(t *testing.T) {
	probs := []float32{0.2, 0.5, 0.3}
	got, err := maskedProbabilities(probs, nil)
	if err != nil {
		t.Fatalf("maskedProbabilities: %v", err)
	}
	for i := range probs {
		if got[i] != probs[i] {
			t.Errorf("got[%d] = %v, want %v (unchanged)", i, got[i], probs[i])
		}
	}
}

func TestMaskedProbabilitiesAllAllowed(t *testing.T) {
	probs := []float32{0.2, 0.5, 0.3}
	got, err := maskedProbabilities(probs, []bool{true, true, true})
	if err != nil {
		t.Fatalf("maskedProbabilities: %v", err)
	}
	for i := range probs {
		if got[i] != probs[i] {
			t.Errorf("got[%d] = %v, want %v (unchanged)", i, got[i], probs[i])
		}
	}
}

func TestMaskedProbabilitiesRenormalizes(t *testing.T) {
	probs := []float32{0.2, 0.5, 0.3}
	got, err := maskedProbabilities(probs, []bool{true, false, true})
	if err != nil {
		t.Fatalf("maskedProbabilities: %v", err)
	}
	if got[1] != 0 {
		t.Errorf("got[1] = %v, want 0 (disallowed)", got[1])
	}
	const tolerance = 1e-6
	if diff := abs32(got[0] - 0.4); diff > tolerance { // 0.2 / (0.2+0.3)
		t.Errorf("got[0] = %v, want 0.4", got[0])
	}
	if diff := abs32(got[2] - 0.6); diff > tolerance { // 0.3 / (0.2+0.3)
		t.Errorf("got[2] = %v, want 0.6", got[2])
	}
}

func TestMaskedProbabilitiesNoLegalAction(t *testing.T) {
	_, err := maskedProbabilities([]float32{0.2, 0.5, 0.3}, []bool{false, false, false})
	if err == nil {
		t.Error("maskedProbabilities with an all-false mask = nil error, want one")
	}
}

func TestMaskedProbabilitiesLengthMismatch(t *testing.T) {
	_, err := maskedProbabilities([]float32{0.2, 0.5, 0.3}, []bool{true, false})
	if err == nil {
		t.Error("maskedProbabilities with a mismatched mask length = nil error, want one")
	}
}

func TestMaskedProbabilitiesUnderflowFallsBackToUniform(t *testing.T) {
	// Every allowed action's probability is exactly 0 (underflowed),
	// but the mask still allows two of them — mirrors the live-observed
	// case pkg/reinforce's own equivalent handles by falling back to a
	// uniform draw rather than erroring, since softmax's output is
	// analytically always >0.
	got, err := maskedProbabilities([]float32{0, 0.9, 0}, []bool{true, false, true})
	if err != nil {
		t.Fatalf("maskedProbabilities: %v", err)
	}
	if got[0] != 0.5 || got[2] != 0.5 {
		t.Errorf("got = %v, want uniform [0.5, _, 0.5] over the two allowed actions", got)
	}
}

func TestGreedyActionBreaksTiesLow(t *testing.T) {
	if got := greedyAction([]float32{0.3, 0.3, 0.4}); got != 2 {
		t.Errorf("greedyAction = %d, want 2", got)
	}
	if got := greedyAction([]float32{0.4, 0.4, 0.2}); got != 0 {
		t.Errorf("greedyAction = %d, want 0 (tie breaks toward lowest index)", got)
	}
}

func TestSampleActionRespectsDistribution(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	probs := []float32{0, 1, 0} // deterministic: only action 1 has any mass
	for i := 0; i < 20; i++ {
		if got := sampleAction(probs, rng); got != 1 {
			t.Fatalf("sampleAction = %d, want 1 (only nonzero-probability action)", got)
		}
	}
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
