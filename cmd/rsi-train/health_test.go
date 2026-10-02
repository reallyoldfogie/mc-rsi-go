package main

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFreshNetworkIsHealthy(t *testing.T) {
	p := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 17, 128, 4)
	h := measureNetworkHealth(p)
	assert.Greater(t, h.Layer1Active, 0.9)
	assert.Greater(t, h.Layer2Active, 0.5)
	assert.Greater(t, h.LogitStd, 1e-3)
	assert.False(t, h.dead())
	// A fresh, randomly-initialized network's action distribution should be
	// close to uniform (near ln(4) nats for 4 actions), not near-zero
	// (deterministic) - and its many independently-initialized hidden units
	// should not look suspiciously redundant of each other.
	assert.Greater(t, h.PolicyEntropy, 1.0, "a fresh network's policy should be close to uniform, not already sharp")
	assert.Less(t, h.Layer1Correlation, 0.5, "independently initialized units shouldn't already look heavily redundant")
	assert.Less(t, h.Layer2Correlation, 0.5)
}

// The failure found live: every second-layer unit permanently off, leaving a
// policy that is just its output biases.
func TestNetworkWithNoActiveSecondLayerIsDead(t *testing.T) {
	p := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 17, 128, 4)
	for i := range p.B1.Data {
		p.B1.Data[i] = -1000
	}
	h := measureNetworkHealth(p)
	require.Equal(t, 0.0, h.Layer2Active)
	assert.InDelta(t, 0, h.LogitStd, 1e-9, "a policy with no active hidden unit is constant across states")
	assert.True(t, h.dead())
}

func TestPartlyDeadNetworkIsNotFatal(t *testing.T) {
	p := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 17, 128, 4)
	for i := 0; i < 100; i++ {
		p.B1.Data[i] = -1000
	}
	h := measureNetworkHealth(p)
	assert.Less(t, h.Layer2Active, 0.3)
	assert.False(t, h.dead(), "28 live units still respond to the input")
}

// The measurements of the network that ran dead for a week (one unit left
// firing, a logit std of 0.002) must count as dead, not as "not exactly zero".
func TestTheLiveDeadNetworkCountsAsDead(t *testing.T) {
	assert.True(t, networkHealth{Layer1Active: 0.67, Layer2Active: 0.0078, LogitStd: 0.0019}.dead())
	assert.False(t, networkHealth{Layer1Active: 1, Layer2Active: 0.99, LogitStd: 0.2}.dead(), "a fresh network")
}

func TestSoftmaxEntropyIsMaximalForUniformLogits(t *testing.T) {
	entropy := softmaxEntropy([]float32{0, 0, 0, 0})
	assert.InDelta(t, math.Log(4), entropy, 1e-6, "equal logits -> a uniform distribution -> ln(n) nats")
}

func TestSoftmaxEntropyIsNearZeroForAStronglyPeakedLogit(t *testing.T) {
	entropy := softmaxEntropy([]float32{50, 0, 0, 0})
	assert.Less(t, entropy, 1e-6, "one logit far ahead of the rest -> an effectively deterministic policy")
}

func TestMeanAbsCorrelationIsOneForIdenticalUnits(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	corr := meanAbsCorrelation([][]float64{a, a, a})
	assert.InDelta(t, 1.0, corr, 1e-9, "every unit tracking every other exactly is maximal redundancy")
}

func TestMeanAbsCorrelationIsOneForPerfectlyAnticorrelatedUnits(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	b := []float64{5, 4, 3, 2, 1}
	corr := meanAbsCorrelation([][]float64{a, b})
	assert.InDelta(t, 1.0, corr, 1e-9, "anticorrelation is still redundancy (one determines the other) - hence the absolute value")
}

func TestMeanAbsCorrelationExcludesConstantUnits(t *testing.T) {
	varying := []float64{1, 2, 3, 4, 5}
	constant := []float64{7, 7, 7, 7, 7}
	// Only one unit actually varies, so there is no defined pair to average -
	// must be 0, not NaN or a misleadingly confident number, from a division
	// by a zero standard deviation.
	corr := meanAbsCorrelation([][]float64{varying, constant})
	assert.Equal(t, 0.0, corr)
}
