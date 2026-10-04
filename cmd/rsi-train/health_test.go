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
	p := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 17, 128, 4, 2)
	h := measureNetworkHealth(p)
	require.Len(t, h.LayerActive, 2)
	assert.Greater(t, h.LayerActive[0], 0.9)
	assert.Greater(t, h.LayerActive[1], 0.5)
	assert.Greater(t, h.LogitStd, 1e-3)
	assert.False(t, h.dead())
	// A fresh, randomly-initialized network's action distribution should be
	// close to uniform (near ln(4) nats for 4 actions), not near-zero
	// (deterministic) - and its many independently-initialized hidden units
	// should not look suspiciously redundant of each other.
	assert.Greater(t, h.PolicyEntropy, 1.0, "a fresh network's policy should be close to uniform, not already sharp")
	require.Len(t, h.LayerCorrelation, 2)
	assert.Less(t, h.LayerCorrelation[0], 0.5, "independently initialized units shouldn't already look heavily redundant")
	assert.Less(t, h.LayerCorrelation[1], 0.5)
}

// The failure found live: every second-layer unit permanently off, leaving a
// policy that is just its output biases.
func TestNetworkWithNoActiveSecondLayerIsDead(t *testing.T) {
	p := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 17, 128, 4, 2)
	for i := range p.Hidden[1].B.Data {
		p.Hidden[1].B.Data[i] = -1000
	}
	h := measureNetworkHealth(p)
	require.Equal(t, 0.0, h.LayerActive[1])
	assert.InDelta(t, 0, h.LogitStd, 1e-9, "a policy with no active hidden unit is constant across states")
	assert.True(t, h.dead())
}

func TestPartlyDeadNetworkIsNotFatal(t *testing.T) {
	p := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 17, 128, 4, 2)
	for i := 0; i < 100; i++ {
		p.Hidden[1].B.Data[i] = -1000
	}
	h := measureNetworkHealth(p)
	assert.Less(t, h.LayerActive[1], 0.3)
	assert.False(t, h.dead(), "28 live units still respond to the input")
}

// The measurements of the network that ran dead for a week (one unit left
// firing, a logit std of 0.002) must count as dead, not as "not exactly zero".
func TestTheLiveDeadNetworkCountsAsDead(t *testing.T) {
	assert.True(t, networkHealth{LayerActive: []float64{0.67, 0.0078}, LogitStd: 0.0019}.dead())
	assert.False(t, networkHealth{LayerActive: []float64{1, 0.99}, LogitStd: 0.2}.dead(), "a fresh network")
}

// TestMeasureNetworkHealthWorksWithOneHiddenLayer and
// TestMeasureNetworkHealthWorksWithFourHiddenLayers pin the fix for the bug
// this type used to have: measureNetworkHealth used to panic on any network
// whose depth wasn't exactly 2 (docs/plans/20 Part A made other depths
// buildable, but this file wasn't updated for it at the time). Neither
// should panic now, and LayerActive/LayerCorrelation must be index-aligned
// with however many hidden layers the network actually has.
func TestMeasureNetworkHealthWorksWithOneHiddenLayer(t *testing.T) {
	p := actorcritic.NewParams(rand.New(rand.NewPCG(21, 22)), 17, 64, 4, 1)
	h := measureNetworkHealth(p)
	require.Len(t, h.LayerActive, 1)
	require.Len(t, h.LayerCorrelation, 1)
	assert.Greater(t, h.LayerActive[0], 0.5)
	assert.False(t, h.dead())
}

func TestMeasureNetworkHealthWorksWithFourHiddenLayers(t *testing.T) {
	p := actorcritic.NewParams(rand.New(rand.NewPCG(23, 24)), 17, 64, 4, 4)
	h := measureNetworkHealth(p)
	require.Len(t, h.LayerActive, 4)
	require.Len(t, h.LayerCorrelation, 4)
	for i := range h.LayerActive {
		assert.Greater(t, h.LayerActive[i], 0.0, "layer %d", i)
	}
	assert.False(t, h.dead())
}

// TestDeadChecksTheLastHiddenLayerRegardlessOfDepth confirms dead()'s
// "last layer feeds the heads directly" reasoning holds for a depth other
// than 2: killing only the last of 3 hidden layers must still be detected,
// even though the two earlier layers remain fully healthy.
func TestDeadChecksTheLastHiddenLayerRegardlessOfDepth(t *testing.T) {
	p := actorcritic.NewParams(rand.New(rand.NewPCG(25, 26)), 17, 64, 4, 3)
	for i := range p.Hidden[2].B.Data {
		p.Hidden[2].B.Data[i] = -1000
	}
	h := measureNetworkHealth(p)
	require.Equal(t, 0.0, h.LayerActive[2])
	assert.Greater(t, h.LayerActive[0], 0.5, "the first hidden layer is untouched and should still look healthy")
	assert.Greater(t, h.LayerActive[1], 0.5, "the second hidden layer is untouched and should still look healthy")
	assert.True(t, h.dead(), "a dead last layer must mark the whole network dead regardless of how many earlier layers are fine")
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
