package main

import (
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
