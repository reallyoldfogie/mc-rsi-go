package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/ppo"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testRollout builds a minimal, otherwise-arbitrary *ppo.RecurrentRollout
// for round-trip tests - its values exist only to be compared back,
// never trained on.
func testRollout(seed float32) *ppo.RecurrentRollout {
	return &ppo.RecurrentRollout{
		Rollout: &ppo.Rollout{
			Episode: &rl.Episode{
				Transitions: []rl.Transition{
					{Observation: rl.Observation{Values: []float32{seed, seed + 1}}, Action: 2, Reward: seed, Done: true},
				},
			},
			LogProbs: []float32{-0.5 * seed},
			Values:   []float32{seed * 2},
		},
		HPrevs:        [][]float32{{seed}},
		CPrevs:        [][]float32{{seed}},
		ValueHPrevs:   [][]float32{{seed}},
		ValueCPrevs:   [][]float32{{seed}},
		PolicyOutputs: [][]float32{{seed, seed}},
		ValueOutputs:  []float32{seed},
	}
}

// TestSaveLoadEpochRollout_RoundTrips covers the save/load mechanics
// Config.OnRolloutCollected/ResumeRollouts rely on: a trajectory saved
// via saveEpochRollout must come back byte-for-byte equivalent (same
// transitions, logprobs, LSTM state) via loadEpochRollouts, for exactly
// the slot index it was saved under - every other slot in the returned,
// RolloutSize-length slice must stay nil.
func TestSaveLoadEpochRollout_RoundTrips(t *testing.T) {
	dir := t.TempDir()
	rollout := testRollout(3)

	require.NoError(t, saveEpochRollout(dir, 7, 12, 2, rollout))

	loaded := loadEpochRollouts(dir, 7, 12, 5)
	require.Len(t, loaded, 5)
	for i, r := range loaded {
		if i == 2 {
			require.NotNil(t, r)
			continue
		}
		assert.Nil(t, r, "slot %d was never saved - it must stay nil", i)
	}

	got := loaded[2]
	assert.Equal(t, rollout.Episode.Transitions, got.Episode.Transitions)
	assert.Equal(t, rollout.LogProbs, got.LogProbs)
	assert.Equal(t, rollout.Values, got.Values)
	assert.Equal(t, rollout.HPrevs, got.HPrevs)
	assert.Equal(t, rollout.CPrevs, got.CPrevs)
	assert.Equal(t, rollout.ValueHPrevs, got.ValueHPrevs)
	assert.Equal(t, rollout.ValueCPrevs, got.ValueCPrevs)
	assert.Equal(t, rollout.PolicyOutputs, got.PolicyOutputs)
	assert.Equal(t, rollout.ValueOutputs, got.ValueOutputs)
}

// TestLoadEpochRollouts_NoCacheReturnsAllNil covers the common case:
// nothing has ever crashed mid-epoch, so there's no cache directory at
// all - loadEpochRollouts must return a plain RolloutSize-length slice
// of nils (meaning "collect everything"), not an error.
func TestLoadEpochRollouts_NoCacheReturnsAllNil(t *testing.T) {
	dir := t.TempDir()
	loaded := loadEpochRollouts(dir, 1, 0, 4)
	require.Len(t, loaded, 4)
	for _, r := range loaded {
		assert.Nil(t, r)
	}
}

// TestLoadEpochRollouts_IgnoresCorruptSlot covers a damaged cache file
// (e.g. a crash mid-write that something other than saveEpochRollout's
// own atomic rename produced): it must be skipped, not treated as a
// fatal error, since every other, intact slot is still worth recovering.
func TestLoadEpochRollouts_IgnoresCorruptSlot(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, saveEpochRollout(dir, 1, 0, 0, testRollout(1)))
	require.NoError(t, saveEpochRollout(dir, 1, 0, 1, testRollout(2)))
	require.NoError(t, os.WriteFile(epochRolloutSlotPath(epochRolloutDir(dir, 1, 0), 1), []byte("{not valid json"), 0o644))

	loaded := loadEpochRollouts(dir, 1, 0, 3)
	require.Len(t, loaded, 3)
	assert.NotNil(t, loaded[0], "slot 0 was saved correctly and must still load")
	assert.Nil(t, loaded[1], "slot 1 is corrupt and must come back nil, not error")
	assert.Nil(t, loaded[2])
}

// TestClearEpochRollouts_RemovesOnlyThatGenerationEpoch covers
// clearEpochRollouts' own precision: wired from OnEpoch to run every
// time any epoch finishes, it must remove exactly that epoch's cache and
// leave every other generation/epoch's cache (in-progress or otherwise)
// untouched.
func TestClearEpochRollouts_RemovesOnlyThatGenerationEpoch(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, saveEpochRollout(dir, 3, 5, 0, testRollout(1)))
	require.NoError(t, saveEpochRollout(dir, 3, 6, 0, testRollout(2)))

	clearEpochRollouts(dir, 3, 5)

	_, err := os.Stat(epochRolloutDir(dir, 3, 5))
	assert.True(t, os.IsNotExist(err), "epoch 5's cache must be gone")
	_, err = os.Stat(epochRolloutDir(dir, 3, 6))
	assert.NoError(t, err, "epoch 6's cache must be untouched")
}

// TestPruneStaleEpochRollouts_KeepsOnlyTheOneBeingResumedInto covers
// pruneStaleEpochRollouts: called at startup before resuming, it must
// remove every cache directory except the exact (generation, epoch) the
// run is about to resume into - guarding against a checkpoint-interval
// gap (resume lands on an earlier epoch than the one that actually
// crashed) leaving stale caches around forever across many restarts.
func TestPruneStaleEpochRollouts_KeepsOnlyTheOneBeingResumedInto(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, saveEpochRollout(dir, 3, 5, 0, testRollout(1)))
	require.NoError(t, saveEpochRollout(dir, 3, 6, 0, testRollout(2)))
	require.NoError(t, saveEpochRollout(dir, 2, 9, 0, testRollout(3)))

	pruneStaleEpochRollouts(dir, 3, 6)

	_, err := os.Stat(epochRolloutDir(dir, 3, 5))
	assert.True(t, os.IsNotExist(err), "a different epoch's cache must be pruned")
	_, err = os.Stat(epochRolloutDir(dir, 2, 9))
	assert.True(t, os.IsNotExist(err), "a different generation's cache must be pruned")
	_, err = os.Stat(epochRolloutDir(dir, 3, 6))
	assert.NoError(t, err, "the generation/epoch being resumed into must survive")
}

// TestPruneStaleEpochRollouts_NoDirYet covers the common case (nothing
// has ever crashed mid-epoch, so baseDir doesn't exist at all) - this
// must be a no-op, not an error.
func TestPruneStaleEpochRollouts_NoDirYet(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "epoch-rollouts")
	pruneStaleEpochRollouts(dir, 1, 0)
}
