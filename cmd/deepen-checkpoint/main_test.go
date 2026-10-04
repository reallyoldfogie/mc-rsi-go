package main

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/lineage"
)

const testEnvironmentID = "mc-agent-rlenv:actions=4:obs=17"

func seedCheckpointDir(t *testing.T, generation int, numHiddenLayers int) string {
	t.Helper()
	dir := t.TempDir()
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 17, 6, 4, numHiddenLayers)
	require.NoError(t, lineage.Save(dir, params, lineage.Record{
		Generation:           generation,
		ParentGeneration:     generation - 1,
		Provenance:           lineage.ProvenanceLeapfrog,
		SeededFromGeneration: -1,
		EnvironmentID:        testEnvironmentID,
		Metadata:             checkpoint.Metadata{Epoch: 12, BestReturn: 5, TotalUpdates: 300},
	}, lineage.PlainCodec))
	return dir
}

func TestRunDeepensTheLatestGenerationIntoAFreshOutDir(t *testing.T) {
	dir := seedCheckpointDir(t, 3, 2)
	outDir := filepath.Join(t.TempDir(), "deepened")

	err := run([]string{
		"-checkpoint-dir", dir,
		"-out-dir", outDir,
		"-at-depth", "1",
	})
	require.NoError(t, err)

	deeper, rec, err := lineage.Load(outDir, 0, testEnvironmentID, lineage.PlainCodec)
	require.NoError(t, err)
	assert.Equal(t, 3, deeper.NumHiddenLayers())
	assert.Equal(t, 6, deeper.HiddenSize())
	assert.Equal(t, 17, deeper.InputSize())
	assert.Equal(t, 4, deeper.OutputSize())
	assert.Equal(t, lineage.ProvenanceDeepened, rec.Provenance)
	assert.Equal(t, -1, rec.ParentGeneration)
	assert.Equal(t, checkpoint.Metadata{}, rec.Metadata, "a deepened generation starts with fresh progress counters, not the source's")

	_, err = os.Stat(filepath.Join(outDir, "source.txt"))
	assert.NoError(t, err, "a human-readable note of where this came from must be written")
}

// TestRunPreservesTheFunctionExactly confirms the CLI wiring doesn't lose
// Net2DeeperNet's own central guarantee: running a probe observation
// through the source and deepened checkpoints (loaded back from disk, not
// the in-memory Params) must produce the same policy and value output.
func TestRunPreservesTheFunctionExactly(t *testing.T) {
	dir := seedCheckpointDir(t, 0, 2)
	outDir := filepath.Join(t.TempDir(), "deepened")

	require.NoError(t, run([]string{
		"-checkpoint-dir", dir,
		"-out-dir", outDir,
		"-at-depth", "2",
		"-verify-probes", "20",
		"-seed", "3",
	}))

	orig, _, err := lineage.Load(dir, 0, testEnvironmentID, lineage.PlainCodec)
	require.NoError(t, err)
	deeper, _, err := lineage.Load(outDir, 0, testEnvironmentID, lineage.PlainCodec)
	require.NoError(t, err)

	rng := rand.New(rand.NewPCG(99, 100))
	for trial := 0; trial < 10; trial++ {
		values := make([]float32, orig.InputSize())
		for i := range values {
			values[i] = float32(rng.NormFloat64())
		}
		obs := rl.Observation{Values: values}
		origOut, err := forward(orig, obs)
		require.NoError(t, err)
		deeperOut, err := forward(deeper, obs)
		require.NoError(t, err)

		for i := range origOut.policy {
			assert.InDelta(t, origOut.policy[i], deeperOut.policy[i], 1e-6, "policy[%d] on trial %d", i, trial)
		}
		assert.InDelta(t, origOut.value, deeperOut.value, 1e-6, "value on trial %d", trial)
	}
}

func TestRunRejectsAnOutDirThatAlreadyHasAGeneration(t *testing.T) {
	dir := seedCheckpointDir(t, 3, 2)
	outDir := t.TempDir()
	outParams := actorcritic.NewParams(rand.New(rand.NewPCG(9, 9)), 17, 6, 4, 2)
	require.NoError(t, lineage.Save(outDir, outParams, lineage.Record{
		Generation: 0, ParentGeneration: -1, Provenance: lineage.ProvenanceLeapfrog,
		SeededFromGeneration: -1, EnvironmentID: testEnvironmentID,
	}, lineage.PlainCodec))

	err := run([]string{
		"-checkpoint-dir", dir,
		"-out-dir", outDir,
		"-at-depth", "1",
	})
	assert.ErrorContains(t, err, "already has a generation record")
}

func TestRunRejectsAnOutOfRangeDepth(t *testing.T) {
	dir := seedCheckpointDir(t, 0, 2)

	err := run([]string{
		"-checkpoint-dir", dir,
		"-out-dir", filepath.Join(t.TempDir(), "deepened"),
		"-at-depth", "3",
	})
	assert.ErrorContains(t, err, "must be between")
}

func TestParseFlagsRequiresCheckpointDirOutDirAndAtDepth(t *testing.T) {
	_, err := parseFlags([]string{"-out-dir", "x", "-at-depth", "1"})
	assert.ErrorContains(t, err, "-checkpoint-dir")

	_, err = parseFlags([]string{"-checkpoint-dir", "x", "-at-depth", "1"})
	assert.ErrorContains(t, err, "-out-dir")

	_, err = parseFlags([]string{"-checkpoint-dir", "x", "-out-dir", "y"})
	assert.ErrorContains(t, err, "-at-depth")

	_, err = parseFlags([]string{"-checkpoint-dir", "x", "-out-dir", "y", "-at-depth", "0"})
	assert.ErrorContains(t, err, "-at-depth")
}
