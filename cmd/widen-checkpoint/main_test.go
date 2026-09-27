package main

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/lineage"
)

const testEnvironmentID = "mc-agent-rlenv:actions=4:obs=17"

func seedCheckpointDir(t *testing.T, generation int, hiddenSize int) string {
	t.Helper()
	dir := t.TempDir()
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 17, hiddenSize, 4)
	require.NoError(t, lineage.Save(dir, params, lineage.Record{
		Generation:           generation,
		ParentGeneration:     generation - 1,
		Provenance:           lineage.ProvenanceLeapfrog,
		SeededFromGeneration: -1,
		EnvironmentID:        testEnvironmentID,
		Metadata:             checkpoint.Metadata{Epoch: 12, BestReturn: 5, TotalUpdates: 300},
	}))
	return dir
}

func TestRunWidensTheLatestGenerationIntoAFreshOutDir(t *testing.T) {
	dir := seedCheckpointDir(t, 3, 6)
	outDir := filepath.Join(t.TempDir(), "widened")

	err := run([]string{
		"-checkpoint-dir", dir,
		"-out-dir", outDir,
		"-new-hidden-size", "10",
		"-seed", "7",
	})
	require.NoError(t, err)

	wide, rec, err := lineage.Load(outDir, 0, testEnvironmentID)
	require.NoError(t, err)
	assert.Equal(t, 10, wide.HiddenSize())
	assert.Equal(t, 17, wide.InputSize())
	assert.Equal(t, 4, wide.OutputSize())
	assert.Equal(t, lineage.ProvenanceWidened, rec.Provenance)
	assert.Equal(t, -1, rec.ParentGeneration)
	assert.Equal(t, checkpoint.Metadata{}, rec.Metadata, "a widened generation starts with fresh progress counters, not the source's")

	_, err = os.Stat(filepath.Join(outDir, "source.txt"))
	assert.NoError(t, err, "a human-readable note of where this came from must be written")
}

func TestRunRejectsAnOutDirThatAlreadyHasAGeneration(t *testing.T) {
	dir := seedCheckpointDir(t, 3, 6)
	outDir := t.TempDir()
	outParams := actorcritic.NewParams(rand.New(rand.NewPCG(9, 9)), 17, 6, 4)
	require.NoError(t, lineage.Save(outDir, outParams, lineage.Record{
		Generation: 0, ParentGeneration: -1, Provenance: lineage.ProvenanceLeapfrog,
		SeededFromGeneration: -1, EnvironmentID: testEnvironmentID,
	}))

	err := run([]string{
		"-checkpoint-dir", dir,
		"-out-dir", outDir,
		"-new-hidden-size", "10",
	})
	assert.ErrorContains(t, err, "already has a generation record")
}

func TestRunRejectsANewHiddenSizeNotGreaterThanTheCurrentOne(t *testing.T) {
	dir := seedCheckpointDir(t, 0, 8)

	err := run([]string{
		"-checkpoint-dir", dir,
		"-out-dir", filepath.Join(t.TempDir(), "widened"),
		"-new-hidden-size", "8",
	})
	assert.ErrorContains(t, err, "must be greater than")
}

func TestParseFlagsRequiresCheckpointDirOutDirAndNewHiddenSize(t *testing.T) {
	_, err := parseFlags([]string{"-out-dir", "x", "-new-hidden-size", "10"})
	assert.ErrorContains(t, err, "-checkpoint-dir")

	_, err = parseFlags([]string{"-checkpoint-dir", "x", "-new-hidden-size", "10"})
	assert.ErrorContains(t, err, "-out-dir")

	_, err = parseFlags([]string{"-checkpoint-dir", "x", "-out-dir", "y"})
	assert.ErrorContains(t, err, "-new-hidden-size")
}

// TestSameSeedReproducesTheSameWidenedNetwork lets a user reproduce a
// widening they liked, or compare noise-std values on an otherwise
// identical draw of which units got cloned.
func TestSameSeedReproducesTheSameWidenedNetwork(t *testing.T) {
	dir := seedCheckpointDir(t, 0, 6)

	out1 := filepath.Join(t.TempDir(), "a")
	require.NoError(t, run([]string{"-checkpoint-dir", dir, "-out-dir", out1, "-new-hidden-size", "9", "-seed", "42"}))
	out2 := filepath.Join(t.TempDir(), "b")
	require.NoError(t, run([]string{"-checkpoint-dir", dir, "-out-dir", out2, "-new-hidden-size", "9", "-seed", "42"}))

	w1, _, err := lineage.Load(out1, 0, testEnvironmentID)
	require.NoError(t, err)
	w2, _, err := lineage.Load(out2, 0, testEnvironmentID)
	require.NoError(t, err)
	assert.Equal(t, w1.W0.Data, w2.W0.Data)
	assert.Equal(t, w1.W1.Data, w2.W1.Data)
}
