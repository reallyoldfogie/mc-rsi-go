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

func seedCheckpointDir(t *testing.T, generation int) string {
	t.Helper()
	dir := t.TempDir()
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 17, 6, 4, 2)
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

func seedRecurrentCheckpointDir(t *testing.T, generation int) string {
	t.Helper()
	dir := t.TempDir()
	params := actorcritic.NewRecurrentParams(rand.New(rand.NewPCG(1, 2)), 17, 6, 4, 2)
	rec := lineage.Record{
		Generation:           generation,
		ParentGeneration:     generation - 1,
		Provenance:           lineage.ProvenanceLeapfrog,
		SeededFromGeneration: -1,
		EnvironmentID:        testEnvironmentID,
		Metadata:             checkpoint.Metadata{Epoch: 12, BestReturn: 5, TotalUpdates: 300},
		Recurrent:            true,
	}
	require.NoError(t, lineage.Save(dir, params, rec, lineage.RecurrentCodec))
	return dir
}

func TestRunExpandsTheLatestGenerationIntoAFreshOutDir(t *testing.T) {
	dir := seedCheckpointDir(t, 3)
	outDir := filepath.Join(t.TempDir(), "expanded")

	err := run([]string{
		"-checkpoint-dir", dir,
		"-out-dir", outDir,
		"-new-actions", "9",
		"-new-obs", "30",
	})
	require.NoError(t, err)

	expanded, rec, err := lineage.Load(outDir, 0, "mc-agent-rlenv:actions=9:obs=30", lineage.PlainCodec)
	require.NoError(t, err)
	assert.Equal(t, 30, expanded.InputSize())
	assert.Equal(t, 9, expanded.OutputSize())
	assert.Equal(t, lineage.ProvenanceExpanded, rec.Provenance)
	assert.Equal(t, -1, rec.ParentGeneration)
	assert.False(t, rec.Recurrent)
	assert.Equal(t, checkpoint.Metadata{}, rec.Metadata, "an expanded generation starts with fresh progress counters, not the source's")

	_, err = os.Stat(filepath.Join(outDir, "source.txt"))
	assert.NoError(t, err, "a human-readable note of where this came from must be written")
}

func TestRunExpandsARecurrentGeneration(t *testing.T) {
	dir := seedRecurrentCheckpointDir(t, 2)
	outDir := filepath.Join(t.TempDir(), "expanded")

	err := run([]string{
		"-checkpoint-dir", dir,
		"-out-dir", outDir,
		"-new-actions", "9",
		"-new-obs", "30",
	})
	require.NoError(t, err)

	expanded, rec, err := lineage.Load(outDir, 0, "mc-agent-rlenv:actions=9:obs=30", lineage.RecurrentCodec)
	require.NoError(t, err)
	assert.Equal(t, 30, expanded.PolicyTrunk.InputSize())
	assert.Equal(t, 9, expanded.PolicyTrunk.OutputSize())
	assert.True(t, rec.Recurrent)
}

func TestRunRejectsAnOutDirThatAlreadyHasAGeneration(t *testing.T) {
	dir := seedCheckpointDir(t, 3)
	outDir := t.TempDir()
	outParams := actorcritic.NewParams(rand.New(rand.NewPCG(9, 9)), 17, 6, 4, 2)
	require.NoError(t, lineage.Save(outDir, outParams, lineage.Record{
		Generation: 0, ParentGeneration: -1, Provenance: lineage.ProvenanceLeapfrog,
		SeededFromGeneration: -1, EnvironmentID: testEnvironmentID,
	}, lineage.PlainCodec))

	err := run([]string{
		"-checkpoint-dir", dir,
		"-out-dir", outDir,
		"-new-actions", "9",
		"-new-obs", "30",
	})
	assert.ErrorContains(t, err, "already has a generation record")
}

func TestRunRejectsShrinking(t *testing.T) {
	dir := seedCheckpointDir(t, 0)

	err := run([]string{
		"-checkpoint-dir", dir,
		"-out-dir", filepath.Join(t.TempDir(), "expanded"),
		"-new-actions", "2",
		"-new-obs", "30",
	})
	assert.ErrorContains(t, err, "shrinking")
}

func TestParseFlagsRequiresCheckpointDirOutDirNewActionsAndNewObs(t *testing.T) {
	_, err := parseFlags([]string{"-out-dir", "x", "-new-actions", "9", "-new-obs", "30"})
	assert.ErrorContains(t, err, "-checkpoint-dir")

	_, err = parseFlags([]string{"-checkpoint-dir", "x", "-new-actions", "9", "-new-obs", "30"})
	assert.ErrorContains(t, err, "-out-dir")

	_, err = parseFlags([]string{"-checkpoint-dir", "x", "-out-dir", "y", "-new-obs", "30"})
	assert.ErrorContains(t, err, "-new-actions")

	_, err = parseFlags([]string{"-checkpoint-dir", "x", "-out-dir", "y", "-new-actions", "9"})
	assert.ErrorContains(t, err, "-new-obs")
}
