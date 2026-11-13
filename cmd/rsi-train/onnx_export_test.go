package main

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/onnxinfer"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/lineage"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/onnxmanifest"
)

const onnxExportTestEnvironmentID = "mc-agent-rlenv:actions=4:obs=17:norm=1"

// TestSavePlainGenerationWithONNXExport verifies savePlainGeneration's
// two jobs: it must do exactly what lineage.Save does (so resuming a run
// isn't affected by the flag at all), and, when exportONNX is true, also
// write a loadable onnx/generation-<N>/model.onnx+manifest.json pair
// alongside it — see docs/plans/23-ONNX-support.md.
func TestSavePlainGenerationWithONNXExport(t *testing.T) {
	dir := t.TempDir()
	rng := rand.New(rand.NewPCG(1, 2))
	params := actorcritic.NewParams(rng, 17, 8, 4, 2)
	rec := lineage.Record{
		Generation:           3,
		ParentGeneration:     2,
		Provenance:           lineage.ProvenanceLeapfrog,
		SeededFromGeneration: -1,
		EnvironmentID:        onnxExportTestEnvironmentID,
	}

	require.NoError(t, savePlainGeneration(dir, params, rec, true))

	// The native checkpoint must exist exactly as lineage.Save would
	// leave it.
	loaded, loadedRec, err := lineage.Load(dir, 3, onnxExportTestEnvironmentID, lineage.PlainCodec)
	require.NoError(t, err)
	assert.Equal(t, 3, loadedRec.Generation)
	assert.Equal(t, params.InputSize(), loaded.InputSize())

	onnxDir := filepath.Join(dir, "onnx", "generation-000000003")
	manifest, onnxData, err := onnxmanifest.Load(onnxDir)
	require.NoError(t, err)
	assert.False(t, manifest.Recurrent)
	assert.Equal(t, onnxExportTestEnvironmentID, manifest.EnvironmentID)
	assert.True(t, manifest.NormalizeObservation)
	assert.Equal(t, 3, manifest.Generation)
	assert.Equal(t, 17, manifest.InputSize)
	assert.Equal(t, 8, manifest.HiddenSize)
	assert.Equal(t, 2, manifest.NumHiddenLayers)
	assert.Equal(t, 4, manifest.OutputSize)

	model, err := onnxinfer.Load(onnxData)
	require.NoError(t, err)
	outs, err := model.Run(map[string]*onnxinfer.Tensor{
		"observation": {Dims: []int64{1, 17}, Data: make([]float32, 17)},
	})
	require.NoError(t, err)
	assert.Len(t, outs["policy"].Data, 4)
}

// TestSaveRecurrentGenerationWithONNXExport mirrors the plain case for
// *actorcritic.RecurrentParams.
func TestSaveRecurrentGenerationWithONNXExport(t *testing.T) {
	dir := t.TempDir()
	rng := rand.New(rand.NewPCG(3, 4))
	params := actorcritic.NewRecurrentParams(rng, 17, 8, 4, 2)
	rec := lineage.Record{
		Generation:           1,
		ParentGeneration:     0,
		Provenance:           lineage.ProvenanceLeapfrog,
		SeededFromGeneration: -1,
		EnvironmentID:        onnxExportTestEnvironmentID,
		Recurrent:            true,
	}

	require.NoError(t, saveRecurrentGeneration(dir, params, rec, true))

	onnxDir := filepath.Join(dir, "onnx", "generation-000000001")
	manifest, onnxData, err := onnxmanifest.Load(onnxDir)
	require.NoError(t, err)
	assert.True(t, manifest.Recurrent)

	model, err := onnxinfer.Load(onnxData)
	require.NoError(t, err)
	zero := func() *onnxinfer.Tensor { return &onnxinfer.Tensor{Dims: []int64{1, 8}, Data: make([]float32, 8)} }
	outs, err := model.Run(map[string]*onnxinfer.Tensor{
		"observation": {Dims: []int64{1, 17}, Data: make([]float32, 17)},
		"policy_h_in": zero(), "policy_c_in": zero(),
		"value_h_in": zero(), "value_c_in": zero(),
	})
	require.NoError(t, err)
	assert.Len(t, outs["policy"].Data, 4)
}

// TestSavePlainGenerationWithoutONNXExportUnchanged confirms the
// opt-in's other half: exportONNX=false (the flag's default) must leave
// no onnx/ directory at all, preserving today's behavior exactly.
func TestSavePlainGenerationWithoutONNXExportUnchanged(t *testing.T) {
	dir := t.TempDir()
	rng := rand.New(rand.NewPCG(5, 6))
	params := actorcritic.NewParams(rng, 17, 8, 4, 2)
	rec := lineage.Record{
		Generation:           0,
		ParentGeneration:     -1,
		Provenance:           lineage.ProvenanceLeapfrog,
		SeededFromGeneration: -1,
		EnvironmentID:        onnxExportTestEnvironmentID,
	}

	require.NoError(t, savePlainGeneration(dir, params, rec, false))

	_, err := os.Stat(filepath.Join(dir, "onnx"))
	assert.True(t, os.IsNotExist(err), "onnx/ must not be created when exportONNX is false")

	_, _, err = lineage.Load(dir, 0, onnxExportTestEnvironmentID, lineage.PlainCodec)
	require.NoError(t, err)
}
