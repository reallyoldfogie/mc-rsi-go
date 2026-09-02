package lineage

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func rootRecord(environmentID string) Record {
	return Record{
		Generation:           0,
		ParentGeneration:     -1,
		Provenance:           ProvenanceLeapfrog,
		SeededFromGeneration: -1,
		CreatedAt:            time.Now(),
		EnvironmentID:        environmentID,
		Metadata:             checkpoint.Metadata{Epoch: 3, BestReturn: 1.5, TotalUpdates: 40},
	}
}

func TestValidateAcceptsWellFormedRecords(t *testing.T) {
	tests := []struct {
		name string
		rec  Record
	}{
		{"root generation", rootRecord("mc:teleport-v1")},
		{"leapfrog child", Record{
			Generation: 1, ParentGeneration: 0, Provenance: ProvenanceLeapfrog,
			SeededFromGeneration: -1, EnvironmentID: "mc:teleport-v1",
		}},
		{"seeded from live", Record{
			Generation: 2, ParentGeneration: 1, Provenance: ProvenanceSeededLive,
			SeededFromGeneration: 1, EnvironmentID: "mc:teleport-v1",
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.NoError(t, tc.rec.Validate())
		})
	}
}

func TestValidateRejectsMalformedRecords(t *testing.T) {
	tests := []struct {
		name string
		rec  Record
	}{
		{"negative generation", Record{Generation: -1, ParentGeneration: -1, Provenance: ProvenanceLeapfrog, SeededFromGeneration: -1, EnvironmentID: "e"}},
		{"parent below -1", Record{Generation: 0, ParentGeneration: -2, Provenance: ProvenanceLeapfrog, SeededFromGeneration: -1, EnvironmentID: "e"}},
		{"non-root with no parent", Record{Generation: 1, ParentGeneration: -1, Provenance: ProvenanceLeapfrog, SeededFromGeneration: -1, EnvironmentID: "e"}},
		{"leapfrog with seeded-from set", Record{Generation: 1, ParentGeneration: 0, Provenance: ProvenanceLeapfrog, SeededFromGeneration: 0, EnvironmentID: "e"}},
		{"seeded-live with no seeded-from", Record{Generation: 1, ParentGeneration: 0, Provenance: ProvenanceSeededLive, SeededFromGeneration: -1, EnvironmentID: "e"}},
		{"unknown provenance", Record{Generation: 0, ParentGeneration: -1, Provenance: "mystery", SeededFromGeneration: -1, EnvironmentID: "e"}},
		{"empty environment ID", Record{Generation: 0, ParentGeneration: -1, Provenance: ProvenanceLeapfrog, SeededFromGeneration: -1, EnvironmentID: ""}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Error(t, tc.rec.Validate())
		})
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	rng := rand.New(rand.NewPCG(1, 2))
	params := actorcritic.NewParams(rng, 12, 8, 5)
	rec := rootRecord("mc:teleport-v1")

	require.NoError(t, Save(dir, params, rec))

	loadedParams, loadedRec, err := Load(dir, 0, "mc:teleport-v1")
	require.NoError(t, err)

	assert.Equal(t, params.W0.Data, loadedParams.W0.Data)
	assert.Equal(t, params.Wpi.Data, loadedParams.Wpi.Data)
	assert.Equal(t, rec.Generation, loadedRec.Generation)
	assert.Equal(t, rec.Provenance, loadedRec.Provenance)
	assert.Equal(t, rec.Metadata, loadedRec.Metadata)
	assert.WithinDuration(t, rec.CreatedAt, loadedRec.CreatedAt, time.Second)
}

func TestSaveRejectsInvalidRecord(t *testing.T) {
	dir := t.TempDir()
	rng := rand.New(rand.NewPCG(1, 2))
	params := actorcritic.NewParams(rng, 4, 4, 2)

	invalid := rootRecord("mc:teleport-v1")
	invalid.Generation = 1 // root generation must be 0 when ParentGeneration is -1

	assert.Error(t, Save(dir, params, invalid))
}

func TestLoadRejectsGenerationMismatch(t *testing.T) {
	dir := t.TempDir()
	rng := rand.New(rand.NewPCG(1, 2))
	params := actorcritic.NewParams(rng, 4, 4, 2)
	require.NoError(t, Save(dir, params, rootRecord("mc:teleport-v1")))

	_, _, err := Load(dir, 1, "mc:teleport-v1")
	assert.Error(t, err)
}

func TestLoadRejectsEnvironmentMismatch(t *testing.T) {
	dir := t.TempDir()
	rng := rand.New(rand.NewPCG(1, 2))
	params := actorcritic.NewParams(rng, 4, 4, 2)
	require.NoError(t, Save(dir, params, rootRecord("mc:teleport-v1")))

	_, _, err := Load(dir, 0, "mc:other-env")
	assert.Error(t, err)
}

func TestLatestReturnsHighestGeneration(t *testing.T) {
	dir := t.TempDir()
	rng := rand.New(rand.NewPCG(1, 2))

	for gen := 0; gen <= 11; gen++ {
		params := actorcritic.NewParams(rng, 4, 4, 2)
		rec := Record{
			Generation:           gen,
			ParentGeneration:     gen - 1,
			Provenance:           ProvenanceLeapfrog,
			SeededFromGeneration: -1,
			EnvironmentID:        "mc:teleport-v1",
		}
		require.NoError(t, Save(dir, params, rec))
	}

	latest, err := Latest(dir)
	require.NoError(t, err)
	assert.Equal(t, 11, latest.Generation)
}

func TestLatestFailsOnEmptyDir(t *testing.T) {
	dir := t.TempDir()
	_, err := Latest(dir)
	assert.Error(t, err)
}

func TestLatestIgnoresUnrelatedFiles(t *testing.T) {
	dir := t.TempDir()
	rng := rand.New(rand.NewPCG(1, 2))
	require.NoError(t, Save(dir, actorcritic.NewParams(rng, 4, 4, 2), rootRecord("mc:teleport-v1")))

	require.NoError(t, os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("not a checkpoint"), 0o644))

	latest, err := Latest(dir)
	require.NoError(t, err)
	assert.Equal(t, 0, latest.Generation)
}
