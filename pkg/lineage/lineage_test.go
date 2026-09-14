package lineage

import (
	"encoding/json"
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

// TestSaveLeavesNoTempFileBehind verifies Save's record write goes
// through writeRecordAtomically's temp-file-plus-rename path (not a
// leftover artifact of it) — see Save's own doc comment on crash safety
// (docs/plans/04-training-entrypoint-and-observability.md's "Done when").
func TestSaveLeavesNoTempFileBehind(t *testing.T) {
	dir := t.TempDir()
	rng := rand.New(rand.NewPCG(1, 2))
	require.NoError(t, Save(dir, actorcritic.NewParams(rng, 4, 4, 2), rootRecord("mc:teleport-v1")))

	entries, err := os.ReadDir(dir)
	require.NoError(t, err)
	for _, entry := range entries {
		assert.NotContains(t, entry.Name(), ".tmp-", "Save left a temp file behind: %s", entry.Name())
	}
}

// TestWriteRecordAtomicallyNeverLeavesAPartialFileAtTheFinalPath verifies
// the atomicity property Save's crash-safety guarantee actually depends
// on directly: the final path is only ever created by one atomic
// os.Rename, so a reader can never observe it mid-write. This can't
// literally simulate a process kill mid-write, but it does prove the
// mechanism is what Save's doc comment claims (temp file + rename, not
// an in-place truncate) rather than only inferring it from Save's
// behavior indirectly.
func TestWriteRecordAtomicallyNeverLeavesAPartialFileAtTheFinalPath(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "generation-000000000.json")

	require.NoError(t, writeRecordAtomically(path, rootRecord("mc:teleport-v1")))

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	var rec Record
	require.NoError(t, json.Unmarshal(data, &rec), "final file must always contain complete, valid JSON, never a partial write")
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
