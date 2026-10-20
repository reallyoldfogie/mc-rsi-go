package main

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadPersistedTrainingStateMissingIsOptional(t *testing.T) {
	_, ok := loadPersistedTrainingState(t.TempDir())
	assert.False(t, ok)
}

func TestLoadPersistedTrainingStateAcceptsLegacyWeightsOnlySidecarShape(t *testing.T) {
	dir := t.TempDir()
	data, err := json.Marshal(struct {
		SchemaVersion    int `json:"schema_version"`
		ParentGeneration int `json:"parent_generation"`
		Epoch            int `json:"epoch"`
	}{SchemaVersion: 1, ParentGeneration: 4, Epoch: 9})
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(dir+"/training-state.json", data, 0o644))

	state, ok := loadPersistedTrainingState(dir)
	require.True(t, ok)
	assert.Equal(t, 4, state.ParentGeneration)
	assert.Equal(t, 9, state.Epoch)
	assert.Nil(t, state.Optimizer)
	assert.Nil(t, state.RNG)
	assert.Nil(t, state.Environments)
}

func TestPersistedTrainingStateRoundTrip(t *testing.T) {
	dir := t.TempDir()
	want := persistedTrainingState{
		SchemaVersion:    1,
		ParentGeneration: 3,
		Epoch:            12,
		Optimizer: &actorcritic.AdamState{
			StepCount: 7,
			Moment1:   [][]float32{{1, 2}},
			Moment2:   [][]float32{{3, 4}},
		},
		RNG:          []byte{1, 2, 3},
		Environments: []environmentState{{Available: false}, {Available: true, Data: json.RawMessage(`{"step":5}`)}},
	}
	require.NoError(t, savePersistedTrainingState(dir, want))

	got, ok := loadPersistedTrainingState(dir)
	require.True(t, ok)
	assert.Equal(t, want, got)
}

func TestLoadPersistedTrainingStateRejectsUnknownSchema(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(dir+"/training-state.json", []byte(`{"schema_version":99}`), 0o644))
	_, ok := loadPersistedTrainingState(dir)
	assert.False(t, ok)
}
