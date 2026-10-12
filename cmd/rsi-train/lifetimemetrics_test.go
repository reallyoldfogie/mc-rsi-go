package main

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadLifetimeMetricsStateAbsentFileFallsBack(t *testing.T) {
	_, ok := loadLifetimeMetricsState(t.TempDir())
	assert.False(t, ok, "no file ever written must behave exactly like no epoch/round has completed yet")
}

func TestLoadLifetimeMetricsStateRejectsInvalidContent(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(lifetimeMetricsStateFile(dir), []byte("not json"), 0o644))
	_, ok := loadLifetimeMetricsState(dir)
	assert.False(t, ok)
}

func TestSaveAndLoadLifetimeMetricsStateRoundTrip(t *testing.T) {
	dir := t.TempDir()
	state := lifetimeMetricsState{
		RoundDurationSeconds: 12345.6,
		EpochDurationSeconds: 789.1,
		Epochs:               42,
		TaskEpisodes:         map[string]int{"mine": 10, "craft": 5},
	}
	require.NoError(t, saveLifetimeMetricsState(dir, state))

	got, ok := loadLifetimeMetricsState(dir)
	require.True(t, ok)
	assert.Equal(t, state, got)
}

func TestSaveLifetimeMetricsStateOverwritesPreviousSave(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, saveLifetimeMetricsState(dir, lifetimeMetricsState{Epochs: 1}))
	require.NoError(t, saveLifetimeMetricsState(dir, lifetimeMetricsState{Epochs: 2, TaskEpisodes: map[string]int{"goto": 3}}))

	got, ok := loadLifetimeMetricsState(dir)
	require.True(t, ok)
	assert.Equal(t, 2, got.Epochs)
	assert.Equal(t, map[string]int{"goto": 3}, got.TaskEpisodes)
}
