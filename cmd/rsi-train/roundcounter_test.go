package main

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadRoundCounterStateAbsentFileFallsBack(t *testing.T) {
	_, ok := loadRoundCounterState(t.TempDir())
	assert.False(t, ok, "no file ever written must behave exactly like no round has completed yet")
}

func TestLoadRoundCounterStateRejectsInvalidContent(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(roundCounterStateFile(dir), []byte("not json"), 0o644))
	_, ok := loadRoundCounterState(dir)
	assert.False(t, ok)
}

func TestSaveAndLoadRoundCounterStateRoundTrip(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, saveRoundCounterState(dir, roundCounterState{Round: 5, Won: 2, Lost: 3}))

	state, ok := loadRoundCounterState(dir)
	require.True(t, ok)
	assert.Equal(t, roundCounterState{Round: 5, Won: 2, Lost: 3}, state)
}

func TestSaveRoundCounterStateOverwritesPreviousRound(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, saveRoundCounterState(dir, roundCounterState{Round: 1, Won: 0, Lost: 1}))
	require.NoError(t, saveRoundCounterState(dir, roundCounterState{Round: 2, Won: 1, Lost: 1}))

	state, ok := loadRoundCounterState(dir)
	require.True(t, ok)
	assert.Equal(t, roundCounterState{Round: 2, Won: 1, Lost: 1}, state)
}
