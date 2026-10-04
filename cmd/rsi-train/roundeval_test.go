package main

import (
	"os"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/leapfrog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadRoundEvalStateAbsentFileFallsBack(t *testing.T) {
	_, ok := loadRoundEvalState(t.TempDir())
	assert.False(t, ok, "no file ever written must behave exactly like no round has completed yet")
}

func TestLoadRoundEvalStateRejectsInvalidContent(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(roundEvalStateFile(dir), []byte("not json"), 0o644))
	_, ok := loadRoundEvalState(dir)
	assert.False(t, ok)
}

func TestSaveAndLoadRoundEvalStateRoundTrip(t *testing.T) {
	dir := t.TempDir()
	result := leapfrog.Result[*actorcritic.Params]{
		StudentWon:    true, // deliberately not persisted/restored - see roundEvalState's doc comment
		TeacherReward: 28.5,
		StudentReward: 29.9,
		EvalEpisodes:  100,
		TeacherStd:    14.3,
		StudentStd:    13.7,
		MeanDiff:      1.4,
		DiffStdErr:    0.6,
	}
	require.NoError(t, saveRoundEvalState(dir, result))

	state, ok := loadRoundEvalState(dir)
	require.True(t, ok)
	assert.Equal(t, roundEvalState{
		TeacherReward: 28.5,
		StudentReward: 29.9,
		EvalEpisodes:  100,
		TeacherStd:    14.3,
		StudentStd:    13.7,
		MeanDiff:      1.4,
		DiffStdErr:    0.6,
	}, state)
}

func TestSaveRoundEvalStateOverwritesPreviousRound(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, saveRoundEvalState(dir, leapfrog.Result[*actorcritic.Params]{TeacherReward: 1, StudentReward: 2}))
	require.NoError(t, saveRoundEvalState(dir, leapfrog.Result[*actorcritic.Params]{TeacherReward: 3, StudentReward: 4}))

	state, ok := loadRoundEvalState(dir)
	require.True(t, ok)
	assert.Equal(t, float32(3), state.TeacherReward)
	assert.Equal(t, float32(4), state.StudentReward)
}
