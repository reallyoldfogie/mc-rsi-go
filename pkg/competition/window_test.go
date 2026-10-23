package competition

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTrainingWindowUsesOneAbsoluteDeadline(t *testing.T) {
	start := time.Date(2026, 10, 19, 12, 0, 0, 0, time.UTC)
	window, err := NewTrainingWindow(start, Duration(10*time.Minute))
	require.NoError(t, err)
	assert.Equal(t, start.Add(10*time.Minute), window.Deadline)
	assert.Equal(t, 5*time.Minute, window.Remaining(start.Add(5*time.Minute)))
	assert.Equal(t, 10*time.Minute, window.Remaining(start.Add(-time.Minute)))
	assert.Zero(t, window.Remaining(start.Add(11*time.Minute)))
}

func TestTrainingWindowContextExpiresAtSharedDeadline(t *testing.T) {
	start := time.Now()
	window, err := NewTrainingWindow(start, Duration(20*time.Millisecond))
	require.NoError(t, err)
	ctx, cancel, err := window.Context(context.Background())
	require.NoError(t, err)
	defer cancel()
	select {
	case <-ctx.Done():
		assert.ErrorIs(t, ctx.Err(), context.DeadlineExceeded)
	case <-time.After(time.Second):
		t.Fatal("training window did not expire")
	}
}

func TestTrainingWindowRejectsInvalidInputs(t *testing.T) {
	_, err := NewTrainingWindow(time.Time{}, Duration(time.Second))
	assert.Error(t, err)
	_, err = NewTrainingWindow(time.Now(), 0)
	assert.Error(t, err)

	_, _, err = (TrainingWindow{}).Context(context.Background())
	assert.Error(t, err)
}
