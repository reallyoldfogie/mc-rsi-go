package pvp

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestInferenceSession_PlaysGameAgainstRealCheckpoint(t *testing.T) {
	dir := t.TempDir()
	checkpointPath := filepath.Join(dir, "p1.json")
	participant := testParticipant()

	// Produce a real checkpoint via the training path first - the same
	// shape loadOrInitParams writes, so loading it back through
	// InferenceSession exercises the genuine environment-ID contract
	// both paths share.
	trainingSession, err := NewParticipantSession("p1", participant, checkpointPath, time.Minute, smallSessionTestSettings(), newTestEnv(t))
	if err != nil {
		t.Fatalf("NewParticipantSession: %v", err)
	}
	stats, err := trainingSession.RunEpoch(context.Background(), 0)
	if err != nil {
		t.Fatalf("RunEpoch: %v", err)
	}
	if err := trainingSession.Checkpoint(0, stats); err != nil {
		t.Fatalf("Checkpoint: %v", err)
	}

	barrier := NewMatchBarrier(1, MatchResetFunc(func(ctx context.Context) error { return nil }))
	wrapped := NewMatchEnvironment(newTestEnv(t), barrier)

	inferenceSession, err := NewInferenceSession("p1", participant, checkpointPath, true, wrapped, 1)
	if err != nil {
		t.Fatalf("NewInferenceSession: %v", err)
	}
	if inferenceSession.plainParams == nil {
		t.Fatal("want a plain params for a non-recurrent participant")
	}

	health, err := inferenceSession.PlayGame(context.Background(), 5)
	if err != nil {
		t.Fatalf("PlayGame: %v", err)
	}
	// gridworldenv has no concept of "health" at all - HealthFromObservation
	// reading index 5 of an observation this toy environment never
	// populates with a combat-specific meaning is still a defined,
	// deterministic float32 (part of its own one-hot position encoding),
	// not a crash or NaN - this just confirms PlayGame completed a real
	// episode and LastObservation got populated, not that it's a
	// meaningful health value for this particular toy environment.
	_ = health
}

func TestInferenceSession_MismatchedCheckpointIsAHardError(t *testing.T) {
	dir := t.TempDir()
	checkpointPath := filepath.Join(dir, "p1.json")

	saver, err := NewParticipantSession("p1", testParticipant(), checkpointPath, time.Minute, smallSessionTestSettings(), newTestEnv(t))
	if err != nil {
		t.Fatalf("NewParticipantSession: %v", err)
	}
	stats, err := saver.RunEpoch(context.Background(), 0)
	if err != nil {
		t.Fatalf("RunEpoch: %v", err)
	}
	if err := saver.Checkpoint(0, stats); err != nil {
		t.Fatalf("Checkpoint: %v", err)
	}

	differentArchitecture := testParticipant()
	differentArchitecture.HiddenLayers = 3
	barrier := NewMatchBarrier(1, MatchResetFunc(func(ctx context.Context) error { return nil }))
	wrapped := NewMatchEnvironment(newTestEnv(t), barrier)

	_, err = NewInferenceSession("p1", differentArchitecture, checkpointPath, true, wrapped, 1)
	if err == nil {
		t.Fatal("want an error when the checkpoint's environment ID does not match")
	}
	if !strings.Contains(err.Error(), "environment") {
		t.Errorf("error = %v, want it to mention the environment ID mismatch", err)
	}
}

func TestInferenceSession_RejectsMissingCheckpoint(t *testing.T) {
	dir := t.TempDir()
	barrier := NewMatchBarrier(1, MatchResetFunc(func(ctx context.Context) error { return nil }))
	wrapped := NewMatchEnvironment(newTestEnv(t), barrier)

	_, err := NewInferenceSession("p1", testParticipant(), filepath.Join(dir, "does-not-exist.json"), true, wrapped, 1)
	if err == nil {
		t.Fatal("want an error when no checkpoint exists for a competition entry")
	}
}

func TestInferenceSession_RecurrentParticipantUsesRecurrentParams(t *testing.T) {
	dir := t.TempDir()
	checkpointPath := filepath.Join(dir, "p1.json")
	participant := Participant{ID: "p1", Managed: true, HiddenLayers: 2, HiddenSize: 8, Recurrent: true, RecurrentChunkLen: 4}

	trainingSession, err := NewParticipantSession("p1", participant, checkpointPath, time.Minute, smallSessionTestSettings(), newTestEnv(t))
	if err != nil {
		t.Fatalf("NewParticipantSession: %v", err)
	}
	stats, err := trainingSession.RunEpoch(context.Background(), 0)
	if err != nil {
		t.Fatalf("RunEpoch: %v", err)
	}
	if err := trainingSession.Checkpoint(0, stats); err != nil {
		t.Fatalf("Checkpoint: %v", err)
	}

	barrier := NewMatchBarrier(1, MatchResetFunc(func(ctx context.Context) error { return nil }))
	wrapped := NewMatchEnvironment(newTestEnv(t), barrier)
	inferenceSession, err := NewInferenceSession("p1", participant, checkpointPath, true, wrapped, 1)
	if err != nil {
		t.Fatalf("NewInferenceSession: %v", err)
	}
	if inferenceSession.recurrentParams == nil {
		t.Fatal("want recurrent params for a recurrent participant")
	}
	if _, err := inferenceSession.PlayGame(context.Background(), 5); err != nil {
		t.Fatalf("PlayGame: %v", err)
	}
}
