package pvp

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/gridworldenv"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// smallSessionTestSettings mirrors cRL-go/pkg/ppo's own smallTestSettings
// fixture - small enough to train real epochs in a unit test without
// meaningfully slowing the suite down.
func smallSessionTestSettings() config.Settings {
	return config.Settings{
		Epochs: 3, RolloutSize: 4, EpisodeLen: 5, Gamma: 0.99, LearningRate: 0.01,
		GridSize: 4, HiddenSize: 8, Seed: 1, Workers: 2,
		ClipEpsilon: 0.2, EntropyCoef: 0.01, ValueCoef: 0.5, GAELambda: 0.95,
		PPOEpochs: 2, MinibatchSize: 8,
	}
}

func newTestEnv(t *testing.T) rl.Environment {
	t.Helper()
	env, err := gridworldenv.New(4)
	if err != nil {
		t.Fatalf("gridworldenv.New: %v", err)
	}
	return gridworldenv.NewAdapter(env)
}

func testParticipant() Participant {
	return Participant{ID: "p1", Managed: true, HiddenLayers: 2, HiddenSize: 8}
}

func TestNewParticipantSession_InitializesFreshParamsWhenNoCheckpointExists(t *testing.T) {
	dir := t.TempDir()
	session, err := NewParticipantSession("p1", testParticipant(), filepath.Join(dir, "p1.json"), time.Minute, smallSessionTestSettings(), newTestEnv(t))
	if err != nil {
		t.Fatalf("NewParticipantSession: %v", err)
	}
	if session.plainTrainer == nil {
		t.Fatal("want a plain trainer for a non-recurrent participant")
	}
}

func TestParticipantSession_RunEpochProducesStats(t *testing.T) {
	dir := t.TempDir()
	session, err := NewParticipantSession("p1", testParticipant(), filepath.Join(dir, "p1.json"), time.Minute, smallSessionTestSettings(), newTestEnv(t))
	if err != nil {
		t.Fatalf("NewParticipantSession: %v", err)
	}
	stats, err := session.RunEpoch(context.Background(), 0)
	if err != nil {
		t.Fatalf("RunEpoch: %v", err)
	}
	if stats.Epoch != 0 {
		t.Errorf("stats.Epoch = %d, want 0", stats.Epoch)
	}
	if stats.SampleCount <= 0 {
		t.Errorf("stats.SampleCount = %d, want > 0", stats.SampleCount)
	}
}

func TestParticipantSession_CheckpointRoundTrip(t *testing.T) {
	dir := t.TempDir()
	checkpointPath := filepath.Join(dir, "p1.json")
	participant := testParticipant()

	session, err := NewParticipantSession("p1", participant, checkpointPath, time.Minute, smallSessionTestSettings(), newTestEnv(t))
	if err != nil {
		t.Fatalf("NewParticipantSession: %v", err)
	}
	stats, err := session.RunEpoch(context.Background(), 0)
	if err != nil {
		t.Fatalf("RunEpoch: %v", err)
	}
	if err := session.Checkpoint(0, stats); err != nil {
		t.Fatalf("Checkpoint: %v", err)
	}

	// A second session constructed against the same path/participant must
	// load the saved checkpoint rather than erroring or silently
	// re-initializing - verified by reusing the exact same environmentID
	// (derived identically from the same Participant/env shape) and
	// confirming the load succeeds.
	reloaded, err := NewParticipantSession("p1", participant, checkpointPath, time.Minute, smallSessionTestSettings(), newTestEnv(t))
	if err != nil {
		t.Fatalf("NewParticipantSession (reload): %v", err)
	}
	if reloaded.EnvironmentID != session.EnvironmentID {
		t.Errorf("EnvironmentID = %q, want %q", reloaded.EnvironmentID, session.EnvironmentID)
	}
}

func TestParticipantSession_MismatchedCheckpointIsAHardError(t *testing.T) {
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

	// A different architecture (hidden_layers) produces a different
	// EnvironmentID, so loading the same file must fail rather than
	// silently accept a shape-incompatible checkpoint.
	differentArchitecture := testParticipant()
	differentArchitecture.HiddenLayers = 3
	_, err = NewParticipantSession("p1", differentArchitecture, checkpointPath, time.Minute, smallSessionTestSettings(), newTestEnv(t))
	if err == nil {
		t.Fatal("want an error when the checkpoint's environment ID does not match")
	}
	if !strings.Contains(err.Error(), "environment") {
		t.Errorf("error = %v, want it to mention the environment ID mismatch", err)
	}
}

func TestParticipantSession_RecurrentParticipantUsesRecurrentTrainer(t *testing.T) {
	dir := t.TempDir()
	participant := Participant{ID: "p1", Managed: true, HiddenLayers: 2, HiddenSize: 8, Recurrent: true, RecurrentChunkLen: 4}
	session, err := NewParticipantSession("p1", participant, filepath.Join(dir, "p1.json"), time.Minute, smallSessionTestSettings(), newTestEnv(t))
	if err != nil {
		t.Fatalf("NewParticipantSession: %v", err)
	}
	if session.recurrentTrainer == nil {
		t.Fatal("want a recurrent trainer for a recurrent participant")
	}
	if _, err := session.RunEpoch(context.Background(), 0); err != nil {
		t.Fatalf("RunEpoch: %v", err)
	}
}

func TestParticipantSession_TrainStopsOnContextDeadlineAndCheckpoints(t *testing.T) {
	dir := t.TempDir()
	checkpointPath := filepath.Join(dir, "p1.json")
	session, err := NewParticipantSession("p1", testParticipant(), checkpointPath, 24*time.Hour, smallSessionTestSettings(), newTestEnv(t))
	if err != nil {
		t.Fatalf("NewParticipantSession: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	err = session.Train(ctx)
	if err != context.DeadlineExceeded {
		t.Fatalf("Train error = %v, want context.DeadlineExceeded", err)
	}

	if _, _, err := actorcritic.LoadFile(checkpointPath, session.EnvironmentID); err != nil {
		t.Fatalf("expected a final checkpoint to have been saved on deadline exceeded: %v", err)
	}
}
