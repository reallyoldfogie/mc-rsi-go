package main

import (
	"context"
	"io"
	"math/rand/v2"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type metricsTestEnvironment struct {
	stepResult rl.StepResult
	onReset    func()
}

func (e *metricsTestEnvironment) Reset(context.Context) (rl.Observation, error) {
	if e.onReset != nil {
		e.onReset()
	}
	return rl.Observation{Values: []float32{0}}, nil
}

func (e *metricsTestEnvironment) Step(context.Context, rl.Action) (rl.StepResult, error) {
	return e.stepResult, nil
}

func (e *metricsTestEnvironment) ObservationSize() int { return 1 }
func (e *metricsTestEnvironment) ActionSpace() int     { return 1 }

func TestInstrumentedEnvironmentRecordsSuccessfulEpisode(t *testing.T) {
	task := "metrics-test-craft"
	env := &instrumentedEnvironment{
		env:  &metricsTestEnvironment{stepResult: rl.StepResult{Reward: 10, Done: true}},
		task: task,
	}
	_, err := env.Reset(context.Background())
	require.NoError(t, err)
	_, err = env.Step(context.Background(), 0)
	require.NoError(t, err)

	body := gatherMetrics(t)
	require.Contains(t, body, `rsi_episodes_started_total{task="`+task+`"}`)
	require.Contains(t, body, `rsi_episodes_total{outcome="success",task="`+task+`"}`)
	require.Contains(t, body, `rsi_episode_steps_total{task="`+task+`"}`)
}

func TestInstrumentedEnvironmentUsesTaskSelectedDuringReset(t *testing.T) {
	selectedTask := "goto"
	env := &instrumentedEnvironment{
		env: &metricsTestEnvironment{
			stepResult: rl.StepResult{Reward: 10, Done: true},
			onReset:    func() { selectedTask = "mine" },
		},
		task:           "curriculum",
		taskForEpisode: func() string { return selectedTask },
	}
	_, err := env.Reset(context.Background())
	require.NoError(t, err)
	_, err = env.Step(context.Background(), 0)
	require.NoError(t, err)

	body := gatherMetrics(t)
	require.Contains(t, body, `rsi_episodes_started_total{task="mine"}`)
	require.Contains(t, body, `rsi_episodes_total{outcome="success",task="mine"}`)
	require.NotContains(t, body, `rsi_episodes_started_total{task="curriculum"}`)
}

func TestInstrumentedEnvironmentRecordsResetFailure(t *testing.T) {
	task := "metrics-test-mine"
	env := &instrumentedEnvironment{env: &resetErrorEnvironment{}, task: task}
	_, err := env.Reset(context.Background())
	require.Error(t, err)

	body := gatherMetrics(t)
	require.Contains(t, body, `rsi_reset_failures_total{task="`+task+`"}`)
}

// TestInstrumentedEnvironmentResetRetriesOnFailureThenSucceeds verifies the
// resetRetries loop added after a single bad task draw (e.g. an
// unreachable goto target) was found live to crash the entire training
// process: Reset should retry a failing wrapped Reset rather than
// propagating the first error, succeeding (with no error and normal
// episode-started bookkeeping) once the wrapped environment does.
func TestInstrumentedEnvironmentResetRetriesOnFailureThenSucceeds(t *testing.T) {
	task := "metrics-test-goto"
	failuresRemaining := 3
	env := &instrumentedEnvironment{
		env:  &flakyResetEnvironment{failuresRemaining: &failuresRemaining},
		task: task,
	}
	_, err := env.Reset(context.Background())
	require.NoError(t, err)
	require.Equal(t, 0, failuresRemaining, "expected exactly 3 failing attempts before the 4th succeeded")

	body := gatherMetrics(t)
	require.Contains(t, body, `rsi_episodes_started_total{task="`+task+`"}`)
}

// TestInstrumentedEnvironmentResetGivesUpAfterExhaustingRetries verifies
// Reset still propagates an error (rather than retrying forever) once a
// wrapped environment fails resetRetries+1 times in a row - retries are a
// resilience improvement against one unlucky task draw, not a promise the
// process can never legitimately fail to reset (e.g. if the server itself
// is genuinely unreachable).
func TestInstrumentedEnvironmentResetGivesUpAfterExhaustingRetries(t *testing.T) {
	task := "metrics-test-mine-exhausted"
	env := &instrumentedEnvironment{env: &resetErrorEnvironment{}, task: task}
	_, err := env.Reset(context.Background())
	require.Error(t, err)
}

func TestSaveFinalCheckpointPreservesMetadata(t *testing.T) {
	dir := t.TempDir()
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 1, 2, 1)
	metadata := checkpoint.Metadata{Epoch: 11, BestReturn: 4.5, TotalUpdates: 37}

	require.NoError(t, saveFinalCheckpoint(dir, params, "test-env", metadata, 7))
	_, loaded, err := actorcritic.LoadFile(filepath.Join(dir, "final.json"), "test-env")
	require.NoError(t, err)
	require.Equal(t, metadata, loaded)

	state, err := loadResumeState(dir)
	require.NoError(t, err)
	assert.Equal(t, 7, state.ParentGeneration, "saveFinalCheckpoint must record which teacher generation this checkpoint was trained against")
}

// TestLoadResumableStudent_AcceptsAMatchingFinalCheckpoint covers the
// straightforward case: a final.json left by a graceful shutdown, whose
// resume-state sidecar names the same parent generation the caller is
// about to train against, must be returned.
func TestLoadResumableStudent_AcceptsAMatchingFinalCheckpoint(t *testing.T) {
	dir := t.TempDir()
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 1, 2, 1)
	metadata := checkpoint.Metadata{Epoch: 18, BestReturn: 12.19, TotalUpdates: 900}
	require.NoError(t, saveFinalCheckpoint(dir, params, "test-env", metadata, 7))

	resumed, epoch, resumedMetadata := loadResumableStudent(dir, "test-env", 7)
	require.NotNil(t, resumed, "a matching-generation final.json must be resumable")
	assert.Equal(t, 18, epoch)
	assert.Equal(t, metadata, resumedMetadata)
}

// TestLoadResumableStudent_PrefersTheMostRecentlyModifiedFile covers two
// related cases with one mechanism: cmd/rsi-train's own "Checkpoint
// interval" gap, where final.json (every graceful shutdown) can record a
// later epoch than the last periodic interim-epoch-*.json save (every
// -checkpoint-interval epochs) did — and the live-found staleness bug
// where a *much older* attempt's higher-epoch leftover file must not
// beat a more recent, lower-epoch one just because its epoch number is
// numerically bigger. Both are decided by file modification time, not
// by comparing Metadata.Epoch — see loadResumableStudent's own doc
// comment. os.Chtimes pins each file's mtime explicitly rather than
// relying on real wall-clock ordering between two fast successive writes
// in a test.
func TestLoadResumableStudent_PrefersTheMostRecentlyModifiedFile(t *testing.T) {
	dir := t.TempDir()
	old := time.Now().Add(-24 * time.Hour)
	recent := time.Now()

	staleHighEpochParams := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 1, 2, 1)
	require.NoError(t, saveInterimCheckpoint(dir, staleHighEpochParams, "test-env", 49, checkpoint.Metadata{Epoch: 49, BestReturn: 17.16, TotalUpdates: 4812}, 7))
	stalePath := filepath.Join(dir, "interim-epoch-000000049.json")
	require.NoError(t, os.Chtimes(stalePath, old, old))

	recentLowEpochParams := actorcritic.NewParams(rand.New(rand.NewPCG(3, 4)), 1, 2, 1)
	require.NoError(t, saveFinalCheckpoint(dir, recentLowEpochParams, "test-env", checkpoint.Metadata{Epoch: 18, BestReturn: 12.19, TotalUpdates: 900}, 7))
	require.NoError(t, os.Chtimes(filepath.Join(dir, "final.json"), recent, recent))

	resumed, epoch, _ := loadResumableStudent(dir, "test-env", 7)
	require.NotNil(t, resumed)
	assert.Equal(t, 18, epoch, "the more recently saved checkpoint must win, even though the stale one recorded a numerically higher epoch")
}

// TestLoadResumableStudent_RejectsAParentGenerationMismatch covers the
// core safety property resumeState exists for: a checkpoint left over
// from a round trained against a different teacher generation than the
// one the caller is about to train against must never be resumed, even
// though it's otherwise perfectly valid.
func TestLoadResumableStudent_RejectsAParentGenerationMismatch(t *testing.T) {
	dir := t.TempDir()
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 1, 2, 1)
	require.NoError(t, saveFinalCheckpoint(dir, params, "test-env", checkpoint.Metadata{Epoch: 18}, 7))

	resumed, epoch, _ := loadResumableStudent(dir, "test-env", 8)
	assert.Nil(t, resumed, "a checkpoint trained against generation 7 must not be resumed when the caller is now on generation 8")
	assert.Equal(t, -1, epoch)
}

// TestLoadResumableStudent_NoOpsOnAFreshCheckpointDirectory covers the
// normal, expected "nothing to resume yet" state — an empty or
// brand-new checkpoint directory — which must not be treated as an
// error.
func TestLoadResumableStudent_NoOpsOnAFreshCheckpointDirectory(t *testing.T) {
	dir := t.TempDir()

	resumed, epoch, _ := loadResumableStudent(dir, "test-env", 7)
	assert.Nil(t, resumed)
	assert.Equal(t, -1, epoch)
}

func gatherMetrics(t *testing.T) string {
	t.Helper()
	recorder := httptest.NewRecorder()
	promhttp.Handler().ServeHTTP(recorder, httptest.NewRequest("GET", "/metrics", nil))
	body, err := io.ReadAll(recorder.Result().Body)
	require.NoError(t, err)
	return strings.ReplaceAll(string(body), "\\n", "\n")
}

type resetErrorEnvironment struct{ metricsTestEnvironment }

func (*resetErrorEnvironment) Reset(context.Context) (rl.Observation, error) {
	return rl.Observation{}, io.ErrUnexpectedEOF
}

// flakyResetEnvironment fails Reset exactly *failuresRemaining times
// (decrementing it each call) before succeeding - a fresh "task draw" per
// attempt, mirroring how rlenv's real TaskSelector is called again on each
// new Reset rather than retrying an identical one.
type flakyResetEnvironment struct {
	metricsTestEnvironment
	failuresRemaining *int
}

func (e *flakyResetEnvironment) Reset(context.Context) (rl.Observation, error) {
	if *e.failuresRemaining > 0 {
		*e.failuresRemaining--
		return rl.Observation{}, io.ErrUnexpectedEOF
	}
	return rl.Observation{Values: []float32{0}}, nil
}
