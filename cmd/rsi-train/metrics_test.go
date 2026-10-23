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
	"github.com/reallyoldfogie/mc-agent/rlenv"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/leapfrog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type metricsTestEnvironment struct {
	stepResult rl.StepResult
	onReset    func()
}

type respawnTestHealth struct{ ready chan struct{} }

func (h respawnTestHealth) Health() (float32, int32, float32, bool) {
	select {
	case <-h.ready:
		return 20, 20, 5, true
	default:
		return 0, 20, 0, true
	}
}

func TestWaitForRespawnWaitsForHealthToRecover(t *testing.T) {
	health := respawnTestHealth{ready: make(chan struct{})}
	go func() {
		time.Sleep(20 * time.Millisecond)
		close(health.ready)
	}()
	require.NoError(t, waitForRespawn(context.Background(), health))
}

func TestWaitForRespawnHonorsContext(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	health := respawnTestHealth{ready: make(chan struct{})}
	err := waitForRespawn(ctx, health)
	require.ErrorIs(t, err, context.DeadlineExceeded)
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

// doneReasonTestEnvironment adds an explicit DoneReason() to
// metricsTestEnvironment so TestInstrumentedEnvironmentExposesCurrentTaskAndDoneReason
// can exercise instrumentedEnvironment.DoneReason's pass-through branch - the
// plain metricsTestEnvironment used elsewhere in this file implements
// neither doneReasonReporter nor leapfrog.EpisodeDoneReasoner.
type doneReasonTestEnvironment struct {
	metricsTestEnvironment
	reason string
}

func (e *doneReasonTestEnvironment) DoneReason() string { return e.reason }

// TestInstrumentedEnvironmentExposesCurrentTaskAndDoneReason covers the two
// exported accessors pkg/leapfrog's per-episode evaluation loop relies on
// (via its own EpisodeTaskReporter/EpisodeDoneReasoner duck-typed
// interfaces) to record each competition evaluation episode's task label
// and termination reason without reaching into the dashboard registry.
func TestInstrumentedEnvironmentExposesCurrentTaskAndDoneReason(t *testing.T) {
	task := "metrics-test-reporter"
	noReason := &instrumentedEnvironment{env: &metricsTestEnvironment{}, task: task}
	assert.Equal(t, task, noReason.CurrentTask())
	assert.Equal(t, "", noReason.DoneReason(), "the wrapped environment doesn't implement DoneReason")

	withReason := &instrumentedEnvironment{env: &doneReasonTestEnvironment{reason: "arrived"}, task: task}
	assert.Equal(t, "arrived", withReason.DoneReason())
}

func TestInstrumentedEnvironmentTracksCompetitionCompletion(t *testing.T) {
	env := &instrumentedEnvironment{
		env:  &metricsTestEnvironment{stepResult: rl.StepResult{Reward: 10, Done: true}},
		task: "metrics-test-eval",
	}
	env.beginCompetitionEvaluation()
	_, err := env.Reset(context.Background())
	require.NoError(t, err)
	_, err = env.Step(context.Background(), 0)
	require.NoError(t, err)
	assert.Equal(t, 1.0, env.endCompetitionEvaluation())
}

func TestInstrumentedEnvironmentRecordsFailedEpisode(t *testing.T) {
	task := "metrics-test-failure"
	env := &instrumentedEnvironment{
		env:  &metricsTestEnvironment{stepResult: rl.StepResult{Reward: 1, Done: true}},
		task: task,
	}
	_, err := env.Reset(context.Background())
	require.NoError(t, err)
	_, err = env.Step(context.Background(), 0)
	require.NoError(t, err)

	body := gatherMetrics(t)
	require.Contains(t, body, `rsi_episodes_total{outcome="failure",task="`+task+`"} 1`)
	require.NotContains(t, body, `rsi_episodes_total{outcome="success",task="`+task+`"}`)
	require.Contains(t, body, `rsi_episode_length_steps_bucket{outcome="failure",phase="training",task="`+task+`",le="1"} 1`)
	require.Contains(t, body, `rsi_episode_terminations_total{phase="training",reason="unspecified_done",task="`+task+`"} 1`)
}

// TestInstrumentedEnvironmentRecordsStepDuration covers the per-step
// timing histogram added to make a class of investigation (correlating
// slow steps with retry loops or blocking sleeps deep in mc-agent, e.g.
// the OpenContainer sleep bug) discoverable from Grafana instead of
// requiring manual log-timestamp correlation. Recorded on both a
// successful and a failing step, since a step that fails slowly is
// exactly the case this metric exists to surface.
func TestInstrumentedEnvironmentRecordsStepDuration(t *testing.T) {
	task := "metrics-test-step-duration"
	env := &instrumentedEnvironment{
		env:  &metricsTestEnvironment{stepResult: rl.StepResult{Reward: 1}},
		task: task,
	}
	_, err := env.Reset(context.Background())
	require.NoError(t, err)
	_, err = env.Step(context.Background(), 0)
	require.NoError(t, err)

	body := gatherMetrics(t)
	require.Contains(t, body, `rsi_step_duration_seconds_count{task="`+task+`"} 1`)
}

func TestInstrumentedEnvironmentRecordsStepDurationOnError(t *testing.T) {
	task := "metrics-test-step-duration-error"
	env := &instrumentedEnvironment{env: &stepErrorEnvironment{}, task: task}
	_, err := env.Step(context.Background(), 0)
	require.Error(t, err)

	body := gatherMetrics(t)
	require.Contains(t, body, `rsi_step_duration_seconds_count{task="`+task+`"} 1`)
}

func TestInstrumentedEnvironmentCountsChosenActionsByTask(t *testing.T) {
	task := "metrics-test-actions"
	env := &instrumentedEnvironment{
		env:  &metricsTestEnvironment{stepResult: rl.StepResult{Reward: 1}},
		task: task,
	}
	_, err := env.Reset(context.Background())
	require.NoError(t, err)
	for _, a := range []rl.Action{rlenv.ActionMine, rlenv.ActionMine, rlenv.ActionGoToTarget, rlenv.ActionWait} {
		_, err = env.Step(context.Background(), a)
		require.NoError(t, err)
	}

	body := gatherMetrics(t)
	require.Contains(t, body, `rsi_actions_total{action="mine",task="`+task+`"} 2`)
	require.Contains(t, body, `rsi_actions_total{action="goto",task="`+task+`"} 1`)
	require.Contains(t, body, `rsi_actions_total{action="wait",task="`+task+`"} 1`)
	assert.Equal(t, "craft", actionName(rlenv.ActionCraft))
	assert.Equal(t, "unknown", actionName(rl.Action(99)))
}

// TestEpisodeLengthStepsTagsPhaseCorrectly guards the confusion found live
// 2026-10-03: without a phase label, a success/truncated episode past the
// 100-step training cap looked impossible on the dashboard, when it was
// actually from a 200-step evaluation episode (-eval-episode-len).
func TestEpisodeLengthStepsTagsPhaseCorrectly(t *testing.T) {
	evaluationInProgress.Store(false)
	defer evaluationInProgress.Store(false)

	taggedTask := "metrics-test-phase-training"
	env := &instrumentedEnvironment{
		env:  &metricsTestEnvironment{stepResult: rl.StepResult{Reward: 10, Done: true}},
		task: taggedTask,
	}
	_, err := env.Reset(context.Background())
	require.NoError(t, err)
	_, err = env.Step(context.Background(), 0)
	require.NoError(t, err)
	require.Contains(t, gatherMetrics(t),
		`rsi_episode_length_steps_bucket{outcome="success",phase="training",task="`+taggedTask+`",le="1"} 1`)

	evaluationInProgress.Store(true)
	taggedTask = "metrics-test-phase-eval"
	env = &instrumentedEnvironment{
		env:  &metricsTestEnvironment{stepResult: rl.StepResult{Reward: 10, Done: true}},
		task: taggedTask,
	}
	_, err = env.Reset(context.Background())
	require.NoError(t, err)
	_, err = env.Step(context.Background(), 0)
	require.NoError(t, err)
	require.Contains(t, gatherMetrics(t),
		`rsi_episode_length_steps_bucket{outcome="success",phase="eval",task="`+taggedTask+`",le="1"} 1`)
}

// TestInstrumentedEnvironmentSkipsEpochTrajectoriesStartedDuringEvaluation
// guards the bug found live 2026-10-03: a round's Teacher+Student
// evaluation episodes run through the same instrumentedEnvironment.Reset as
// training rollout collection, so without this gate they counted toward
// rsi_epoch_trajectories_started too, pushing the "epoch progress"
// dashboard bar (rsi_epoch_trajectories_started / rsi_epoch_rollout_size)
// past 100% once evaluation ran after the last training epoch's own reset.
func TestInstrumentedEnvironmentSkipsEpochTrajectoriesStartedDuringEvaluation(t *testing.T) {
	metricEpochTrajectoriesStarted.Set(0)
	evaluationInProgress.Store(false)
	defer evaluationInProgress.Store(false)

	env := &instrumentedEnvironment{
		env:  &metricsTestEnvironment{stepResult: rl.StepResult{Reward: 10, Done: true}},
		task: "metrics-test-eval-gate",
	}

	_, err := env.Reset(context.Background())
	require.NoError(t, err)
	require.Contains(t, gatherMetrics(t), "rsi_epoch_trajectories_started 1")

	evaluationInProgress.Store(true)
	_, err = env.Reset(context.Background())
	require.NoError(t, err)
	require.Contains(t, gatherMetrics(t), "rsi_epoch_trajectories_started 1",
		"an evaluation-episode Reset must not count toward epoch rollout progress")

	evaluationInProgress.Store(false)
	_, err = env.Reset(context.Background())
	require.NoError(t, err)
	require.Contains(t, gatherMetrics(t), "rsi_epoch_trajectories_started 2",
		"a training-episode Reset after evaluation ends must resume counting")
}

func TestMarkEvaluationInProgressSetsFlagAndCallsWrapped(t *testing.T) {
	evaluationInProgress.Store(false)
	defer evaluationInProgress.Store(false)

	var calledWith leapfrog.EvalSide
	wrapped := markEvaluationInProgress(func(side leapfrog.EvalSide) { calledWith = side })

	wrapped(leapfrog.EvalStudent)
	assert.True(t, evaluationInProgress.Load())
	assert.Equal(t, leapfrog.EvalStudent, calledWith)
}

func TestMarkEvaluationInProgressToleratesNilWrapped(t *testing.T) {
	evaluationInProgress.Store(false)
	defer evaluationInProgress.Store(false)

	wrapped := markEvaluationInProgress(nil)
	assert.NotPanics(t, func() { wrapped(leapfrog.EvalTeacher) })
	assert.True(t, evaluationInProgress.Load())
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
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 1, 2, 1, 2)
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
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 1, 2, 1, 2)
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

	staleHighEpochParams := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 1, 2, 1, 2)
	require.NoError(t, saveInterimCheckpoint(dir, staleHighEpochParams, "test-env", 49, checkpoint.Metadata{Epoch: 49, BestReturn: 17.16, TotalUpdates: 4812}, 7))
	stalePath := filepath.Join(dir, "interim-epoch-000000049.json")
	require.NoError(t, os.Chtimes(stalePath, old, old))

	recentLowEpochParams := actorcritic.NewParams(rand.New(rand.NewPCG(3, 4)), 1, 2, 1, 2)
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
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 1, 2, 1, 2)
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

// TestClearInterimCheckpointsPreventsResumingAFinishedRound: after a round
// ends (a lost round keeps the same parent generation), a restart early in
// the next round must not find the finished student's checkpoints.
func TestClearInterimCheckpointsPreventsResumingAFinishedRound(t *testing.T) {
	dir := t.TempDir()
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 1, 2, 1, 2)
	require.NoError(t, saveInterimCheckpoint(dir, params, "test-env", 99, checkpoint.Metadata{Epoch: 99}, 7))
	require.NoError(t, saveFinalCheckpoint(dir, params, "test-env", checkpoint.Metadata{Epoch: 101}, 7))
	resumed, _, _ := loadResumableStudent(dir, "test-env", 7)
	require.NotNil(t, resumed, "precondition: the finished round's checkpoints are resumable")

	clearInterimCheckpoints(dir)

	resumed, epoch, _ := loadResumableStudent(dir, "test-env", 7)
	assert.Nil(t, resumed)
	assert.Equal(t, -1, epoch)
	entries, err := os.ReadDir(dir)
	require.NoError(t, err)
	assert.Empty(t, entries, "every recovery file must be gone")

	clearInterimCheckpoints(dir) // and clearing an already-clear directory is fine
}

func TestRecordRoundEvaluationPublishesTheStatisticsThePromotionRestsOn(t *testing.T) {
	recordRoundEvaluation(leapfrog.Result[*actorcritic.Params]{
		TeacherReward: 9.5, StudentReward: 11.0, EvalEpisodes: 35,
		TeacherStd: 3.25, StudentStd: 2.75, MeanDiff: 1.5, DiffStdErr: 0.75,
	})
	body := gatherMetrics(t)
	require.Contains(t, body, "rsi_teacher_reward 9.5")
	require.Contains(t, body, "rsi_student_reward 11")
	require.Contains(t, body, "rsi_eval_episodes 35")
	require.Contains(t, body, `rsi_eval_reward_std{side="teacher"} 3.25`)
	require.Contains(t, body, `rsi_eval_reward_std{side="student"} 2.75`)
	require.Contains(t, body, "rsi_eval_paired_diff 1.5")
	require.Contains(t, body, "rsi_eval_paired_diff_stderr 0.75")
	require.Contains(t, body, "rsi_eval_paired_diff_zscore 2", "1.5 / 0.75")

	// A zero standard error must not produce +Inf/NaN.
	recordRoundEvaluation(leapfrog.Result[*actorcritic.Params]{MeanDiff: 1, DiffStdErr: 0})
	require.Contains(t, gatherMetrics(t), "rsi_eval_paired_diff_zscore 0")
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

type stepErrorEnvironment struct{ metricsTestEnvironment }

func (*stepErrorEnvironment) Step(context.Context, rl.Action) (rl.StepResult, error) {
	return rl.StepResult{}, io.ErrUnexpectedEOF
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
