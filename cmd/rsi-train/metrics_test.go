package main

import (
	"context"
	"io"
	"math/rand/v2"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
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

func TestSaveFinalCheckpointPreservesMetadata(t *testing.T) {
	dir := t.TempDir()
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 1, 2, 1)
	metadata := checkpoint.Metadata{Epoch: 11, BestReturn: 4.5, TotalUpdates: 37}

	require.NoError(t, saveFinalCheckpoint(dir, params, "test-env", metadata))
	_, loaded, err := actorcritic.LoadFile(filepath.Join(dir, "final.json"), "test-env")
	require.NoError(t, err)
	require.Equal(t, metadata, loaded)
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
