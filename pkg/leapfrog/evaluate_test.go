package leapfrog

import (
	"context"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/stretchr/testify/require"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

type evaluationTestEnv struct{ done bool }

func (e *evaluationTestEnv) Reset(context.Context) (rl.Observation, error) {
	e.done = false
	return rl.Observation{Values: []float32{0}}, nil
}

func (e *evaluationTestEnv) Step(context.Context, rl.Action) (rl.StepResult, error) {
	e.done = true
	return rl.StepResult{Reward: 2, Done: true}, nil
}

func (e *evaluationTestEnv) ObservationSize() int { return 1 }
func (e *evaluationTestEnv) ActionSpace() int     { return 1 }

// reportingTestEnv implements EpisodeTaskReporter and EpisodeDoneReasoner
// (both optional, duck-typed interfaces evaluateEpisode checks) so tests can
// drive what EpisodeResult.Task/TerminationReason record. doneAtStep <= 0
// means the episode never sees Done=true, exercising the "timeout"
// classification instead.
type reportingTestEnv struct {
	task       string
	doneReason string
	doneAtStep int
	step       int
}

func (e *reportingTestEnv) Reset(context.Context) (rl.Observation, error) {
	e.step = 0
	return rl.Observation{Values: []float32{0}}, nil
}

func (e *reportingTestEnv) Step(context.Context, rl.Action) (rl.StepResult, error) {
	e.step++
	done := e.doneAtStep > 0 && e.step >= e.doneAtStep
	return rl.StepResult{Reward: 1, Done: done}, nil
}

func (e *reportingTestEnv) ObservationSize() int { return 1 }
func (e *reportingTestEnv) ActionSpace() int     { return 1 }
func (e *reportingTestEnv) CurrentTask() string  { return e.task }
func (e *reportingTestEnv) DoneReason() string   { return e.doneReason }

func TestEvaluateRecordsTaskAndTerminationReasonFromAnOptionalReporter(t *testing.T) {
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 1, 4, 1, 2)
	env := &reportingTestEnv{task: "mine", doneReason: "mined", doneAtStep: 1}

	scores, err := Evaluate(context.Background(), []rl.Environment{env}, params, 2, 5, false, PlainSeams, rand.New(rand.NewPCG(3, 4)))
	require.NoError(t, err)
	require.Len(t, scores[0], 2)
	for _, result := range scores[0] {
		require.Equal(t, "mine", result.Task)
		require.Equal(t, "mined", result.TerminationReason)
	}
}

func TestEvaluateClassifiesATimedOutEpisodeDistinctlyFromAnUnreportedDone(t *testing.T) {
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 1, 4, 1, 2)

	timedOut := &reportingTestEnv{task: "goto", doneAtStep: 0}
	scores, err := Evaluate(context.Background(), []rl.Environment{timedOut}, params, 1, 3, false, PlainSeams, rand.New(rand.NewPCG(3, 4)))
	require.NoError(t, err)
	require.Equal(t, "timeout", scores[0][0].TerminationReason, "every step ran with Done always false")

	unreported := &reportingTestEnv{task: "goto", doneAtStep: 1}
	scores, err = Evaluate(context.Background(), []rl.Environment{unreported}, params, 1, 3, false, PlainSeams, rand.New(rand.NewPCG(3, 4)))
	require.NoError(t, err)
	require.Equal(t, "unspecified_done", scores[0][0].TerminationReason, "Done=true but DoneReason() returned \"\"")
}

func TestEvaluateExposesReadOnlyEvaluation(t *testing.T) {
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 1, 4, 1, 2)
	scores, err := Evaluate(context.Background(), []rl.Environment{&evaluationTestEnv{}}, params, 3, 5, false, PlainSeams, rand.New(rand.NewPCG(3, 4)))
	require.NoError(t, err)
	require.Len(t, scores, 1)
	require.Len(t, scores[0], 3)
	for _, result := range scores[0] {
		require.Equal(t, float32(2), result.Reward)
	}
}
