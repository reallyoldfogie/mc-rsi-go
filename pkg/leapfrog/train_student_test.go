package leapfrog

import (
	"context"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/require"
)

type trainStudentTestEnv struct{}

func (trainStudentTestEnv) Reset(context.Context) (rl.Observation, error) {
	return rl.Observation{Values: []float32{0}}, nil
}

func (trainStudentTestEnv) Step(context.Context, rl.Action) (rl.StepResult, error) {
	return rl.StepResult{Observation: rl.Observation{Values: []float32{0}}, Reward: 1, Done: true}, nil
}

func (trainStudentTestEnv) ObservationSize() int { return 1 }
func (trainStudentTestEnv) ActionSpace() int     { return 1 }

func TestTrainStudentExposesTrainingWithoutEvaluation(t *testing.T) {
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 1, 4, 1, 2)
	settings := config.Default()
	settings.HiddenSize = 4
	settings.RolloutSize = 1
	settings.EpisodeLen = 1
	settings.MinibatchSize = 1
	settings.PPOEpochs = 1
	settings.Workers = 1
	cfg := Config[*actorcritic.Params]{Trainer: settings, EpochsPerGeneration: 1, EvalEpisodes: 1, EvalEpisodeLen: 1}
	student, err := TrainStudent(context.Background(), []rl.Environment{trainStudentTestEnv{}}, params, cfg, PlainSeams)
	require.NoError(t, err)
	require.NotNil(t, student)
	require.Equal(t, params.InputSize(), student.InputSize())
}
