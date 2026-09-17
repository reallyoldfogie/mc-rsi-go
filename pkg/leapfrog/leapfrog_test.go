package leapfrog

import (
	"context"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/gridworldenv"
	"github.com/reallyoldfogie/cRL-go/pkg/ppo"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestDecideWinner exercises the win/lose comparison logic directly,
// independent of any real training or environment — see
// docs/plans/03-single-task-leapfrog-evaluation-loop.md's "Done when"
// for why this is worth testing in isolation from a full Round.
func TestDecideWinner(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	studentParams := actorcritic.NewParams(rng, 4, 4, 2)

	tests := []struct {
		name          string
		teacherReward float32
		studentReward float32
		wantWon       bool
	}{
		{"student strictly better wins", 1.0, 2.0, true},
		{"student strictly worse loses", 2.0, 1.0, false},
		{"exact tie does not count as a win", 1.5, 1.5, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := decideWinner(tc.teacherReward, tc.studentReward, studentParams)
			assert.Equal(t, tc.wantWon, result.StudentWon)
			assert.Equal(t, tc.teacherReward, result.TeacherReward)
			assert.Equal(t, tc.studentReward, result.StudentReward)
			if tc.wantWon {
				assert.Same(t, studentParams, result.StudentParams)
			} else {
				assert.Nil(t, result.StudentParams, "a losing (or tied) Student's params must not be returned")
			}
		})
	}
}

func TestConfigValidateRejectsNonPositiveFields(t *testing.T) {
	base := Config{EpochsPerGeneration: 1, EvalEpisodes: 1, EvalEpisodeLen: 1}

	tests := []struct {
		name string
		cfg  Config
	}{
		{"zero epochs per generation", Config{EpochsPerGeneration: 0, EvalEpisodes: 1, EvalEpisodeLen: 1}},
		{"negative epochs per generation", Config{EpochsPerGeneration: -1, EvalEpisodes: 1, EvalEpisodeLen: 1}},
		{"zero eval episodes", Config{EpochsPerGeneration: 1, EvalEpisodes: 0, EvalEpisodeLen: 1}},
		{"zero eval episode length", Config{EpochsPerGeneration: 1, EvalEpisodes: 1, EvalEpisodeLen: 0}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Error(t, tc.cfg.Validate())
		})
	}

	assert.NoError(t, base.Validate())
}

func TestRoundRejectsNilTeacherParams(t *testing.T) {
	env, _ := newTestGridEnv(t)
	_, err := Round(context.Background(), []rl.Environment{env}, nil, testConfig(env), rand.New(rand.NewPCG(1, 2)))
	assert.Error(t, err)
}

func TestRoundRejectsEmptyEnvs(t *testing.T) {
	env, teacherParams := newTestGridEnv(t)
	_, err := Round(context.Background(), nil, teacherParams, testConfig(env), rand.New(rand.NewPCG(1, 2)))
	assert.Error(t, err)
}

// TestRoundEndToEndAgainstGridworld runs a real (if tiny) round against
// cRL-go's own gridworldenv — not a live rlenv session, per
// docs/plans/03's "Done when" — checking the whole Round pipeline wires
// together (training, both evaluations, the win decision) without
// asserting which side happens to win: that outcome depends on real,
// randomly-initialized training and isn't something this test should
// pin down, per this repo's own instructions not to assert on
// incidental behavior.
func TestRoundEndToEndAgainstGridworld(t *testing.T) {
	env, teacherParams := newTestGridEnv(t)
	cfg := testConfig(env)
	rng := rand.New(rand.NewPCG(7, 11))

	result, err := Round(context.Background(), []rl.Environment{env}, teacherParams, cfg, rng)
	require.NoError(t, err)

	assert.False(t, isNaN32(result.TeacherReward))
	assert.False(t, isNaN32(result.StudentReward))

	if result.StudentWon {
		require.NotNil(t, result.StudentParams, "a winning Student must return its params")
		assert.Equal(t, teacherParams.InputSize(), result.StudentParams.InputSize())
		assert.Equal(t, teacherParams.HiddenSize(), result.StudentParams.HiddenSize())
		assert.Equal(t, teacherParams.OutputSize(), result.StudentParams.OutputSize())
	} else {
		assert.Nil(t, result.StudentParams)
	}
}

// TestRoundCallsOnEpochOncePerStudentTrainingEpoch verifies
// Config.OnEpoch (added for docs/plans/04-training-entrypoint-and-observability.md's
// interim-checkpoint/progress-logging needs) fires exactly
// cfg.EpochsPerGeneration times, with strictly increasing Epoch numbers
// and a non-nil params snapshot each time — and does not fire at all
// during evaluation (which doesn't train).
func TestRoundCallsOnEpochOncePerStudentTrainingEpoch(t *testing.T) {
	env, teacherParams := newTestGridEnv(t)
	cfg := testConfig(env)
	cfg.EpochsPerGeneration = 3

	var seenEpochs []int
	cfg.OnEpoch = func(stats ppo.EpochStats, params *actorcritic.Params) {
		require.NotNil(t, params)
		seenEpochs = append(seenEpochs, stats.Epoch)
	}

	_, err := Round(context.Background(), []rl.Environment{env}, teacherParams, cfg, rand.New(rand.NewPCG(1, 2)))
	require.NoError(t, err)

	require.Equal(t, []int{0, 1, 2}, seenEpochs)
}

// TestRoundEndToEndAgainstMultipleGridworlds is TestRoundEndToEndAgainstGridworld's
// len(envs) > 1 counterpart: proves Round's pooled-training branch
// (ppo.NewWithPersistentEnvPool, via trainStudent) is actually wired
// together end to end in this package, not just unit-tested in
// isolation inside cRL-go itself.
func TestRoundEndToEndAgainstMultipleGridworlds(t *testing.T) {
	envs, teacherParams := newTestGridEnvPool(t, 2)
	cfg := testConfig(envs[0].(*gridworldenv.Adapter))
	rng := rand.New(rand.NewPCG(7, 11))

	result, err := Round(context.Background(), envs, teacherParams, cfg, rng)
	require.NoError(t, err)

	assert.False(t, isNaN32(result.TeacherReward))
	assert.False(t, isNaN32(result.StudentReward))

	if result.StudentWon {
		require.NotNil(t, result.StudentParams, "a winning Student must return its params")
		assert.Equal(t, teacherParams.InputSize(), result.StudentParams.InputSize())
		assert.Equal(t, teacherParams.HiddenSize(), result.StudentParams.HiddenSize())
		assert.Equal(t, teacherParams.OutputSize(), result.StudentParams.OutputSize())
	} else {
		assert.Nil(t, result.StudentParams)
	}
}

// newTestGridEnv builds a small, deterministic gridworldenv instance
// (gridworldenv's own layout is fixed regardless of rng — see
// pkg/reinforce.EnvFactory's doc comment in cRL-go) plus a freshly
// initialized *actorcritic.Params shaped to match it.
func newTestGridEnv(t *testing.T) (*gridworldenv.Adapter, *actorcritic.Params) {
	t.Helper()

	const gridSize = 4 // 2x2 grid: smallest valid perfect-square size.
	rawEnv, err := gridworldenv.New(gridSize)
	require.NoError(t, err)
	env := gridworldenv.NewAdapter(rawEnv)

	rng := rand.New(rand.NewPCG(1, 2))
	teacherParams := actorcritic.NewParams(rng, env.ObservationSize(), 8, env.ActionSpace())
	return env, teacherParams
}

// newTestGridEnvPool builds n independent gridworldenv.Adapter instances
// (mirroring newTestGridEnv, but n separate envs rather than one)
// sharing a single teacherParams, for exercising Round/trainStudent's
// len(envs) > 1 (ppo.NewWithPersistentEnvPool) branch.
func newTestGridEnvPool(t *testing.T, n int) ([]rl.Environment, *actorcritic.Params) {
	t.Helper()

	const gridSize = 4 // 2x2 grid: smallest valid perfect-square size.
	envs := make([]rl.Environment, n)
	var observationSize, actionSpace int
	for i := range n {
		rawEnv, err := gridworldenv.New(gridSize)
		require.NoError(t, err)
		env := gridworldenv.NewAdapter(rawEnv)
		envs[i] = env
		observationSize, actionSpace = env.ObservationSize(), env.ActionSpace()
	}

	rng := rand.New(rand.NewPCG(1, 2))
	teacherParams := actorcritic.NewParams(rng, observationSize, 8, actionSpace)
	return envs, teacherParams
}

// testConfig returns a Config sized for a fast unit test: one training
// epoch over a two-trajectory rollout, two short evaluation episodes per
// side.
func testConfig(env *gridworldenv.Adapter) Config {
	return Config{
		Trainer: config.Settings{
			RolloutSize:   2,
			EpisodeLen:    10,
			Gamma:         0.99,
			LearningRate:  0.05,
			GridSize:      4, // must be a perfect square (config.Settings.Validate) even though env, not GridSize, is what's actually driven — see Config.Trainer's own doc comment.
			HiddenSize:    8,
			Seed:          42,
			Workers:       1,
			ClipEpsilon:   0.2,
			EntropyCoef:   0.01,
			ValueCoef:     0.5,
			GAELambda:     0.95,
			PPOEpochs:     1,
			MinibatchSize: 4,
		},
		EpochsPerGeneration: 1,
		EvalEpisodes:        2,
		EvalEpisodeLen:      10,
	}
}

func isNaN32(f float32) bool {
	return f != f
}
