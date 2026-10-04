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
	studentParams := actorcritic.NewParams(rng, 4, 4, 2, 2)

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
	base := Config[*actorcritic.Params]{EpochsPerGeneration: 1, EvalEpisodes: 1, EvalEpisodeLen: 1}

	tests := []struct {
		name string
		cfg  Config[*actorcritic.Params]
	}{
		{"zero epochs per generation", Config[*actorcritic.Params]{EpochsPerGeneration: 0, EvalEpisodes: 1, EvalEpisodeLen: 1}},
		{"negative epochs per generation", Config[*actorcritic.Params]{EpochsPerGeneration: -1, EvalEpisodes: 1, EvalEpisodeLen: 1}},
		{"zero eval episodes", Config[*actorcritic.Params]{EpochsPerGeneration: 1, EvalEpisodes: 0, EvalEpisodeLen: 1}},
		{"zero eval episode length", Config[*actorcritic.Params]{EpochsPerGeneration: 1, EvalEpisodes: 1, EvalEpisodeLen: 0}},
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
	_, err := Round(context.Background(), []rl.Environment{env}, nil, testConfig(env), PlainSeams, rand.New(rand.NewPCG(1, 2)))
	assert.Error(t, err)
}

func TestRoundRejectsEmptyEnvs(t *testing.T) {
	env, teacherParams := newTestGridEnv(t)
	_, err := Round(context.Background(), nil, teacherParams, testConfig(env), PlainSeams, rand.New(rand.NewPCG(1, 2)))
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

	result, err := Round(context.Background(), []rl.Environment{env}, teacherParams, cfg, PlainSeams, rng)
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

// TestRoundEndToEndAgainstGridworldRecurrent mirrors
// TestRoundEndToEndAgainstGridworld exactly, but trains and evaluates
// *actorcritic.RecurrentParams via RecurrentSeams instead of
// *actorcritic.Params via PlainSeams — the smoke test confirming Round's
// generic core actually works for both type arguments, not just the one
// every other test here exercises.
func TestRoundEndToEndAgainstGridworldRecurrent(t *testing.T) {
	env, teacherParams := newTestGridEnvRecurrent(t)
	cfg := testRecurrentConfig(env)
	rng := rand.New(rand.NewPCG(7, 11))

	result, err := Round(context.Background(), []rl.Environment{env}, teacherParams, cfg, RecurrentSeams, rng)
	require.NoError(t, err)

	assert.False(t, isNaN32(result.TeacherReward))
	assert.False(t, isNaN32(result.StudentReward))

	if result.StudentWon {
		require.NotNil(t, result.StudentParams, "a winning Student must return its params")
		assert.Equal(t, teacherParams.HiddenSize(), result.StudentParams.HiddenSize())
	} else {
		assert.Nil(t, result.StudentParams)
	}
}

// newTestGridEnvRecurrent mirrors newTestGridEnv, but returns a freshly
// initialized *actorcritic.RecurrentParams instead.
func newTestGridEnvRecurrent(t *testing.T) (*gridworldenv.Adapter, *actorcritic.RecurrentParams) {
	t.Helper()

	const gridSize = 4 // 2x2 grid: smallest valid perfect-square size.
	rawEnv, err := gridworldenv.New(gridSize)
	require.NoError(t, err)
	env := gridworldenv.NewAdapter(rawEnv)

	rng := rand.New(rand.NewPCG(1, 2))
	teacherParams := actorcritic.NewRecurrentParams(rng, env.ObservationSize(), 8, env.ActionSpace(), 2)
	return env, teacherParams
}

// testRecurrentConfig mirrors testConfig, with RecurrentChunkLen set so
// the trainer actually trains recurrently rather than rejecting the
// RecurrentParams/non-recurrent-settings mismatch.
func testRecurrentConfig(env *gridworldenv.Adapter) Config[*actorcritic.RecurrentParams] {
	cfg := testConfig(env)
	cfg.Trainer.RecurrentChunkLen = 2
	return Config[*actorcritic.RecurrentParams]{
		Trainer:             cfg.Trainer,
		EpochsPerGeneration: cfg.EpochsPerGeneration,
		EvalEpisodes:        cfg.EvalEpisodes,
		EvalEpisodeLen:      cfg.EvalEpisodeLen,
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

	_, err := Round(context.Background(), []rl.Environment{env}, teacherParams, cfg, PlainSeams, rand.New(rand.NewPCG(1, 2)))
	require.NoError(t, err)

	require.Equal(t, []int{0, 1, 2}, seenEpochs)
}

// TestRoundResumesStudentFromConfigInsteadOfCloningTeacher verifies
// Config.ResumeStudent/ResumeStudentStartEpoch (added for cmd/rsi-train's
// interim-checkpoint resume feature): when set, trainStudent must start
// from the given params — not teacherParams.Snapshot() — and OnEpoch's
// reported epoch numbers must continue from ResumeStudentStartEpoch
// rather than restarting at 0, with EpochsPerGeneration the round's total
// budget (so resuming at epoch 19 of 21 runs epochs 19 and 20).
func TestRoundResumesStudentFromConfigInsteadOfCloningTeacher(t *testing.T) {
	env, teacherParams := newTestGridEnv(t)
	resumeParams := teacherParams.Snapshot()
	cfg := testConfig(env)
	cfg.EpochsPerGeneration = 21
	cfg.ResumeStudent = resumeParams
	cfg.ResumeStudentStartEpoch = 19

	var seenEpochs []int
	var sawResumedParamsFirst bool
	cfg.OnEpoch = func(stats ppo.EpochStats, params *actorcritic.Params) {
		seenEpochs = append(seenEpochs, stats.Epoch)
		if len(seenEpochs) == 1 {
			sawResumedParamsFirst = params == resumeParams
		}
	}

	_, err := Round(context.Background(), []rl.Environment{env}, teacherParams, cfg, PlainSeams, rand.New(rand.NewPCG(1, 2)))
	require.NoError(t, err)

	assert.Equal(t, []int{19, 20}, seenEpochs, "epoch numbers must continue from ResumeStudentStartEpoch and stop at the EpochsPerGeneration total")
	assert.True(t, sawResumedParamsFirst, "the very first epoch must train the resumed params object itself, not a fresh clone of teacherParams")
}

// TestRoundResumedPastTheEpochBudgetSkipsTrainingAndStillEvaluates: a
// resume at or beyond EpochsPerGeneration must not re-arm a fresh budget
// (the live bug: every restart added another full round of epochs); it
// trains nothing and goes straight to the round's evaluation.
func TestRoundResumedPastTheEpochBudgetSkipsTrainingAndStillEvaluates(t *testing.T) {
	env, teacherParams := newTestGridEnv(t)
	cfg := testConfig(env)
	cfg.EpochsPerGeneration = 50
	cfg.ResumeStudent = teacherParams.Snapshot()
	cfg.ResumeStudentStartEpoch = 101

	var seenEpochs []int
	cfg.OnEpoch = func(stats ppo.EpochStats, _ *actorcritic.Params) {
		seenEpochs = append(seenEpochs, stats.Epoch)
	}

	var evaluatedSides []EvalSide
	cfg.OnBeforeEvaluation = func(side EvalSide) { evaluatedSides = append(evaluatedSides, side) }

	_, err := Round(context.Background(), []rl.Environment{env}, teacherParams, cfg, PlainSeams, rand.New(rand.NewPCG(1, 2)))
	require.NoError(t, err)

	assert.Empty(t, seenEpochs, "no epochs may run once the resumed epoch is past the budget")
	assert.Len(t, evaluatedSides, 2, "the round must still evaluate both the Teacher and the resumed Student")
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

	result, err := Round(context.Background(), envs, teacherParams, cfg, PlainSeams, rng)
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

// TestRoundCallsOnBeforeEvaluationOncePerSideInOrder verifies
// Config.OnBeforeEvaluation (added so a caller with per-episode random
// task selection, e.g. cmd/rsi-train's curriculum wiring, can pin
// Teacher's and Student's eval episodes to an identical task sequence)
// fires exactly twice per Round, EvalTeacher immediately before
// EvalStudent, and not at all during Student training.
func TestRoundCallsOnBeforeEvaluationOncePerSideInOrder(t *testing.T) {
	env, teacherParams := newTestGridEnv(t)
	cfg := testConfig(env)

	var seenSides []EvalSide
	cfg.OnBeforeEvaluation = func(side EvalSide) {
		seenSides = append(seenSides, side)
	}

	_, err := Round(context.Background(), []rl.Environment{env}, teacherParams, cfg, PlainSeams, rand.New(rand.NewPCG(1, 2)))
	require.NoError(t, err)

	assert.Equal(t, []EvalSide{EvalTeacher, EvalStudent}, seenSides)
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
	teacherParams := actorcritic.NewParams(rng, env.ObservationSize(), 8, env.ActionSpace(), 2)
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
	teacherParams := actorcritic.NewParams(rng, observationSize, 8, actionSpace, 2)
	return envs, teacherParams
}

// testConfig returns a Config sized for a fast unit test: one training
// epoch over a two-trajectory rollout, two short evaluation episodes per
// side.
func testConfig(env *gridworldenv.Adapter) Config[*actorcritic.Params] {
	return Config[*actorcritic.Params]{
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

// scriptedEnv is an environment whose every episode is one step long and
// returns a fixed reward, recording how many episodes it served - enough to
// test evaluation scheduling and statistics without any learning.
type scriptedEnv struct {
	rewards []float32 // reward of episode k is rewards[k % len]
	served  int
}

func (e *scriptedEnv) Reset(context.Context) (rl.Observation, error) {
	return rl.Observation{Values: []float32{0}}, nil
}
func (e *scriptedEnv) Step(context.Context, rl.Action) (rl.StepResult, error) {
	r := e.rewards[e.served%len(e.rewards)]
	e.served++
	return rl.StepResult{Observation: rl.Observation{Values: []float32{0}}, Reward: r, Done: true}, nil
}
func (e *scriptedEnv) ObservationSize() int { return 1 }
func (e *scriptedEnv) ActionSpace() int     { return 2 }

func TestEvaluateAllRunsEveryEnvironmentAndKeepsPerEnvironmentOrder(t *testing.T) {
	a := &scriptedEnv{rewards: []float32{1, 2, 3}}
	b := &scriptedEnv{rewards: []float32{10, 20, 30}}
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 1, 4, 2, 2)

	scores, err := evaluateAll(context.Background(), []rl.Environment{a, b}, params, 3, 5, true, PlainSeams, rand.New(rand.NewPCG(3, 4)))
	require.NoError(t, err)

	assert.Equal(t, [][]float32{{1, 2, 3}, {10, 20, 30}}, scores, "each environment evaluates EvalEpisodes episodes, in order, concurrently")
}

// TestSummarizeRoundPairedStatistics: the paired difference, its standard
// error and the win rules, on hand-computable numbers.
func TestSummarizeRoundPairedStatistics(t *testing.T) {
	params := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 1, 4, 2, 2)
	teacher := [][]float32{{10, 10, 10, 10}}
	student := [][]float32{{11, 11, 11, 11}} // +1 every pair: zero spread, so any z-score is beaten

	result := summarizeRound(teacher, student, params, Config[*actorcritic.Params]{})
	assert.True(t, result.StudentWon)
	assert.InDelta(t, 1.0, result.MeanDiff, 1e-6)
	assert.InDelta(t, 0.0, result.DiffStdErr, 1e-6)
	assert.Equal(t, 4, result.EvalEpisodes)

	// A noisy tie: same mean advantage of +1, but per-pair differences swing
	// wildly, so a 1.645 z-score must reject what the plain rule accepts.
	noisyTeacher := [][]float32{{10, 10, 10, 10}}
	noisyStudent := [][]float32{{20, 0, 18, 6}} // mean 11, diffs +10 -10 +8 -4
	plain := summarizeRound(noisyTeacher, noisyStudent, params, Config[*actorcritic.Params]{})
	assert.True(t, plain.StudentWon, "the unguarded rule promotes on any positive mean difference")
	guarded := summarizeRound(noisyTeacher, noisyStudent, params, Config[*actorcritic.Params]{WinZScore: 1.645})
	assert.False(t, guarded.StudentWon, "a +1 mean over a per-pair spread of ~9 is noise, not improvement")
	assert.Nil(t, guarded.StudentParams, "a rejected Student's params must be dropped")

	// Margin: +1 is not enough when 2 is demanded.
	assert.False(t, summarizeRound(teacher, student, params, Config[*actorcritic.Params]{MinWinMargin: 2}).StudentWon)
	assert.True(t, summarizeRound(teacher, student, params, Config[*actorcritic.Params]{MinWinMargin: 1}).StudentWon)
}

func TestConfigValidateRejectsNegativeWinRules(t *testing.T) {
	base := Config[*actorcritic.Params]{EpochsPerGeneration: 1, EvalEpisodes: 1, EvalEpisodeLen: 1}
	bad := base
	bad.WinZScore = -1
	assert.Error(t, bad.Validate())
	bad = base
	bad.MinWinMargin = -1
	assert.Error(t, bad.Validate())
	assert.NoError(t, base.Validate())
}

// Evaluation takes the sampled action by default, and the most probable one
// only when asked.
func TestEvalActionSamplesUnlessGreedy(t *testing.T) {
	decision := rl.Decision{Action: 1, Probabilities: []float32{0.9, 0.1}} // sampled the unlikely action
	assert.Equal(t, rl.Action(1), evalAction(decision, false), "sampling evaluation acts on the sampled action")
	assert.Equal(t, rl.Action(0), evalAction(decision, true), "greedy evaluation acts on the most probable action")
}
