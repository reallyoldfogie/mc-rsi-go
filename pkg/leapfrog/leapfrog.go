// Package leapfrog implements the single-task Teacher/Student leapfrog
// evaluation loop described in
// docs/plans/03-single-task-leapfrog-evaluation-loop.md: clone a Teacher
// into a Student, keep training the Student, then compare both against
// a live rl.Environment and report whether the Student won.
//
// Round accepts one or more rl.Environment instances (see
// docs/plans/08-parallel-environments-and-scaling.md). A single
// environment (the original v1 design — see that document's "Why
// sequential, not concurrent" for the mc-agent concurrency bugs, since
// fixed via docs/plans/05-mc-agent-concurrency-fixes.md, that originally
// motivated this) still trains and evaluates entirely sequentially,
// unchanged from before this package supported more than one. Multiple
// environments train the Student concurrently across all of them (one
// real bot session driving each, via ppo.NewWithPersistentEnvPool) for
// throughput, while evaluation stays sequential against envs[0] only —
// see trainStudent's own doc comment for why concurrency is bounded to
// len(envs) rather than some other worker count, and Round's for why
// evaluation isn't parallelized too.
//
// Round does not persist anything: pkg/lineage.Save, called by the
// caller (docs/plans/04), owns turning a winning Result into a saved
// generation, keeping this package free of filesystem concerns exactly
// like pkg/lineage stays free of "when to create a generation"
// decisions.
package leapfrog

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/ppo"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// Config configures one leapfrog round. This is narrower than the
// Config sketched in docs/plans/03's own pseudocode: that sketch's
// EnvironmentID field is dropped here since Round never persists
// anything (see this package's doc comment) and so never needs it — the
// caller (docs/plans/04) already has its own -environment-id flag value
// for the lineage.Record it builds from Round's Result, and round-tripping
// the same string through Config/Result would be dead weight.
type Config struct {
	// Trainer holds the PPO hyperparameters used to keep training the
	// Student. Trainer.Epochs is ignored: EpochsPerGeneration below
	// controls how many epochs one round trains for, since a leapfrog
	// round is deliberately shorter than a whole training run's worth
	// of epochs. Trainer.Workers is likewise unused: Round always
	// drives env sequentially against one persistent instance (see
	// this package's own doc comment), matching how
	// ppo.NewWithPersistentEnv itself ignores Workers.
	//
	// Trainer.GridSize must still be set to a perfect square even
	// though env (not GridSize) determines the actual environment —
	// config.Settings.Validate enforces this regardless of which
	// environment a Trainer actually drives; 1 is a reasonable
	// placeholder when env isn't one of cRL-go's own grid-shaped toy
	// environments.
	Trainer config.Settings

	// EpochsPerGeneration is how many PPO epochs the Student trains for
	// before facing evaluation against the Teacher. Must be positive.
	EpochsPerGeneration int

	// EvalEpisodes is how many episodes each side (Teacher, then
	// Student) is evaluated over. Must be positive.
	EvalEpisodes int

	// EvalEpisodeLen caps the number of steps in one evaluation
	// episode, mirroring config.Settings.EpisodeLen's role during
	// training. Must be positive.
	EvalEpisodeLen int

	// OnEpoch, if set, is called after each Student-training epoch inside
	// Round (never during evaluation, which doesn't train) with that
	// epoch's own EpochStats and the Student's params as they stand right
	// after that epoch's update. Added per
	// docs/plans/04-training-entrypoint-and-observability.md's own
	// "Checkpoint interval" section, which flagged this as a needed
	// refinement once a real caller (cmd/rsi-train) wanted to save an
	// interim checkpoint and log progress partway through a round's
	// potentially-long Student-training phase, not only at Round's very
	// end. nil (the default) costs nothing extra — Round behaves exactly
	// as it did before this field existed.
	OnEpoch func(stats ppo.EpochStats, params *actorcritic.Params)

	// OnBeforeEvaluation, if set, is called exactly twice per Round:
	// once with EvalTeacher immediately before Teacher's evaluation
	// episodes begin, once with EvalStudent immediately before
	// Student's. Added so a caller whose env.Reset draws a per-episode
	// task at random (e.g. cmd/rsi-train's curriculum wiring) can pin
	// both sides to the identical sequence of sampled tasks for this
	// round's comparison — without it, Round's win condition (strictly
	// greater mean reward over EvalEpisodes, no margin) conflates "which
	// random tasks each side happened to draw" with actual skill
	// difference, especially once both sides' rewards are close. nil
	// (the default) costs nothing extra — Round behaves exactly as it
	// did before this field existed, drawing eval tasks from wherever
	// the shared per-env task rng naturally continues.
	OnBeforeEvaluation func(side EvalSide)

	// ResumeStudent, if set, is used as the Student's starting params
	// for this Round call instead of teacherParams.Snapshot() — a
	// caller resuming a mid-round interim checkpoint from an earlier,
	// interrupted attempt against this exact teacher (cmd/rsi-train's
	// own resumeState sidecar validates that precondition before ever
	// setting this). ResumeStudentStartEpoch is the first epoch number
	// OnEpoch/the trainer see (instead of 0) — cfg.EpochsPerGeneration
	// still controls how many *more* epochs this call trains for, not
	// the total including whatever epochs the resumed checkpoint
	// already completed, so callers don't need "epochs remaining"
	// arithmetic. Both fields are the zero value (nil, 0) by default,
	// which is exactly today's unchanged from-a-teacher-clone behavior.
	//
	// A checkpoint only ever saves Params (weights), never Adam's own
	// per-parameter moment estimates — trainStudent always constructs a
	// fresh optimizer regardless of ResumeStudent, so this is a warm
	// start from good weights, not a byte-for-byte continuation of the
	// original optimizer state. That's the standard, well-understood
	// tradeoff of a weights-only checkpoint format, not a bug.
	//
	// Round does not clear these fields itself once used: cfg is passed
	// by value, so a caller that builds one Config and reuses it across
	// several Round calls (cmd/rsi-train's own round loop) must clear
	// them after the first call, or every later round would incorrectly
	// resume the same stale checkpoint instead of cloning from that
	// round's own (possibly different, if an earlier round won) teacher.
	ResumeStudent           *actorcritic.Params
	ResumeStudentStartEpoch int
}

// EvalSide identifies which side of a leapfrog comparison
// Config.OnBeforeEvaluation is about to run evaluation episodes for.
type EvalSide int

const (
	EvalTeacher EvalSide = iota
	EvalStudent
)

// Validate reports whether cfg's own fields are usable. It does not
// validate cfg.Trainer — Round surfaces that separately, via
// ppo.NewWithPersistentEnv's own settings.Validate() call, so a
// Trainer-shaped error and a leapfrog-shaped error read distinctly to a
// caller instead of being merged into one.
func (cfg Config) Validate() error {
	if cfg.EpochsPerGeneration <= 0 {
		return fmt.Errorf("leapfrog: epochs per generation must be positive, got %d", cfg.EpochsPerGeneration)
	}
	if cfg.EvalEpisodes <= 0 {
		return fmt.Errorf("leapfrog: eval episodes must be positive, got %d", cfg.EvalEpisodes)
	}
	if cfg.EvalEpisodeLen <= 0 {
		return fmt.Errorf("leapfrog: eval episode length must be positive, got %d", cfg.EvalEpisodeLen)
	}
	return nil
}

// Result is the outcome of one leapfrog round.
type Result struct {
	// StudentWon is true if the Student's mean evaluation reward
	// strictly exceeded the Teacher's (see docs/plans/03's "Win
	// condition" — no tie-breaking margin).
	StudentWon bool
	// TeacherReward and StudentReward are each side's mean reward
	// across Config.EvalEpisodes evaluation episodes.
	TeacherReward float32
	StudentReward float32
	// StudentParams is the Student's trained params. Only set when
	// StudentWon is true — a losing Student's params are discarded
	// rather than handed back, matching pkg/lineage's own convention of
	// never recording a generation that didn't win.
	StudentParams *actorcritic.Params
}

// Round runs one leapfrog round against envs: clones teacherParams into
// a Student (actorcritic.Params.Snapshot — teacherParams itself is
// never mutated), continues training the clone for
// cfg.EpochsPerGeneration PPO epochs against envs (concurrently across
// all of them if len(envs) > 1 — see trainStudent), then evaluates both
// Teacher and Student — Teacher first, Student second, both against
// fresh envs[0].Reset episodes — over cfg.EvalEpisodes episodes each.
// Evaluation deliberately stays sequential against envs[0] only even
// when len(envs) > 1: cfg.EvalEpisodes is small relative to training
// rollout volume (see docs/plans/08-parallel-environments-and-scaling.md's
// own numbers), so parallelizing it isn't worth the added complexity
// this round.
//
// Evaluation picks each step's highest-probability action (via
// Actor.ActWithInfo) rather than sampling stochastically, matching
// rsi-with-rl.md section 2's "Teacher: low exploration" for both sides
// during comparison — only Student *training* (via ppo.Trainer's own
// rollout collection) explores; evaluation should measure each side's
// best judgment, not exploration noise.
func Round(ctx context.Context, envs []rl.Environment, teacherParams *actorcritic.Params, cfg Config, rng *rand.Rand) (Result, error) {
	if len(envs) == 0 {
		return Result{}, fmt.Errorf("leapfrog: at least one environment is required")
	}
	if teacherParams == nil {
		return Result{}, fmt.Errorf("leapfrog: teacher params must not be nil")
	}
	if err := cfg.Validate(); err != nil {
		return Result{}, err
	}

	studentParams, err := trainStudent(ctx, envs, teacherParams, cfg)
	if err != nil {
		return Result{}, fmt.Errorf("leapfrog: training student: %w", err)
	}

	if cfg.OnBeforeEvaluation != nil {
		cfg.OnBeforeEvaluation(EvalTeacher)
	}
	teacherReward, err := evaluate(ctx, envs[0], teacherParams, cfg.EvalEpisodes, cfg.EvalEpisodeLen, rng)
	if err != nil {
		return Result{}, fmt.Errorf("leapfrog: evaluating teacher: %w", err)
	}

	if cfg.OnBeforeEvaluation != nil {
		cfg.OnBeforeEvaluation(EvalStudent)
	}
	studentReward, err := evaluate(ctx, envs[0], studentParams, cfg.EvalEpisodes, cfg.EvalEpisodeLen, rng)
	if err != nil {
		return Result{}, fmt.Errorf("leapfrog: evaluating student: %w", err)
	}

	return decideWinner(teacherReward, studentReward, studentParams), nil
}

// trainStudent clones teacherParams and trains the clone for
// cfg.EpochsPerGeneration PPO epochs against envs. A single environment
// (len(envs) == 1) is driven exactly as before this package supported
// more than one — ppo.NewWithPersistentEnv, sequential rollout
// collection — so single-environment callers see zero behavioral
// change. Multiple environments (len(envs) > 1) are driven concurrently
// via ppo.NewWithPersistentEnvPool, one real bot session per goroutine
// for the pool's entire lifetime (see that constructor's own doc
// comment for why concurrency is bounded to len(envs) rather than
// cfg.Trainer.Workers). Either way, every env is reused as a persistent
// environment (Reset between episodes, never rebuilt), matching
// rlenv.Environment's own "long-lived" contract — a live bot session is
// far too expensive to reconnect per episode.
func trainStudent(ctx context.Context, envs []rl.Environment, teacherParams *actorcritic.Params, cfg Config) (*actorcritic.Params, error) {
	studentParams := teacherParams.Snapshot()
	startEpoch := 0
	if cfg.ResumeStudent != nil {
		studentParams = cfg.ResumeStudent
		startEpoch = cfg.ResumeStudentStartEpoch
	}

	var trainer *ppo.Trainer
	var err error
	if len(envs) == 1 {
		env := envs[0]
		persistentFactory := func(*rand.Rand) (rl.Environment, error) {
			return env, nil
		}
		trainer, err = ppo.NewWithPersistentEnv(cfg.Trainer, persistentFactory, studentParams)
	} else {
		trainer, err = ppo.NewWithPersistentEnvPool(cfg.Trainer, envs, studentParams)
	}
	if err != nil {
		return nil, fmt.Errorf("constructing trainer: %w", err)
	}

	for i := range cfg.EpochsPerGeneration {
		epoch := startEpoch + i
		stats, err := trainer.RunEpoch(ctx, epoch)
		if err != nil {
			return nil, fmt.Errorf("epoch %d: %w", epoch, err)
		}
		if cfg.OnEpoch != nil {
			cfg.OnEpoch(stats, trainer.Params())
		}
	}

	return trainer.Params(), nil
}

// evaluate runs episodes episodes of at most episodeLen steps each
// against env using params, greedily (see Round's doc comment), and
// returns the mean total reward per episode.
func evaluate(ctx context.Context, env rl.Environment, params *actorcritic.Params, episodes, episodeLen int, rng *rand.Rand) (float32, error) {
	actor, err := actorcritic.NewActor(params)
	if err != nil {
		return 0, fmt.Errorf("building actor: %w", err)
	}

	var totalReward float32
	for range episodes {
		reward, err := evaluateEpisode(ctx, env, actor, episodeLen, rng)
		if err != nil {
			return 0, err
		}
		totalReward += reward
	}
	return totalReward / float32(episodes), nil
}

// evaluateEpisode runs one evaluation episode and returns its total
// reward, mirroring cRL-go/pkg/reinforce's own collectTrajectoryFromEnv
// mask-lookup pattern (a type assertion to rl.ActionMasker, nil mask if
// env doesn't implement it) so this package behaves identically to
// training rollouts with respect to action masking.
func evaluateEpisode(ctx context.Context, env rl.Environment, actor *actorcritic.Actor, episodeLen int, rng *rand.Rand) (float32, error) {
	observation, err := env.Reset(ctx)
	if err != nil {
		return 0, fmt.Errorf("resetting environment: %w", err)
	}

	var totalReward float32
	for range episodeLen {
		var mask []bool
		if masker, ok := env.(rl.ActionMasker); ok {
			mask = masker.ActionMask()
		}

		decision, err := actor.ActWithInfo(observation, mask, rng)
		if err != nil {
			return 0, fmt.Errorf("deciding action: %w", err)
		}

		result, err := env.Step(ctx, greedyAction(decision.Probabilities))
		if err != nil {
			return 0, fmt.Errorf("stepping environment: %w", err)
		}
		totalReward += result.Reward
		observation = result.Observation
		if result.Done {
			break
		}
	}
	return totalReward, nil
}

// greedyAction returns the index of the highest probability in probs,
// breaking ties toward the lowest index. Used instead of sampling so
// evaluation compares each side's best judgment rather than exploration
// noise (see Round's doc comment).
func greedyAction(probs []float32) rl.Action {
	best := 0
	for i, p := range probs {
		if p > probs[best] {
			best = i
		}
	}
	return rl.Action(best)
}

// decideWinner applies Round's win condition (studentReward strictly
// greater than teacherReward, no tie-breaking margin — see
// docs/plans/03's "Win condition") and omits studentParams from the
// returned Result when the Student didn't win.
func decideWinner(teacherReward, studentReward float32, studentParams *actorcritic.Params) Result {
	result := Result{TeacherReward: teacherReward, StudentReward: studentReward}
	if studentReward > teacherReward {
		result.StudentWon = true
		result.StudentParams = studentParams
	}
	return result
}
