// Package leapfrog implements the single-task Teacher/Student leapfrog
// evaluation loop described in
// docs/plans/03-single-task-leapfrog-evaluation-loop.md: clone a Teacher
// into a Student, keep training the Student, then compare both against
// the same rl.Environment and report whether the Student won.
//
// Deliberately narrow for v1 (see that document's "Why sequential, not
// concurrent"): Round drives exactly one rl.Environment instance,
// evaluating Teacher and Student one after another against fresh
// episodes of it rather than via two concurrent bot sessions, so it
// doesn't need mc-agent's two outstanding concurrency bugs
// (docs/plans/05-mc-agent-concurrency-fixes.md) fixed first.
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
}

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

// Round runs one leapfrog round against env: clones teacherParams into a
// Student (actorcritic.Params.Snapshot — teacherParams itself is never
// mutated), continues training the clone for cfg.EpochsPerGeneration PPO
// epochs against env, then evaluates both Teacher and Student — Teacher
// first, Student second, both against fresh env.Reset episodes of the
// same env instance — over cfg.EvalEpisodes episodes each.
//
// Evaluation picks each step's highest-probability action (via
// Actor.ActWithInfo) rather than sampling stochastically, matching
// rsi-with-rl.md section 2's "Teacher: low exploration" for both sides
// during comparison — only Student *training* (via ppo.Trainer's own
// rollout collection) explores; evaluation should measure each side's
// best judgment, not exploration noise.
func Round(ctx context.Context, env rl.Environment, teacherParams *actorcritic.Params, cfg Config, rng *rand.Rand) (Result, error) {
	if teacherParams == nil {
		return Result{}, fmt.Errorf("leapfrog: teacher params must not be nil")
	}
	if err := cfg.Validate(); err != nil {
		return Result{}, err
	}

	studentParams, err := trainStudent(ctx, env, teacherParams, cfg)
	if err != nil {
		return Result{}, fmt.Errorf("leapfrog: training student: %w", err)
	}

	teacherReward, err := evaluate(ctx, env, teacherParams, cfg.EvalEpisodes, cfg.EvalEpisodeLen, rng)
	if err != nil {
		return Result{}, fmt.Errorf("leapfrog: evaluating teacher: %w", err)
	}

	studentReward, err := evaluate(ctx, env, studentParams, cfg.EvalEpisodes, cfg.EvalEpisodeLen, rng)
	if err != nil {
		return Result{}, fmt.Errorf("leapfrog: evaluating student: %w", err)
	}

	return decideWinner(teacherReward, studentReward, studentParams), nil
}

// trainStudent clones teacherParams and trains the clone for
// cfg.EpochsPerGeneration PPO epochs against env, reused as a persistent
// environment (ppo.NewWithPersistentEnv) rather than rebuilt per
// episode, matching rlenv.Environment's own "long-lived, Reset between
// episodes" contract (a live bot session is far too expensive to
// reconnect per episode).
func trainStudent(ctx context.Context, env rl.Environment, teacherParams *actorcritic.Params, cfg Config) (*actorcritic.Params, error) {
	studentParams := teacherParams.Snapshot()

	persistentFactory := func(*rand.Rand) (rl.Environment, error) {
		return env, nil
	}

	trainer, err := ppo.NewWithPersistentEnv(cfg.Trainer, persistentFactory, studentParams)
	if err != nil {
		return nil, fmt.Errorf("constructing trainer: %w", err)
	}

	for epoch := range cfg.EpochsPerGeneration {
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
