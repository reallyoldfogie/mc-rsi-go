package pvp

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"math/rand/v2"
	"time"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	crlconfig "github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/ppo"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// EnvironmentID is a stable architecture/environment identity string for
// checkpoint compatibility checking - mirrors
// pkg/competition.EnvironmentID's own shape, independently (see
// docs/plans/19-combat-pvp-training.md).
func EnvironmentID(p Participant, observationSize, actionSpace int) string {
	mode := "plain"
	chunk := 0
	if p.Recurrent {
		mode = "recurrent"
		chunk = p.RecurrentChunkLen
	}
	return fmt.Sprintf("pvp:layers=%d:width=%d:mode=%s:chunk=%d:obs=%d:actions=%d",
		p.HiddenLayers, p.HiddenSize, mode, chunk, observationSize, actionSpace)
}

// ParticipantSession owns one managed combatant's continuous training
// loop - an independent learner (COMBAT_COMPETITION_PLAN.md decision 1)
// training against whatever its live opponent(s) currently do. Unlike
// pkg/competition.Session, there is no separate evaluation phase: combat
// training has no shared curriculum to evaluate against, so a session is
// just "keep calling RunEpoch, checkpoint periodically" - see
// docs/plans/19-combat-pvp-training.md for why pkg/leapfrog's teacher/
// student machinery is not used here either.
type ParticipantSession struct {
	ParticipantID      string
	EnvironmentID      string
	CheckpointPath     string
	CheckpointInterval time.Duration

	recurrent        bool
	plainTrainer     *ppo.Trainer
	recurrentTrainer *ppo.RecurrentTrainer
}

// NewParticipantSession constructs one managed participant's trainer,
// loading an existing checkpoint at checkpointPath if present (and its
// EnvironmentID matches - a mismatch is a hard error, never a silent
// re-init, so a config change can't accidentally resume into an
// incompatible checkpoint), or initializing fresh params otherwise. env
// should already be wrapped in a MatchEnvironment (barrier.go) by the
// caller - this constructor has no awareness that match synchronization
// exists.
func NewParticipantSession(participantID string, p Participant, checkpointPath string, checkpointInterval time.Duration, settings crlconfig.Settings, env rl.Environment) (*ParticipantSession, error) {
	if checkpointInterval <= 0 {
		return nil, fmt.Errorf("pvp: checkpoint interval must be positive")
	}
	environmentID := EnvironmentID(p, env.ObservationSize(), env.ActionSpace())
	s := &ParticipantSession{
		ParticipantID:      participantID,
		EnvironmentID:      environmentID,
		CheckpointPath:     checkpointPath,
		CheckpointInterval: checkpointInterval,
		recurrent:          p.Recurrent,
	}

	if p.Recurrent {
		params, err := loadOrInitRecurrentParams(checkpointPath, environmentID, env.ObservationSize(), p.HiddenSize, env.ActionSpace(), p.HiddenLayers)
		if err != nil {
			return nil, err
		}
		settings.RecurrentChunkLen = p.RecurrentChunkLen
		trainer, err := ppo.NewRecurrentWithPersistentEnvPool(settings, []rl.Environment{env}, params)
		if err != nil {
			return nil, fmt.Errorf("pvp: building recurrent trainer for %s: %w", participantID, err)
		}
		s.recurrentTrainer = trainer
		return s, nil
	}

	params, err := loadOrInitParams(checkpointPath, environmentID, env.ObservationSize(), p.HiddenSize, env.ActionSpace(), p.HiddenLayers)
	if err != nil {
		return nil, err
	}
	trainer, err := ppo.NewWithPersistentEnvPool(settings, []rl.Environment{env}, params)
	if err != nil {
		return nil, fmt.Errorf("pvp: building trainer for %s: %w", participantID, err)
	}
	s.plainTrainer = trainer
	return s, nil
}

func loadOrInitParams(path, environmentID string, observationSize, hiddenSize, actionSpace, hiddenLayers int) (*actorcritic.Params, error) {
	params, _, err := actorcritic.LoadFile(path, environmentID)
	if err == nil {
		return params, nil
	}
	if !errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("pvp: loading checkpoint %s: %w", path, err)
	}
	rng := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 1))
	return actorcritic.NewParams(rng, observationSize, hiddenSize, actionSpace, hiddenLayers), nil
}

func loadOrInitRecurrentParams(path, environmentID string, observationSize, hiddenSize, actionSpace, hiddenLayers int) (*actorcritic.RecurrentParams, error) {
	params, _, err := actorcritic.RecurrentLoadFile(path, environmentID)
	if err == nil {
		return params, nil
	}
	if !errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("pvp: loading recurrent checkpoint %s: %w", path, err)
	}
	rng := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 1))
	return actorcritic.NewRecurrentParams(rng, observationSize, hiddenSize, actionSpace, hiddenLayers), nil
}

// RunEpoch runs exactly one training epoch (rollout collection + PPO
// update) and returns its stats.
func (s *ParticipantSession) RunEpoch(ctx context.Context, epoch int) (ppo.EpochStats, error) {
	if s.recurrent {
		return s.recurrentTrainer.RunEpoch(ctx, epoch)
	}
	return s.plainTrainer.RunEpoch(ctx, epoch)
}

// Checkpoint saves the session's current params to CheckpointPath,
// overwriting any existing checkpoint in place - combat training has no
// generation/promotion concept (COMBAT_COMPETITION_PLAN.md decision 1),
// so unlike mc-rsi-trainer's lineage package, there is only ever one
// live checkpoint per participant, not a history of generations.
func (s *ParticipantSession) Checkpoint(epoch int, stats ppo.EpochStats) error {
	metadata := checkpoint.Metadata{Epoch: epoch, TotalUpdates: stats.UpdateCount}
	if s.recurrent {
		return s.recurrentTrainer.Params().SaveFile(s.CheckpointPath, s.EnvironmentID, metadata)
	}
	return actorcritic.SaveFile(s.CheckpointPath, s.plainTrainer.Params(), s.EnvironmentID, metadata)
}

// Train runs epochs continuously, checkpointing every CheckpointInterval
// of wall-clock time, until ctx is done. A context.DeadlineExceeded/
// context.Canceled return is a normal, expected stop (the caller's
// shared deadline or shutdown signal), not a failure; any other error
// from RunEpoch is returned immediately. The final checkpoint is always
// saved before returning, regardless of how the loop ended, so training
// progress is never lost on a clean shutdown.
func (s *ParticipantSession) Train(ctx context.Context) error {
	epoch := 0
	lastCheckpoint := time.Now()
	var lastStats ppo.EpochStats
	for {
		stats, err := s.RunEpoch(ctx, epoch)
		if err != nil {
			_ = s.Checkpoint(epoch, lastStats)
			return err
		}
		lastStats = stats
		epoch++
		if time.Since(lastCheckpoint) >= s.CheckpointInterval {
			if err := s.Checkpoint(epoch, stats); err != nil {
				return fmt.Errorf("pvp: checkpointing %s at epoch %d: %w", s.ParticipantID, epoch, err)
			}
			lastCheckpoint = time.Now()
		}
		if ctx.Err() != nil {
			_ = s.Checkpoint(epoch, lastStats)
			return ctx.Err()
		}
	}
}
