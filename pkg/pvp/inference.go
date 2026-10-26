package pvp

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/leapfrog"
)

// InferenceSession owns one managed competition-mode participant's frozen
// policy: no training, no checkpointing, just PlayGame reused once per
// best-of-N game. Built on leapfrog.Evaluate's own episode-evaluation
// loop, which has no dependency on leapfrog's teacher/student lineage
// concept at all when used purely for evaluation - see
// docs/plans/20-combat-pvp-competition.md. Unlike NewParticipantSession
// (Phase 1 training), there is no fresh-init fallback: a competition
// entry with no existing checkpoint is a configuration error, not
// something to paper over by competing with an untrained network.
type InferenceSession struct {
	ParticipantID string
	EnvironmentID string
	Greedy        bool

	env             *MatchEnvironment
	recurrent       bool
	plainParams     *actorcritic.Params
	recurrentParams *actorcritic.RecurrentParams
	rng             *rand.Rand
}

// NewInferenceSession loads a frozen checkpoint (hard error on an
// environment-ID mismatch, same as NewParticipantSession) and wraps env
// (already a MatchEnvironment, built the same way a Phase 1 training
// session's environment is). greedy selects deterministic (argmax) vs.
// sampled action selection - COMBAT_COMPETITION_PLAN.md doesn't mandate
// one; a competition result is more reproducible with greedy on, but the
// choice is left to the caller.
func NewInferenceSession(participantID string, p Participant, checkpointPath string, greedy bool, env *MatchEnvironment, seed uint64) (*InferenceSession, error) {
	environmentID := EnvironmentID(p, env.ObservationSize(), env.ActionSpace())
	s := &InferenceSession{
		ParticipantID: participantID,
		EnvironmentID: environmentID,
		Greedy:        greedy,
		env:           env,
		recurrent:     p.Recurrent,
		rng:           rand.New(rand.NewPCG(seed, 0)),
	}
	if p.Recurrent {
		params, _, err := actorcritic.RecurrentLoadFile(checkpointPath, environmentID)
		if err != nil {
			return nil, fmt.Errorf("pvp: loading recurrent checkpoint for %s: %w", participantID, err)
		}
		s.recurrentParams = params
		return s, nil
	}
	params, _, err := actorcritic.LoadFile(checkpointPath, environmentID)
	if err != nil {
		return nil, fmt.Errorf("pvp: loading checkpoint for %s: %w", participantID, err)
	}
	s.plainParams = params
	return s, nil
}

// PlayGame plays exactly one game (episode) to completion - Done=true, or
// episodeLen steps elapse without it - and returns this participant's own
// health at the end, for the caller to feed into ComputeGameOutcome. The
// match's synchronized barrier (inside env) is what actually makes every
// participant's "one game" line up to the same round, the same way a
// training ParticipantSession's own Reset/Step calls do; this session has
// no awareness of synchronization itself.
func (s *InferenceSession) PlayGame(ctx context.Context, episodeLen int) (health float64, err error) {
	envs := []rl.Environment{s.env}
	if s.recurrent {
		if _, err := leapfrog.Evaluate(ctx, envs, s.recurrentParams, 1, episodeLen, s.Greedy, leapfrog.RecurrentSeams, s.rng); err != nil {
			return 0, fmt.Errorf("pvp: playing game for %s: %w", s.ParticipantID, err)
		}
	} else {
		if _, err := leapfrog.Evaluate(ctx, envs, s.plainParams, 1, episodeLen, s.Greedy, leapfrog.PlainSeams, s.rng); err != nil {
			return 0, fmt.Errorf("pvp: playing game for %s: %w", s.ParticipantID, err)
		}
	}
	return HealthFromObservation(s.env.LastObservation()), nil
}
