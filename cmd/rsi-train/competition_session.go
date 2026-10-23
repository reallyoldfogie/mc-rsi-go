package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	rand "math/rand/v2"
	"sync"
	"time"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	crlconfig "github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/ppo"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"

	competitioncfg "github.com/reallyoldfogie/mc-rsi-trainer/pkg/competition"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/leapfrog"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/lineage"
)

var errCompetitionAgentConnectionLost = errors.New("competition agent connection lost")

// liveCompetitionSession adapts one existing mc-agent environment to the
// package-level competition.Session contract. It deliberately reuses the
// same leapfrog round mechanics as rsi-train; the competition runner controls
// the outer deadline and final evaluation barrier.
type liveCompetitionSession struct {
	identity      competitioncfg.Identity
	competitor    competitioncfg.Competitor
	connected     connectedEnvironment
	env           rl.Environment
	settings      crlconfig.Settings
	checkpointDir string
	environmentID string

	epochsPerGeneration int
	evalEpisodeLen      int
	evalGreedy          bool
	metrics             *competitioncfg.CompetitorMetrics

	mu             sync.Mutex
	reconnectMu    sync.Mutex
	instrumented   *instrumentedEnvironment
	plainFinal     *actorcritic.Params
	recurrentFinal *actorcritic.RecurrentParams
	closed         bool
	closeOnce      sync.Once
}

func (s *liveCompetitionSession) Identity() competitioncfg.Identity { return s.identity }

func (s *liveCompetitionSession) Train(ctx context.Context) (competitioncfg.TrainingSummary, error) {
	sessionCtx, cancel := s.agentLifecycleContext(ctx)
	defer cancel()
	var summary competitioncfg.TrainingSummary
	var err error
	if s.competitor.Recurrent {
		summary, err = s.trainRecurrent(sessionCtx)
	} else {
		summary, err = s.trainPlain(sessionCtx)
	}
	err = s.classifyLifecycleError(ctx, err)
	if s.metrics != nil {
		normalDeadline := err == nil || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled)
		s.metrics.SetTrainingComplete(normalDeadline)
		s.metrics.SetTrainingFailed(!normalDeadline)
	}
	return summary, err
}

func (s *liveCompetitionSession) agentLifecycleContext(parent context.Context) (context.Context, context.CancelFunc) {
	// A dropped socket must not cancel the competitor's entire training
	// context: competition sessions can reconnect and resume before the shared
	// deadline. The rollout operation itself returns the socket error, which
	// the training/evaluation loops handle below.
	return context.WithCancel(parent)
}

const (
	competitionReconnectAttempts = 5
	competitionReconnectBackoff  = 2 * time.Second
)

func (s *liveCompetitionSession) reconnectIfNeeded(ctx context.Context) error {
	s.reconnectMu.Lock()
	defer s.reconnectMu.Unlock()
	if ctx.Err() != nil {
		return ctx.Err()
	}
	select {
	case <-s.connected.agent.Done():
	default:
		return nil
	}
	if s.connected.reconnect == nil {
		return errCompetitionAgentConnectionLost
	}
	var lastErr error
	for attempt := 1; attempt <= competitionReconnectAttempts; attempt++ {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		fresh, err := s.connected.reconnect(ctx)
		if err == nil {
			old := s.connected.agent
			if old != nil {
				closeConnectedAgent(old)
			}
			s.connected.agent = fresh.agent
			s.connected.taskForEpisode = fresh.taskForEpisode
			s.connected.goalForEpisode = fresh.goalForEpisode
			s.connected.targetOffsetForEpisode = fresh.targetOffsetForEpisode
			s.connected.reseedEvalTasks = fresh.reseedEvalTasks
			s.connected.setEvaluationPlan = fresh.setEvaluationPlan
			if s.instrumented != nil {
				s.instrumented.replaceAgentEnvironment(fresh.env, healthProviderForAgent(fresh.agent), fresh.taskForEpisode, fresh.goalForEpisode, fresh.targetOffsetForEpisode)
				s.connected.env = s.instrumented
			} else {
				s.connected.env = fresh.env
				s.env = fresh.env
			}
			if s.connected.updateDashboardAgent != nil {
				s.connected.updateDashboardAgent(fresh.agent)
			}
			log.Printf("competition competitor %s reconnected on attempt %d", s.identity.CompetitorID, attempt)
			return nil
		}
		lastErr = err
		log.Printf("competition competitor %s reconnect attempt %d/%d failed: %v", s.identity.CompetitorID, attempt, competitionReconnectAttempts, err)
		if attempt < competitionReconnectAttempts {
			timer := time.NewTimer(competitionReconnectBackoff)
			select {
			case <-ctx.Done():
				timer.Stop()
				return ctx.Err()
			case <-timer.C:
			}
		}
	}
	return fmt.Errorf("%w: reconnect attempts exhausted: %v", errCompetitionAgentConnectionLost, lastErr)
}

func (s *liveCompetitionSession) reconnectAfterError(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return err
	}
	select {
	case <-s.connected.agent.Done():
		if reconnectErr := s.reconnectIfNeeded(ctx); reconnectErr != nil {
			return reconnectErr
		}
		return nil
	default:
		return err
	}
}

func (s *liveCompetitionSession) classifyLifecycleError(parent context.Context, err error) error {
	if parent.Err() != nil {
		return err
	}
	select {
	case <-s.connected.agent.Done():
		return fmt.Errorf("competitor %s: %w", s.identity.CompetitorID, errCompetitionAgentConnectionLost)
	default:
		return err
	}
}

func (s *liveCompetitionSession) trainPlain(ctx context.Context) (competitioncfg.TrainingSummary, error) {
	teacher, record, err := loadOrInitTeacher(s.checkpointDir, s.environmentID, s.env.ObservationSize(), s.settings.HiddenSize, s.env.ActionSpace(), s.competitor.HiddenLayers)
	if err != nil {
		return competitioncfg.TrainingSummary{}, err
	}
	latest := teacher
	var summary competitioncfg.TrainingSummary
	cfg := leapfrog.Config[*actorcritic.Params]{
		Trainer:             s.settings,
		EpochsPerGeneration: s.epochsPerGeneration,
		EvalEpisodes:        1,
		EvalEpisodeLen:      s.evalEpisodeLen,
		OnEpoch: func(stats ppo.EpochStats, params *actorcritic.Params) {
			if s.metrics != nil {
				s.metrics.EpochCompleted(stats.SampleCount, stats.UpdateCount)
			}
			s.mu.Lock()
			latest = params
			summary.Epochs++
			summary.Samples += stats.SampleCount
			summary.GradientUpdates += stats.UpdateCount
			s.mu.Unlock()
		},
	}
	if s.connected.reseedEvalTasks != nil {
		cfg.OnBeforeEvaluation = pairedEvalTaskReseeder([]func(uint64, uint64){s.connected.reseedEvalTasks})
	}
	for ctx.Err() == nil {
		result, roundErr := leapfrog.Round(ctx, []rl.Environment{s.env}, teacher, cfg, leapfrog.PlainSeams, rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), uint64(summary.Rounds))))
		if roundErr != nil {
			if reconnectErr := s.reconnectAfterError(ctx, roundErr); reconnectErr == nil {
				continue
			} else if reconnectErr != roundErr {
				return summary, reconnectErr
			}
			s.mu.Lock()
			s.plainFinal = latest
			s.mu.Unlock()
			return summary, roundErr
		}
		summary.Rounds++
		if result.StudentWon {
			teacher = result.StudentParams
			record = nextCompetitionRecord(record, s.environmentID, false)
			if err := lineage.Save(s.checkpointDir, teacher, record, lineage.PlainCodec); err != nil {
				return summary, fmt.Errorf("saving competitor %s generation %d: %w", s.identity.CompetitorID, record.Generation, err)
			}
		}
	}
	s.mu.Lock()
	s.plainFinal = latest
	s.mu.Unlock()
	return summary, ctx.Err()
}

func (s *liveCompetitionSession) trainRecurrent(ctx context.Context) (competitioncfg.TrainingSummary, error) {
	teacher, record, err := loadOrInitTeacherRecurrent(s.checkpointDir, s.environmentID, s.env.ObservationSize(), s.settings.HiddenSize, s.env.ActionSpace(), s.competitor.HiddenLayers)
	if err != nil {
		return competitioncfg.TrainingSummary{}, err
	}
	latest := teacher
	var summary competitioncfg.TrainingSummary
	cfg := leapfrog.Config[*actorcritic.RecurrentParams]{
		Trainer:             s.settings,
		EpochsPerGeneration: s.epochsPerGeneration,
		EvalEpisodes:        1,
		EvalEpisodeLen:      s.evalEpisodeLen,
		OnEpoch: func(stats ppo.EpochStats, params *actorcritic.RecurrentParams) {
			if s.metrics != nil {
				s.metrics.EpochCompleted(stats.SampleCount, stats.UpdateCount)
			}
			s.mu.Lock()
			latest = params
			summary.Epochs++
			summary.Samples += stats.SampleCount
			summary.GradientUpdates += stats.UpdateCount
			s.mu.Unlock()
		},
	}
	if s.connected.reseedEvalTasks != nil {
		cfg.OnBeforeEvaluation = pairedEvalTaskReseeder([]func(uint64, uint64){s.connected.reseedEvalTasks})
	}
	for ctx.Err() == nil {
		result, roundErr := leapfrog.Round(ctx, []rl.Environment{s.env}, teacher, cfg, leapfrog.RecurrentSeams, rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), uint64(summary.Rounds))))
		if roundErr != nil {
			if reconnectErr := s.reconnectAfterError(ctx, roundErr); reconnectErr == nil {
				continue
			} else if reconnectErr != roundErr {
				return summary, reconnectErr
			}
			s.mu.Lock()
			s.recurrentFinal = latest
			s.mu.Unlock()
			return summary, roundErr
		}
		summary.Rounds++
		if result.StudentWon {
			teacher = result.StudentParams
			record = nextCompetitionRecord(record, s.environmentID, true)
			if err := lineage.Save(s.checkpointDir, teacher, record, lineage.RecurrentCodec); err != nil {
				return summary, fmt.Errorf("saving competitor %s generation %d: %w", s.identity.CompetitorID, record.Generation, err)
			}
		}
	}
	s.mu.Lock()
	s.recurrentFinal = latest
	s.mu.Unlock()
	return summary, ctx.Err()
}

func nextCompetitionRecord(previous lineage.Record, environmentID string, recurrent bool) lineage.Record {
	return lineage.Record{
		Generation: previous.Generation + 1, ParentGeneration: previous.Generation,
		Provenance: lineage.ProvenanceLeapfrog, SeededFromGeneration: -1,
		CreatedAt: time.Now(), EnvironmentID: environmentID, Metadata: previous.Metadata,
		Recurrent: recurrent,
	}
}

func (s *liveCompetitionSession) Evaluate(ctx context.Context, plan competitioncfg.EvaluationPlan) (competitioncfg.EvaluationSummary, error) {
	sessionCtx, cancel := s.agentLifecycleContext(ctx)
	defer cancel()
	if s.connected.setEvaluationPlan == nil {
		return competitioncfg.EvaluationSummary{}, fmt.Errorf("competitor %s has no curriculum evaluation selector", s.identity.CompetitorID)
	}
	s.mu.Lock()
	plain, recurrent := s.plainFinal, s.recurrentFinal
	s.mu.Unlock()
	if plain == nil && recurrent == nil {
		return competitioncfg.EvaluationSummary{}, fmt.Errorf("competitor %s has no final policy", s.identity.CompetitorID)
	}
	s.connected.setEvaluationPlan(&plan)
	defer s.connected.setEvaluationPlan(nil)
	rng := rand.New(rand.NewPCG(plan.Seed, uint64(s.identity.AreaIndex)))
	var results [][]leapfrog.EpisodeResult
	var err error
	for {
		if s.instrumented != nil {
			s.instrumented.beginCompetitionEvaluation()
		}
		s.connected.setEvaluationPlan(&plan)
		rng = rand.New(rand.NewPCG(plan.Seed, uint64(s.identity.AreaIndex)))
		if s.competitor.Recurrent {
			results, err = leapfrog.Evaluate(sessionCtx, []rl.Environment{s.env}, recurrent, len(plan.Scenarios), s.evalEpisodeLen, s.evalGreedy, leapfrog.RecurrentSeams, rng)
		} else {
			results, err = leapfrog.Evaluate(sessionCtx, []rl.Environment{s.env}, plain, len(plan.Scenarios), s.evalEpisodeLen, s.evalGreedy, leapfrog.PlainSeams, rng)
		}
		if err == nil {
			break
		}
		if s.instrumented != nil {
			_ = s.instrumented.endCompetitionEvaluation()
		}
		if reconnectErr := s.reconnectAfterError(ctx, err); reconnectErr == nil {
			continue
		}
		return competitioncfg.EvaluationSummary{}, s.classifyLifecycleError(ctx, err)
	}
	summary := summarizeCompetitionScores(results)
	if s.instrumented != nil {
		summary.CompletionRate = s.instrumented.endCompetitionEvaluation()
	}
	if s.metrics != nil {
		s.metrics.SetEvaluationStatistics(summary)
	}
	return summary, nil
}

func summarizeCompetitionScores(results [][]leapfrog.EpisodeResult) competitioncfg.EvaluationSummary {
	var rewards []float64
	var tasks, reasons []string
	for _, environmentResults := range results {
		for _, result := range environmentResults {
			rewards = append(rewards, float64(result.Reward))
			tasks = append(tasks, result.Task)
			reasons = append(reasons, result.TerminationReason)
		}
	}
	summary := competitioncfg.SummarizeRewards(rewards)
	summary.EpisodeTasks = tasks
	summary.EpisodeTerminationReasons = reasons
	return summary
}

func (s *liveCompetitionSession) Close() error {
	var closeErr error
	s.closeOnce.Do(func() {
		if s.connected.stopWatch != nil {
			close(s.connected.stopWatch)
		}
		closeConnectedAgent(s.connected.agent)
		s.mu.Lock()
		s.closed = true
		s.mu.Unlock()
	})
	return closeErr
}
