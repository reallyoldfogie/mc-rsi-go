package competition

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Session is the live-independent contract implemented by one competitor's
// trainer/environment adapter. The runner owns concurrency and deadlines;
// the session owns model-specific plain/recurrent training details.
type Session interface {
	Identity() Identity
	Train(context.Context) (TrainingSummary, error)
	Evaluate(context.Context, EvaluationPlan) (EvaluationSummary, error)
	Close() error
}

type TrainingSummary struct {
	Epochs          int     `json:"epochs"`
	Rounds          int     `json:"rounds"`
	Samples         int     `json:"samples"`
	GradientUpdates int     `json:"gradient_updates"`
	FinalCheckpoint string  `json:"final_checkpoint"`
	BestReturn      float64 `json:"best_return"`
}

type EvaluationSummary struct {
	Seed     uint64    `json:"seed,omitempty"`
	Episodes int       `json:"episodes"`
	// EpisodeRewards, EpisodeTasks, and EpisodeTerminationReasons are
	// parallel, index-aligned per-episode records: episode i's reward,
	// task label, and termination reason. Task/TerminationReason let a
	// result explain *why* a reward happened (e.g. "arrived" vs "timeout")
	// without reverse-engineering it from the reward's magnitude alone -
	// see docs/plans/21-training-competition-framework.md.
	EpisodeRewards            []float64 `json:"episode_rewards,omitempty"`
	EpisodeTasks              []string  `json:"episode_tasks,omitempty"`
	EpisodeTerminationReasons []string  `json:"episode_termination_reasons,omitempty"`
	MeanReward                float64   `json:"mean_reward"`
	RewardStdDev              float64   `json:"reward_stddev"`
	ConfidenceLevel           float64   `json:"confidence_level"`
	ConfidenceLower           float64   `json:"confidence_lower"`
	ConfidenceUpper           float64   `json:"confidence_upper"`
	CompletionRate            float64   `json:"completion_rate"`
	Error                     string    `json:"error,omitempty"`
}

type EvaluationRunResult struct {
	Seed        uint64                       `json:"seed"`
	PlanHash    string                       `json:"evaluation_plan_hash"`
	Competitors []EvaluationCompetitorResult `json:"competitors"`
}

type EvaluationCompetitorResult struct {
	Identity   Identity          `json:"identity"`
	Evaluation EvaluationSummary `json:"evaluation"`
	Status     string            `json:"status"`
	Error      string            `json:"error,omitempty"`
}

type CompetitorResult struct {
	Identity    Identity            `json:"identity"`
	Training    TrainingSummary     `json:"training"`
	Evaluation  EvaluationSummary   `json:"evaluation"`
	Evaluations []EvaluationSummary `json:"evaluations,omitempty"`
	Status      string              `json:"status"`
	Error       string              `json:"error,omitempty"`
}

type CompetitionResult struct {
	Window         TrainingWindow        `json:"training_window"`
	StartedAt      time.Time             `json:"started_at"`
	FinishedAt     time.Time             `json:"finished_at"`
	PlanHash       string                `json:"evaluation_plan_hash"`
	PlanHashes     []string              `json:"evaluation_plan_hashes,omitempty"`
	EvaluationRuns []EvaluationRunResult `json:"evaluation_runs,omitempty"`
	Competitors    []CompetitorResult    `json:"competitors"`
}

// Run starts all sessions from one barrier, gives them one absolute training
// deadline, then evaluates successful sessions against the same immutable
// plan, itself bounded by evaluationTimeLimit (see
// TrainingConfig.EvaluationTimeLimit - the caller resolves the heuristic
// once and passes the resulting duration here, rather than Run reaching
// back into Config). A session whose Train returns context.DeadlineExceeded
// is considered a normal fixed-budget completion; other errors are recorded
// as failures. A session whose Evaluate instead runs past
// evaluationTimeLimit is recorded as "evaluation_time_limit_exceeded", not
// a failure, keeping whatever plans it did complete rather than discarding
// them.
func Run(ctx context.Context, sessions []Session, window TrainingWindow, plans []EvaluationPlan, evaluationTimeLimit time.Duration) (CompetitionResult, error) {
	if len(sessions) == 0 {
		return CompetitionResult{}, fmt.Errorf("competition: at least one session is required")
	}
	if len(plans) == 0 {
		return CompetitionResult{}, fmt.Errorf("competition: at least one evaluation plan is required")
	}
	planHashes := make([]string, len(plans))
	for i, plan := range plans {
		if err := plan.Validate(); err != nil {
			return CompetitionResult{}, err
		}
		hash, err := plan.Hash()
		if err != nil {
			return CompetitionResult{}, fmt.Errorf("hashing evaluation plan %d: %w", i, err)
		}
		planHashes[i] = hash
	}
	if window.StartedAt.IsZero() || window.Deadline.IsZero() || !window.Deadline.After(window.StartedAt) {
		return CompetitionResult{}, fmt.Errorf("competition: invalid training window")
	}
	if evaluationTimeLimit <= 0 {
		return CompetitionResult{}, fmt.Errorf("competition: evaluation time limit must be positive")
	}
	result := CompetitionResult{Window: window, StartedAt: window.StartedAt, PlanHash: planHashes[0], PlanHashes: planHashes, Competitors: make([]CompetitorResult, len(sessions))}
	start := make(chan struct{})
	trainCtx, cancel, err := window.Context(ctx)
	if err != nil {
		return CompetitionResult{}, err
	}
	defer cancel()

	var wg sync.WaitGroup
	trainErrs := make([]error, len(sessions))
	for i, session := range sessions {
		result.Competitors[i].Identity = session.Identity()
		result.Competitors[i].Status = "training"
		wg.Add(1)
		go func(index int, session Session) {
			defer wg.Done()
			<-start
			summary, err := session.Train(trainCtx)
			trainErrs[index] = err
			result.Competitors[index].Training = summary
		}(i, session)
	}
	close(start)
	wg.Wait()

	// Training is complete for every session before any evaluation starts. This
	// prevents a fast architecture from evaluating while a slower one is still
	// mutating its world or final checkpoint.
	//
	// Evaluation is parallelized across competitors, but each Session still
	// owns exactly one environment, so its own episodes remain sequential.
	// This preserves the one-agent-per-area isolation while avoiding an
	// unnecessary wall-clock delay between independent competitors.
	evalCtx, evalCancel := context.WithTimeout(ctx, evaluationTimeLimit)
	defer evalCancel()

	var evaluationWG sync.WaitGroup
	evaluationErrs := make([][]error, len(sessions))
	evaluationSummaries := make([][]EvaluationSummary, len(sessions))
	for i, session := range sessions {
		if err := trainErrs[i]; err != nil && !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled) {
			result.Competitors[i].Status = "training_failed"
			result.Competitors[i].Error = err.Error()
			continue
		}
		result.Competitors[i].Status = "evaluating"
		evaluationWG.Add(1)
		go func(index int, session Session) {
			defer evaluationWG.Done()
			evaluationErrs[index] = make([]error, len(plans))
			for planIndex, plan := range plans {
				summary, evalErr := session.Evaluate(evalCtx, plan)
				if evalErr != nil {
					evaluationErrs[index][planIndex] = evalErr
					break
				}
				summary.Seed = plan.Seed
				evaluationSummaries[index] = append(evaluationSummaries[index], summary)
			}
		}(i, session)
	}
	evaluationWG.Wait()
	for i := range sessions {
		if result.Competitors[i].Status != "evaluating" {
			continue
		}
		result.Competitors[i].Evaluations = evaluationSummaries[i]
		result.Competitors[i].Evaluation = AggregateEvaluationSummaries(evaluationSummaries[i])
		if len(evaluationSummaries[i]) != len(plans) {
			result.Competitors[i].Status = "evaluation_failed"
			for _, evalErr := range evaluationErrs[i] {
				if evalErr != nil {
					if errors.Is(evalErr, context.DeadlineExceeded) || errors.Is(evalErr, context.Canceled) {
						result.Competitors[i].Status = "evaluation_time_limit_exceeded"
					}
					result.Competitors[i].Error = evalErr.Error()
					break
				}
			}
		} else {
			result.Competitors[i].Status = "complete"
		}
	}
	result.EvaluationRuns = make([]EvaluationRunResult, len(plans))
	for planIndex, plan := range plans {
		run := EvaluationRunResult{Seed: plan.Seed, PlanHash: planHashes[planIndex], Competitors: make([]EvaluationCompetitorResult, len(sessions))}
		for i, session := range sessions {
			run.Competitors[i].Identity = session.Identity()
			if planIndex < len(evaluationSummaries[i]) {
				run.Competitors[i].Status = "complete"
				run.Competitors[i].Evaluation = evaluationSummaries[i][planIndex]
			} else {
				run.Competitors[i].Status = "evaluation_failed"
				if i < len(evaluationErrs) && planIndex < len(evaluationErrs[i]) && evaluationErrs[i][planIndex] != nil {
					evalErr := evaluationErrs[i][planIndex]
					if errors.Is(evalErr, context.DeadlineExceeded) || errors.Is(evalErr, context.Canceled) {
						run.Competitors[i].Status = "evaluation_time_limit_exceeded"
					}
					run.Competitors[i].Error = evalErr.Error()
				}
			}
		}
		result.EvaluationRuns[planIndex] = run
	}

	var closeErr error
	for _, session := range sessions {
		if err := session.Close(); err != nil && closeErr == nil {
			closeErr = err
		}
	}
	result.FinishedAt = time.Now()
	if closeErr != nil {
		return result, fmt.Errorf("closing competition sessions: %w", closeErr)
	}
	return result, nil
}
