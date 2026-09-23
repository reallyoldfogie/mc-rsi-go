// Prometheus instrumentation for rsi-train, so a training run's progress
// is visible to anyone watching a Grafana dashboard (see
// monitoring/README.md), not only to whoever launched the process and is
// tailing its log. See docs/glossary.md's "Metrics" section for what
// each of these means in plain language, and monitoring/grafana's
// provisioned dashboard for how they're graphed.
//
// Metrics are process-global (package-level, via promauto) rather than
// threaded through run()/leapfrog.Round's own parameters — this file is
// the only one that touches them, and every other call site just calls
// the small record*/observe* helpers below, so nothing else in this
// command needs to know Prometheus exists.
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"

	"github.com/reallyoldfogie/mc-agent/rlenv"
)

var (
	metricGeneration = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_generation",
		Help: "Current Teacher generation number (increases by one each time a Student beats its Teacher and is promoted).",
	})

	metricCurrentRound = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_current_round",
		Help: "The round number currently in progress (1-indexed).",
	})

	metricCurrentEpoch = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_current_epoch",
		Help: "The Student-training epoch currently in progress within rsi_current_round (0-indexed). Reset to -1 at the start of each round, before its first epoch completes.",
	})

	metricRoundsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_rounds_total",
		Help: "Completed leapfrog rounds, by outcome.",
	}, []string{"outcome"}) // "won" (Student beat Teacher) | "lost" (Teacher held)

	metricRoundDurationSeconds = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "rsi_round_duration_seconds",
		Help:    "Wall-clock duration of each completed leapfrog round.",
		Buckets: []float64{60, 300, 600, 1200, 1800, 3600, 5400, 7200, 10800, 14400},
	})

	metricTeacherReward = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_teacher_reward",
		Help: "Teacher's mean evaluation reward in the most recently completed round.",
	})

	metricStudentReward = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_student_reward",
		Help: "Student's mean evaluation reward in the most recently completed round.",
	})

	metricEpochAvgReturn = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_epoch_avg_return",
		Help: "Average rollout return of the most recently completed Student-training epoch.",
	})

	metricEpochsTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "rsi_epochs_total",
		Help: "Total Student-training epochs completed across every round so far.",
	})

	metricEpochSamplesTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "rsi_epoch_samples_total",
		Help: "Total rollout steps (samples) consumed by Student training across every epoch so far — rate() of this is the training throughput.",
	})

	metricGradientUpdatesTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "rsi_gradient_updates_total",
		Help: "Total gradient-update steps applied during Student training.",
	})

	metricEpisodesTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_episodes_total",
		Help: "Completed training episodes, by task and outcome.",
	}, []string{"task", "outcome"}) // outcome: success | failure | truncated

	metricEpisodesStartedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_episodes_started_total",
		Help: "Training episodes successfully started, by task.",
	}, []string{"task"})

	metricEpisodeStepsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_episode_steps_total",
		Help: "Environment steps taken during training episodes, by task.",
	}, []string{"task"})

	metricResetFailuresTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_reset_failures_total",
		Help: "Environment reset failures, by task.",
	}, []string{"task"})

	metricStepErrorsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_step_errors_total",
		Help: "Environment step failures, by task.",
	}, []string{"task"})

	// metricTaskEpisodesTotal is only ever incremented when -curriculum-config
	// is set (see recordTaskSelection) — without a curriculum, rsi-train has
	// no hook into per-episode task selection at all (Reset is called deep
	// inside cRL-go's own rollout/evaluate loops, not by this command), so
	// this metric simply never appears rather than reporting a misleading
	// single-task 100%.
	metricTaskEpisodesTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_task_episodes_total",
		Help: "Episodes posed by the curriculum, by task type. Only emitted when -curriculum-config is set.",
	}, []string{"task"}) // "goto" | "mine" | "craft"
)

// startMetricsServer serves /metrics on addr in the background. Errors
// (e.g. the port is already taken) are logged, not fatal — a metrics
// server going down is an observability problem, not a reason to stop an
// otherwise-healthy training run.
func startMetricsServer(addr string) {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	go func() {
		log.Printf("metrics: serving /metrics on %s", addr)
		if err := http.ListenAndServe(addr, mux); err != nil {
			log.Printf("metrics: server error: %v", err)
		}
	}()
}

const episodeSuccessRewardThreshold float32 = 5

// instrumentedEnvironment records episode-level health and throughput while
// preserving optional action-mask support from the wrapped environment.
type instrumentedEnvironment struct {
	env            rl.Environment
	task           string
	taskForEpisode func() string
	started        bool
}

func (e *instrumentedEnvironment) currentTask() string {
	if e.taskForEpisode != nil {
		if task := e.taskForEpisode(); task != "" {
			return task
		}
	}
	return e.task
}

func (e *instrumentedEnvironment) ObservationSize() int { return e.env.ObservationSize() }
func (e *instrumentedEnvironment) ActionSpace() int     { return e.env.ActionSpace() }

func (e *instrumentedEnvironment) ActionMask() []bool {
	if masker, ok := e.env.(rl.ActionMasker); ok {
		return masker.ActionMask()
	}
	return nil
}

// resetRetries bounds how many extra times instrumentedEnvironment.Reset
// retries after the wrapped environment's own Reset fails, before giving
// up and propagating the error. Distinct from rlenv.Environment.Reset's
// own internal retry (which retries the *same* already-selected episode
// task - useful for transient RCON/network hiccups): this retries with a
// *fresh* task draw instead, since rlenv calls Config.TaskSelector again
// on every new Reset attempt, and some failures are about that specific
// task rather than a transient glitch - retrying the identical task would
// just fail the same way again. Found live: a single bad terrain draw
// ("no walkable+reachable cell found near goto target ... within 4
// blocks") crashed the entire training process outright, taking down
// every -parallel-envs bot's progress with it - not because anything was
// actually broken, just because one of many possible random tasks
// happened to be unreachable this time.
const resetRetries = 5

func (e *instrumentedEnvironment) Reset(ctx context.Context) (rl.Observation, error) {
	task := e.currentTask()
	if e.started {
		metricEpisodesTotal.WithLabelValues(task, "truncated").Inc()
	}
	var obs rl.Observation
	var err error
	for attempt := 0; attempt <= resetRetries; attempt++ {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return obs, ctxErr
		}
		obs, err = e.env.Reset(ctx)
		if err == nil {
			break
		}
		metricResetFailuresTotal.WithLabelValues(task).Inc()
		if attempt < resetRetries {
			log.Printf("instrumentedEnvironment: Reset attempt %d/%d failed (%v), retrying with a fresh task draw", attempt+1, resetRetries+1, err)
		}
	}
	if err != nil {
		e.started = false
		return obs, fmt.Errorf("resetting environment after %d attempts: %w", resetRetries+1, err)
	}
	// TaskSelector runs inside the wrapped environment's Reset. Read the
	// selected task again after Reset for the new episode's counters.
	task = e.currentTask()
	e.started = true
	metricEpisodesStartedTotal.WithLabelValues(task).Inc()
	return obs, nil
}

func (e *instrumentedEnvironment) Step(ctx context.Context, action rl.Action) (rl.StepResult, error) {
	task := e.currentTask()
	result, err := e.env.Step(ctx, action)
	if err != nil {
		metricStepErrorsTotal.WithLabelValues(task).Inc()
		return result, err
	}
	metricEpisodeStepsTotal.WithLabelValues(task).Inc()
	if result.Done {
		outcome := "failure"
		if result.Reward >= episodeSuccessRewardThreshold {
			outcome = "success"
		}
		metricEpisodesTotal.WithLabelValues(task, outcome).Inc()
		e.started = false
	}
	return result, nil
}

// recordTaskSelection increments metricTaskEpisodesTotal for whichever
// single task override describes as active, mirroring
// rlenvadapter.TaskSelector's own "exactly one task active" invariant
// (GoToTargetDisabled defaults true; MineTargetBlock/CraftTargetItem are
// empty unless that task was chosen).
func recordTaskSelection(override rlenv.TaskOverride) {
	switch {
	case !override.GoToTargetDisabled:
		metricTaskEpisodesTotal.WithLabelValues("goto").Inc()
	case override.MineTargetBlock != "":
		metricTaskEpisodesTotal.WithLabelValues("mine").Inc()
	case override.CraftTargetItem != "":
		metricTaskEpisodesTotal.WithLabelValues("craft").Inc()
	}
}
