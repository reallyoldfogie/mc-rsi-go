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
	"net/http/pprof"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"

	"github.com/reallyoldfogie/mc-agent/rlenv"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/leapfrog"
)

// evaluationInProgress is true from the first OnBeforeEvaluation call of a
// round (see markEvaluationInProgress) until that round's leapfrog.Round
// call returns. instrumentedEnvironment.Reset fires for every episode on
// the shared connected environments regardless of whether leapfrog.Round is
// currently training or evaluating, but rsi_epoch_trajectories_started is
// specifically meant to track training rollout collection toward
// rsi_epoch_rollout_size — without this gate, each round's Teacher+Student
// evaluation episodes (EvalEpisodes * len(envs) each, entirely unrelated to
// RolloutSize) also incremented it after the last training epoch's own
// reset, pushing the "epoch progress" bar past 100% during evaluation.
// Found live 2026-10-03: a round ending showed the round-progress bar at
// 100% and the epoch-progress bar at 147%.
var evaluationInProgress atomic.Bool

// markEvaluationInProgress wraps a leapfrog.Config.OnBeforeEvaluation
// callback to set evaluationInProgress before running it. The caller is
// responsible for clearing the flag once leapfrog.Round returns (it has no
// "evaluation ended" hook of its own to clear it from inside).
func markEvaluationInProgress(onBeforeEvaluation func(leapfrog.EvalSide)) func(leapfrog.EvalSide) {
	return func(side leapfrog.EvalSide) {
		evaluationInProgress.Store(true)
		if onBeforeEvaluation != nil {
			onBeforeEvaluation(side)
		}
	}
}

var (
	metricNetworkActiveUnits = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "rsi_network_active_units_fraction",
		Help: "Fraction of a policy hidden layer's ReLU units that fire for at least one probe observation, by layer (1 or 2). A dead unit never recovers; 0 in layer 2 means the network is dead.",
	}, []string{"layer"})

	metricPolicyLogitStd = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_policy_logit_std",
		Help: "Mean standard deviation of the policy's output logits across probe observations. 0 means the policy is the same in every state (a dead network).",
	})

	metricPolicyEntropy = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_policy_entropy",
		Help: "Mean Shannon entropy (nats) of softmax(logits) across probe observations - how uncertain the action distribution actually is. Falls as the policy sharpens; near 0 means it's effectively deterministic for every probe (softmax saturation).",
	})

	metricNetworkUnitCorrelation = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "rsi_network_unit_correlation",
		Help: "Mean absolute pairwise Pearson correlation between a hidden layer's unit activations across probe observations, by layer (1 or 2). A capacity-direction proxy, not a dead-unit check: high means many units behave near-identically (redundant, unused headroom - growing hidden_size would likely just add more redundancy); low means units are already doing mutually distinct work (closer to actually capacity-constrained). See docs/plans/12's 'Diagnosing a training plateau' section.",
	}, []string{"layer"})

	metricBestReturnAllTime = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_best_return_alltime",
		Help: "The best Student-training epoch average return ever seen in this run (checkpoint.Metadata.BestReturn) - persisted across restarts, rounds, AND promotions (it is carried forward in rec.Metadata, never reset). A value that stays flat across many rounds despite continued training is one signal (not proof on its own) that the architecture, not optimization, may be the ceiling - see docs/plans/12's 'Diagnosing a training plateau' section.",
	})

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

	metricEpochRolloutSize = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_epoch_rollout_size",
		Help: "Target trajectory count for the epoch currently in progress (cRL-go's config.Settings.RolloutSize) - the denominator for rsi_epoch_trajectories_started / rsi_epoch_rollout_size as an in-progress-epoch completion percentage. Set once at startup; doesn't change mid-run.",
	})

	metricEpochsPerGeneration = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_epochs_per_generation",
		Help: "How many Student-training epochs make up one round (-epochs-per-generation) - the denominator for (rsi_current_epoch + 1) / rsi_epochs_per_generation as a round-progress percentage (how far through its epoch budget the current round is, distinct from rsi_epoch_rollout_size's within-epoch progress). Set once at startup; doesn't change mid-run.",
	})

	metricEpochTrajectoriesStarted = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_epoch_trajectories_started",
		Help: "Trajectories (episodes) started so far toward the current epoch's rsi_epoch_rollout_size target. Reset to 0 whenever rsi_current_epoch advances to a new epoch - there is no mid-epoch callback in cRL-go to hook directly, so this is inferred from instrumentedEnvironment.Reset, which already fires once per trajectory in real time regardless of which cRL-go collection function is driving it.",
	})

	// Buckets are scaled for one epoch (rsi_epoch_rollout_size trajectories
	// collected, then one minibatch-Adam pass over them), not a whole round
	// like metricRoundDurationSeconds - deliberately distinct, smaller
	// buckets. Measured as wall-clock time between consecutive OnEpoch
	// callback firings (see run()/runRecurrentRounds's own lastEpochEnd),
	// so the very first epoch of a round is timed from the round's own
	// start, not from whatever the prior round's last epoch happened to be.
	metricEpochDurationSeconds = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "rsi_epoch_duration_seconds",
		Help:    "Wall-clock duration of each completed Student-training epoch (rsi_epoch_rollout_size trajectories collected plus the PPO update over them) - rsi_round_duration_seconds divided roughly by rsi_epochs_per_generation, but measured directly rather than inferred, since task-mix variance across an epoch's trajectories (see rsi_episode_length_steps) means epochs are not actually uniform in length.",
		Buckets: []float64{10, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200},
	})

	metricParallelEnvs = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_parallel_envs",
		Help: "The number of parallel bot sessions currently connected. Can change mid-run via POST /config/parallel-envs, applied at the next round boundary.",
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

	metricEvalEpisodes = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_eval_episodes",
		Help: "Evaluation episodes per side (Teacher and Student each) in the most recently completed round, summed over every environment.",
	})

	metricEvalRewardStd = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "rsi_eval_reward_std",
		Help: "Sample standard deviation of per-episode evaluation reward in the most recently completed round, by side.",
	}, []string{"side"}) // "teacher" | "student"

	metricEvalPairedDiff = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_eval_paired_diff",
		Help: "Mean paired per-episode evaluation reward difference (Student minus Teacher, same task on each side) in the most recently completed round.",
	})

	metricEvalPairedDiffStdErr = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_eval_paired_diff_stderr",
		Help: "Standard error of rsi_eval_paired_diff. A difference within a couple of these of zero is noise, not a real gap between the two policies.",
	})

	metricEvalPairedDiffZ = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_eval_paired_diff_zscore",
		Help: "rsi_eval_paired_diff divided by rsi_eval_paired_diff_stderr (0 when the standard error is 0): how many standard errors the Student's advantage is. Compare with rsi_eval_win_zscore_threshold.",
	})

	metricRewardTimePenalty = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_reward_time_penalty",
		Help: "The per-step time penalty this run's reward uses (-time-penalty, or rlenv's 0.01 default). Returns from runs with different values aren't comparable.",
	})

	metricEvalWinZThreshold = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rsi_eval_win_zscore_threshold",
		Help: "The -win-z-score this run promotes at: a Student must beat the Teacher by at least this many standard errors (0 = plain strictly-greater rule).",
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

	// Buckets are scaled for a single environment step (network round-trip
	// plus in-game action time, e.g. a mine/craft tick wait), not a whole
	// round like metricRoundDurationSeconds - deliberately distinct from
	// its buckets. Observed regardless of error (see Step): a step that
	// fails slowly, like a stuck retry loop waiting on a missing
	// ingredient or an unresponsive container, is exactly what this metric
	// exists to surface.
	metricActionsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_actions_total",
		Help: "Environment steps by the action the policy chose, by task. With rsi_step_duration_seconds this shows where an episode's steps go: most steps of a policy still learning a sequence are instant no-ops (Mine before the block is in view, GoToTarget after arriving, Wait), not slow ones.",
	}, []string{"task", "action"}) // action: wait | goto | mine | craft

	metricStepDurationSeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "rsi_step_duration_seconds",
		Help:    "Wall-clock duration of each environment step, by task.",
		Buckets: []float64{0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30},
	}, []string{"task"})

	// Buckets are Fibonacci-ish rather than evenly spaced: most episodes
	// (see docs/plans/13's step-budget investigation) finish in the
	// single-to-low-double digits even for the longest chain tasks, so
	// resolution matters most there, with just enough buckets near the
	// active step caps (100 for a training episode, 200 for an evaluation
	// one - see the phase label below) to see how much truncated mass piles
	// up at each.
	metricEpisodeLengthSteps = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "rsi_episode_length_steps",
		Help:    "How many environment steps an episode actually ran before ending, by task, outcome, and phase (\"training\" | \"eval\" - a training episode's step cap is 100, an evaluation one's is 200, so mixing them without this label made a success/truncated episode well past 100 look impossible when it's actually just from evaluation; see -eval-episode-len). truncated episodes land at (or just under) the active step cap almost by definition - what this metric actually reveals is where success/failure episodes fall relative to that cap: a tail hugging the cap means some near-misses might finish with a slightly larger budget; a tail that's short, with truncation making up the rest, means the episodes that don't finish are stuck early rather than merely short on room. See docs/plans/13's step-budget investigation.",
		Buckets: []float64{1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 100, 130, 160, 200},
	}, []string{"task", "outcome", "phase"})

	// metricTaskEpisodesTotal is only ever incremented when -curriculum-config
	// is set (see recordTaskSelection) — without a curriculum, rsi-train has
	// no hook into per-episode task selection at all (Reset is called deep
	// inside cRL-go's own rollout/evaluate loops, not by this command), so
	// this metric simply never appears rather than reporting a misleading
	// single-task 100%.
	metricTaskEpisodesTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_task_episodes_total",
		Help: "Episodes posed by the curriculum, by task type. Only emitted when -curriculum-config is set.",
	}, []string{"task"}) // "goto" | "mine" | "craft" | "mine_far" | "craft_far" | "chain_gather" | "chain_table" | "chain_place" | "chain_use"
)

// recordNetworkHealth publishes h. Labels its per-layer gauges "1", "2", ...
// in trunk order (see networkHealth.LayerActive), whatever the network's
// actual depth happens to be.
func recordNetworkHealth(h networkHealth) {
	for i, active := range h.LayerActive {
		metricNetworkActiveUnits.WithLabelValues(strconv.Itoa(i + 1)).Set(active)
	}
	metricPolicyLogitStd.Set(h.LogitStd)
	metricPolicyEntropy.Set(h.PolicyEntropy)
	for i, corr := range h.LayerCorrelation {
		metricNetworkUnitCorrelation.WithLabelValues(strconv.Itoa(i + 1)).Set(corr)
	}
}

// recordRoundEvaluation publishes a completed round's evaluation statistics
// (see leapfrog.Result): the two sides' mean rewards and spread, how many
// episodes each was evaluated on, and the paired difference with its
// standard error and z-score - the numbers the promotion decision actually
// rests on, so a dashboard can show whether a win was signal or noise.
func recordRoundEvaluation[P leapfrog.RecurrentOrNot](result leapfrog.Result[P]) {
	metricTeacherReward.Set(float64(result.TeacherReward))
	metricStudentReward.Set(float64(result.StudentReward))
	metricEvalEpisodes.Set(float64(result.EvalEpisodes))
	metricEvalRewardStd.WithLabelValues("teacher").Set(float64(result.TeacherStd))
	metricEvalRewardStd.WithLabelValues("student").Set(float64(result.StudentStd))
	metricEvalPairedDiff.Set(float64(result.MeanDiff))
	metricEvalPairedDiffStdErr.Set(float64(result.DiffStdErr))
	z := 0.0
	if result.DiffStdErr > 0 {
		z = float64(result.MeanDiff) / float64(result.DiffStdErr)
	}
	metricEvalPairedDiffZ.Set(z)
}

// startMetricsServer serves /metrics, exporter's POST /checkpoint/export,
// envCount's POST /config/parallel-envs, restart's POST
// /control/restart-at-epoch-boundary, and net/http/pprof's standard
// profiling endpoints (/debug/pprof/..., e.g.
// `go tool pprof 'http://host:PORT/debug/pprof/profile?seconds=30'` for a
// CPU profile of this exact live process - the reason these are wired in
// at all, rather than left to pprof's usual DefaultServeMux auto-register
// trick, is that this file already runs its own mux for /metrics) on addr
// in the background. Errors (e.g. the port is already taken) are logged,
// not fatal — this server going down is an observability problem, not a
// reason to stop an otherwise-healthy training run.
func startMetricsServer(addr string, exporter *checkpointExporter, envCount *envCountRequest, restart *restartRequest, agents *agentRegistry) {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	mux.Handle("/checkpoint/export", exporter)
	mux.Handle("/config/parallel-envs", envCount)
	mux.Handle("/control/restart-at-epoch-boundary", restart)
	mux.Handle("/agents", agents)
	mux.HandleFunc("/agents/world", agents.worldStreamHandler)
	mux.HandleFunc("/dashboard", serveDashboardPage)
	mux.Handle("/icons/", serveItemIcons())
	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
	go func() {
		log.Printf("metrics: serving /metrics, POST /checkpoint/export, POST /config/parallel-envs, POST /control/restart-at-epoch-boundary, GET /agents, /agents/world, /dashboard, /icons/, and /debug/pprof/ on %s", addr)
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
	// goalForEpisode reports the current episode's actual success
	// condition in plain language ("mine dirt", "craft stick", "goto
	// (+5, +0, +0) from spawn", ...) for the dashboard's Goal stat -
	// taskForEpisode's own counterpart, naming WHAT the episode ends by
	// achieving rather than which task type it is. Always set alongside
	// taskForEpisode at every connectedEnvironment construction site
	// (main.go); nil only in a test fixture that doesn't care.
	goalForEpisode func() string
	// targetOffsetForEpisode reports this episode's rlenv.TaskOverride.
	// TargetOffset and whether it's meaningful at all (ok is false for a
	// plain mine/craft episode, which has no predetermined location - see
	// agentSnapshot.Target's own doc comment) — the dashboard's own
	// agentRegistry combines this with the bot's own Spawn (captured
	// separately, since this closure only knows the *offset*, not an
	// absolute position) to plot a target marker.
	targetOffsetForEpisode func() (offset [3]float64, ok bool)
	started                bool
	steps                  int // steps taken so far in the in-progress episode; see metricEpisodeLengthSteps

	// registry and index are the dashboard's agentRegistry and this env's
	// index within it (matching connected/envs — see agentRegistry's own
	// doc comment). registry is always non-nil (run always constructs one,
	// whether or not -metrics-addr ends up serving it), so these updates
	// never need a nil check.
	registry *agentRegistry
	index    int
}

func (e *instrumentedEnvironment) currentTask() string {
	if e.taskForEpisode != nil {
		if task := e.taskForEpisode(); task != "" {
			return task
		}
	}
	return e.task
}

func (e *instrumentedEnvironment) currentGoal() string {
	if e.goalForEpisode != nil {
		return e.goalForEpisode()
	}
	return ""
}

func (e *instrumentedEnvironment) currentTargetOffset() (offset [3]float64, ok bool) {
	if e.targetOffsetForEpisode != nil {
		return e.targetOffsetForEpisode()
	}
	return [3]float64{}, false
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

// episodeLengthPhase is metricEpisodeLengthSteps' third label: whether the
// episode just recorded was part of training rollout collection or one of a
// round's Teacher/Student evaluation episodes, reusing evaluationInProgress
// (the same flag that gates rsi_epoch_trajectories_started). Training's
// step cap is 100, evaluation's is -eval-episode-len (default 200) - without
// this, a success/truncated episode past 100 looked impossible when it was
// actually just from evaluation.
func episodeLengthPhase() string {
	if evaluationInProgress.Load() {
		return "eval"
	}
	return "training"
}

func (e *instrumentedEnvironment) Reset(ctx context.Context) (rl.Observation, error) {
	task := e.currentTask()
	if e.started {
		metricEpisodesTotal.WithLabelValues(task, "truncated").Inc()
		metricEpisodeLengthSteps.WithLabelValues(task, "truncated", episodeLengthPhase()).Observe(float64(e.steps))
		e.steps = 0
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
	goal := e.currentGoal()
	targetOffset, hasTarget := e.currentTargetOffset()
	e.started = true
	metricEpisodesStartedTotal.WithLabelValues(task).Inc()
	if !evaluationInProgress.Load() {
		metricEpochTrajectoriesStarted.Inc()
	}
	e.registry.updateEpisode(e.index, func(s *agentSnapshot) {
		s.Task = task
		s.Goal = goal
		s.LastAction = ""
		s.EpisodeStep = 0
		s.LastOutcome = ""
		// A new episode's Spawn is wherever the bot's position is next
		// polled, not its last episode's — re-arm the capture, and clear
		// the stale Target until Spawn (and this one) are recombined.
		s.needsSpawnCapture = true
		s.hasTarget = hasTarget
		s.targetOffset = targetOffset
		s.Target = nil
	})
	return obs, nil
}

func (e *instrumentedEnvironment) Step(ctx context.Context, action rl.Action) (rl.StepResult, error) {
	task := e.currentTask()
	metricActionsTotal.WithLabelValues(task, actionName(action)).Inc()
	start := time.Now()
	result, err := e.env.Step(ctx, action)
	metricStepDurationSeconds.WithLabelValues(task).Observe(time.Since(start).Seconds())
	if err != nil {
		metricStepErrorsTotal.WithLabelValues(task).Inc()
		return result, err
	}
	metricEpisodeStepsTotal.WithLabelValues(task).Inc()
	e.steps++
	outcome := ""
	if result.Done {
		outcome = "failure"
		if result.Reward >= episodeSuccessRewardThreshold {
			outcome = "success"
		}
		metricEpisodesTotal.WithLabelValues(task, outcome).Inc()
		metricEpisodeLengthSteps.WithLabelValues(task, outcome, episodeLengthPhase()).Observe(float64(e.steps))
		e.started = false
		e.steps = 0
	}
	e.registry.updateEpisode(e.index, func(s *agentSnapshot) {
		s.Task = task
		s.LastAction = actionName(action)
		s.EpisodeStep = e.steps
		if outcome != "" {
			s.LastOutcome = outcome
		}
	})
	return result, nil
}

// defaultTimePenalty mirrors rlenv's built-in per-step penalty, which applies
// when Config.TimePenalty is 0 (see -time-penalty).
const defaultTimePenalty = 0.01

// effectiveTimePenalty is the penalty the reward actually uses for flag value
// v: 0 means rlenv's default.
func effectiveTimePenalty(v float64) float64 {
	if v <= 0 {
		return defaultTimePenalty
	}
	return v
}

// actionName is the metric label for one of rlenv's discrete actions.
func actionName(action rl.Action) string {
	switch action {
	case rlenv.ActionWait:
		return "wait"
	case rlenv.ActionGoToTarget:
		return "goto"
	case rlenv.ActionMine:
		return "mine"
	case rlenv.ActionCraft:
		return "craft"
	case rlenv.ActionPlace:
		return "place"
	default:
		return "unknown"
	}
}

// recordTaskSelection increments metricTaskEpisodesTotal for name (an
// rlenvadapter.Task.Name() value, e.g. "mine_far" — see main.go's own
// TaskSelector closure, its only caller).
func recordTaskSelection(name string) {
	if name != "" && name != "unknown" {
		metricTaskEpisodesTotal.WithLabelValues(name).Inc()
	}
}
