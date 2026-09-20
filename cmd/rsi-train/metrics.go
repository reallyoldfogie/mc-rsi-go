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
	"log"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"

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
