package competition

import (
	"fmt"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
)

// Metrics owns a registry for one competition run. It is deliberately not
// backed by package-global promauto metrics, so multiple competitions can be
// embedded or tested without cross-run state collisions.
type Metrics struct {
	Registry *prometheus.Registry
	runID    string

	epochsTotal        *prometheus.CounterVec
	samplesTotal       *prometheus.CounterVec
	gradientUpdates    *prometheus.CounterVec
	evaluationReward   *prometheus.GaugeVec
	evaluationStdDev   *prometheus.GaugeVec
	evaluationLower    *prometheus.GaugeVec
	evaluationUpper    *prometheus.GaugeVec
	evaluationEpisodes *prometheus.GaugeVec
	trainingComplete   *prometheus.GaugeVec
	trainingFailed     *prometheus.GaugeVec
	agentExits         *prometheus.CounterVec
	stopReason         *prometheus.GaugeVec
	stopReasonOnce     sync.Once
}

// NewMetrics creates the metrics used by the competition coordinator. The
// complete label set is fixed here; arbitrary display names are never labels.
func NewMetrics(runID string) (*Metrics, error) {
	if runID == "" {
		return nil, fmt.Errorf("metrics run ID must not be empty")
	}
	labels := []string{"run_id", "competitor_id"}
	m := &Metrics{Registry: prometheus.NewRegistry(), runID: runID}
	m.epochsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_competition_epochs_total",
		Help: "Completed training epochs by competition competitor.",
	}, labels)
	m.samplesTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_competition_samples_total",
		Help: "Training samples consumed by competition competitor.",
	}, labels)
	m.gradientUpdates = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_competition_gradient_updates_total",
		Help: "Gradient updates applied by competition competitor.",
	}, labels)
	m.evaluationReward = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "rsi_competition_evaluation_reward",
		Help: "Mean final evaluation reward by competition competitor.",
	}, labels)
	m.evaluationStdDev = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "rsi_competition_evaluation_reward_stddev",
		Help: "Sample standard deviation of episode rewards by competition competitor.",
	}, labels)
	m.evaluationLower = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "rsi_competition_evaluation_confidence_lower",
		Help: "Lower bound of the 95 percent evaluation reward confidence interval.",
	}, labels)
	m.evaluationUpper = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "rsi_competition_evaluation_confidence_upper",
		Help: "Upper bound of the 95 percent evaluation reward confidence interval.",
	}, labels)
	m.evaluationEpisodes = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "rsi_competition_evaluation_episodes",
		Help: "Number of completed evaluation episodes by competition competitor.",
	}, labels)
	m.trainingComplete = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "rsi_competition_training_complete",
		Help: "Whether competition training completed for a competitor.",
	}, labels)
	m.trainingFailed = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "rsi_competition_training_failed",
		Help: "Whether competition training failed for a competitor.",
	}, labels)
	m.agentExits = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_competition_agent_exits_total",
		Help: "Unexpected live-agent exits observed during competition.",
	}, []string{"run_id", "competitor_id", "username", "reason"})
	m.stopReason = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "rsi_competition_stop_reason",
		Help: "The first competition stop reason, represented by a 1-valued reason label.",
	}, []string{"run_id", "reason"})
	for _, collector := range []prometheus.Collector{
		m.epochsTotal, m.samplesTotal, m.gradientUpdates,
		m.evaluationReward, m.evaluationStdDev, m.evaluationLower, m.evaluationUpper,
		m.evaluationEpisodes, m.trainingComplete, m.trainingFailed,
		m.agentExits, m.stopReason,
	} {
		if err := m.Registry.Register(collector); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// AgentExited records the shutdown signal that can otherwise be hidden in the
// cascade of context-canceled cleanup messages.
func (m *Metrics) AgentExited(competitorID, username, reason string) {
	m.agentExits.WithLabelValues(m.runID, competitorID, username, reason).Inc()
}

// RecordStopReason records only the first competition-wide stop reason.
func (m *Metrics) RecordStopReason(reason string) {
	m.stopReasonOnce.Do(func() {
		m.stopReason.WithLabelValues(m.runID, reason).Set(1)
	})
}

// Competitor returns a label-bound view for one configured competitor.
func (m *Metrics) Competitor(competitorID string) *CompetitorMetrics {
	return &CompetitorMetrics{
		parent: m, runID: m.runID, competitorID: competitorID,
	}
}

type CompetitorMetrics struct {
	parent       *Metrics
	runID        string
	competitorID string
}

func (m *CompetitorMetrics) EpochCompleted(samples, gradientUpdates int) {
	m.parent.epochsTotal.WithLabelValues(m.runID, m.competitorID).Inc()
	m.parent.samplesTotal.WithLabelValues(m.runID, m.competitorID).Add(float64(samples))
	m.parent.gradientUpdates.WithLabelValues(m.runID, m.competitorID).Add(float64(gradientUpdates))
}

func (m *CompetitorMetrics) SetEvaluationReward(reward float64) {
	m.parent.evaluationReward.WithLabelValues(m.runID, m.competitorID).Set(reward)
}

func (m *CompetitorMetrics) SetEvaluationStatistics(summary EvaluationSummary) {
	labels := []string{m.runID, m.competitorID}
	m.parent.evaluationReward.WithLabelValues(labels...).Set(summary.MeanReward)
	m.parent.evaluationStdDev.WithLabelValues(labels...).Set(summary.RewardStdDev)
	m.parent.evaluationLower.WithLabelValues(labels...).Set(summary.ConfidenceLower)
	m.parent.evaluationUpper.WithLabelValues(labels...).Set(summary.ConfidenceUpper)
	m.parent.evaluationEpisodes.WithLabelValues(labels...).Set(float64(summary.Episodes))
}

func (m *CompetitorMetrics) SetTrainingComplete(complete bool) {
	value := 0.0
	if complete {
		value = 1
	}
	m.parent.trainingComplete.WithLabelValues(m.runID, m.competitorID).Set(value)
}

func (m *CompetitorMetrics) SetTrainingFailed(failed bool) {
	value := 0.0
	if failed {
		value = 1
	}
	m.parent.trainingFailed.WithLabelValues(m.runID, m.competitorID).Set(value)
}
