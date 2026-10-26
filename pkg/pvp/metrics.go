package pvp

import (
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
)

// Metrics owns a registry for one training-mode (Phase 1) run. Mirrors
// pkg/competition.Metrics's own shape (not backed by package-global
// promauto metrics, so multiple runs can be embedded/tested without
// cross-run state collisions), labeled run_id/participant_id/phase per
// COMBAT_COMPETITION_PLAN.md Phase 3 item 1 - phase is always "training"
// for this package's own metrics (pkg/tournament's own Metrics uses
// "match"), kept as an explicit label rather than baked into the metric
// name so a single combined /metrics endpoint can distinguish the two
// modes' series from each other without name collisions.
type Metrics struct {
	Registry *prometheus.Registry
	runID    string

	epochsTotal          *prometheus.CounterVec
	samplesTotal         *prometheus.CounterVec
	gradientUpdatesTotal *prometheus.CounterVec
	checkpointSavesTotal *prometheus.CounterVec
	trainingStopped      *prometheus.GaugeVec
}

const trainingPhaseLabel = "training"

// NewMetrics creates the metrics used by one pvp.Run call. The complete
// label set is fixed here; arbitrary display names are never labels.
func NewMetrics(runID string) (*Metrics, error) {
	if runID == "" {
		return nil, fmt.Errorf("pvp: metrics run ID must not be empty")
	}
	labels := []string{"run_id", "participant_id", "phase"}
	m := &Metrics{Registry: prometheus.NewRegistry(), runID: runID}
	m.epochsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_pvp_epochs_total",
		Help: "Completed training epochs by pvp training participant.",
	}, labels)
	m.samplesTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_pvp_samples_total",
		Help: "Training samples consumed by pvp training participant.",
	}, labels)
	m.gradientUpdatesTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_pvp_gradient_updates_total",
		Help: "Gradient updates applied by pvp training participant.",
	}, labels)
	m.checkpointSavesTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_pvp_checkpoint_saves_total",
		Help: "Checkpoint saves by pvp training participant.",
	}, labels)
	m.trainingStopped = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "rsi_pvp_training_stopped",
		Help: "Whether this pvp training participant's session has stopped (1) or is still running (0).",
	}, labels)
	for _, collector := range []prometheus.Collector{
		m.epochsTotal, m.samplesTotal, m.gradientUpdatesTotal,
		m.checkpointSavesTotal, m.trainingStopped,
	} {
		if err := m.Registry.Register(collector); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// Participant returns a label-bound view for one managed participant.
func (m *Metrics) Participant(participantID string) *ParticipantMetrics {
	return &ParticipantMetrics{parent: m, runID: m.runID, participantID: participantID}
}

// ParticipantMetrics is a label-bound recorder for one managed training
// participant - see Metrics.Participant.
type ParticipantMetrics struct {
	parent        *Metrics
	runID         string
	participantID string
}

func (m *ParticipantMetrics) EpochCompleted(samples, gradientUpdates int) {
	labels := []string{m.runID, m.participantID, trainingPhaseLabel}
	m.parent.epochsTotal.WithLabelValues(labels...).Inc()
	m.parent.samplesTotal.WithLabelValues(labels...).Add(float64(samples))
	m.parent.gradientUpdatesTotal.WithLabelValues(labels...).Add(float64(gradientUpdates))
}

func (m *ParticipantMetrics) CheckpointSaved() {
	m.parent.checkpointSavesTotal.WithLabelValues(m.runID, m.participantID, trainingPhaseLabel).Inc()
}

func (m *ParticipantMetrics) SetStopped(stopped bool) {
	value := 0.0
	if stopped {
		value = 1
	}
	m.parent.trainingStopped.WithLabelValues(m.runID, m.participantID, trainingPhaseLabel).Set(value)
}
