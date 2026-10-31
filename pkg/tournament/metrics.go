package tournament

import (
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/pvp"
)

// Metrics owns a registry for one competition-mode (Phase 2) run.
// Mirrors pkg/competition.Metrics's own shape, labeled run_id/
// competitor_id/phase per COMBAT_COMPETITION_PLAN.md Phase 3 item 1 -
// phase is always "match" here (pkg/pvp's own Metrics uses "training"),
// an explicit label rather than baked into the metric name, so a
// combined /metrics endpoint can tell the two modes' series apart
// without name collisions. competitor_id is used as the label name for
// consistency with pkg/competition/pkg/pvp's own convention, even though
// a tournament.Competitor is semantically a team, not a lone agent.
type Metrics struct {
	Registry *prometheus.Registry
	runID    string

	gamesPlayedTotal    *prometheus.CounterVec
	gamesWonTotal       *prometheus.CounterVec
	damageDealt         *prometheus.GaugeVec
	pairingsPlayedTotal *prometheus.CounterVec
	bracketRound        prometheus.Gauge
	championDecided     prometheus.Gauge
}

const matchPhaseLabel = "match"

// NewMetrics creates the metrics used by one tournament.Run call.
func NewMetrics(runID string) (*Metrics, error) {
	if runID == "" {
		return nil, fmt.Errorf("tournament: metrics run ID must not be empty")
	}
	labels := []string{"run_id", "competitor_id", "phase"}
	m := &Metrics{Registry: prometheus.NewRegistry(), runID: runID}
	m.gamesPlayedTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_tournament_games_played_total",
		Help: "Games played by tournament competitor.",
	}, labels)
	m.gamesWonTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_tournament_games_won_total",
		Help: "Games won by tournament competitor.",
	}, labels)
	m.damageDealt = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "rsi_tournament_damage_dealt",
		Help: "Damage dealt by tournament competitor in its most recently completed game.",
	}, labels)
	m.pairingsPlayedTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "rsi_tournament_pairings_played_total",
		Help: "Bracket pairings played by tournament competitor (win or loss).",
	}, labels)
	m.bracketRound = prometheus.NewGauge(prometheus.GaugeOpts{
		Name:        "rsi_tournament_bracket_round",
		Help:        "The current bracket round index being played (0 = first round).",
		ConstLabels: prometheus.Labels{"run_id": runID},
	})
	m.championDecided = prometheus.NewGauge(prometheus.GaugeOpts{
		Name:        "rsi_tournament_champion_decided",
		Help:        "Whether this tournament run has a decided champion (1) or not (0).",
		ConstLabels: prometheus.Labels{"run_id": runID},
	})
	for _, collector := range []prometheus.Collector{
		m.gamesPlayedTotal, m.gamesWonTotal, m.damageDealt, m.pairingsPlayedTotal,
		m.bracketRound, m.championDecided,
	} {
		if err := m.Registry.Register(collector); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// SetBracketRound records the bracket round currently being played.
func (m *Metrics) SetBracketRound(round int) {
	m.bracketRound.Set(float64(round))
}

// SetChampionDecided records whether a champion has been decided.
func (m *Metrics) SetChampionDecided(decided bool) {
	value := 0.0
	if decided {
		value = 1
	}
	m.championDecided.Set(value)
}

// RecordGame records one game's outcome for both competitors in a
// pairing - gamesPlayedTotal always increments for both; gamesWonTotal
// only for the winner (a draw increments neither's win count, matching
// how pkg/tournament's own PairingOutcome tallies game wins).
func (m *Metrics) RecordGame(competitorA, competitorB string, outcome pvp.GameOutcome) {
	for _, id := range []string{competitorA, competitorB} {
		m.gamesPlayedTotal.WithLabelValues(m.runID, id, matchPhaseLabel).Inc()
	}
	if !outcome.Draw && outcome.Winner != "" {
		m.gamesWonTotal.WithLabelValues(m.runID, outcome.Winner, matchPhaseLabel).Inc()
	}
	for team, dealt := range outcome.DamageDealt {
		m.damageDealt.WithLabelValues(m.runID, team, matchPhaseLabel).Set(dealt)
	}
}

// RecordPairing records one completed pairing's outcome for both
// competitors.
func (m *Metrics) RecordPairing(competitorA, competitorB string) {
	for _, id := range []string{competitorA, competitorB} {
		m.pairingsPlayedTotal.WithLabelValues(m.runID, id, matchPhaseLabel).Inc()
	}
}
