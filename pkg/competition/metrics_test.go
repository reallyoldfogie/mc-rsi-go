package competition

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCompetitionMetricsRemainDistinctByCompetitor(t *testing.T) {
	metrics, err := NewMetrics("run-1")
	require.NoError(t, err)
	basic := metrics.Competitor("basic")
	wide := metrics.Competitor("wide")
	basic.EpochCompleted(10, 2)
	wide.EpochCompleted(20, 4)
	basic.SetEvaluationReward(1.5)
	wide.SetEvaluationReward(2.5)
	basic.SetTrainingComplete(true)
	wide.SetTrainingFailed(true)
	metrics.AgentExited("wide", "RSICompetition-1", "agent_done")
	metrics.RecordStopReason("agent_exit")

	assert.Equal(t, 1.5, gatheredGauge(t, metrics, "rsi_competition_evaluation_reward", "basic"))
	assert.Equal(t, 2.5, gatheredGauge(t, metrics, "rsi_competition_evaluation_reward", "wide"))
	assert.Equal(t, 1.0, gatheredGauge(t, metrics, "rsi_competition_training_complete", "basic"))
	assert.Equal(t, 1.0, gatheredGauge(t, metrics, "rsi_competition_training_failed", "wide"))
	assert.Equal(t, 1.0, gatheredLabeledMetric(t, metrics, "rsi_competition_stop_reason", "reason", "agent_exit"))
	assert.Equal(t, 1.0, gatheredLabeledMetric(t, metrics, "rsi_competition_agent_exits_total", "competitor_id", "wide"))
}

func gatheredLabeledMetric(t *testing.T, metrics *Metrics, name, labelName, labelValue string) float64 {
	t.Helper()
	families, err := metrics.Registry.Gather()
	require.NoError(t, err)
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		for _, metric := range family.GetMetric() {
			for _, label := range metric.GetLabel() {
				if label.GetName() != labelName || label.GetValue() != labelValue {
					continue
				}
				if metric.Gauge != nil {
					return metric.GetGauge().GetValue()
				}
				return metric.GetCounter().GetValue()
			}
		}
	}
	t.Fatalf("metric %s with %s=%s not found", name, labelName, labelValue)
	return 0
}

func gatheredGauge(t *testing.T, metrics *Metrics, name, competitorID string) float64 {
	t.Helper()
	families, err := metrics.Registry.Gather()
	require.NoError(t, err)
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		for _, metric := range family.GetMetric() {
			for _, label := range metric.GetLabel() {
				if label.GetName() == "competitor_id" && label.GetValue() == competitorID {
					return metric.GetGauge().GetValue()
				}
			}
		}
	}
	t.Fatalf("metric %s for competitor %s not found", name, competitorID)
	return 0
}

func TestCompetitionMetricsRejectEmptyRunID(t *testing.T) {
	_, err := NewMetrics("")
	require.Error(t, err)
}
