package pvp

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestNewMetrics_RejectsEmptyRunID(t *testing.T) {
	if _, err := NewMetrics(""); err == nil {
		t.Fatal("want an error for an empty run ID")
	}
}

func TestParticipantMetrics_EpochCompletedRecordsAllThreeSeries(t *testing.T) {
	m, err := NewMetrics("run1")
	if err != nil {
		t.Fatalf("NewMetrics: %v", err)
	}
	p := m.Participant("red1")
	p.EpochCompleted(64, 8)
	p.EpochCompleted(64, 8)

	if got := testutil.ToFloat64(m.epochsTotal.WithLabelValues("run1", "red1", "training")); got != 2 {
		t.Errorf("epochsTotal = %v, want 2", got)
	}
	if got := testutil.ToFloat64(m.samplesTotal.WithLabelValues("run1", "red1", "training")); got != 128 {
		t.Errorf("samplesTotal = %v, want 128", got)
	}
	if got := testutil.ToFloat64(m.gradientUpdatesTotal.WithLabelValues("run1", "red1", "training")); got != 16 {
		t.Errorf("gradientUpdatesTotal = %v, want 16", got)
	}
}

func TestParticipantMetrics_CheckpointSaved(t *testing.T) {
	m, err := NewMetrics("run1")
	if err != nil {
		t.Fatalf("NewMetrics: %v", err)
	}
	p := m.Participant("red1")
	p.CheckpointSaved()
	p.CheckpointSaved()
	p.CheckpointSaved()

	if got := testutil.ToFloat64(m.checkpointSavesTotal.WithLabelValues("run1", "red1", "training")); got != 3 {
		t.Errorf("checkpointSavesTotal = %v, want 3", got)
	}
}

func TestParticipantMetrics_SetStopped(t *testing.T) {
	m, err := NewMetrics("run1")
	if err != nil {
		t.Fatalf("NewMetrics: %v", err)
	}
	p := m.Participant("red1")

	p.SetStopped(false)
	if got := testutil.ToFloat64(m.trainingStopped.WithLabelValues("run1", "red1", "training")); got != 0 {
		t.Errorf("trainingStopped = %v, want 0", got)
	}
	p.SetStopped(true)
	if got := testutil.ToFloat64(m.trainingStopped.WithLabelValues("run1", "red1", "training")); got != 1 {
		t.Errorf("trainingStopped = %v, want 1", got)
	}
}

func TestMetrics_DistinctParticipantsHaveIndependentSeries(t *testing.T) {
	m, err := NewMetrics("run1")
	if err != nil {
		t.Fatalf("NewMetrics: %v", err)
	}
	m.Participant("red1").EpochCompleted(10, 1)
	m.Participant("blue1").EpochCompleted(20, 2)

	if got := testutil.ToFloat64(m.samplesTotal.WithLabelValues("run1", "red1", "training")); got != 10 {
		t.Errorf("red1 samplesTotal = %v, want 10", got)
	}
	if got := testutil.ToFloat64(m.samplesTotal.WithLabelValues("run1", "blue1", "training")); got != 20 {
		t.Errorf("blue1 samplesTotal = %v, want 20", got)
	}
}
