package main

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// lifetimeMetricsState is the cumulative, restart-surviving totals behind
// the dashboard's "Average round duration", "Average epoch duration",
// "Task mix" and "Task mix balance" panels - the same restart-reset
// problem rsi_rounds_total had (see roundCounterState's own doc comment),
// for rsi_round_duration_seconds/rsi_epoch_duration_seconds (Histograms)
// and rsi_task_episodes_total (a CounterVec).
//
// rsi_task_episodes_total needs no parallel field here beyond TaskEpisodes
// itself: it's a plain Counter per label, so (like rsi_rounds_total)
// seeding it directly with Add() at startup is enough to make it restart-
// surviving at the source - panels can read it raw. rsi_epochs_total is
// the same story and doesn't even need a field here; see Epochs below for
// why it still has one.
//
// rsi_round_duration_seconds/rsi_epoch_duration_seconds are different:
// Histograms have no public API to seed their internal sum/count from a
// persisted scalar the way a plain Counter's Add does. Rather than try to
// reconstruct those two Histograms themselves across a restart, this
// tracks the same lifetime sum (RoundDurationSeconds/EpochDurationSeconds)
// in parallel, plain float64 fields, exported as their own small Gauges
// (metricRoundDurationSecondsLifetimeTotal/metricEpochDurationSecondsLifetimeTotal)
// - enough for the dashboard's own "total / count" average panels, which
// never needed percentile buckets anyway (checked live 2026-10-12: nothing
// else references these two Histograms). Epochs is this sum's own
// denominator, kept here too even though rsi_epochs_total (once seeded)
// would already give the same number, so this file's own round-trip test
// doesn't depend on main.go's wiring being correct.
type lifetimeMetricsState struct {
	RoundDurationSeconds float64        `json:"round_duration_seconds"`
	EpochDurationSeconds float64        `json:"epoch_duration_seconds"`
	Epochs               int            `json:"epochs"`
	TaskEpisodes         map[string]int `json:"task_episodes"`
}

// lifetimeMetricsStateFile is where saveLifetimeMetricsState/
// loadLifetimeMetricsState keep the latest totals, alongside
// round-counter.json and the interim student checkpoint in interimDir.
func lifetimeMetricsStateFile(interimDir string) string {
	return filepath.Join(interimDir, "lifetime-metrics.json")
}

// loadLifetimeMetricsState reads back the totals saved by
// saveLifetimeMetricsState, or returns ok=false if none exists yet (a
// fresh run, or one from before this existed) - never an error a caller
// needs to handle, mirroring loadRoundCounterState's own convention.
func loadLifetimeMetricsState(interimDir string) (lifetimeMetricsState, bool) {
	data, err := os.ReadFile(lifetimeMetricsStateFile(interimDir))
	if err != nil {
		return lifetimeMetricsState{}, false
	}
	var state lifetimeMetricsState
	if err := json.Unmarshal(data, &state); err != nil {
		return lifetimeMetricsState{}, false
	}
	return state, true
}

// saveLifetimeMetricsState atomically (temp file plus rename, so a crash
// mid-write can never leave a truncated/corrupt file behind) records
// state, overwriting whatever was there before.
func saveLifetimeMetricsState(interimDir string, state lifetimeMetricsState) error {
	path := lifetimeMetricsStateFile(interimDir)
	tmp, err := os.CreateTemp(interimDir, filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := json.NewEncoder(tmp).Encode(state); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}
