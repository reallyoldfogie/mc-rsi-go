package main

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// roundCounterState is the lifetime round count and won/lost tally this run
// has accumulated, persisted next to round-eval.json (see roundEvalState's
// own doc comment) so Grafana's "round" and "rounds won/lost" panels -
// rsi_current_round (a Gauge) and rsi_rounds_total (a Counter) - don't
// silently reset to 1/0/0 on every process restart: neither Prometheus
// type has any restart-survival of its own, unlike rec.Generation and
// checkpoint.Metadata, which are already file-backed. Found live
// 2026-10-12: a restart to pick up a new binary mid-run left the dashboard
// showing round 1 / 0 won / 0 lost, even though the run was actually many
// rounds and a promotion into its history.
type roundCounterState struct {
	Round int `json:"round"`
	Won   int `json:"won"`
	Lost  int `json:"lost"`
}

// roundCounterStateFile is where saveRoundCounterState/loadRoundCounterState
// keep the latest completed round's cumulative counts, alongside
// round-eval.json and the interim student checkpoint in interimDir.
func roundCounterStateFile(interimDir string) string {
	return filepath.Join(interimDir, "round-counter.json")
}

// loadRoundCounterState reads back the counts saved by
// saveRoundCounterState, or returns ok=false if none exists yet (a fresh
// run, or one from before this existed) - never an error a caller needs to
// handle, mirroring loadRoundEvalState's own convention.
func loadRoundCounterState(interimDir string) (roundCounterState, bool) {
	data, err := os.ReadFile(roundCounterStateFile(interimDir))
	if err != nil {
		return roundCounterState{}, false
	}
	var state roundCounterState
	if err := json.Unmarshal(data, &state); err != nil {
		return roundCounterState{}, false
	}
	return state, true
}

// saveRoundCounterState atomically (temp file plus rename, so a crash
// mid-write can never leave a truncated/corrupt file behind) records
// state, overwriting whatever the previous round left behind.
func saveRoundCounterState(interimDir string, state roundCounterState) error {
	path := roundCounterStateFile(interimDir)
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
