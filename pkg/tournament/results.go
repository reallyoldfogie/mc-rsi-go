package tournament

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Manifest is the run-level metadata written once at the start.
type Manifest struct {
	RunID       string    `json:"run_id"`
	StartedAt   time.Time `json:"started_at"`
	BestOf      int       `json:"best_of"`
	EpisodeLen  int       `json:"episode_len"`
	Competitors []string  `json:"competitors"`
}

// GameResult is one game's persisted outcome.
type GameResult struct {
	Winner      string             `json:"winner,omitempty"`
	Draw        bool               `json:"draw"`
	DamageDealt map[string]float64 `json:"damage_dealt"`
}

// PairingResult is one completed pairing's full record.
type PairingResult struct {
	Round       int          `json:"round"`
	Slot        int          `json:"slot"`
	CompetitorA string       `json:"competitor_a"`
	CompetitorB string       `json:"competitor_b"`
	WinnerID    string       `json:"winner_id"`
	GamesWonA   int          `json:"games_won_a"`
	GamesWonB   int          `json:"games_won_b"`
	Games       []GameResult `json:"games"`
}

// Results is the full run's persisted state.
type Results struct {
	Manifest   Manifest        `json:"manifest"`
	Pairings   []PairingResult `json:"pairings"`
	Champion   string          `json:"champion,omitempty"`
	FinishedAt *time.Time      `json:"finished_at,omitempty"`
}

// ResultsWriter owns incremental, atomic persistence to
// Config.ResultsRoot/results.json - written after every pairing (not
// just at the end), so an interrupted run leaves the most recently
// completed pairing's result inspectable, matching pkg/competition's
// own artifact-writing convention without importing that package
// directly (same "mirrored, not imported" choice Phase 1 made for its
// own config schema).
type ResultsWriter struct {
	path    string
	results Results
}

// NewResultsWriter creates Config.ResultsRoot if needed and writes the
// initial manifest-only results file.
func NewResultsWriter(cfg Config) (*ResultsWriter, error) {
	if err := os.MkdirAll(cfg.ResultsRoot, 0o755); err != nil {
		return nil, fmt.Errorf("tournament: creating results directory %s: %w", cfg.ResultsRoot, err)
	}
	ids := make([]string, len(cfg.Competitors))
	for i, c := range cfg.Competitors {
		ids[i] = c.ID
	}
	rw := &ResultsWriter{
		path: filepath.Join(cfg.ResultsRoot, "results.json"),
		results: Results{Manifest: Manifest{
			RunID: cfg.RunID, StartedAt: time.Now(), BestOf: cfg.BestOf,
			EpisodeLen: cfg.EpisodeLen, Competitors: ids,
		}},
	}
	if err := rw.write(); err != nil {
		return nil, err
	}
	return rw, nil
}

// RecordPairing appends one completed pairing's result and persists it
// immediately.
func (rw *ResultsWriter) RecordPairing(pairing *Pairing, outcome PairingOutcome) error {
	games := make([]GameResult, len(outcome.Games))
	for i, g := range outcome.Games {
		games[i] = GameResult{Winner: g.Winner, Draw: g.Draw, DamageDealt: g.DamageDealt}
	}
	rw.results.Pairings = append(rw.results.Pairings, PairingResult{
		Round: pairing.Round, Slot: pairing.Slot,
		CompetitorA: pairing.CompetitorA, CompetitorB: pairing.CompetitorB,
		WinnerID: outcome.WinnerCompetitorID, GamesWonA: outcome.GamesWonA, GamesWonB: outcome.GamesWonB,
		Games: games,
	})
	return rw.write()
}

// Finalize records the tournament champion and finish time.
func (rw *ResultsWriter) Finalize(champion string) error {
	now := time.Now()
	rw.results.Champion = champion
	rw.results.FinishedAt = &now
	return rw.write()
}

// write persists the current results atomically (temp file + rename),
// so an interrupted run leaves either the old or the new complete file,
// never a half-written one.
func (rw *ResultsWriter) write() error {
	data, err := json.MarshalIndent(rw.results, "", "  ")
	if err != nil {
		return fmt.Errorf("tournament: marshaling results: %w", err)
	}
	tmp := rw.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("tournament: writing results: %w", err)
	}
	if err := os.Rename(tmp, rw.path); err != nil {
		return fmt.Errorf("tournament: finalizing results write: %w", err)
	}
	return nil
}
