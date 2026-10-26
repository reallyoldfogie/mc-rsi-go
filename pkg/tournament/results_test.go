package tournament

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/pvp"
)

func TestNewResultsWriter_WritesInitialManifest(t *testing.T) {
	dir := t.TempDir()
	cfg := validTournamentConfig()
	cfg.ResultsRoot = dir

	if _, err := NewResultsWriter(cfg); err != nil {
		t.Fatalf("NewResultsWriter: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(dir, "results.json"))
	if err != nil {
		t.Fatalf("reading results.json: %v", err)
	}
	var results Results
	if err := json.Unmarshal(data, &results); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if results.Manifest.RunID != cfg.RunID {
		t.Errorf("Manifest.RunID = %q, want %q", results.Manifest.RunID, cfg.RunID)
	}
	if results.Manifest.BestOf != cfg.BestOf {
		t.Errorf("Manifest.BestOf = %d, want %d", results.Manifest.BestOf, cfg.BestOf)
	}
	if len(results.Pairings) != 0 {
		t.Errorf("Pairings = %+v, want empty before any pairing is recorded", results.Pairings)
	}
	if results.FinishedAt != nil {
		t.Error("FinishedAt should be nil before Finalize")
	}
}

func TestResultsWriter_RecordPairingPersistsIncrementally(t *testing.T) {
	dir := t.TempDir()
	cfg := validTournamentConfig()
	cfg.ResultsRoot = dir

	rw, err := NewResultsWriter(cfg)
	if err != nil {
		t.Fatalf("NewResultsWriter: %v", err)
	}

	pairing := &Pairing{Round: 0, Slot: 0, CompetitorA: "red", CompetitorB: "blue"}
	outcome := PairingOutcome{
		WinnerCompetitorID: "red",
		GamesWonA:          2, GamesWonB: 1,
		Games: []pvp.GameOutcome{
			{Winner: "red", DamageDealt: map[string]float64{"red": 20, "blue": 5}},
			{Winner: "blue", DamageDealt: map[string]float64{"red": 5, "blue": 20}},
			{Winner: "red", DamageDealt: map[string]float64{"red": 20, "blue": 10}},
		},
	}
	if err := rw.RecordPairing(pairing, outcome); err != nil {
		t.Fatalf("RecordPairing: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(dir, "results.json"))
	if err != nil {
		t.Fatalf("reading results.json: %v", err)
	}
	var results Results
	if err := json.Unmarshal(data, &results); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(results.Pairings) != 1 {
		t.Fatalf("Pairings = %+v, want 1 entry", results.Pairings)
	}
	recorded := results.Pairings[0]
	if recorded.WinnerID != "red" || recorded.GamesWonA != 2 || recorded.GamesWonB != 1 {
		t.Errorf("recorded pairing = %+v, want WinnerID=red GamesWonA=2 GamesWonB=1", recorded)
	}
	if len(recorded.Games) != 3 {
		t.Fatalf("Games = %+v, want 3 entries", recorded.Games)
	}
}

func TestResultsWriter_FinalizeRecordsChampionAndFinishTime(t *testing.T) {
	dir := t.TempDir()
	cfg := validTournamentConfig()
	cfg.ResultsRoot = dir

	rw, err := NewResultsWriter(cfg)
	if err != nil {
		t.Fatalf("NewResultsWriter: %v", err)
	}
	if err := rw.Finalize("red"); err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(dir, "results.json"))
	if err != nil {
		t.Fatalf("reading results.json: %v", err)
	}
	var results Results
	if err := json.Unmarshal(data, &results); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if results.Champion != "red" {
		t.Errorf("Champion = %q, want %q", results.Champion, "red")
	}
	if results.FinishedAt == nil {
		t.Fatal("FinishedAt should be set after Finalize")
	}
}

func TestNewResultsWriter_RejectsUnwritableDirectory(t *testing.T) {
	dir := t.TempDir()
	// Make the parent read-only so MkdirAll for a nested results root fails.
	if err := os.Chmod(dir, 0o500); err != nil {
		t.Skipf("could not set up read-only directory: %v", err)
	}
	defer os.Chmod(dir, 0o700)

	cfg := validTournamentConfig()
	cfg.ResultsRoot = filepath.Join(dir, "nested", "results")
	if _, err := NewResultsWriter(cfg); err == nil {
		t.Fatal("want an error when the results directory cannot be created")
	}
}
