package tournament

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestLiveStatus_RefreshReflectsBracketState(t *testing.T) {
	bracket, err := NewBracket([]string{"red", "blue"})
	if err != nil {
		t.Fatalf("NewBracket: %v", err)
	}
	status := NewLiveStatus("run1")
	status.Refresh(bracket, nil)

	snap := status.Snapshot()
	if snap.RunID != "run1" {
		t.Errorf("RunID = %q, want run1", snap.RunID)
	}
	if len(snap.Pairings) != 1 {
		t.Fatalf("len(Pairings) = %d, want 1", len(snap.Pairings))
	}
	if snap.Pairings[0].CompetitorA != "red" || snap.Pairings[0].CompetitorB != "blue" {
		t.Errorf("unexpected pairing: %+v", snap.Pairings[0])
	}
	if snap.ChampionDecided {
		t.Error("ChampionDecided = true before any result recorded")
	}

	if err := bracket.RecordResult(0, 0, "red"); err != nil {
		t.Fatalf("RecordResult: %v", err)
	}
	status.Refresh(bracket, nil)
	snap = status.Snapshot()
	if !snap.ChampionDecided || snap.Champion != "red" {
		t.Errorf("champion = %q decided=%v, want red/true", snap.Champion, snap.ChampionDecided)
	}
	if snap.Pairings[0].WinnerID != "red" {
		t.Errorf("WinnerID = %q, want red", snap.Pairings[0].WinnerID)
	}
}

func TestLiveStatus_RefreshSetsCurrentPairing(t *testing.T) {
	bracket, err := NewBracket([]string{"red", "blue"})
	if err != nil {
		t.Fatalf("NewBracket: %v", err)
	}
	status := NewLiveStatus("run1")
	pairing, ok := bracket.NextReadyPairing()
	if !ok {
		t.Fatal("expected a ready pairing")
	}
	status.Refresh(bracket, pairing)

	snap := status.Snapshot()
	if snap.CurrentPairing == nil {
		t.Fatal("CurrentPairing is nil")
	}
	if snap.CurrentPairing.CompetitorA != "red" || snap.CurrentPairing.CompetitorB != "blue" {
		t.Errorf("unexpected current pairing: %+v", snap.CurrentPairing)
	}
}

func TestLiveStatus_ServeStatusReturnsJSON(t *testing.T) {
	bracket, err := NewBracket([]string{"red", "blue"})
	if err != nil {
		t.Fatalf("NewBracket: %v", err)
	}
	status := NewLiveStatus("run1")
	status.Refresh(bracket, nil)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/status", nil)
	status.serveStatus(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status code = %d, want 200", rec.Code)
	}
	var got StatusSnapshot
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("unmarshaling response: %v", err)
	}
	if got.RunID != "run1" {
		t.Errorf("RunID = %q, want run1", got.RunID)
	}
}

func TestLiveStatus_ServeScoreboardRendersCompetitorsAndChampion(t *testing.T) {
	bracket, err := NewBracket([]string{"red", "blue"})
	if err != nil {
		t.Fatalf("NewBracket: %v", err)
	}
	if err := bracket.RecordResult(0, 0, "blue"); err != nil {
		t.Fatalf("RecordResult: %v", err)
	}
	status := NewLiveStatus("run1")
	status.Refresh(bracket, nil)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	status.serveScoreboard(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status code = %d, want 200", rec.Code)
	}
	body := rec.Body.String()
	for _, want := range []string{"red", "blue", "Champion: blue"} {
		if !strings.Contains(body, want) {
			t.Errorf("scoreboard HTML missing %q:\n%s", want, body)
		}
	}
}

func TestLiveStatus_SnapshotIsSafeForConcurrentAccess(t *testing.T) {
	bracket, err := NewBracket([]string{"red", "blue", "green", "yellow"})
	if err != nil {
		t.Fatalf("NewBracket: %v", err)
	}
	status := NewLiveStatus("run1")

	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 100; i++ {
			status.Refresh(bracket, nil)
		}
	}()
	for i := 0; i < 100; i++ {
		_ = status.Snapshot()
	}
	<-done
}
