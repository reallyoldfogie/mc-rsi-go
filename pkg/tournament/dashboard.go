package tournament

import (
	"encoding/json"
	"html/template"
	"log"
	"net/http"
	"sync"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/liveview"
)

// PairingSnapshot is one pairing's plain-data, JSON-serializable state -
// a copy, never a live *Pairing, so it can be read concurrently with
// Run's own bracket mutations without synchronizing on the bracket
// itself at all (see LiveStatus's own doc comment).
type PairingSnapshot struct {
	Round       int    `json:"round"`
	Slot        int    `json:"slot"`
	CompetitorA string `json:"competitor_a"`
	CompetitorB string `json:"competitor_b"`
	WinnerID    string `json:"winner_id"`
	GamesWonA   int    `json:"games_won_a"`
	GamesWonB   int    `json:"games_won_b"`
	Bye         bool   `json:"bye"`
}

// StatusSnapshot is a tournament run's full live state at one instant -
// the data layer for the /status JSON endpoint and the HTML scoreboard
// page (COMBAT_COMPETITION_PLAN.md Phase 3 item 2).
type StatusSnapshot struct {
	RunID           string            `json:"run_id"`
	Pairings        []PairingSnapshot `json:"pairings"`
	CurrentPairing  *PairingSnapshot  `json:"current_pairing,omitempty"`
	Champion        string            `json:"champion,omitempty"`
	ChampionDecided bool              `json:"champion_decided"`
}

// LiveStatus is a thread-safe, continuously replaced snapshot of one
// tournament run's progress. Run calls Refresh (from its own single
// goroutine, the only one that ever touches a live *Bracket) every time
// something changes; dashboard HTTP handlers call Snapshot from
// whichever goroutine net/http happens to use for that request. Because
// Refresh always builds a brand new StatusSnapshot value (never mutating
// a previously published one in place) and Snapshot returns a copy of
// the stored value under the same mutex, there is no need to ever touch
// the live Bracket/Pairing pointers themselves from outside Run's own
// goroutine - sidesteps synchronizing on those entirely.
type LiveStatus struct {
	mu       sync.RWMutex
	snapshot StatusSnapshot
}

// NewLiveStatus creates an empty status for runID.
func NewLiveStatus(runID string) *LiveStatus {
	return &LiveStatus{snapshot: StatusSnapshot{RunID: runID}}
}

// Refresh rebuilds the published snapshot from bracket's current state
// (read here, safely, since the caller is assumed to be the same
// goroutine that owns all of bracket's own mutations) and the pairing
// currently being played, if any.
func (s *LiveStatus) Refresh(bracket *Bracket, currentPairing *Pairing) {
	pairings := make([]PairingSnapshot, 0)
	for _, round := range bracket.Rounds {
		for _, p := range round {
			pairings = append(pairings, PairingSnapshot{
				Round: p.Round, Slot: p.Slot,
				CompetitorA: p.CompetitorA, CompetitorB: p.CompetitorB,
				WinnerID: p.WinnerID, GamesWonA: p.GamesWonA, GamesWonB: p.GamesWonB,
				Bye: p.IsBye(),
			})
		}
	}
	var current *PairingSnapshot
	if currentPairing != nil {
		current = &PairingSnapshot{
			Round: currentPairing.Round, Slot: currentPairing.Slot,
			CompetitorA: currentPairing.CompetitorA, CompetitorB: currentPairing.CompetitorB,
			WinnerID: currentPairing.WinnerID, GamesWonA: currentPairing.GamesWonA, GamesWonB: currentPairing.GamesWonB,
		}
	}
	champion, decided := bracket.Champion()

	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot = StatusSnapshot{
		RunID: s.snapshot.RunID, Pairings: pairings, CurrentPairing: current,
		Champion: champion, ChampionDecided: decided,
	}
}

// Snapshot returns the currently published status.
func (s *LiveStatus) Snapshot() StatusSnapshot {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.snapshot
}

func (s *LiveStatus) serveStatus(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(s.Snapshot()); err != nil {
		log.Printf("tournament: encoding /status response: %v", err)
	}
}

var scoreboardTemplate = template.Must(template.New("scoreboard").Parse(`<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Tournament {{.RunID}}</title>
<meta http-equiv="refresh" content="3">
<style>
body { font-family: sans-serif; margin: 2em; }
table { border-collapse: collapse; margin-bottom: 1.5em; }
th, td { border: 1px solid #ccc; padding: 0.4em 0.8em; text-align: left; }
.current { background: #ffe8a3; }
.bye { color: #888; }
.champion { font-size: 1.3em; font-weight: bold; color: #2a7a2a; }
</style>
</head>
<body>
<h1>Tournament {{.RunID}}</h1>
{{if .ChampionDecided}}<p class="champion">Champion: {{.Champion}}</p>{{end}}
<table>
<tr><th>Round</th><th>Competitor A</th><th>Competitor B</th><th>Score</th><th>Winner</th></tr>
{{range .Pairings}}
<tr{{if .Bye}} class="bye"{{end}}>
<td>{{.Round}}</td>
<td>{{.CompetitorA}}</td>
<td>{{.CompetitorB}}</td>
<td>{{.GamesWonA}} - {{.GamesWonB}}</td>
<td>{{.WinnerID}}</td>
</tr>
{{end}}
</table>
{{if .CurrentPairing}}<p>Now playing: {{.CurrentPairing.CompetitorA}} vs {{.CurrentPairing.CompetitorB}} ({{.CurrentPairing.GamesWonA}} - {{.CurrentPairing.GamesWonB}})</p>{{end}}
</body>
</html>
`))

func (s *LiveStatus) serveScoreboard(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := scoreboardTemplate.Execute(w, s.Snapshot()); err != nil {
		log.Printf("tournament: rendering scoreboard page: %v", err)
	}
}

// StartDashboardServer serves the HTML scoreboard page at "/", the JSON
// status snapshot at "/status", metrics' own registry at "/metrics", and -
// if view is non-nil - pkg/liveview's own live 3D agent view at
// "/dashboard" (plus its "/agents", "/agents/world", "/icons/", and
// "/vendor/" support routes): the actual "watch the competitors fight"
// view (COMBAT_COMPETITION_PLAN.md's dashboard item), since the scoreboard
// alone only shows bracket/game results, not what's happening live inside
// a game. All served on addr, in the background. Errors (e.g. the port is
// already taken) are logged, not fatal - this server going down is an
// observability problem, not a reason to stop an otherwise-healthy
// tournament run (same convention as cmd/rsi-train's own
// startMetricsServer).
func StartDashboardServer(addr string, status *LiveStatus, metrics *Metrics, view *liveview.Registry) {
	mux := http.NewServeMux()
	mux.HandleFunc("/", status.serveScoreboard)
	mux.HandleFunc("/status", status.serveStatus)
	if metrics != nil {
		mux.Handle("/metrics", promhttp.HandlerFor(metrics.Registry, promhttp.HandlerOpts{}))
	}
	routes := "/ (scoreboard), /status (JSON), /metrics (Prometheus)"
	if view != nil {
		mux.HandleFunc("/dashboard", liveview.ServeDashboardPage)
		mux.Handle("/agents", view)
		mux.HandleFunc("/agents/world", view.WorldStreamHandler)
		mux.Handle("/icons/", liveview.ServeItemIcons())
		mux.Handle("/vendor/", liveview.ServeVendor())
		routes += ", /dashboard (live 3D agent view)"
	}
	go func() {
		log.Printf("tournament: serving %s on %s", routes, addr)
		if err := http.ListenAndServe(addr, mux); err != nil {
			log.Printf("tournament: dashboard server on %s stopped: %v", addr, err)
		}
	}()
}
