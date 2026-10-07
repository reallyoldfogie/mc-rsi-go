package main

import (
	"context"
	_ "embed"
	"net/http"
	"sync"
	"time"

	"github.com/reallyoldfogie/mc-agent/models"
)

//go:embed dashboard.html
var dashboardPage []byte

// serveDashboardPage serves GET /dashboard: the static page that polls
// GET /agents. No templating — the page is pure client-side JS against the
// JSON endpoint, so the same bytes work unmodified regardless of -metrics-addr.
func serveDashboardPage(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(dashboardPage)
}

// agentPositionPollInterval is how often agentRegistry.runPoller refreshes
// every bot's position/health from its live agent. Independent of (and much
// slower than) the per-tick rate mc-agent's own movement executor polls at —
// this is for a human watching a dashboard, not control logic.
const agentPositionPollInterval = time.Second

// agentSnapshot is one bot's point-in-time state for the dashboard's
// GET /agents endpoint. Two different writers keep it current:
// instrumentedEnvironment.Reset/Step update the episode/task/action fields
// as training produces them, and agentRegistry.runPoller independently
// refreshes the position/health fields (data instrumentedEnvironment has no
// access to, since it only wraps rl.Environment, not models.Agent) on its
// own timer. agent itself is never serialized.
type agentSnapshot struct {
	Index       int       `json:"index"`
	Username    string    `json:"username"`
	Task        string    `json:"task"`
	LastAction  string    `json:"last_action"`
	EpisodeStep int       `json:"episode_step"`
	LastOutcome string    `json:"last_outcome"`
	X           float64   `json:"x"`
	Y           float64   `json:"y"`
	Z           float64   `json:"z"`
	Yaw         float64   `json:"yaw"`
	Pitch       float64   `json:"pitch"`
	Health      float32   `json:"health"`
	Food        int32     `json:"food"`
	UpdatedAt   time.Time `json:"updated_at"`

	agent models.Agent
}

// agentRegistry is the dashboard's live view of every connected bot,
// indexed exactly like connected/envs in main.go: add appends (matching
// connectEnvironmentRange/reconcileEnvCount's own append-only growth) and
// truncate drops the highest-indexed entries (matching reconcileEnvCount's
// scale-down). Safe for concurrent use by instrumentedEnvironment's
// Reset/Step (any goroutine driving rollout collection), runPoller's own
// background goroutine, and ServeHTTP.
type agentRegistry struct {
	mu   sync.Mutex
	bots []*agentSnapshot
}

func newAgentRegistry() *agentRegistry {
	return &agentRegistry{}
}

// add registers a newly connected bot at the next index.
func (r *agentRegistry) add(username string, agent models.Agent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.bots = append(r.bots, &agentSnapshot{Index: len(r.bots), Username: username, agent: agent})
}

// truncate drops every entry at index >= n. A no-op if n >= the current
// count, so callers don't need to check first.
func (r *agentRegistry) truncate(n int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if n < len(r.bots) {
		r.bots = r.bots[:n]
	}
}

// updateEpisode applies fn to bot i's snapshot and stamps UpdatedAt. A no-op
// on a nil receiver (every instrumentedEnvironment test fixture below
// builds one without ever wiring in a *agentRegistry — run() always
// supplies a real one) or for an out-of-range index — the only caller is
// instrumentedEnvironment, which has no sensible way to react to a stale
// index, and an index is only ever stale in between a scale-down and that
// instrumentedEnvironment itself being discarded.
func (r *agentRegistry) updateEpisode(i int, fn func(*agentSnapshot)) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if i < 0 || i >= len(r.bots) {
		return
	}
	fn(r.bots[i])
	r.bots[i].UpdatedAt = time.Now()
}

// pollOnce refreshes every bot's position/yaw/pitch (via models.Position,
// which models.Agent always satisfies through CommandAgent/MovementAgent)
// and health/food (via models.HealthProvider, satisfied structurally by the
// concrete bot type but not part of the models.Agent interface itself,
// hence the type assertion). GetPosition/Health are both plain accessors
// over state mc-agent already tracks from the connection's own packet
// stream, not new network calls, so this is cheap even at this poller's own
// independent cadence.
func (r *agentRegistry) pollOnce() {
	r.mu.Lock()
	bots := make([]*agentSnapshot, len(r.bots))
	copy(bots, r.bots)
	r.mu.Unlock()

	for _, b := range bots {
		pos, yaw, pitch, ok := b.agent.GetPosition()
		var health float32
		var food int32
		healthKnown := false
		if hp, ok := b.agent.(models.HealthProvider); ok {
			health, food, _, healthKnown = hp.Health()
		}

		r.mu.Lock()
		if ok {
			b.X, b.Y, b.Z, b.Yaw, b.Pitch = pos.X, pos.Y, pos.Z, yaw, pitch
		}
		if healthKnown {
			b.Health, b.Food = health, food
		}
		r.mu.Unlock()
	}
}

// runPoller calls pollOnce every agentPositionPollInterval until ctx is
// done. Safe to start before any bot is registered (pollOnce on zero bots
// is a no-op) — see run's own call site for why that matters.
func (r *agentRegistry) runPoller(ctx context.Context) {
	ticker := time.NewTicker(agentPositionPollInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			r.pollOnce()
		}
	}
}

// snapshot returns a point-in-time copy of every bot's state, safe for the
// caller to serialize or read without holding r.mu.
func (r *agentRegistry) snapshot() []agentSnapshot {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]agentSnapshot, len(r.bots))
	for i, b := range r.bots {
		out[i] = *b
	}
	return out
}

// ServeHTTP serves GET /agents: the JSON array the dashboard page polls.
func (r *agentRegistry) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		http.Error(w, "GET only", http.StatusMethodNotAllowed)
		return
	}
	respondJSON(w, http.StatusOK, r.snapshot())
}
