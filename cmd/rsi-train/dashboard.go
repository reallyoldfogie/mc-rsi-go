package main

import (
	"context"
	"embed"
	"io/fs"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/reallyoldfogie/mc-agent/models"
	"github.com/reallyoldfogie/mc-bot-go/bot/screen"
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

// itemIconsFS is every vanilla item and block texture (16x16 PNGs, ~7MB)
// from a 1.21.11 client jar extraction already sitting in this ecosystem's
// own mc-protocol-go data-gen cache (see docs on that repo for how it got
// there) — copied in once rather than referenced in place, since go:embed
// can only embed files inside this module. Flat block textures stand in
// for the item icon of block-type items (e.g. oak_log, crafting_table):
// not the isometric cube the real client renders, but enough to recognize
// at a glance next to the item's name, which the dashboard always shows
// alongside it. Served at /icons/<item-name-without-"minecraft:">.png;
// an item with no match (a non-vanilla ID, or one newer than 1.21.11)
// just 404s, and the dashboard's own JS falls back to a text badge.
//
//go:embed itemicons
var itemIconsFS embed.FS

func serveItemIcons() http.Handler {
	sub, err := fs.Sub(itemIconsFS, "itemicons")
	if err != nil {
		panic(err) // itemicons is embedded above; only a packaging bug could make this fail.
	}
	return http.StripPrefix("/icons/", http.FileServer(http.FS(sub)))
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
	Index       int              `json:"index"`
	Username    string           `json:"username"`
	Task        string           `json:"task"`
	LastAction  string           `json:"last_action"`
	EpisodeStep int              `json:"episode_step"`
	LastOutcome string           `json:"last_outcome"`
	X           float64          `json:"x"`
	Y           float64          `json:"y"`
	Z           float64          `json:"z"`
	Yaw         float64          `json:"yaw"`
	Pitch       float64          `json:"pitch"`
	Health      float32          `json:"health"`
	Food        int32            `json:"food"`
	Inventory   []inventoryEntry `json:"inventory"`
	UpdatedAt   time.Time        `json:"updated_at"`

	agent models.Agent
}

// inventoryEntry is one distinct item (summed across every slot it occupies
// in the bot's main inventory + hotbar) for the dashboard's inventory
// strip. Name has its "minecraft:" prefix already stripped, matching
// itemIconsFS's own filenames.
type inventoryEntry struct {
	Name  string `json:"name"`
	Count int    `json:"count"`
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
		inventory := readInventory(b.agent)

		r.mu.Lock()
		if ok {
			b.X, b.Y, b.Z, b.Yaw, b.Pitch = pos.X, pos.Y, pos.Z, yaw, pitch
		}
		if healthKnown {
			b.Health, b.Food = health, food
		}
		if inventory != nil {
			b.Inventory = inventory
		}
		r.mu.Unlock()
	}
}

// readInventory sums a's main inventory + hotbar slots by item name, or
// returns nil if a's screen/item-name subsystems aren't ready yet (mirrors
// GetInventory/GetItemManager's own "nil means not ready" convention —
// neither is a new network call, both just read state mc-agent already
// tracks from the connection's own packet stream, same as GetPosition).
func readInventory(a models.Agent) []inventoryEntry {
	inv := a.GetInventory()
	if inv == nil {
		return nil
	}
	itemMgr := a.GetItemManager()
	if itemMgr == nil {
		return nil
	}
	return summarizeInventory(itemMgr, inv.Main(), inv.Hotbar())
}

// summarizeInventory sums each distinct item across every slot group
// passed (e.g. main inventory + hotbar), stripping each resolved name's
// "minecraft:" prefix and skipping empty/air slots. Split out from
// readInventory so it's testable without a models.Agent fake for the
// whole (large) interface — see reconcileEnvCount's own tests for why
// this repo avoids those.
func summarizeInventory(itemMgr models.ItemManager, slotGroups ...[]screen.Slot) []inventoryEntry {
	counts := map[string]int{}
	for _, slots := range slotGroups {
		for _, s := range slots {
			if s.Count == 0 {
				continue
			}
			name := strings.TrimPrefix(itemMgr.GetItemNameByID(int32(s.ID)), "minecraft:")
			if name == "" || name == "air" {
				continue
			}
			counts[name] += int(s.Count)
		}
	}

	entries := make([]inventoryEntry, 0, len(counts))
	for name, count := range counts {
		entries = append(entries, inventoryEntry{Name: name, Count: count})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name < entries[j].Name })
	return entries
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
