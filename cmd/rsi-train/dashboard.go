package main

import (
	"context"
	"embed"
	"fmt"
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
	return http.StripPrefix("/icons/", itemIconFallback{fsys: sub, next: http.FileServer(http.FS(sub))})
}

// itemIconFallbackSuffixes are the directional-texture suffixes Minecraft
// uses for a block with no single uniform face — a bare "<name>.png" only
// exists for blocks that look the same from every side (dirt, oak_planks,
// ...); one like crafting_table, furnace, or grass_block only has
// "<name>_top.png"/"_front.png"/"_side.png"/"_all.png", never a bare file.
// Tried in this order (_front, then _side, then _top) to match how an
// inventory slot icon actually reads at a glance.
var itemIconFallbackSuffixes = []string{"_front", "_side", "_top", "_all"}

// itemIconFallback serves a bare "<name>.png" request directly when it
// exists, and otherwise rewrites it to the first itemIconFallbackSuffixes
// variant that does, instead of just 404ing — without this, every
// multi-face block-type item in the bot's inventory 404s, not just the
// handful whose icon happens to need its own flat texture.
type itemIconFallback struct {
	fsys fs.FS
	next http.Handler
}

func (h itemIconFallback) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	name, ok := strings.CutSuffix(strings.TrimPrefix(req.URL.Path, "/"), ".png")
	if !ok || name == "" || strings.ContainsRune(name, '/') {
		h.next.ServeHTTP(w, req)
		return
	}
	if _, err := fs.Stat(h.fsys, name+".png"); err == nil {
		h.next.ServeHTTP(w, req)
		return
	}
	for _, suffix := range itemIconFallbackSuffixes {
		if _, err := fs.Stat(h.fsys, name+suffix+".png"); err == nil {
			req = req.Clone(req.Context())
			req.URL.Path = "/" + name + suffix + ".png"
			h.next.ServeHTTP(w, req)
			return
		}
	}
	h.next.ServeHTTP(w, req) // no variant either; let it 404 on the original path.
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
	Goal        string           `json:"goal"`
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
	Nearby      []nearbyEntity   `json:"nearby"`
	// Spawn is where this episode started — not something rlenv exposes
	// directly (its own resolved reset position is unexported, same
	// reason Goal's own doc comment gives for "goto <loc>" not being a
	// live absolute coordinate), so it's approximated as wherever the
	// bot's position is first polled after Reset — see needsSpawnCapture.
	// nil until that first poll lands.
	Spawn *vec3 `json:"spawn"`
	// Target is this episode's fixed destination, Spawn plus whichever
	// TargetOffset the selected Task carried (see targetOffset) — real
	// for goto/mine_far/craft_far/every chain stage, since all of those
	// seed something at (or travel to) a point offset from Spawn; nil for
	// a plain mine/craft episode, which has no predetermined location to
	// mark (the bot finds its own block/table near wherever it already
	// is). Recomputed every poll once Spawn is known, so it never goes
	// stale relative to it.
	Target    *vec3     `json:"target"`
	UpdatedAt time.Time `json:"updated_at"`

	agent models.Agent

	// needsSpawnCapture is set whenever a new episode starts (Reset) and
	// cleared the next time pollOnce captures a position for it into
	// Spawn - see agentRegistry.pollOnce. hasTarget/targetOffset are this
	// episode's own TargetOffset and whether it's meaningful (see
	// instrumentedEnvironment.currentTargetOffset), carried alongside
	// Task/Goal from the same TaskSelector draw, used to (re)compute
	// Target once Spawn is known.
	needsSpawnCapture bool
	hasTarget         bool
	targetOffset      [3]float64
}

// vec3 is a plain 3D point — Spawn/Target are pointers to one so they
// encode as JSON null (checked directly in the dashboard's own JS, no
// separate "is this known/meaningful" flag needed) until there's a real
// position to report.
type vec3 struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	Z float64 `json:"z"`
}

// inventoryEntry is one inventory slot (main inventory or hotbar) for the
// dashboard's inventory strip - the zero value represents an empty slot,
// shown as an empty box so the strip reads as "this bot's whole inventory,
// slot for slot," not just a compacted list of what it's holding. Name has
// its "minecraft:" prefix already stripped, matching itemIconsFS's own
// filenames.
type inventoryEntry struct {
	// Slot is this entry's 1-based position in the displayed bar (1-27
	// are main-inventory slots, 28-36 are hotbar slots, matching the
	// Main()-then-Hotbar() order summarizeInventory builds entries in) —
	// not mc-agent/mc-bot-go's own raw protocol slot numbers (screen.
	// MainSlotStart=9, HotbarSlotStart=36, with the crafting grid and
	// armor ahead of those), which would be confusing here since this
	// strip never shows those other slot kinds at all.
	Slot  int    `json:"slot"`
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
		var nearby []nearbyEntity
		if ok {
			nearby = readNearbyEntities(b.agent, pos.X, pos.Y, pos.Z)
		}

		r.mu.Lock()
		if ok {
			b.X, b.Y, b.Z, b.Yaw, b.Pitch = pos.X, pos.Y, pos.Z, yaw, pitch
			b.Nearby = nearby
			if b.needsSpawnCapture {
				b.Spawn = &vec3{X: pos.X, Y: pos.Y, Z: pos.Z}
				b.needsSpawnCapture = false
			}
			if b.hasTarget && b.Spawn != nil {
				b.Target = &vec3{
					X: b.Spawn.X + b.targetOffset[0],
					Y: b.Spawn.Y + b.targetOffset[1],
					Z: b.Spawn.Z + b.targetOffset[2],
				}
			} else {
				b.Target = nil
			}
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

// nearbyEntityRadius bounds how far from a bot's own position
// readNearbyEntities looks, and nearbyEntityMax caps how many of those it
// keeps (nearest first) - a generous margin beyond the dashboard's own
// ZOOM_RADIUS (dashboard.html), so nothing visible in the zoomed view is
// missing, without unbounded JSON payload growth in a crowded area.
const (
	nearbyEntityRadius = 32.0
	nearbyEntityMax    = 40
)

// nearbyEntity is one non-bot entity (a dropped item, a mob, ...) near a
// bot's own position, for the dashboard's zoomed single-agent view -
// context for "what is around this bot," not just where the bot itself is.
type nearbyEntity struct {
	Type string  `json:"type"`
	X    float64 `json:"x"`
	Y    float64 `json:"y"`
	Z    float64 `json:"z"`
}

// readNearbyEntities summarizes a's already-tracked nearby entities around
// (originX, originY, originZ), or nil if it has none within
// nearbyEntityRadius. GetTrackedEntities is a plain read of state mc-agent
// already tracks from the connection's own packet stream (every entity in
// the bot's own loaded view distance), not a new network call - same cost
// class as GetPosition/GetInventory.
func readNearbyEntities(a models.Agent, originX, originY, originZ float64) []nearbyEntity {
	tracked := a.GetTrackedEntities()
	if len(tracked) == 0 {
		return nil
	}
	return summarizeNearbyEntities(tracked, a.GetRegistry("minecraft:entity_type"), originX, originY, originZ)
}

// summarizeNearbyEntities filters tracked to live (non-Removed) entities
// within nearbyEntityRadius of the given origin, resolves each one's type
// name via reg (falling back to a numeric placeholder if reg is nil or
// doesn't recognize the ID), and returns the nearbyEntityMax nearest ones
// sorted by distance. Split out from readNearbyEntities so it's testable
// without a models.Agent fake for the whole (large) interface - see
// summarizeInventory's own doc comment for why this repo avoids those.
func summarizeNearbyEntities(tracked map[int32]models.TrackedEntityInfo, reg models.CustomRegistry, originX, originY, originZ float64) []nearbyEntity {
	type candidate struct {
		nearbyEntity
		distSq float64
	}
	maxDistSq := nearbyEntityRadius * nearbyEntityRadius
	candidates := make([]candidate, 0, len(tracked))
	for _, info := range tracked {
		if info.Removed {
			continue
		}
		dx, dy, dz := info.X-originX, info.Y-originY, info.Z-originZ
		distSq := dx*dx + dy*dy + dz*dz
		if distSq > maxDistSq {
			continue
		}
		name := ""
		if reg != nil {
			if n, ok := reg.GetNameByID(info.EntityType); ok {
				name = strings.TrimPrefix(n, "minecraft:")
			}
		}
		if name == "" {
			name = fmt.Sprintf("entity-%d", info.EntityType)
		}
		candidates = append(candidates, candidate{
			nearbyEntity: nearbyEntity{Type: name, X: info.X, Y: info.Y, Z: info.Z},
			distSq:       distSq,
		})
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].distSq < candidates[j].distSq })
	if len(candidates) > nearbyEntityMax {
		candidates = candidates[:nearbyEntityMax]
	}

	out := make([]nearbyEntity, len(candidates))
	for i, c := range candidates {
		out[i] = c.nearbyEntity
	}
	return out
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

// summarizeInventory returns one entry PER SLOT (including empty ones, as
// the zero value — see inventoryEntry's own doc comment) across every slot
// group passed, in the order given (readInventory passes Main() then
// Hotbar(), matching the real inventory screen's own top-to-bottom slot
// order — see screen.MainSlotStart/HotbarSlotStart) and preserving each
// group's own slot order within it. Deliberately NOT merged/summed by
// name — two separate stacks of the same item in different slots stay two
// separate entries, exactly as the player's own inventory would show them,
// per the dashboard's own "same order as the player's inventory" design.
// Each resolved name has its "minecraft:" prefix stripped; air (a slot
// technically "holding" the air item, not actually empty) counts as empty
// too. Split out from readInventory so it's testable without a
// models.Agent fake for the whole (large) interface — see
// reconcileEnvCount's own tests for why this repo avoids those.
func summarizeInventory(itemMgr models.ItemManager, slotGroups ...[]screen.Slot) []inventoryEntry {
	entries := make([]inventoryEntry, 0, 36) // 27 main + 9 hotbar, the common case
	for _, slots := range slotGroups {
		for _, s := range slots {
			slot := len(entries) + 1
			if s.Count == 0 {
				entries = append(entries, inventoryEntry{Slot: slot})
				continue
			}
			name := strings.TrimPrefix(itemMgr.GetItemNameByID(int32(s.ID)), "minecraft:")
			if name == "" || name == "air" {
				entries = append(entries, inventoryEntry{Slot: slot})
				continue
			}
			entries = append(entries, inventoryEntry{Slot: slot, Name: name, Count: int(s.Count)})
		}
	}
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
