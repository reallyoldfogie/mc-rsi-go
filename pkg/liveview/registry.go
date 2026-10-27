// Package liveview is the shared live-dashboard/world-stream machinery
// originally built for cmd/rsi-train's own architecture-competition
// dashboard (/dashboard, /agents, /agents/world), extracted here so
// pkg/pvp and pkg/tournament's combat competitions can reuse the exact
// same mechanism - real ray-traced visible blocks, tracked nearby
// entities (which already includes other players, not just items/mobs -
// confirmed live, this is what lets a combat bot's opponent show up in
// its own view with no new tracking code), inventory, and position - read
// straight off each bot's own live connection, rendered in a Three.js 3D
// view in the browser. cmd/rsi-train keeps its own copy of this
// mechanism unchanged (a bigger, riskier refactor of a working,
// heavily-tested production command wasn't worth it just to deduplicate
// ~900 lines); this package is the new, generalized home other
// commands build on.
package liveview

import (
	"context"
	"embed"
	"encoding/json"
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

// ServeDashboardPage serves GET /dashboard: the static page that polls
// GET /agents. No templating — the page is pure client-side JS against the
// JSON endpoint, so the same bytes work unmodified regardless of command.
func ServeDashboardPage(w http.ResponseWriter, req *http.Request) {
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

// ServeItemIcons serves the embedded item/block icon set at /icons/.
func ServeItemIcons() http.Handler {
	sub, err := fs.Sub(itemIconsFS, "itemicons")
	if err != nil {
		panic(err) // itemicons is embedded above; only a packaging bug could make this fail.
	}
	return http.StripPrefix("/icons/", itemIconFallback{fsys: sub, next: http.FileServer(http.FS(sub))})
}

// vendorFS is third-party frontend JS vendored straight into the binary
// (currently just a minified Three.js ES module build, for the dashboard's
// 3D view) so the dashboard stays fully self-contained with no external
// CDN dependency at runtime, same as itemIconsFS above.
//
//go:embed vendor
var vendorFS embed.FS

// ServeVendor serves the embedded third-party frontend JS at /vendor/.
func ServeVendor() http.Handler {
	sub, err := fs.Sub(vendorFS, "vendor")
	if err != nil {
		panic(err) // vendor is embedded above; only a packaging bug could make this fail.
	}
	return http.StripPrefix("/vendor/", http.FileServer(http.FS(sub)))
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
// exists, and otherwise rewrites it to a matching directional variant. For
// 3D block rendering it also handles requests for a missing face such as
// "oak_log_side.png" by falling back to the block's real bare texture
// "oak_log.png" (or one of its real directional variants). This keeps the
// browser from logging a 404 for every face of a uniform-texture block.
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
	if resolved := h.resolveIcon(name); resolved != "" {
		req = req.Clone(req.Context())
		req.URL.Path = "/" + resolved
		h.next.ServeHTTP(w, req)
		return
	}
	h.next.ServeHTTP(w, req) // no variant either; let it 404 on the original path.
}

// resolveIcon returns the embedded filename that best represents name. A
// request for a missing directional face first retries the block's bare name,
// then the available directional variants. The exact request is always
// preferred, so this does not interfere with genuine per-face textures.
func (h itemIconFallback) resolveIcon(name string) string {
	if _, err := fs.Stat(h.fsys, name+".png"); err == nil {
		return name + ".png"
	}
	for _, suffix := range itemIconFallbackSuffixes {
		if _, err := fs.Stat(h.fsys, name+suffix+".png"); err == nil {
			return name + suffix + ".png"
		}
	}
	for _, face := range []string{"_top", "_bottom", "_side", "_front", "_all"} {
		base, ok := strings.CutSuffix(name, face)
		if !ok || base == "" {
			continue
		}
		if _, err := fs.Stat(h.fsys, base+".png"); err == nil {
			return base + ".png"
		}
		for _, suffix := range itemIconFallbackSuffixes {
			if _, err := fs.Stat(h.fsys, base+suffix+".png"); err == nil {
				return base + suffix + ".png"
			}
		}
	}
	return ""
}

// AgentPollInterval is how often Registry.RunPoller refreshes every bot's
// dashboard state from its live agent. Independent of the per-tick rate
// mc-agent's own movement executor polls at — this is for a human
// watching a dashboard, not control logic.
const AgentPollInterval = 250 * time.Millisecond

// Snapshot is one bot's point-in-time state for the dashboard's GET
// /agents endpoint. Two different writers can keep it current: a caller
// driving some notion of episodes/steps via UpdateEpisode (as
// cmd/rsi-train's own instrumentedEnvironment does for the Task/Goal/
// Episode* fields below - all optional, left empty by a caller with no
// such concept, e.g. a combat match), and Registry.RunPoller
// independently refreshing the position/health fields (data an episode
// driver has no access to, since it only ever sees whatever environment
// abstraction it's wrapping, not models.Agent) on its own timer. agent
// itself is never serialized.
type Snapshot struct {
	Index    int    `json:"index"`
	Username string `json:"username"`
	// Competition identity is empty unless the caller sets it via
	// AddCompetition - kept in the common snapshot so one dashboard page
	// renders every kind of run (plain training, architecture
	// competition, combat match) without a second viewer or endpoint.
	RunID        string `json:"run_id,omitempty"`
	CompetitorID string `json:"competitor_id,omitempty"`
	DisplayName  string `json:"display_name,omitempty"`
	Architecture string `json:"architecture,omitempty"`
	Task         string `json:"task"`
	// Phase is a caller-defined free-form label (cmd/rsi-train uses
	// "training"/"eval"; a combat caller might use "training"/"match") -
	// training rollout collection and a round's teacher/student
	// evaluation episodes (or a match's own games) can run through this
	// exact same bot connection and Registry slot, one after another, so
	// without this field the dashboard has no way to show which one is
	// currently happening.
	Phase                 string             `json:"phase"`
	Goal                  string             `json:"goal"`
	LastAction            string             `json:"last_action"`
	EpisodeStep           int                `json:"episode_step"`
	LastOutcome           string             `json:"last_outcome"`
	LastTerminationReason string             `json:"last_termination_reason"`
	X                     float64            `json:"x"`
	Y                     float64            `json:"y"`
	Z                     float64            `json:"z"`
	Yaw                   float64            `json:"yaw"`
	Pitch                 float64            `json:"pitch"`
	Health                float32            `json:"health"`
	Food                  int32              `json:"food"`
	Inventory             *InventorySnapshot `json:"inventory"`
	Nearby                []NearbyEntity     `json:"nearby"`
	// Spawn is where this episode started — an approximation (wherever
	// the bot's position was first polled after UpdateEpisode set
	// needsSpawnCapture), nil until that first poll lands.
	Spawn *Vec3 `json:"spawn"`
	// Target is this episode's fixed destination, Spawn plus whichever
	// offset SetTarget carried - nil if SetTarget was never called for
	// this episode. Recomputed every poll once Spawn is known, so it
	// never goes stale relative to it.
	Target *Vec3 `json:"target"`
	// UpdatedAt is stamped by UpdateEpisode alone - only when the caller
	// has something new to report, never by PollOnce. A single step/game
	// can legitimately take several seconds of real time to physically
	// complete, so this is an activity signal, not a connection-health
	// one - don't use it to decide whether the bot is still connected
	// (see PolledAt for that; conflating the two misreads "mid-step" as
	// "disconnected").
	UpdatedAt time.Time `json:"updated_at"`
	// PolledAt is stamped by PollOnce every time it successfully reads
	// this bot's live position (runs at RunPoller's own cadence
	// regardless of whether anything else is in progress) - the
	// connection-health signal UpdatedAt isn't.
	PolledAt time.Time `json:"polled_at"`

	agent models.Agent

	// needsSpawnCapture is set by SetNeedsSpawnCapture and cleared the
	// next time PollOnce captures a position for it into Spawn.
	// hasTarget/targetOffset are set together by SetTarget, used to
	// (re)compute Target once Spawn is known.
	needsSpawnCapture bool
	hasTarget         bool
	targetOffset      [3]float64
}

// Vec3 is a plain 3D point — Spawn/Target are pointers to one so they
// encode as JSON null (checked directly in the dashboard's own JS, no
// separate "is this known/meaningful" flag needed) until there's a real
// position to report.
type Vec3 struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	Z float64 `json:"z"`
}

// InventoryEntry is one inventory slot for the dashboard. The zero value
// represents an empty slot, shown as an empty box so the dashboard reads as
// the bot's whole inventory, slot for slot, rather than a compacted list.
type InventoryEntry struct {
	// Slot is 1-based within its displayed group, not the raw protocol slot.
	Slot  int    `json:"slot"`
	Name  string `json:"name"`
	Count int    `json:"count"`
}

// InventorySnapshot keeps equipment visually separate from the main
// inventory and hotbar. Armor is ordered head, chest, legs, feet, matching
// screen.Inventory.Armor(). Offhand is a single slot rather than a one-item
// array so the browser can label it unambiguously.
type InventorySnapshot struct {
	Armor   []InventoryEntry `json:"armor"`
	Main    []InventoryEntry `json:"main"`
	Hotbar  []InventoryEntry `json:"hotbar"`
	Offhand InventoryEntry   `json:"offhand"`
}

// Registry is the dashboard's live view of every connected bot, indexed
// in registration order: Add/AddCompetition append, Truncate drops the
// highest-indexed entries. Safe for concurrent use by any caller driving
// UpdateEpisode, RunPoller's own background goroutine, and ServeHTTP.
type Registry struct {
	mu   sync.Mutex
	bots []*Snapshot
}

// NewRegistry creates an empty Registry.
func NewRegistry() *Registry {
	return &Registry{}
}

// Add registers a newly connected bot at the next index, with no
// competition/match identity (RunID/CompetitorID/DisplayName/Architecture
// stay empty) - see AddCompetition for a bot that has one. Returns the
// assigned index, for a caller that wants to UpdateEpisode it right away
// (e.g. to set Phase).
func (r *Registry) Add(username string, agent models.Agent) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	index := len(r.bots)
	r.bots = append(r.bots, &Snapshot{Index: index, Username: username, agent: agent})
	return index
}

// AddCompetition registers a bot with a stable identity for the
// dashboard to group/label by - originally the architecture-competition
// manifest/Prometheus-label identity, reused as-is for a combat match's
// own run/competitor/display-name identity (architecture left empty,
// since a combat participant has no analogous concept). Returns the
// assigned index, same as Add.
func (r *Registry) AddCompetition(username string, agent models.Agent, runID, competitorID, displayName, architecture string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	index := len(r.bots)
	r.bots = append(r.bots, &Snapshot{
		Index: index, Username: username, RunID: runID,
		CompetitorID: competitorID, DisplayName: displayName,
		Architecture: architecture, agent: agent,
	})
	return index
}

// ReplaceAgent swaps the live models.Agent backing bot index without
// changing its identity fields - for a caller that reconnects a bot under
// the same dashboard slot. A no-op on a nil receiver or out-of-range
// index.
func (r *Registry) ReplaceAgent(index int, agent models.Agent) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if index >= 0 && index < len(r.bots) {
		r.bots[index].agent = agent
	}
}

// Truncate drops every entry at index >= n. A no-op if n >= the current
// count, so callers don't need to check first.
func (r *Registry) Truncate(n int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if n < len(r.bots) {
		r.bots = r.bots[:n]
	}
}

// UpdateEpisode applies fn to bot i's snapshot and stamps UpdatedAt. A
// no-op on a nil receiver or an out-of-range index, so a caller that
// doesn't always have a real Registry (e.g. a unit test fixture) doesn't
// need to guard every call.
func (r *Registry) UpdateEpisode(i int, fn func(*Snapshot)) {
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

// SetNeedsSpawnCapture marks that bot i's current episode/game hasn't had
// its Spawn captured yet - the next PollOnce call will set it from
// whatever position it reads. Call from inside an UpdateEpisode fn, or
// directly for a caller that doesn't need the UpdatedAt stamp.
func (s *Snapshot) SetNeedsSpawnCapture() {
	s.needsSpawnCapture = true
	s.Spawn = nil
}

// SetTarget marks that bot i's current episode has a fixed target at
// Spawn+offset once Spawn is known (computed on each subsequent
// PollOnce); ClearTarget removes it.
func (s *Snapshot) SetTarget(offsetX, offsetY, offsetZ float64) {
	s.hasTarget = true
	s.targetOffset = [3]float64{offsetX, offsetY, offsetZ}
}

// ClearTarget removes any target set by SetTarget.
func (s *Snapshot) ClearTarget() {
	s.hasTarget = false
	s.Target = nil
}

// AgentAt returns bot i's live models.Agent — the same connection driving
// it, not a separate spectator session — for a caller that needs to read
// live world state directly (worldstream.go's SSE handler). false on a
// nil receiver or an out-of-range index.
func (r *Registry) AgentAt(i int) (models.Agent, bool) {
	if r == nil {
		return nil, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if i < 0 || i >= len(r.bots) {
		return nil, false
	}
	return r.bots[i].agent, true
}

// PollOnce refreshes every bot's position/yaw/pitch (via models.Position,
// which models.Agent always satisfies through CommandAgent/MovementAgent)
// and health/food (via models.HealthProvider, satisfied structurally by the
// concrete bot type but not part of the models.Agent interface itself,
// hence the type assertion). GetPosition/Health are both plain accessors
// over state mc-agent already tracks from the connection's own packet
// stream, not new network calls, so this is cheap even at this poller's own
// independent cadence.
func (r *Registry) PollOnce() {
	r.mu.Lock()
	bots := make([]*Snapshot, len(r.bots))
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
		var nearby []NearbyEntity
		if ok {
			nearby = readNearbyEntities(b.agent, pos.X, pos.Y, pos.Z)
		}

		r.mu.Lock()
		if ok {
			b.PolledAt = time.Now()
			b.X, b.Y, b.Z, b.Yaw, b.Pitch = pos.X, pos.Y, pos.Z, yaw, pitch
			b.Nearby = nearby
			if b.needsSpawnCapture {
				b.Spawn = &Vec3{X: pos.X, Y: pos.Y, Z: pos.Z}
				b.needsSpawnCapture = false
			}
			if b.hasTarget && b.Spawn != nil {
				b.Target = &Vec3{
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

// NearbyEntity is one other entity near a bot's own position, for the
// dashboard's zoomed single-agent view - context for "what is around this
// bot," not just where the bot itself is. Not filtered to non-players:
// GetTrackedEntities tracks everything mc-agent sees, including other
// connected players - which is what lets a combat opponent show up here
// automatically, with no combat-specific tracking code of its own.
type NearbyEntity struct {
	ID        int32   `json:"id"`
	Type      string  `json:"type"`
	ItemName  string  `json:"item_name,omitempty"`
	ItemCount int32   `json:"item_count,omitempty"`
	X         float64 `json:"x"`
	Y         float64 `json:"y"`
	Z         float64 `json:"z"`
}

// readNearbyEntities summarizes a's already-tracked nearby entities around
// (originX, originY, originZ), or nil if it has none within
// nearbyEntityRadius. GetTrackedEntities is a plain read of state mc-agent
// already tracks from the connection's own packet stream (every entity in
// the bot's own loaded view distance), not a new network call - same cost
// class as GetPosition/GetInventory.
func readNearbyEntities(a models.Agent, originX, originY, originZ float64) []NearbyEntity {
	tracked := a.GetTrackedEntities()
	if len(tracked) == 0 {
		return nil
	}
	return summarizeNearbyEntities(tracked, a.GetRegistry("minecraft:entity_type"), a.GetItemManager(), originX, originY, originZ)
}

// summarizeNearbyEntities filters tracked to live (non-Removed) entities
// within nearbyEntityRadius of the given origin, resolves each one's type
// name via reg (falling back to a numeric placeholder if reg is nil or
// doesn't recognize the ID), and returns the nearbyEntityMax nearest ones
// sorted by distance. Split out from readNearbyEntities so it's testable
// without a models.Agent fake for the whole (large) interface.
func summarizeNearbyEntities(tracked map[int32]models.TrackedEntityInfo, reg models.CustomRegistry, items models.ItemManager, originX, originY, originZ float64) []NearbyEntity {
	type candidate struct {
		NearbyEntity
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
		itemName := ""
		if name == "item" && info.HasItem && items != nil {
			itemName = strings.TrimPrefix(items.GetItemNameByID(info.ItemID), "minecraft:")
		}
		candidates = append(candidates, candidate{
			NearbyEntity: NearbyEntity{ID: info.EntityID, Type: name, ItemName: itemName, ItemCount: info.ItemCount, X: info.X, Y: info.Y, Z: info.Z},
			distSq:       distSq,
		})
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].distSq < candidates[j].distSq })
	if len(candidates) > nearbyEntityMax {
		candidates = candidates[:nearbyEntityMax]
	}

	out := make([]NearbyEntity, len(candidates))
	for i, c := range candidates {
		out[i] = c.NearbyEntity
	}
	return out
}

// readInventory reads the grouped player inventory, or returns nil if a's
// screen/item-name subsystems aren't ready yet (mirrors
// GetInventory/GetItemManager's own "nil means not ready" convention —
// neither is a new network call, both just read state mc-agent already
// tracks from the connection's own packet stream, same as GetPosition).
func readInventory(a models.Agent) *InventorySnapshot {
	inv := a.GetInventory()
	if inv == nil {
		return nil
	}
	itemMgr := a.GetItemManager()
	if itemMgr == nil {
		return nil
	}
	offhand := InventoryEntry{}
	if slot := inv.Offhand(); slot != nil {
		offhand = summarizeInventory(itemMgr, []screen.Slot{*slot})[0]
	}
	return &InventorySnapshot{
		Armor:   summarizeInventory(itemMgr, inv.Armor()),
		Main:    summarizeInventory(itemMgr, inv.Main()),
		Hotbar:  summarizeInventory(itemMgr, inv.Hotbar()),
		Offhand: offhand,
	}
}

// summarizeInventory returns one entry PER SLOT (including empty ones, as
// the zero value — see InventoryEntry's own doc comment) across every slot
// group passed, in the order given, preserving each group's own slot order
// within it. Deliberately NOT merged/summed by name — two separate stacks
// of the same item in different slots stay two separate entries, exactly
// as the player's own inventory would show them. Each resolved name has
// its "minecraft:" prefix stripped; air (a slot technically "holding" the
// air item, not actually empty) counts as empty too. Split out from
// readInventory so it's testable without a models.Agent fake for the
// whole (large) interface.
func summarizeInventory(itemMgr models.ItemManager, slotGroups ...[]screen.Slot) []InventoryEntry {
	entries := make([]InventoryEntry, 0, 36) // 27 main + 9 hotbar, the common case
	for _, slots := range slotGroups {
		for _, s := range slots {
			slot := len(entries) + 1
			if s.Count == 0 {
				entries = append(entries, InventoryEntry{Slot: slot})
				continue
			}
			name := strings.TrimPrefix(itemMgr.GetItemNameByID(int32(s.ID)), "minecraft:")
			if name == "" || name == "air" {
				entries = append(entries, InventoryEntry{Slot: slot})
				continue
			}
			entries = append(entries, InventoryEntry{Slot: slot, Name: name, Count: int(s.Count)})
		}
	}
	return entries
}

// RunPoller calls PollOnce every AgentPollInterval until ctx is done. Safe
// to start before any bot is registered (PollOnce on zero bots is a
// no-op).
func (r *Registry) RunPoller(ctx context.Context) {
	ticker := time.NewTicker(AgentPollInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			r.PollOnce()
		}
	}
}

// Snapshots returns a point-in-time copy of every bot's state, safe for
// the caller to serialize or read without holding r.mu.
func (r *Registry) Snapshots() []Snapshot {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]Snapshot, len(r.bots))
	for i, b := range r.bots {
		out[i] = *b
	}
	return out
}

// ServeHTTP serves GET /agents: the JSON array the dashboard page polls.
func (r *Registry) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		http.Error(w, "GET only", http.StatusMethodNotAllowed)
		return
	}
	respondJSON(w, http.StatusOK, r.Snapshots())
}

func respondJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
