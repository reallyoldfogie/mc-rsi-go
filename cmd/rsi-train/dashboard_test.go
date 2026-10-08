package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"testing/fstest"

	"github.com/reallyoldfogie/mc-agent/models"
	"github.com/reallyoldfogie/mc-bot-go/bot/screen"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeItemManager resolves item IDs to names from a fixed map, the one
// piece of models.ItemManager summarizeInventory actually needs - a small,
// honest fake, unlike a whole models.Agent (see this file's own doc
// comment just above).
type fakeItemManager map[int32]string

func (f fakeItemManager) GetItemNameByID(id int32) string { return f[id] }

// pollOnce (and anything else that calls a models.Agent method) is
// deliberately not unit-tested here, consistent with reconcileEnvCount's
// own tests: this repo's live bot-connection code is exercised by its
// integration test suites, not hand-rolled models.Agent fakes for the
// whole (large) interface. A nil agent is fine for everything below since
// none of it calls into agent.

func TestAgentRegistryAddAssignsSequentialIndices(t *testing.T) {
	r := newAgentRegistry()
	r.add("bot-0", nil)
	r.add("bot-1", nil)

	snap := r.snapshot()
	require.Len(t, snap, 2)
	assert.Equal(t, 0, snap[0].Index)
	assert.Equal(t, "bot-0", snap[0].Username)
	assert.Equal(t, 1, snap[1].Index)
	assert.Equal(t, "bot-1", snap[1].Username)
}

func TestAgentRegistryTruncateDropsHighestIndexed(t *testing.T) {
	r := newAgentRegistry()
	r.add("bot-0", nil)
	r.add("bot-1", nil)
	r.add("bot-2", nil)

	r.truncate(1)

	snap := r.snapshot()
	require.Len(t, snap, 1)
	assert.Equal(t, "bot-0", snap[0].Username)
}

func TestAgentRegistryTruncateAboveCountIsNoop(t *testing.T) {
	r := newAgentRegistry()
	r.add("bot-0", nil)

	r.truncate(5)

	assert.Len(t, r.snapshot(), 1)
}

func TestAgentRegistryUpdateEpisodeAppliesFnAndStampsUpdatedAt(t *testing.T) {
	r := newAgentRegistry()
	r.add("bot-0", nil)

	r.updateEpisode(0, func(s *agentSnapshot) {
		s.Task = "goto"
		s.EpisodeStep = 3
	})

	snap := r.snapshot()
	require.Len(t, snap, 1)
	assert.Equal(t, "goto", snap[0].Task)
	assert.Equal(t, 3, snap[0].EpisodeStep)
	assert.False(t, snap[0].UpdatedAt.IsZero())
}

func TestAgentRegistryUpdateEpisodeOutOfRangeIsNoop(t *testing.T) {
	r := newAgentRegistry()
	r.add("bot-0", nil)

	assert.NotPanics(t, func() {
		r.updateEpisode(5, func(s *agentSnapshot) { s.Task = "should-not-apply" })
	})
}

func TestAgentRegistryUpdateEpisodeNilReceiverIsNoop(t *testing.T) {
	var r *agentRegistry
	assert.NotPanics(t, func() {
		r.updateEpisode(0, func(s *agentSnapshot) { s.Task = "unreachable" })
	})
}

func TestAgentRegistryServeHTTPReturnsSnapshotAsJSON(t *testing.T) {
	r := newAgentRegistry()
	r.add("bot-0", nil)
	r.updateEpisode(0, func(s *agentSnapshot) { s.Task = "mine" })

	req := httptest.NewRequest(http.MethodGet, "/agents", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	var got []agentSnapshot
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &got))
	require.Len(t, got, 1)
	assert.Equal(t, "bot-0", got[0].Username)
	assert.Equal(t, "mine", got[0].Task)
}

func TestAgentRegistryServeHTTPRejectsNonGet(t *testing.T) {
	r := newAgentRegistry()

	req := httptest.NewRequest(http.MethodPost, "/agents", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
}

func TestServeDashboardPageServesEmbeddedHTML(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/dashboard", nil)
	w := httptest.NewRecorder()
	serveDashboardPage(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Header().Get("Content-Type"), "text/html")
	assert.Contains(t, w.Body.String(), "<!doctype html>")
}

func TestItemIconFallbackServesBareNameDirectly(t *testing.T) {
	fsys := fstest.MapFS{"stick.png": &fstest.MapFile{Data: []byte("stick-bytes")}}
	h := itemIconFallback{fsys: fsys, next: http.FileServer(http.FS(fsys))}

	req := httptest.NewRequest(http.MethodGet, "/stick.png", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "stick-bytes", w.Body.String())
}

func TestItemIconFallbackFallsBackToDirectionalSuffix(t *testing.T) {
	fsys := fstest.MapFS{"crafting_table_top.png": &fstest.MapFile{Data: []byte("top-bytes")}}
	h := itemIconFallback{fsys: fsys, next: http.FileServer(http.FS(fsys))}

	req := httptest.NewRequest(http.MethodGet, "/crafting_table.png", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "top-bytes", w.Body.String())
}

func TestItemIconFallbackPrefersEarlierSuffixOverLater(t *testing.T) {
	fsys := fstest.MapFS{
		"furnace_top.png":  &fstest.MapFile{Data: []byte("top-bytes")},
		"furnace_side.png": &fstest.MapFile{Data: []byte("side-bytes")},
	}
	h := itemIconFallback{fsys: fsys, next: http.FileServer(http.FS(fsys))}

	req := httptest.NewRequest(http.MethodGet, "/furnace.png", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "side-bytes", w.Body.String(), "itemIconFallbackSuffixes' own order (_side before _top) must be honored")
}

func TestItemIconFallback404sWhenNoVariantExists(t *testing.T) {
	fsys := fstest.MapFS{}
	h := itemIconFallback{fsys: fsys, next: http.FileServer(http.FS(fsys))}

	req := httptest.NewRequest(http.MethodGet, "/totally_not_a_real_item.png", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusNotFound, w.Code)
}

func TestSummarizeInventoryPreservesSlotOrderAcrossGroupsWithoutMerging(t *testing.T) {
	itemMgr := fakeItemManager{1: "minecraft:oak_log", 2: "minecraft:stick"}
	main := []screen.Slot{{ID: 1, Count: 3}, {ID: 2, Count: 1}}
	hotbar := []screen.Slot{{ID: 1, Count: 2}}

	got := summarizeInventory(itemMgr, main, hotbar)

	// Two oak_log stacks in different slots stay two separate entries, in
	// main-then-hotbar slot order, matching the real inventory screen -
	// not merged/summed the way the dashboard used to.
	assert.Equal(t, []inventoryEntry{
		{Slot: 1, Name: "oak_log", Count: 3},
		{Slot: 2, Name: "stick", Count: 1},
		{Slot: 3, Name: "oak_log", Count: 2},
	}, got)
}

func TestSummarizeInventoryRepresentsEmptyAndAirSlotsAsEmptyEntriesInPlace(t *testing.T) {
	itemMgr := fakeItemManager{0: "minecraft:air", 1: "minecraft:stick"}
	slots := []screen.Slot{{ID: 0, Count: 0}, {ID: 0, Count: 1}, {ID: 1, Count: 0}, {ID: 1, Count: 4}}

	got := summarizeInventory(itemMgr, slots)

	// One entry per slot, in slot order - empty/air slots become the zero
	// value plus their own Slot number (an empty, numbered box) rather
	// than being dropped, so the strip shows which slot each item is
	// actually in.
	assert.Equal(t, []inventoryEntry{
		{Slot: 1}, {Slot: 2}, {Slot: 3}, {Slot: 4, Name: "stick", Count: 4},
	}, got)
}

func TestSummarizeInventoryEmptyInventoryReturnsEmptyNotNil(t *testing.T) {
	got := summarizeInventory(fakeItemManager{}, nil)

	assert.NotNil(t, got, "an empty inventory must still report zero items, not be mistaken for a 'not ready yet' nil")
	assert.Empty(t, got)
}

func TestSummarizeInventoryAllEmptySlotsReportsOneEntryPerSlot(t *testing.T) {
	main := make([]screen.Slot, 27) // a bot holding nothing: every slot genuinely empty.
	hotbar := make([]screen.Slot, 9)

	got := summarizeInventory(fakeItemManager{}, main, hotbar)

	assert.Len(t, got, 36, "an all-empty inventory is still 36 slots, not 0 - the dashboard needs every box to draw a full strip")
	for i, e := range got {
		assert.Equal(t, inventoryEntry{Slot: i + 1}, e)
	}
}

// fakeEntityTypeRegistry resolves entity type IDs to names from a fixed
// map, the one piece of models.CustomRegistry summarizeNearbyEntities
// actually needs.
type fakeEntityTypeRegistry map[int32]string

func (f fakeEntityTypeRegistry) GetID() string                       { return "minecraft:entity_type" }
func (f fakeEntityTypeRegistry) GetNameByID(id int32) (string, bool) { n, ok := f[id]; return n, ok }
func (f fakeEntityTypeRegistry) GetIDByName(string) (int32, bool)    { return 0, false }
func (f fakeEntityTypeRegistry) IsReady() bool                       { return true }
func (f fakeEntityTypeRegistry) Dump()                               {}

func TestSummarizeNearbyEntitiesFiltersByRadiusAndSortsByDistance(t *testing.T) {
	reg := fakeEntityTypeRegistry{1: "minecraft:item", 2: "minecraft:zombie"}
	tracked := map[int32]models.TrackedEntityInfo{
		10: {EntityID: 10, EntityType: 2, X: 100, Y: 0, Z: 0},              // far away - excluded
		11: {EntityID: 11, EntityType: 1, X: 5, Y: 0, Z: 0},                // nearest
		12: {EntityID: 12, EntityType: 1, X: 0, Y: 0, Z: 10},               // next nearest
		13: {EntityID: 13, EntityType: 2, X: 0, Y: 0, Z: 0, Removed: true}, // excluded
	}

	got := summarizeNearbyEntities(tracked, reg, 0, 0, 0)

	assert.Equal(t, []nearbyEntity{
		{Type: "item", X: 5, Y: 0, Z: 0},
		{Type: "item", X: 0, Y: 0, Z: 10},
	}, got)
}

func TestSummarizeNearbyEntitiesUnresolvedTypeFallsBackToNumericName(t *testing.T) {
	tracked := map[int32]models.TrackedEntityInfo{1: {EntityID: 1, EntityType: 99, X: 1, Y: 0, Z: 0}}

	got := summarizeNearbyEntities(tracked, fakeEntityTypeRegistry{}, 0, 0, 0)

	require.Len(t, got, 1)
	assert.Equal(t, "entity-99", got[0].Type)
}

func TestSummarizeNearbyEntitiesNilRegistryFallsBackToNumericName(t *testing.T) {
	tracked := map[int32]models.TrackedEntityInfo{1: {EntityID: 1, EntityType: 7, X: 1, Y: 0, Z: 0}}

	got := summarizeNearbyEntities(tracked, nil, 0, 0, 0)

	require.Len(t, got, 1)
	assert.Equal(t, "entity-7", got[0].Type)
}

func TestSummarizeNearbyEntitiesCapsAtMax(t *testing.T) {
	reg := fakeEntityTypeRegistry{1: "minecraft:item"}
	tracked := make(map[int32]models.TrackedEntityInfo, nearbyEntityMax+10)
	for i := int32(0); i < nearbyEntityMax+10; i++ {
		// Spaced well under a block apart so every one of them - not just
		// the radius filter's own survivors - is within nearbyEntityRadius,
		// exercising the max-count cap specifically.
		tracked[i] = models.TrackedEntityInfo{EntityID: i, EntityType: 1, X: float64(i) * 0.1, Y: 0, Z: 0}
	}

	got := summarizeNearbyEntities(tracked, reg, 0, 0, 0)

	assert.Len(t, got, nearbyEntityMax)
}
