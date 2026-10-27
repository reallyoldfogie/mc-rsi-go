package liveview

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// worldStreamRadius bounds how far out Registry.WorldStreamHandler asks
// FindAllVisibleBlocksInSphere to ray-trace each tick. Its own doc comment
// (mc-agent's models/agent_actions.go) benchmarks radius=32 at ~6000 rays;
// 24 sits deliberately short of that as a starting point until this is
// actually measured against a live bot - trivial to retune once it is.
const worldStreamRadius = 24

// worldStreamTickInterval reuses AgentPollInterval's own cadence rather
// than inventing a second polling rate for essentially the same "how
// often is a human watching this actually going to notice" tradeoff.
const worldStreamTickInterval = AgentPollInterval

// worldStreamBlockInterval is intentionally slower than position/entity
// updates. Ray-tracing the visible block set is the expensive part of the
// stream and does not need to run four times per second to make a fast-moving
// agent view feel responsive.
const worldStreamBlockInterval = time.Second

// worldStreamPositionEvent is one "position" SSE message — the bot's own
// eye position and look direction, same fields Snapshot already reports
// over GET /agents, just pushed instead of polled.
type worldStreamPositionEvent struct {
	Index int     `json:"index"`
	Type  string  `json:"type"`
	X     float64 `json:"x"`
	Y     float64 `json:"y"`
	Z     float64 `json:"z"`
	Yaw   float64 `json:"yaw"`
	Pitch float64 `json:"pitch"`
}

// worldStreamBlock is one visible block, directly off
// models.VisibleBlockInfo — BlockName is already resolved there, so there's
// nothing left for this layer to look up.
type worldStreamBlock struct {
	X    int    `json:"x"`
	Y    int    `json:"y"`
	Z    int    `json:"z"`
	Name string `json:"name"`
}

// worldStreamBlocksEvent is one "blocks" SSE message — every non-air block
// currently visible from the bot's own position (see
// agent.FindAllVisibleBlocksInSphere's own doc comment for how "visible"
// is defined: real ray-traced line of sight, not just "within some box").
type worldStreamBlocksEvent struct {
	Index  int                `json:"index"`
	Type   string             `json:"type"`
	Blocks []worldStreamBlock `json:"blocks"`
}

// worldStreamEntitiesEvent is the current tracked-entity set near the bot.
// The browser uses each entity ID to update/remove billboards without
// accumulating stale Three.js objects.
type worldStreamEntitiesEvent struct {
	Index    int            `json:"index"`
	Type     string         `json:"type"`
	Entities []NearbyEntity `json:"entities"`
}

// worldStreamGoneEvent is the final message sent before the handler
// returns because the requested agent index no longer resolves (a
// parallel-envs scale-down, or a combat bot disconnecting at match end) —
// lets the frontend drop back to 2D with a clear reason instead of just
// watching the connection go quiet.
type worldStreamGoneEvent struct {
	Index  int    `json:"index"`
	Type   string `json:"type"`
	Reason string `json:"reason"`
}

// isAirBlockName reports whether name is one of Minecraft's air variants
// (regular, cave, or void air) - see WorldStreamHandler's own call site
// for why a ray-traced "visible" block can still come back named this.
func isAirBlockName(name string) bool {
	switch strings.TrimPrefix(name, "minecraft:") {
	case "air", "cave_air", "void_air":
		return true
	default:
		return false
	}
}

// WorldStreamHandler serves GET /agents/world?index=N (or index=all): a
// Server-Sent Events stream of that bot's own live position, nearby
// entities, and visible-block data, read straight off the same
// models.Agent connection already driving it — no separate spectator bot,
// no RCON, no third-party Minecraft-version data of any kind. SSE rather
// than WebSocket because the data only ever flows server to browser; the
// browser's one piece of input (which bot to watch) is the URL itself, so
// there's nothing a bidirectional protocol would buy here that plain
// net/http doesn't already give for free.
func (r *Registry) WorldStreamHandler(w http.ResponseWriter, req *http.Request) {
	indexParam := req.URL.Query().Get("index")
	if indexParam == "all" {
		r.worldStreamAllHandler(w, req)
		return
	}
	index, err := strconv.Atoi(indexParam)
	if err != nil {
		http.Error(w, "index query parameter is required", http.StatusBadRequest)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	ticker := time.NewTicker(worldStreamTickInterval)
	defer ticker.Stop()
	lastBlockScan := time.Time{}

	for {
		agent, ok := r.AgentAt(index)
		if !ok {
			writeSSE(w, worldStreamGoneEvent{Type: "gone", Reason: fmt.Sprintf("agent %d is no longer connected", index)})
			flusher.Flush()
			return
		}

		pos, yaw, pitch, initialized := agent.GetPosition()
		if initialized {
			writeSSE(w, worldStreamPositionEvent{Type: "position", X: pos.X, Y: pos.Y, Z: pos.Z, Yaw: yaw, Pitch: pitch})
			writeSSE(w, worldStreamEntitiesEvent{
				Type:     "entities",
				Entities: readNearbyEntities(agent, pos.X, pos.Y, pos.Z),
			})
		}

		if time.Since(lastBlockScan) < worldStreamBlockInterval {
			select {
			case <-req.Context().Done():
				return
			case <-ticker.C:
			}
			continue
		}
		lastBlockScan = time.Now()

		// Bounded independently of the tick interval: a single slow
		// ray-trace (e.g. a momentarily overloaded host) should time out
		// and let the loop try again next tick, not stall the whole
		// stream indefinitely.
		scanCtx, cancel := context.WithTimeout(req.Context(), 5*time.Second)
		visible, err := agent.FindAllVisibleBlocksInSphere(scanCtx, worldStreamRadius)
		cancel()
		if err != nil {
			log.Printf("liveview: FindAllVisibleBlocksInSphere for agent %d: %v", index, err)
		} else {
			blocks := make([]worldStreamBlock, 0, len(visible))
			for _, b := range visible {
				if isAirBlockName(b.BlockName) {
					// FindAllVisibleBlocksInSphere's own ray-trace only
					// ever hits a loaded, non-passable (so non-air)
					// block - but it re-reads each hit coordinate's
					// stateID/name a second time afterward to build its
					// result, which races a concurrent mine elsewhere on
					// a shared server: if another bot mines that exact
					// block between the ray hit and that re-read, the
					// re-read can see air. Harmless to whatever this
					// bot itself is doing - just skip it rather than
					// hand the 3D view a block named "air" to texture
					// as a solid cube.
					continue
				}
				blocks = append(blocks, worldStreamBlock{X: b.X, Y: b.Y, Z: b.Z, Name: b.BlockName})
			}
			writeSSE(w, worldStreamBlocksEvent{Type: "blocks", Blocks: blocks})
		}
		flusher.Flush()

		select {
		case <-req.Context().Done():
			return
		case <-ticker.C:
		}
	}
}

// worldStreamAllHandler multiplexes every agent's world updates onto one SSE
// connection. Browsers commonly cap the number of concurrent HTTP/1.1
// connections per origin; opening one persistent EventSource per agent made
// the later cells in a large All-agents grid silently starve. Every event is
// tagged with its agent index so the browser can route it to that cell. For
// a combat match, this is the grid view that shows both fighters' own
// visible-block/nearby-entity views side by side - each one's own view
// already includes the opponent as a tracked entity, with no extra code.
func (r *Registry) worldStreamAllHandler(w http.ResponseWriter, req *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	ticker := time.NewTicker(worldStreamTickInterval)
	defer ticker.Stop()
	lastBlockScan := make(map[int]time.Time)
	known := make(map[int]bool)

	for {
		snapshots := r.Snapshots()
		current := make(map[int]bool, len(snapshots))
		type initializedAgent struct {
			snapshot Snapshot
		}
		initialized := make([]initializedAgent, 0, len(snapshots))

		// Send every fast-changing position/entity update before doing any
		// expensive block ray traces. This prevents a slow early agent from
		// starving the later cells in the grid.
		for _, snapshot := range snapshots {
			current[snapshot.Index] = true
			known[snapshot.Index] = true
			if snapshot.agent == nil {
				continue
			}
			pos, yaw, pitch, ready := snapshot.agent.GetPosition()
			if !ready {
				continue
			}
			writeSSE(w, worldStreamPositionEvent{Index: snapshot.Index, Type: "position", X: pos.X, Y: pos.Y, Z: pos.Z, Yaw: yaw, Pitch: pitch})
			writeSSE(w, worldStreamEntitiesEvent{
				Index:    snapshot.Index,
				Type:     "entities",
				Entities: readNearbyEntities(snapshot.agent, pos.X, pos.Y, pos.Z),
			})
			initialized = append(initialized, initializedAgent{snapshot: snapshot})
		}

		for _, item := range initialized {
			if time.Since(lastBlockScan[item.snapshot.Index]) < worldStreamBlockInterval {
				continue
			}
			lastBlockScan[item.snapshot.Index] = time.Now()
			scanCtx, cancel := context.WithTimeout(req.Context(), 5*time.Second)
			visible, err := item.snapshot.agent.FindAllVisibleBlocksInSphere(scanCtx, worldStreamRadius)
			cancel()
			if err != nil {
				log.Printf("liveview: FindAllVisibleBlocksInSphere for agent %d: %v", item.snapshot.Index, err)
				continue
			}
			blocks := make([]worldStreamBlock, 0, len(visible))
			for _, b := range visible {
				if isAirBlockName(b.BlockName) {
					continue
				}
				blocks = append(blocks, worldStreamBlock{X: b.X, Y: b.Y, Z: b.Z, Name: b.BlockName})
			}
			writeSSE(w, worldStreamBlocksEvent{Index: item.snapshot.Index, Type: "blocks", Blocks: blocks})
		}

		for index := range known {
			if current[index] {
				continue
			}
			writeSSE(w, worldStreamGoneEvent{Index: index, Type: "gone", Reason: fmt.Sprintf("agent %d is no longer connected", index)})
			delete(known, index)
			delete(lastBlockScan, index)
		}
		flusher.Flush()

		select {
		case <-req.Context().Done():
			return
		case <-ticker.C:
		}
	}
}

// writeSSE writes v as a single Server-Sent Events "data:" message. Errors
// are deliberately swallowed — a write failing here means the client's
// connection is already going away, which the next loop iteration's
// req.Context().Done() check will notice and exit on; there's nothing
// more this call itself can usefully do about it.
func writeSSE(w http.ResponseWriter, v any) {
	data, err := json.Marshal(v)
	if err != nil {
		log.Printf("liveview: marshal %T: %v", v, err)
		return
	}
	fmt.Fprintf(w, "data: %s\n\n", data)
}
