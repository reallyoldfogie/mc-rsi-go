package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"
)

// envCountRequest serves POST /config/parallel-envs?n=N: an operator's way
// to ask for a different number of parallel bot sessions without
// restarting the trainer (and so without resetting to a fresh random
// generation, losing checkpoint state, or any of the rest of what a
// restart costs - see checkpointExporter's own doc comment for that
// history). It only ever records the latest request; applying it (adding
// or closing bot sessions) is run's own job, done once between rounds via
// reconcileEnvCount - never from this handler's own goroutine, since nothing
// about live-adjusting connected/envs is safe to do while leapfrog.Round is
// concurrently driving rollout collection against them.
type envCountRequest struct {
	mu      sync.Mutex
	pending *int
}

func newEnvCountRequest() *envCountRequest {
	return &envCountRequest{}
}

// take returns the most recently requested count, if one hasn't already
// been taken, and clears it. Called once per round-loop iteration.
func (r *envCountRequest) take() (int, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.pending == nil {
		return 0, false
	}
	n := *r.pending
	r.pending = nil
	return n, true
}

func (r *envCountRequest) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	n, err := strconv.Atoi(req.URL.Query().Get("n"))
	if err != nil || n < 1 {
		http.Error(w, "n must be a positive integer", http.StatusBadRequest)
		return
	}

	r.mu.Lock()
	r.pending = &n
	r.mu.Unlock()

	respondJSON(w, http.StatusAccepted, map[string]string{
		"status": "requested " + strconv.Itoa(n) + " parallel envs; will apply at the next round boundary",
	})
}

// parallelEnvsStabilityWindow is how long a live-scaled parallel-envs
// count must keep running, with no crash, before it gets persisted for
// future restarts to pick up in place of -parallel-envs's own flag
// value. Chosen from a live incident: a scale-up to 10 envs disconnected
// one of the newly added bots (a timeout) about 5 minutes in, taking the
// whole process down with it (any one bot session ending stops the
// run). 10 minutes gives real margin above that without making the
// feature feel unresponsive - most of a round still elapses before a
// scale is "trusted."
const parallelEnvsStabilityWindow = 10 * time.Minute

// parallelEnvsStateFile is where the last stable parallel-envs count is
// persisted (plain JSON, not a pkg/lineage/pkg/checkpoint file - this
// isn't training state, just the operator's last-proven-good scaling
// choice).
func parallelEnvsStateFile(checkpointDir string) string {
	return filepath.Join(checkpointDir, "parallel-envs.json")
}

type parallelEnvsState struct {
	Count int `json:"count"`
}

// loadPersistedParallelEnvs reads the last count a live scale proved
// stable for parallelEnvsStabilityWindow, if any. Absent or unreadable —
// including the case this exists specifically to handle, a scale whose
// parallelEnvsConfirmer timer never got to fire because the process
// crashed first — means the caller falls back to -parallel-envs's own
// flag value, exactly as if no scale had ever been requested.
func loadPersistedParallelEnvs(checkpointDir string) (int, bool) {
	data, err := os.ReadFile(parallelEnvsStateFile(checkpointDir))
	if err != nil {
		return 0, false
	}
	var state parallelEnvsState
	if err := json.Unmarshal(data, &state); err != nil || state.Count < 1 {
		return 0, false
	}
	return state.Count, true
}

// savePersistedParallelEnvs atomically (temp file plus rename, so a
// crash mid-write can never leave a truncated/corrupt file behind)
// records n as the new last-proven-stable parallel-envs count.
func savePersistedParallelEnvs(checkpointDir string, n int) error {
	tmp, err := os.CreateTemp(checkpointDir, filepath.Base(parallelEnvsStateFile(checkpointDir))+".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := json.NewEncoder(tmp).Encode(parallelEnvsState{Count: n}); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, parallelEnvsStateFile(checkpointDir))
}

// parallelEnvsConfirmer schedules savePersistedParallelEnvs to run only
// once a live-scaled count has survived parallelEnvsStabilityWindow
// without the process exiting — see that constant's own doc comment.
// Deliberately never writes anything itself at startup, even when
// resuming from an already-persisted count: only run's round loop, on
// an actual successful reconcileEnvCount, ever calls scheduleConfirm, so
// a count that was never (re-)scaled this process's lifetime is simply
// left untouched on disk.
type parallelEnvsConfirmer struct {
	checkpointDir string
	// delay defaults to parallelEnvsStabilityWindow; a field (not a
	// direct reference to that const) only so tests can shrink it rather
	// than waiting out the real window.
	delay time.Duration
	mu    sync.Mutex
	timer *time.Timer
}

func newParallelEnvsConfirmer(checkpointDir string) *parallelEnvsConfirmer {
	return &parallelEnvsConfirmer{checkpointDir: checkpointDir, delay: parallelEnvsStabilityWindow}
}

// scheduleConfirm (re)starts the stability timer for n, cancelling
// whichever still-pending confirmation (for a since-superseded count)
// this replaces — so rapidly scaling 7 -> 8 -> 10 only ever persists 10,
// never a value already abandoned before it could prove itself.
func (c *parallelEnvsConfirmer) scheduleConfirm(n int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.timer != nil {
		c.timer.Stop()
	}
	c.timer = time.AfterFunc(c.delay, func() {
		if err := savePersistedParallelEnvs(c.checkpointDir, n); err != nil {
			log.Printf("persisting parallel-envs count %d: %v", n, err)
		} else {
			log.Printf("parallel-envs count %d held stable for %s; persisted for future restarts", n, c.delay)
		}
	})
}
