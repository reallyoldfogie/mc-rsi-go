package main

import (
	"net/http"
	"strconv"
	"sync"
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
