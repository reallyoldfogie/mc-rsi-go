package main

import (
	"net/http"
	"sync/atomic"
)

// restartRequest serves POST /control/restart-at-epoch-boundary: an
// operator's way to ask the trainer to exit cleanly right after its
// current Student-training epoch finishes, instead of having to watch
// rsi_current_epoch and kill -TERM the process by hand at the right
// moment (the manual process this exists to replace: swapping in a
// rebuilt bin/rsi-train-multistep, or a config edit, without losing more
// than the one epoch in flight).
//
// Checked once per OnEpoch firing (run()/runRecurrentRounds's own OnEpoch
// callback), where a request calls the same stop() context.CancelFunc
// the networkDead path already uses - ctx cancellation unwinds the round
// loop cleanly (saving a final recovery checkpoint, same as any other
// between-epochs stop) and run() returns nil, unlike networkDead's own
// errNetworkDead. That distinction matters: main() exits 0 for a nil
// return, which supervise-rsi-train.sh treats as an ordinary stop and
// restarts automatically - exactly the point of asking for this instead
// of a plain kill. A nonzero/errNetworkDead exit is supervise-rsi-train.sh's
// own signal not to restart.
type restartRequest struct {
	pending atomic.Bool
}

func newRestartRequest() *restartRequest {
	return &restartRequest{}
}

// take reports whether a restart was requested, clearing the flag so a
// later epoch doesn't also stop after this run already has. Called once
// per OnEpoch firing.
func (r *restartRequest) take() bool {
	return r.pending.CompareAndSwap(true, false)
}

func (r *restartRequest) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	r.pending.Store(true)
	respondJSON(w, http.StatusAccepted, map[string]string{
		"status": "requested a restart at the end of the current epoch",
	})
}
