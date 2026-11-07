package main

import (
	"log"
	"sync"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
)

// recurrentCheckpointJob is one already-safe-to-serialize snapshot a
// backgroundCheckpointWriter writes to disk - every field is either a
// plain value or an independent copy (Params via RecurrentParams.Snapshot,
// Optimizer via Adam.State's own deep copy, RNGState via
// rand.PCG.MarshalBinary, EnvStates via captureEnvironmentStates), never a
// reference into the live trainer's own, still-mutating state. That
// matters because the job is written on a goroutine that keeps running
// after the epoch loop has already moved on to the next epoch, which
// mutates the trainer's real parameters/optimizer moments in place - see
// backgroundCheckpointWriter's own doc comment.
type recurrentCheckpointJob struct {
	interimDir       string
	params           *actorcritic.RecurrentParams
	environmentID    string
	epoch            int
	progress         checkpoint.Metadata
	parentGeneration int
	optimizer        actorcritic.AdamState
	rngState         []byte
	envStates        []environmentState
}

// write performs the job's own slow disk I/O (JSON marshaling plus an
// atomic rename for each of two files) - the part backgroundCheckpointWriter
// exists to keep off the epoch loop's own critical path. Errors are
// logged directly, not returned: nothing downstream of a background
// write is waiting on its result the way the old synchronous call's
// caller could be.
func (j recurrentCheckpointJob) write() {
	if err := saveInterimRecurrentCheckpoint(j.interimDir, j.params, j.environmentID, j.epoch, j.progress, j.parentGeneration); err != nil {
		log.Printf("  saving recurrent interim checkpoint: %v", err)
		return
	}
	if err := savePersistedTrainingState(j.interimDir, persistedTrainingState{
		SchemaVersion:    1,
		ParentGeneration: j.parentGeneration,
		Epoch:            j.epoch,
		Optimizer:        &j.optimizer,
		RNG:              j.rngState,
		Environments:     j.envStates,
	}); err != nil {
		log.Printf("  saving recurrent continuation state: %v", err)
	}
}

// checkpointJob is the one thing backgroundCheckpointWriter needs from a
// submitted job - recurrentCheckpointJob's own write method satisfies
// this directly. Pulled out as its own interface so
// backgroundCheckpointWriter's coalescing/ordering behavior is
// unit-testable with a fake job that never touches disk, instead of only
// through a live training run.
type checkpointJob interface{ write() }

// backgroundCheckpointWriter runs one recurrentCheckpointJob's slow disk
// write in the background while training continues, instead of blocking
// the next epoch's rollout collection for the write's full duration.
// Found live, 2026-11-07 (docs/plans/21-buried-mining-success.md): the
// Grafana "training throughput" panel showed a recurring multi-minute
// zero-throughput stretch once episodes got much shorter (an earlier,
// separate fix); log analysis confirmed a real ~97s stretch of total
// agent silence at an epoch boundary, not just a metrics-granularity
// artifact - and since sample count per epoch had dropped roughly 50x
// without a matching drop in epoch duration, the per-epoch
// -checkpoint-interval-1 interim-checkpoint save (a multi-megabyte
// training-state.json written synchronously, inline in the same callback
// that gated the next epoch's rollout collection from starting at all)
// was the leading suspect, not the backward pass itself.
//
// Coalesces rather than queues: Submit while a write is already running
// replaces whatever was pending (if anything) instead of piling up a
// backlog - an older pending job is always superseded by a newer one
// before it would ever be written, so dropping it costs nothing. At most
// one write runs at a time, so the last rename to each checkpoint file
// always reflects the most recently submitted job that got written, never
// an out-of-order overwrite by a slower, earlier one racing a faster,
// later one.
//
// Callers must construct one and immediately `defer w.Wait()`, so every
// return path out of the enclosing function blocks until any in-flight
// background write finishes - and must also call Wait explicitly before
// performing a final, synchronous save of the same files once Submit will
// never be called again (see runRecurrentRounds for why the deferred Wait
// alone isn't enough there: a defer runs after the function body, which
// includes that final save, not before it).
//
// Submit/Wait are not safe to call concurrently with themselves - both
// are only ever called from the single goroutine driving the epoch loop
// (Submit from its own per-epoch callback, Wait once after that loop
// exits), never from two goroutines at once.
type backgroundCheckpointWriter struct {
	mu      sync.Mutex
	pending checkpointJob
	busy    bool
	wg      sync.WaitGroup
}

// Submit hands job off to the background writer, starting one if none is
// currently running.
func (w *backgroundCheckpointWriter) Submit(job checkpointJob) {
	w.mu.Lock()
	w.pending = job
	if w.busy {
		w.mu.Unlock()
		return
	}
	w.busy = true
	w.wg.Add(1)
	w.mu.Unlock()
	go w.run()
}

func (w *backgroundCheckpointWriter) run() {
	defer w.wg.Done()
	for {
		w.mu.Lock()
		job := w.pending
		w.pending = nil
		if job == nil {
			w.busy = false
			w.mu.Unlock()
			return
		}
		w.mu.Unlock()
		job.write()
	}
}

// Wait blocks until the currently in-flight write, if any, finishes -
// including one that became in-flight after a job pending when Wait was
// called got picked up, so this never returns while run is still working
// through anything Submit had already handed it.
func (w *backgroundCheckpointWriter) Wait() {
	w.wg.Wait()
}
