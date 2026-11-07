package main

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeCheckpointJob is a checkpointJob that never touches disk - it
// records that it ran (via ran, incremented once per write call) and
// optionally blocks until release is closed, so a test can hold one write
// in flight while submitting more jobs behind it. started, if set, is
// closed the instant write begins - a test that needs to submit more jobs
// only after this one has actually been picked up by run() (not merely
// handed to Submit, which races the background goroutine's own startup)
// waits on it first.
type fakeCheckpointJob struct {
	id      int
	ran     *int32
	order   *[]int
	orderMu *sync.Mutex
	started chan struct{}
	release <-chan struct{}
}

func (j fakeCheckpointJob) write() {
	if j.started != nil {
		close(j.started)
	}
	if j.release != nil {
		<-j.release
	}
	atomic.AddInt32(j.ran, 1)
	if j.order != nil {
		j.orderMu.Lock()
		*j.order = append(*j.order, j.id)
		j.orderMu.Unlock()
	}
}

func TestBackgroundCheckpointWriterRunsASingleSubmittedJob(t *testing.T) {
	var w backgroundCheckpointWriter
	var ran int32
	w.Submit(fakeCheckpointJob{id: 1, ran: &ran})
	w.Wait()
	if ran != 1 {
		t.Fatalf("ran = %d, want 1", ran)
	}
}

func TestBackgroundCheckpointWriterWaitWithNothingSubmittedReturnsImmediately(t *testing.T) {
	var w backgroundCheckpointWriter
	done := make(chan struct{})
	go func() {
		w.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Wait blocked with nothing ever submitted")
	}
}

// TestBackgroundCheckpointWriterCoalescesPendingJobs is the core property
// backgroundCheckpointWriter exists for (see its own doc comment): while
// one job is still writing, submitting several more must not queue them
// all up - only the *last* one submitted before the first write finishes
// should ever actually run, and it must run after the first, never
// concurrently with it.
func TestBackgroundCheckpointWriterCoalescesPendingJobs(t *testing.T) {
	var w backgroundCheckpointWriter
	var ran int32
	var order []int
	var orderMu sync.Mutex
	release := make(chan struct{})
	started := make(chan struct{})

	// job 1 blocks on release, simulating a slow write still in flight.
	w.Submit(fakeCheckpointJob{id: 1, ran: &ran, order: &order, orderMu: &orderMu, started: started, release: release})
	<-started // wait until job 1 has actually been picked up by run(), not just handed to Submit

	// Submitted while job 1 is still blocked: job 2 must be superseded by
	// job 3 before either ever gets a chance to run, so only job 3 (the
	// last one submitted) should ever actually write.
	w.Submit(fakeCheckpointJob{id: 2, ran: &ran, order: &order, orderMu: &orderMu})
	w.Submit(fakeCheckpointJob{id: 3, ran: &ran, order: &order, orderMu: &orderMu})

	close(release) // let job 1 proceed
	w.Wait()

	if ran != 2 {
		t.Fatalf("ran = %d jobs, want exactly 2 (job 1, then only the last-submitted job 3 - job 2 must be coalesced away)", ran)
	}
	orderMu.Lock()
	defer orderMu.Unlock()
	if len(order) != 2 || order[0] != 1 || order[1] != 3 {
		t.Fatalf("write order = %v, want [1 3] (job 2 coalesced, job 3 runs strictly after job 1)", order)
	}
}

// TestBackgroundCheckpointWriterWaitBlocksUntilInFlightWriteFinishes
// confirms Wait doesn't return early while a write - even one picked up
// only after Wait was already waiting on the WaitGroup - is still running.
func TestBackgroundCheckpointWriterWaitBlocksUntilInFlightWriteFinishes(t *testing.T) {
	var w backgroundCheckpointWriter
	var ran int32
	release := make(chan struct{})
	w.Submit(fakeCheckpointJob{id: 1, ran: &ran, release: release})

	waitReturned := make(chan struct{})
	go func() {
		w.Wait()
		close(waitReturned)
	}()

	select {
	case <-waitReturned:
		t.Fatal("Wait returned before the in-flight job's release was closed")
	case <-time.After(50 * time.Millisecond):
	}

	close(release)
	select {
	case <-waitReturned:
	case <-time.After(time.Second):
		t.Fatal("Wait never returned after the in-flight job finished")
	}
	if ran != 1 {
		t.Fatalf("ran = %d, want 1", ran)
	}
}
