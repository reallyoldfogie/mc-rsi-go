package pvp

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// scriptedEnv is a fake rl.Environment whose Step results are scripted per
// call, so tests can simulate one participant dying on a specific tick
// while a sibling keeps playing.
type scriptedEnv struct {
	mu        sync.Mutex
	steps     []rl.StepResult // consumed in order; Done defaults to false past the end
	stepCalls int
	resets    int
}

func (e *scriptedEnv) ObservationSize() int { return 1 }
func (e *scriptedEnv) ActionSpace() int     { return 1 }

func (e *scriptedEnv) Reset(ctx context.Context) (rl.Observation, error) {
	e.mu.Lock()
	e.resets++
	e.mu.Unlock()
	return rl.Observation{Values: []float32{0}}, nil
}

func (e *scriptedEnv) Step(ctx context.Context, action rl.Action) (rl.StepResult, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	idx := e.stepCalls
	e.stepCalls++
	if idx < len(e.steps) {
		return e.steps[idx], nil
	}
	return rl.StepResult{Observation: rl.Observation{Values: []float32{0}}, Reward: 0, Done: false}, nil
}

func (e *scriptedEnv) stepCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.stepCalls
}

func (e *scriptedEnv) resetCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.resets
}

func countingReset(calls *int32) MatchResetFunc {
	return func(ctx context.Context) error {
		atomic.AddInt32(calls, 1)
		return nil
	}
}

func TestMatchEnvironment_SiblingDeathForcesOtherDone(t *testing.T) {
	var resetCalls int32
	barrier := newMatchBarrier(2, countingReset(&resetCalls))

	dyingUnderlying := &scriptedEnv{steps: []rl.StepResult{
		{Observation: rl.Observation{Values: []float32{1}}, Reward: 1, Done: true}, // dies on step 0
	}}
	survivingUnderlying := &scriptedEnv{steps: []rl.StepResult{
		{Observation: rl.Observation{Values: []float32{1}}, Reward: 1, Done: false},
		{Observation: rl.Observation{Values: []float32{1}}, Reward: 1, Done: false},
	}}

	dying := newMatchEnvironment(dyingUnderlying, barrier)
	surviving := newMatchEnvironment(survivingUnderlying, barrier)

	ctx := context.Background()

	result, err := dying.Step(ctx, 0)
	if err != nil {
		t.Fatalf("dying.Step: %v", err)
	}
	if !result.Done {
		t.Fatal("dying combatant's own death should report Done=true")
	}

	// The surviving combatant's own underlying environment says Done=false
	// on this tick, but the barrier already knows the match ended - its
	// wrapped Step must override Done to true.
	result, err = surviving.Step(ctx, 0)
	if err != nil {
		t.Fatalf("surviving.Step: %v", err)
	}
	if !result.Done {
		t.Fatal("surviving combatant's episode should be forced Done=true once a sibling dies")
	}
	if survivingUnderlying.stepCount() != 1 {
		t.Fatalf("surviving underlying step count = %d, want exactly 1 (the real action was still taken)", survivingUnderlying.stepCount())
	}

	// Both now call Reset - this should unblock once both have arrived and
	// perform the shared reset exactly once.
	var wg sync.WaitGroup
	errs := make([]error, 2)
	wg.Add(2)
	go func() { defer wg.Done(); _, errs[0] = dying.Reset(ctx) }()
	go func() { defer wg.Done(); _, errs[1] = surviving.Reset(ctx) }()
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("Reset[%d]: %v", i, err)
		}
	}
	if got := atomic.LoadInt32(&resetCalls); got != 1 {
		t.Fatalf("ResetMatch called %d times, want exactly 1", got)
	}
	if dyingUnderlying.resetCount() != 1 || survivingUnderlying.resetCount() != 1 {
		t.Fatalf("underlying Reset counts = %d/%d, want 1/1", dyingUnderlying.resetCount(), survivingUnderlying.resetCount())
	}

	// After the round reset, the barrier must be clean for the next round:
	// a fresh Step should NOT be forced Done just because the previous
	// round ended with Done=true.
	result, err = surviving.Step(ctx, 0)
	if err != nil {
		t.Fatalf("surviving.Step (round 2): %v", err)
	}
	if result.Done {
		t.Fatal("a fresh round must not inherit the previous round's Done state")
	}
}

func TestMatchBarrier_ResetBlocksUntilAllArrive(t *testing.T) {
	var resetCalls int32
	barrier := newMatchBarrier(3, countingReset(&resetCalls))

	ctx := context.Background()
	done := make(chan struct{}, 2)
	for i := 0; i < 2; i++ {
		go func() {
			_ = barrier.arriveAtReset(ctx)
			done <- struct{}{}
		}()
	}

	select {
	case <-done:
		t.Fatal("arriveAtReset returned before all 3 participants arrived")
	case <-time.After(100 * time.Millisecond):
		// expected: still blocked with only 2/3 arrived
	}

	if err := barrier.arriveAtReset(ctx); err != nil {
		t.Fatalf("third arriveAtReset: %v", err)
	}
	<-done
	<-done

	if got := atomic.LoadInt32(&resetCalls); got != 1 {
		t.Fatalf("ResetMatch called %d times, want exactly 1", got)
	}
}

func TestMatchBarrier_IsCyclicAcrossMultipleRounds(t *testing.T) {
	var resetCalls int32
	barrier := newMatchBarrier(2, countingReset(&resetCalls))
	ctx := context.Background()

	for round := 1; round <= 3; round++ {
		var wg sync.WaitGroup
		wg.Add(2)
		for i := 0; i < 2; i++ {
			go func() {
				defer wg.Done()
				if err := barrier.arriveAtReset(ctx); err != nil {
					t.Errorf("round arriveAtReset: %v", err)
				}
			}()
		}
		wg.Wait()
		if got := atomic.LoadInt32(&resetCalls); got != int32(round) {
			t.Fatalf("after round %d: ResetMatch called %d times, want %d", round, got, round)
		}
	}
}

func TestMatchBarrier_PropagatesResetError(t *testing.T) {
	wantErr := fmt.Errorf("boom")
	barrier := newMatchBarrier(1, MatchResetFunc(func(ctx context.Context) error { return wantErr }))
	if err := barrier.arriveAtReset(context.Background()); err != wantErr {
		t.Fatalf("arriveAtReset error = %v, want %v", err, wantErr)
	}
}

func TestMatchBarrier_ContextCancellationUnblocksWaiter(t *testing.T) {
	barrier := newMatchBarrier(2, MatchResetFunc(func(ctx context.Context) error { return nil }))
	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() { errCh <- barrier.arriveAtReset(ctx) }()

	// Give the goroutine a moment to actually start waiting before canceling.
	time.Sleep(20 * time.Millisecond)
	cancel()

	select {
	case err := <-errCh:
		if err != context.Canceled {
			t.Fatalf("arriveAtReset error = %v, want context.Canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("arriveAtReset did not unblock after context cancellation")
	}
}
