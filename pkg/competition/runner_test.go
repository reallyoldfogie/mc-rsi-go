package competition

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum"
)

type fakeSession struct {
	identity Identity
	mu       sync.Mutex
	deadline time.Time
	planHash string
	closed   bool
}

func (s *fakeSession) Identity() Identity { return s.identity }

func (s *fakeSession) Train(ctx context.Context) (TrainingSummary, error) {
	deadline, ok := ctx.Deadline()
	if !ok {
		return TrainingSummary{}, context.Canceled
	}
	s.mu.Lock()
	s.deadline = deadline
	s.mu.Unlock()
	<-ctx.Done()
	return TrainingSummary{Epochs: 3, Samples: 30}, ctx.Err()
}

func (s *fakeSession) Evaluate(_ context.Context, plan EvaluationPlan) (EvaluationSummary, error) {
	hash, err := plan.Hash()
	if err != nil {
		return EvaluationSummary{}, err
	}
	s.mu.Lock()
	s.planHash = hash
	s.mu.Unlock()
	return EvaluationSummary{Episodes: len(plan.Scenarios), MeanReward: 4.5, CompletionRate: 0.5}, nil
}

func (s *fakeSession) Close() error {
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
	return nil
}

func (s *fakeSession) snapshot() (time.Time, string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.deadline, s.planHash, s.closed
}

func TestRunUsesOneDeadlineAndEveryEvaluationPlanForEverySession(t *testing.T) {
	start := time.Now()
	window, err := NewTrainingWindow(start, Duration(20*time.Millisecond))
	require.NoError(t, err)
	plan, err := NewEvaluationPlan(testPool(), 99, 3)
	require.NoError(t, err)
	sessions := []Session{
		&fakeSession{identity: Identity{RunID: "run", CompetitorID: "basic"}},
		&fakeSession{identity: Identity{RunID: "run", CompetitorID: "wide"}},
	}

	result, err := Run(context.Background(), sessions, window, []EvaluationPlan{plan}, time.Minute)
	require.NoError(t, err)
	assert.Len(t, result.Competitors, 2)
	assert.Equal(t, "complete", result.Competitors[0].Status)
	assert.Equal(t, "complete", result.Competitors[1].Status)
	deadline, _, _ := sessions[0].(*fakeSession).snapshot()
	assert.Equal(t, window.Deadline, deadline)

	wantHash, err := plan.Hash()
	require.NoError(t, err)
	for _, raw := range sessions {
		deadline, planHash, closed := raw.(*fakeSession).snapshot()
		assert.Equal(t, window.Deadline, deadline)
		assert.Equal(t, wantHash, planHash)
		assert.True(t, closed)
	}
}

type evaluationBarrier struct {
	mu       sync.Mutex
	entered  int
	allReady chan struct{}
	release  chan struct{}
}

type parallelEvalSession struct {
	identity Identity
	barrier  *evaluationBarrier
}

func (s *parallelEvalSession) Identity() Identity { return s.identity }
func (s *parallelEvalSession) Train(ctx context.Context) (TrainingSummary, error) {
	<-ctx.Done()
	return TrainingSummary{}, ctx.Err()
}
func (s *parallelEvalSession) Evaluate(ctx context.Context, plan EvaluationPlan) (EvaluationSummary, error) {
	s.barrier.mu.Lock()
	s.barrier.entered++
	if s.barrier.entered == 2 {
		close(s.barrier.allReady)
	}
	s.barrier.mu.Unlock()
	select {
	case <-s.barrier.release:
		return EvaluationSummary{Episodes: len(plan.Scenarios)}, nil
	case <-ctx.Done():
		return EvaluationSummary{}, ctx.Err()
	}
}
func (s *parallelEvalSession) Close() error { return nil }

func TestRunEvaluatesCompetitorsInParallelButKeepsOneSessionPerEnvironment(t *testing.T) {
	window, err := NewTrainingWindow(time.Now(), Duration(10*time.Millisecond))
	require.NoError(t, err)
	plan, err := NewEvaluationPlan(testPool(), 7, 2)
	require.NoError(t, err)
	barrier := &evaluationBarrier{allReady: make(chan struct{}), release: make(chan struct{})}
	sessions := []Session{
		&parallelEvalSession{identity: Identity{RunID: "run", CompetitorID: "a"}, barrier: barrier},
		&parallelEvalSession{identity: Identity{RunID: "run", CompetitorID: "b"}, barrier: barrier},
	}

	resultCh := make(chan CompetitionResult, 1)
	errCh := make(chan error, 1)
	go func() {
		result, runErr := Run(context.Background(), sessions, window, []EvaluationPlan{plan}, time.Minute)
		resultCh <- result
		errCh <- runErr
	}()

	select {
	case <-barrier.allReady:
		// Both Evaluate calls reached their barrier before either was released.
		// This proves the runner started them concurrently.
		close(barrier.release)
	case <-time.After(time.Second):
		t.Fatal("evaluation calls did not overlap")
	}
	result := <-resultCh
	require.NoError(t, <-errCh)
	assert.Equal(t, "complete", result.Competitors[0].Status)
	assert.Equal(t, "complete", result.Competitors[1].Status)
}

func TestRunStopsEvaluationAtTheTimeLimitWithoutWaitingForSlowCompetitors(t *testing.T) {
	window, err := NewTrainingWindow(time.Now(), Duration(10*time.Millisecond))
	require.NoError(t, err)
	plan, err := NewEvaluationPlan(testPool(), 7, 2)
	require.NoError(t, err)
	// barrier.release is never closed, so Evaluate blocks until its context
	// is done - exactly what a competitor whose episodes never terminate
	// early would look like.
	barrier := &evaluationBarrier{allReady: make(chan struct{}), release: make(chan struct{})}
	sessions := []Session{&parallelEvalSession{identity: Identity{RunID: "run", CompetitorID: "slow"}, barrier: barrier}}

	start := time.Now()
	result, err := Run(context.Background(), sessions, window, []EvaluationPlan{plan}, 30*time.Millisecond)
	require.NoError(t, err)
	assert.Less(t, time.Since(start), time.Second, "Run must not wait past the evaluation time limit")
	assert.Equal(t, "evaluation_time_limit_exceeded", result.Competitors[0].Status)
	assert.Equal(t, context.DeadlineExceeded.Error(), result.Competitors[0].Error)
}

func TestRunRejectsNonPositiveEvaluationTimeLimit(t *testing.T) {
	window, err := NewTrainingWindow(time.Now(), Duration(time.Second))
	require.NoError(t, err)
	plan, err := NewEvaluationPlan(testPool(), 1, 1)
	require.NoError(t, err)
	sessions := []Session{&fakeSession{identity: Identity{RunID: "run", CompetitorID: "a"}}}

	_, err = Run(context.Background(), sessions, window, []EvaluationPlan{plan}, 0)
	require.Error(t, err)
	assert.ErrorContains(t, err, "evaluation time limit")
}

func TestRunRecordsTrainingFailureAndStillClosesEverySession(t *testing.T) {
	window, err := NewTrainingWindow(time.Now(), Duration(time.Second))
	require.NoError(t, err)
	plan, err := NewEvaluationPlan(testPool(), 1, 1)
	require.NoError(t, err)
	failing := &errorSession{identity: Identity{RunID: "run", CompetitorID: "bad"}}
	good := &fakeSession{identity: Identity{RunID: "run", CompetitorID: "good"}}
	result, err := Run(context.Background(), []Session{failing, good}, window, []EvaluationPlan{plan}, time.Minute)
	require.NoError(t, err)
	assert.Equal(t, "training_failed", result.Competitors[0].Status)
	assert.Equal(t, "complete", result.Competitors[1].Status)
	assert.True(t, good.snapshotClosed())
}

type errorSession struct{ identity Identity }

func (s *errorSession) Identity() Identity { return s.identity }
func (s *errorSession) Train(context.Context) (TrainingSummary, error) {
	return TrainingSummary{}, assert.AnError
}
func (s *errorSession) Evaluate(context.Context, EvaluationPlan) (EvaluationSummary, error) {
	return EvaluationSummary{}, nil
}
func (s *errorSession) Close() error { return nil }

func (s *fakeSession) snapshotClosed() bool {
	_, _, closed := s.snapshot()
	return closed
}

func testPool() curriculum.Pool {
	return curriculum.Pool{TargetOffsets: [][3]float64{{5, 0, 0}}}
}
