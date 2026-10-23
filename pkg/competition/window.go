package competition

import (
	"context"
	"fmt"
	"time"
)

// TrainingWindow gives every competitor the same absolute deadline. Sessions
// should receive the context returned by Context and use StartedAt/Deadline
// for reporting rather than calculating independent durations themselves.
type TrainingWindow struct {
	StartedAt time.Time `json:"started_at"`
	Deadline  time.Time `json:"deadline"`
}

func NewTrainingWindow(start time.Time, duration Duration) (TrainingWindow, error) {
	if start.IsZero() {
		return TrainingWindow{}, fmt.Errorf("training window start must not be zero")
	}
	if time.Duration(duration) <= 0 {
		return TrainingWindow{}, fmt.Errorf("training window duration must be positive")
	}
	return TrainingWindow{StartedAt: start, Deadline: start.Add(time.Duration(duration))}, nil
}

// Context derives a deadline context from the competition window. The caller
// owns the returned cancel function and must call it when its session exits.
func (w TrainingWindow) Context(parent context.Context) (context.Context, context.CancelFunc, error) {
	if w.StartedAt.IsZero() || w.Deadline.IsZero() || !w.Deadline.After(w.StartedAt) {
		return nil, nil, fmt.Errorf("invalid training window")
	}
	ctx, cancel := context.WithDeadline(parent, w.Deadline)
	return ctx, cancel, nil
}

func (w TrainingWindow) Remaining(now time.Time) time.Duration {
	if now.After(w.Deadline) {
		return 0
	}
	if now.Before(w.StartedAt) {
		return w.Deadline.Sub(w.StartedAt)
	}
	return w.Deadline.Sub(now)
}
