package pvp

import (
	"context"
	"fmt"
	"testing"
)

// TestIsUnexpectedStopError_SeesThroughWrappedSentinels guards against a
// live-found regression: a deadline-exceeded/canceled error reaching
// runMatch/Run is typically wrapped by several layers (ppo.Trainer's own
// "resetting environment: %w", the RCON library's "rcon exec timeout:
// %w", arena.go's "teleport %q: %w", ...) by the time it gets here. A
// direct == comparison against the bare sentinel never matches a wrapped
// error, which live-misreported a clean deadline-based training stop as
// a hard match failure.
func TestIsUnexpectedStopError_SeesThroughWrappedSentinels(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"bare deadline exceeded", context.DeadlineExceeded, false},
		{"bare canceled", context.Canceled, false},
		{"wrapped once", fmt.Errorf("ppo: resetting environment: %w", context.DeadlineExceeded), false},
		{
			"wrapped through the real live chain",
			fmt.Errorf("ppo: resetting environment: %w",
				fmt.Errorf("teleport %q: %w", "Bot",
					fmt.Errorf("rcon exec timeout: %w", context.DeadlineExceeded))),
			false,
		},
		{"wrapped canceled", fmt.Errorf("session %q: %w", "red", context.Canceled), false},
		{"genuine failure", fmt.Errorf("agent disconnected unexpectedly"), true},
		{"wrapped genuine failure", fmt.Errorf("pvp: match %q: %w", "arena-1", fmt.Errorf("agent disconnected")), true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := isUnexpectedStopError(c.err); got != c.want {
				t.Errorf("isUnexpectedStopError(%v) = %v, want %v", c.err, got, c.want)
			}
		})
	}
}
