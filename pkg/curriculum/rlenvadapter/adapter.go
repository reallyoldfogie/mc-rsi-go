// Package rlenvadapter converts a pkg/curriculum.Generator into a real
// mc-agent rlenv.Config.TaskSelector — the "thin adapter"
// docs/plans/07-curriculum-generator.md's own "Package layout" section
// calls for, kept in its own package specifically so pkg/curriculum
// itself never needs to import mc-agent/rlenv (see that package's own
// doc comment). Anything that wants to actually drive a live
// rlenv.Environment with a curriculum.Generator imports this package;
// anything that only wants to test/use curriculum selection logic
// imports pkg/curriculum alone and never pulls in mc-agent at all.
package rlenvadapter

import (
	oldrand "math/rand" // rlenv.TaskSelector's own rng parameter type — see TaskSelector's doc comment for why it's unused here.
	"math/rand/v2"

	"github.com/reallyoldfogie/mc-agent/rlenv"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum"
)

// TaskSelector wraps gen as an rlenv.TaskSelector: each call asks gen for
// this episode's TaskSpec (via rng, not rlenv's own passed-in
// *math/rand.Rand — curriculum.Generator is deliberately built around
// math/rand/v2, matching the rest of this repo, so the caller supplies
// its own v2 source here up front rather than this adapter bridging
// rlenv's older math/rand type on every call) and converts it into a
// TaskOverride that activates exactly that one task, disabling the other
// two — matching docs/plans/07's "Selection strategy": one task per
// episode, not composed multi-task episodes (see that document's own
// "Reward composition" section on why composing was deliberately not
// attempted).
func TaskSelector(gen curriculum.Generator, rng *rand.Rand) rlenv.TaskSelector {
	return func(episode int, _ *oldrand.Rand) rlenv.TaskOverride {
		spec := gen.Next(episode, rng)

		override := rlenv.TaskOverride{GoToTargetDisabled: true}
		switch spec.Type {
		case curriculum.TaskGoto:
			override.GoToTargetDisabled = false
			override.TargetOffset = spec.TargetOffset
		case curriculum.TaskMine:
			override.MineTargetBlock = spec.MineTargetBlock
			override.MineSearchRadius = spec.MineSearchRadius
		case curriculum.TaskCraft:
			override.CraftTargetItem = spec.CraftTargetItem
		}
		return override
	}
}
