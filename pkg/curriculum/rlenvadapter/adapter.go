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

// Convert turns one curriculum.TaskSpec into the rlenv.TaskOverride that
// poses it — the mechanical half of what TaskSelector does, exported
// separately so a caller that also wants the TaskSpec itself (its Type,
// for labeling/logging/metrics — see TaskType.String()/TaskSpec.
// GoalDescription — not just the task it poses) can draw one from its own
// curriculum.Generator and convert it, rather than re-deriving the same
// information downstream from TaskOverride's raw field combination (which
// Convert's own switch below would then have to stay in sync with by
// hand). Activates exactly one task, disabling the other two — matching
// docs/plans/07's "Selection strategy": one task per episode. The
// exception is the far types (TaskMineFar/TaskCraftFar): "go there, then
// mine/craft" composites, which activate the goto task alongside a mine
// or craft task and ask rlenv to seed the mine block or table beside the
// goto target (see rlenv.TaskOverride.SeedAtGoal).
func Convert(spec curriculum.TaskSpec) rlenv.TaskOverride {
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
	case curriculum.TaskMineFar:
		override.GoToTargetDisabled = false
		override.TargetOffset = spec.TargetOffset
		override.MineTargetBlock = spec.MineTargetBlock
		override.MineSearchRadius = spec.MineSearchRadius
		override.SeedAtGoal = true
	case curriculum.TaskChainGather, curriculum.TaskChainTable, curriculum.TaskChainPlace, curriculum.TaskChainUse:
		// The multi-step chain (docs/plans/12): a tree seeded out of
		// sight beside the goto target; mine its logs (collecting the
		// drops), craft, place, and craft the goal at the placed table.
		override.GoToTargetDisabled = false
		override.TargetOffset = spec.TargetOffset
		override.MineTargetBlock = spec.MineTargetBlock
		override.MineSearchRadius = spec.MineSearchRadius
		override.CraftTargetItem = spec.CraftTargetItem
		override.ChainStage = spec.Type.ChainStage()
		override.CollectDrops = true
		override.SeedAtGoal = true
	case curriculum.TaskCraftFar:
		override.GoToTargetDisabled = false
		override.TargetOffset = spec.TargetOffset
		override.CraftTargetItem = spec.CraftTargetItem
		override.CraftSearchRadius = spec.CraftSearchRadius
		override.SeedAtGoal = true
	}
	return override
}

// TaskSelector wraps gen as an rlenv.TaskSelector: each call asks gen for
// this episode's TaskSpec (via rng, not rlenv's own passed-in
// *math/rand.Rand — curriculum.Generator is deliberately built around
// math/rand/v2, matching the rest of this repo, so the caller supplies
// its own v2 source here up front rather than this adapter bridging
// rlenv's older math/rand type on every call) and Converts it. A caller
// that only needs to pose the task rlenv.Config.TaskSelector expects uses
// this directly; one that also wants the chosen TaskSpec itself calls
// gen.Next and Convert separately instead (see Convert's own doc comment)
// rather than drawing a second, different TaskSpec from gen here.
func TaskSelector(gen curriculum.Generator, rng *rand.Rand) rlenv.TaskSelector {
	return func(episode int, _ *oldrand.Rand) rlenv.TaskOverride {
		return Convert(gen.Next(episode, rng))
	}
}
