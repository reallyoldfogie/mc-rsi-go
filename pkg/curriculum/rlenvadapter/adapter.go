// Package rlenvadapter defines every task curriculum.Pool can pose and
// turns one of them into a real mc-agent rlenv.Config.TaskSelector — the
// "thin adapter" docs/plans/07-curriculum-generator.md's own "Package
// layout" section calls for, kept in its own package specifically so
// pkg/curriculum itself never needs to import mc-agent/rlenv (see that
// package's own doc comment). Anything that wants to actually drive a
// live rlenv.Environment against a curriculum.Pool imports this package;
// anything that only wants to load/validate a curriculum config imports
// pkg/curriculum alone and never pulls in mc-agent at all.
//
// Task kinds (goto, mine, craft, the "far" composites, each chain stage)
// are each their own small struct implementing Task, registered in
// allFactories — see Task's own doc comment for why: the previous design
// here (and in cmd/rsi-train, before that) classified an episode's task
// from which combination of rlenv.TaskOverride fields happened to be set,
// in a switch that a new task kind had to be added to correctly, in sync
// with every other switch doing the same classification elsewhere. A new
// task kind is now a new struct (+ a factory for it, appended to
// allFactories) — self-contained, nothing elsewhere to keep in sync.
package rlenvadapter

import (
	"fmt"
	oldrand "math/rand" // rlenv.TaskSelector's own rng parameter type — see TaskSelector's doc comment for why it's unused here.
	"math/rand/v2"
	"strings"

	"github.com/reallyoldfogie/mc-agent/rlenv"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum"
)

// Task is one episode's chosen task: a concrete value drawn from a
// TaskFactory, carrying whatever parameters that draw settled on (a
// goto's target offset, a mine's block, ...). Name labels the task kind
// (metric labels, the dashboard's Task stat); GoalDescription states its
// actual success condition in plain language (the dashboard's Goal
// stat); ToOverride poses it to rlenv. Every task kind is its own small
// struct satisfying this interface — see allFactories.
type Task interface {
	Name() string
	GoalDescription() string
	ToOverride() rlenv.TaskOverride
}

// TaskFactory draws a random Task of its own kind from pool, or reports
// that pool has nothing to draw (Available) so Select leaves it out of
// that episode's draw.
type TaskFactory interface {
	Available(pool curriculum.Pool) bool
	Draw(pool curriculum.Pool, rng *rand.Rand) Task
}

// allFactories is every task kind Select can draw, in a fixed order (so
// that, for a given rng stream, which kinds are available never itself
// depends on map iteration or similar non-determinism) — the one place a
// new task kind's factory needs to be registered to make it selectable.
// Every other property of a task kind (its name, its goal, its override)
// lives entirely on the kind's own Task/TaskFactory implementation below,
// not here.
var allFactories = []TaskFactory{
	gotoFactory{},
	mineFactory{},
	craftFactory{},
	mineFarFactory{},
	craftFarFactory{},
	mineBuriedFactory{},
	chainGatherFactory{},
	chainTableFactory{},
	chainPlaceFactory{},
	chainUseFactory{},
	chainMineFactory{},
}

func shortName(id string) string { return strings.TrimPrefix(id, "minecraft:") }

func gotoGoalDescription(offset [3]float64) string {
	return fmt.Sprintf("goto (%+.0f, %+.0f, %+.0f) from spawn", offset[0], offset[1], offset[2])
}

// gotoTask poses "reach a point offset from wherever Reset finds the bot."
type gotoTask struct{ offset [3]float64 }

func (t gotoTask) Name() string            { return "goto" }
func (t gotoTask) GoalDescription() string { return gotoGoalDescription(t.offset) }
func (t gotoTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{GoToTargetDisabled: false, TargetOffset: t.offset}
}

type gotoFactory struct{}

func (gotoFactory) Available(pool curriculum.Pool) bool { return len(pool.TargetOffsets) > 0 }
func (gotoFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	return gotoTask{offset: pool.TargetOffsets[rng.IntN(len(pool.TargetOffsets))]}
}

// mineTask poses mining a block already within reach.
type mineTask struct {
	block        string
	searchRadius int
}

func (t mineTask) Name() string            { return "mine" }
func (t mineTask) GoalDescription() string { return "mine " + shortName(t.block) }
func (t mineTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{GoToTargetDisabled: true, MineTargetBlock: t.block, MineSearchRadius: t.searchRadius}
}

type mineFactory struct{}

func (mineFactory) Available(pool curriculum.Pool) bool { return len(pool.MineTargetBlocks) > 0 }
func (mineFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	block := pool.MineTargetBlocks[rng.IntN(len(pool.MineTargetBlocks))]
	return mineTask{block: block, searchRadius: pool.MineSearchRadius}
}

// mineBuriedTask poses mining a block buried in the shared template's
// underground stone band (docs/plans/14) — structurally identical to
// mineTask (no goto/far-seed component: the target is already within
// reach, just not yet visible), plus a tool seeded into inventory first
// since the target can't realistically be reached bare-handed.
type mineBuriedTask struct {
	block                string
	searchRadius         int
	requiredTool         string
	requiredToolEnchants map[string]int
}

func (t mineBuriedTask) Name() string            { return "mine_buried" }
func (t mineBuriedTask) GoalDescription() string { return "mine buried " + shortName(t.block) }
func (t mineBuriedTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled:           true,
		MineTargetBlock:              t.block,
		MineSearchRadius:             t.searchRadius,
		MineRequiredTool:             t.requiredTool,
		MineRequiredToolEnchantments: t.requiredToolEnchants,
		// The target already exists, baked into the shared clone
		// template — DefaultEpisodeSeeder must not also place a fresh,
		// immediately-visible instance beside the bot (caught live,
		// against a real server: it otherwise does exactly that, since
		// the real buried instance is never visible by design).
		MineTargetBuried: true,
	}
}

type mineBuriedFactory struct{}

func (mineBuriedFactory) Available(pool curriculum.Pool) bool {
	return len(pool.MineBuriedTargetBlocks) > 0
}
func (mineBuriedFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	block := pool.MineBuriedTargetBlocks[rng.IntN(len(pool.MineBuriedTargetBlocks))]
	return mineBuriedTask{
		block:                block,
		searchRadius:         pool.MineBuriedSearchRadius,
		requiredTool:         pool.MineBuriedRequiredTool,
		requiredToolEnchants: pool.MineBuriedRequiredToolEnchantments,
	}
}

// craftTask poses crafting an item already craftable from held ingredients.
type craftTask struct{ item string }

func (t craftTask) Name() string            { return "craft" }
func (t craftTask) GoalDescription() string { return "craft " + shortName(t.item) }
func (t craftTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{GoToTargetDisabled: true, CraftTargetItem: t.item}
}

type craftFactory struct{}

func (craftFactory) Available(pool curriculum.Pool) bool { return len(pool.CraftTargetItems) > 0 }
func (craftFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	return craftTask{item: pool.CraftTargetItems[rng.IntN(len(pool.CraftTargetItems))]}
}

// mineFarTask poses "go there, then mine": the block is seeded beside the
// goto target, out of sight until the bot travels there.
type mineFarTask struct {
	offset       [3]float64
	block        string
	searchRadius int
}

func (t mineFarTask) Name() string            { return "mine_far" }
func (t mineFarTask) GoalDescription() string { return "mine " + shortName(t.block) }
func (t mineFarTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled: false,
		TargetOffset:       t.offset,
		MineTargetBlock:    t.block,
		MineSearchRadius:   t.searchRadius,
		SeedAtGoal:         true,
	}
}

type mineFarFactory struct{}

func (mineFarFactory) Available(pool curriculum.Pool) bool {
	return len(pool.MineFarTargetOffsets) > 0 && len(pool.MineFarBlocks) > 0
}
func (mineFarFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	offset := pool.MineFarTargetOffsets[rng.IntN(len(pool.MineFarTargetOffsets))]
	block := pool.MineFarBlocks[rng.IntN(len(pool.MineFarBlocks))]
	return mineFarTask{offset: offset, block: block, searchRadius: pool.MineFarSearchRadius}
}

// craftFarTask poses "go there, then craft": a crafting table is seeded
// beside the goto target, out of the craft-ready search radius until the
// bot travels there.
type craftFarTask struct {
	offset       [3]float64
	item         string
	searchRadius int
}

func (t craftFarTask) Name() string            { return "craft_far" }
func (t craftFarTask) GoalDescription() string { return "craft " + shortName(t.item) }
func (t craftFarTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled: false,
		TargetOffset:       t.offset,
		CraftTargetItem:    t.item,
		CraftSearchRadius:  t.searchRadius,
		SeedAtGoal:         true,
	}
}

type craftFarFactory struct{}

func (craftFarFactory) Available(pool curriculum.Pool) bool {
	return len(pool.CraftFarTargetOffsets) > 0 && len(pool.CraftFarItems) > 0 && pool.CraftFarSearchRadius > 0
}
func (craftFarFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	offset := pool.CraftFarTargetOffsets[rng.IntN(len(pool.CraftFarTargetOffsets))]
	item := pool.CraftFarItems[rng.IntN(len(pool.CraftFarItems))]
	return craftFarTask{offset: offset, item: item, searchRadius: pool.CraftFarSearchRadius}
}

// chainStageNames names each stage of the multi-step crafting chain
// (docs/plans/12 and, for stage 5, docs/plans/14) — the one list that
// needs a new entry if rlenv ever adds another stage; everything else
// about a chain episode (its goal, its override) is shared across all
// five stages below, since they differ only in stage number and which
// craft item/mine target (if any) they carry.
var chainStageNames = map[int]string{1: "chain_gather", 2: "chain_table", 3: "chain_place", 4: "chain_use", 5: "chain_mine"}

// chainTask poses one stage of the multi-step crafting chain: a tree of
// curriculum.ChainTreeBlock is seeded beside the goto target, out of
// sight from the start, and the episode asks for stage 1 (gather a log),
// 2 (craft planks and a table), 3 (place the table), 4 (craft the goal
// item at it), or 5/ChainMine (then find and mine mineTarget, a block
// buried in the shared template's underground stone band —
// docs/plans/14-buried-block-mining-and-memory-map.md's
// "chain_buried_mining": unlike mine_buried, the pickaxe is crafted from
// scratch here, never given). All five stages are modeled as one struct,
// not five near-identical ones, since they share everything but
// stage/craftItem/mineTarget.
type chainTask struct {
	stage        int
	offset       [3]float64
	searchRadius int
	craftItem    string // "" for stage 1; curriculum.ChainTableItem for 2/3; the goal item for 4/5.
	mineTarget   string // only meaningful at stage 5 (ChainMine): the buried block to mine once craftItem is crafted.
}

func (t chainTask) Name() string {
	if name, ok := chainStageNames[t.stage]; ok {
		return name
	}
	return "chain_unknown"
}

// GoalDescription states the chain's actual, final success condition —
// crafting the goal item for every stage except ChainMine, where the
// chain continues past that craft and only mining the buried block
// afterward actually ends the episode.
func (t chainTask) GoalDescription() string {
	if t.stage == rlenv.ChainMine {
		return "mine buried " + shortName(t.mineTarget)
	}
	return "craft " + shortName(t.craftItem)
}
func (t chainTask) ToOverride() rlenv.TaskOverride {
	return rlenv.TaskOverride{
		GoToTargetDisabled:   false,
		TargetOffset:         t.offset,
		MineTargetBlock:      curriculum.ChainTreeBlock,
		MineSearchRadius:     t.searchRadius,
		CraftTargetItem:      t.craftItem,
		ChainStage:           t.stage,
		ChainMineTargetBlock: t.mineTarget,
		CollectDrops:         true,
		SeedAtGoal:           true,
	}
}

// chainAvailable is shared by every chain stage's factory: all four need
// the same offsets/radius; only stage 4 (chainUseFactory) additionally
// needs a goal item.
func chainAvailable(pool curriculum.Pool) bool {
	return len(pool.ChainTargetOffsets) > 0 && pool.ChainSearchRadius > 0
}

type chainGatherFactory struct{}

func (chainGatherFactory) Available(pool curriculum.Pool) bool { return chainAvailable(pool) }
func (chainGatherFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	offset := pool.ChainTargetOffsets[rng.IntN(len(pool.ChainTargetOffsets))]
	return chainTask{stage: 1, offset: offset, searchRadius: pool.ChainSearchRadius}
}

type chainTableFactory struct{}

func (chainTableFactory) Available(pool curriculum.Pool) bool { return chainAvailable(pool) }
func (chainTableFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	offset := pool.ChainTargetOffsets[rng.IntN(len(pool.ChainTargetOffsets))]
	return chainTask{stage: 2, offset: offset, searchRadius: pool.ChainSearchRadius, craftItem: curriculum.ChainTableItem}
}

type chainPlaceFactory struct{}

func (chainPlaceFactory) Available(pool curriculum.Pool) bool { return chainAvailable(pool) }
func (chainPlaceFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	offset := pool.ChainTargetOffsets[rng.IntN(len(pool.ChainTargetOffsets))]
	return chainTask{stage: 3, offset: offset, searchRadius: pool.ChainSearchRadius, craftItem: curriculum.ChainTableItem}
}

type chainUseFactory struct{}

func (chainUseFactory) Available(pool curriculum.Pool) bool {
	return chainAvailable(pool) && len(pool.ChainGoalItems) > 0
}
func (chainUseFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	offset := pool.ChainTargetOffsets[rng.IntN(len(pool.ChainTargetOffsets))]
	item := pool.ChainGoalItems[rng.IntN(len(pool.ChainGoalItems))]
	return chainTask{stage: 4, offset: offset, searchRadius: pool.ChainSearchRadius, craftItem: item}
}

// chainMineFactory draws "chain_buried_mining" (docs/plans/14): needs a
// goal item to craft (the same pool ChainUse draws from) *and* at least
// one buried block to mine afterward — missing either leaves the type
// unavailable, the same way craftFarFactory needs both its offsets and
// its items.
type chainMineFactory struct{}

func (chainMineFactory) Available(pool curriculum.Pool) bool {
	return chainAvailable(pool) && len(pool.ChainGoalItems) > 0 && len(pool.ChainMineTargetBlocks) > 0
}
func (chainMineFactory) Draw(pool curriculum.Pool, rng *rand.Rand) Task {
	offset := pool.ChainTargetOffsets[rng.IntN(len(pool.ChainTargetOffsets))]
	item := pool.ChainGoalItems[rng.IntN(len(pool.ChainGoalItems))]
	mineTarget := pool.ChainMineTargetBlocks[rng.IntN(len(pool.ChainMineTargetBlocks))]
	return chainTask{stage: rlenv.ChainMine, offset: offset, searchRadius: pool.ChainSearchRadius, craftItem: item, mineTarget: mineTarget}
}

// errEmptyPool is ValidatePool's error when pool has nothing to draw.
var errEmptyPool = fmt.Errorf("rlenvadapter: pool has no candidate values for any task kind")

// ValidatePool reports an error if pool has no task kind with at least one
// candidate value (otherwise Select would have nothing to draw) — call
// once at startup against an operator-supplied curriculum config, so a
// misconfigured pool fails fast with a clear error instead of panicking
// the first time an episode needs a task.
func ValidatePool(pool curriculum.Pool) error {
	for _, f := range allFactories {
		if f.Available(pool) {
			return nil
		}
	}
	return errEmptyPool
}

// Select implements docs/plans/07's "Selection strategy": picks a task
// kind uniformly at random from whichever of allFactories pool actually
// has candidates for (equal probability among however many are available
// this episode), then asks that kind's own factory to draw a uniformly
// random parameter set from pool. No difficulty progression — see
// docs/plans/07's own "deliberate v2 deferral" note on why that's not
// attempted here. Panics if pool is empty — callers validate with
// ValidatePool first, same two-step shape this package's predecessor
// (NewUniformRandom, then Next) already had.
func Select(pool curriculum.Pool, rng *rand.Rand) Task {
	var available []TaskFactory
	for _, f := range allFactories {
		if f.Available(pool) {
			available = append(available, f)
		}
	}
	if len(available) == 0 {
		panic("rlenvadapter: Select called with an empty pool; call ValidatePool first")
	}
	return available[rng.IntN(len(available))].Draw(pool, rng)
}

// TaskSelector wraps pool as an rlenv.TaskSelector for a caller that only
// needs to pose the task rlenv.Config.TaskSelector expects, not the Task
// itself (its Name/GoalDescription — see cmd/rsi-train's own
// TaskSelector closure, which calls Select directly instead, for exactly
// that). Each call draws a fresh Task via Select and converts it.
func TaskSelector(pool curriculum.Pool, rng *rand.Rand) rlenv.TaskSelector {
	return func(int, *oldrand.Rand) rlenv.TaskOverride {
		return Select(pool, rng).ToOverride()
	}
}
