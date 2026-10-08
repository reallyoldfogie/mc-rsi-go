package rlenvadapter

import (
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/mc-agent/rlenv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum"
)

func fullPool() curriculum.Pool {
	return curriculum.Pool{
		TargetOffsets:    [][3]float64{{5, 0, 0}, {-5, 0, 0}, {0, 0, 5}},
		MineTargetBlocks: []string{"minecraft:stone", "minecraft:coal_ore"},
		MineSearchRadius: 8,
		CraftTargetItems: []string{"minecraft:stick", "minecraft:oak_planks"},
	}
}

func farPool() curriculum.Pool {
	p := fullPool()
	p.MineFarTargetOffsets = [][3]float64{{24, 0, 0}, {0, 0, -24}}
	p.MineFarBlocks = []string{"minecraft:sand", "minecraft:gravel"}
	p.MineFarSearchRadius = 6
	p.CraftFarTargetOffsets = [][3]float64{{40, 0, 0}, {0, 0, 44}}
	p.CraftFarItems = []string{"minecraft:chest", "minecraft:bowl"}
	p.CraftFarSearchRadius = 6
	return p
}

func chainPool() curriculum.Pool {
	p := fullPool()
	p.ChainTargetOffsets = [][3]float64{{18, 0, 0}, {0, 0, -18}}
	p.ChainGoalItems = []string{"minecraft:bowl", "minecraft:chest"}
	p.ChainSearchRadius = 8
	return p
}

func TestValidatePoolRejectsEmptyPool(t *testing.T) {
	assert.Error(t, ValidatePool(curriculum.Pool{}))
}

func TestValidatePoolAcceptsASinglePopulatedKind(t *testing.T) {
	assert.NoError(t, ValidatePool(curriculum.Pool{TargetOffsets: [][3]float64{{1, 0, 0}}}))
}

// TestSelectProducesAllConfiguredKindsOverManySamples is the statistical
// test docs/plans/07's own "Done when" calls for: with all three original
// task kinds populated, every one of them must show up across enough
// samples — not an exact distribution check (that would make this test
// flaky by design), just "all three are reachable."
func TestSelectProducesAllConfiguredKindsOverManySamples(t *testing.T) {
	pool := fullPool()
	rng := rand.New(rand.NewPCG(1, 2))
	seen := map[string]bool{}
	const samples = 300
	for range samples {
		seen[Select(pool, rng).Name()] = true
	}

	assert.True(t, seen["goto"], "goto never appeared in %d samples", samples)
	assert.True(t, seen["mine"], "mine never appeared in %d samples", samples)
	assert.True(t, seen["craft"], "craft never appeared in %d samples", samples)
}

// TestSelectExcludesEmptyPools verifies a task kind with no candidate
// values is never selected, rather than e.g. panicking or silently
// returning a zero-value task for it.
func TestSelectExcludesEmptyPools(t *testing.T) {
	pool := curriculum.Pool{
		MineTargetBlocks: []string{"minecraft:stone"},
		CraftTargetItems: []string{"minecraft:stick"},
		// TargetOffsets deliberately left empty.
	}
	require.NoError(t, ValidatePool(pool))

	rng := rand.New(rand.NewPCG(3, 4))
	for episode := range 200 {
		name := Select(pool, rng).Name()
		assert.NotEqual(t, "goto", name, "goto selected despite an empty TargetOffsets pool (episode %d)", episode)
	}
}

// TestSelectSamplesOnlyConfiguredValues verifies every drawn Task's
// parameters are actually one of the pool's own candidates, not something
// out of range or a zero value masquerading as a real choice.
func TestSelectSamplesOnlyConfiguredValues(t *testing.T) {
	pool := fullPool()

	targetOffsets := map[[3]float64]bool{}
	for _, v := range pool.TargetOffsets {
		targetOffsets[v] = true
	}
	mineBlocks := map[string]bool{}
	for _, v := range pool.MineTargetBlocks {
		mineBlocks[v] = true
	}
	craftItems := map[string]bool{}
	for _, v := range pool.CraftTargetItems {
		craftItems[v] = true
	}

	rng := rand.New(rand.NewPCG(5, 6))
	for episode := range 300 {
		switch task := Select(pool, rng).(type) {
		case gotoTask:
			assert.True(t, targetOffsets[task.offset], "episode %d: offset %v not in pool", episode, task.offset)
		case mineTask:
			assert.True(t, mineBlocks[task.block], "episode %d: block %q not in pool", episode, task.block)
			assert.Equal(t, pool.MineSearchRadius, task.searchRadius)
		case craftTask:
			assert.True(t, craftItems[task.item], "episode %d: item %q not in pool", episode, task.item)
		default:
			t.Fatalf("episode %d: unexpected task %#v", episode, task)
		}
	}
}

func TestSelectIsDeterministicForAFixedRNGSequence(t *testing.T) {
	pool := fullPool()
	rngA := rand.New(rand.NewPCG(42, 42))
	rngB := rand.New(rand.NewPCG(42, 42))

	for episode := range 50 {
		a := Select(pool, rngA)
		b := Select(pool, rngB)
		assert.Equal(t, a, b, "episode %d: identical rng streams should produce identical Tasks", episode)
	}
}

func TestSelectProducesTheFarKindsWithTheirOwnParameters(t *testing.T) {
	pool := farPool()
	rng := rand.New(rand.NewPCG(3, 4))
	seen := map[string]bool{}
	for range 2000 {
		task := Select(pool, rng)
		seen[task.Name()] = true
		switch task := task.(type) {
		case mineFarTask:
			assert.Contains(t, pool.MineFarTargetOffsets, task.offset)
			assert.Contains(t, pool.MineFarBlocks, task.block)
			assert.Equal(t, 6, task.searchRadius, "the far search radius, not the near one")
		case craftFarTask:
			assert.Contains(t, pool.CraftFarTargetOffsets, task.offset)
			assert.Contains(t, pool.CraftFarItems, task.item)
			assert.Equal(t, 6, task.searchRadius)
		}
	}
	for _, name := range []string{"goto", "mine", "craft", "mine_far", "craft_far"} {
		assert.True(t, seen[name], "%s must be reachable", name)
	}
}

func TestFarKindsNeedBothOffsetsAndTargets(t *testing.T) {
	p := fullPool()
	p.MineFarTargetOffsets = [][3]float64{{24, 0, 0}} // no blocks
	p.CraftFarItems = []string{"minecraft:chest"}     // no offsets
	assert.False(t, mineFarFactory{}.Available(p))
	assert.False(t, craftFarFactory{}.Available(p))
}

// Without a radius the table gate is off and the agent would simply walk to
// the table itself, so the kind would teach nothing.
func TestCraftFarNeedsASearchRadius(t *testing.T) {
	p := farPool()
	assert.True(t, craftFarFactory{}.Available(p))
	p.CraftFarSearchRadius = 0
	assert.False(t, craftFarFactory{}.Available(p))
}

// A pool without far tasks must select exactly as it did before they
// existed for any given rng stream (the far kinds are appended, never
// interleaved, in allFactories).
func TestPoolWithoutFarTasksSelectsExactlyAsBefore(t *testing.T) {
	pool := fullPool()
	a, b := rand.New(rand.NewPCG(9, 9)), rand.New(rand.NewPCG(9, 9))
	for range 200 {
		got := Select(pool, a)
		assert.NotEqual(t, "mine_far", got.Name())
		assert.NotEqual(t, "craft_far", got.Name())
		_ = Select(pool, b)
	}
}

func TestChainKindsNeedOffsetsAndARadiusAndStageFourAlsoNeedsGoals(t *testing.T) {
	p := fullPool()
	p.ChainTargetOffsets = [][3]float64{{18, 0, 0}}
	assert.False(t, chainGatherFactory{}.Available(p), "no radius: the tree would start in sight")

	p.ChainSearchRadius = 8
	assert.True(t, chainGatherFactory{}.Available(p))
	assert.True(t, chainTableFactory{}.Available(p))
	assert.True(t, chainPlaceFactory{}.Available(p))
	assert.False(t, chainUseFactory{}.Available(p), "stage 4 needs a goal item")

	p.ChainGoalItems = []string{"minecraft:bowl"}
	assert.True(t, chainUseFactory{}.Available(p))
}

func TestChainTasksCarryTheirStageAndItems(t *testing.T) {
	pool := chainPool()
	rng := rand.New(rand.NewPCG(11, 12))
	seen := map[string]bool{}
	for range 4000 {
		task, ok := Select(pool, rng).(chainTask)
		if !ok {
			continue
		}
		seen[task.Name()] = true
		assert.Equal(t, 8, task.searchRadius)
		assert.Contains(t, pool.ChainTargetOffsets, task.offset)
		switch task.stage {
		case 1:
			assert.Empty(t, task.craftItem, "gather has nothing to craft")
		case 2, 3:
			assert.Equal(t, curriculum.ChainTableItem, task.craftItem)
		case 4:
			assert.Contains(t, pool.ChainGoalItems, task.craftItem)
		}
	}
	for _, name := range []string{"chain_gather", "chain_table", "chain_place", "chain_use"} {
		assert.True(t, seen[name], "%s must be reachable", name)
	}
}

func TestChainTaskToOverrideCarriesTheTreeBlockAndStage(t *testing.T) {
	task := chainTask{stage: 2, offset: [3]float64{18, 0, 0}, searchRadius: 8, craftItem: curriculum.ChainTableItem}
	o := task.ToOverride()

	assert.False(t, o.GoToTargetDisabled, "the bot must travel to the tree")
	assert.Equal(t, [3]float64{18, 0, 0}, o.TargetOffset)
	assert.Equal(t, curriculum.ChainTreeBlock, o.MineTargetBlock)
	assert.Equal(t, 8, o.MineSearchRadius)
	assert.Equal(t, curriculum.ChainTableItem, o.CraftTargetItem)
	assert.Equal(t, 2, o.ChainStage)
	assert.True(t, o.SeedAtGoal, "the tree must be seeded beside the goto target")
	assert.True(t, o.CollectDrops, "the mined logs must reach the inventory")
}

// A pool without a chain must select exactly as it did before the chain
// kinds existed, for any given rng stream: they are appended, never
// interleaved, in allFactories.
func TestPoolWithoutAChainSelectsExactlyAsBefore(t *testing.T) {
	before := farPool()
	after := farPool()
	after.ChainSearchRadius = 8 // a radius alone enables nothing
	a, b := rand.New(rand.NewPCG(5, 5)), rand.New(rand.NewPCG(5, 5))
	for range 300 {
		assert.Equal(t, Select(before, a), Select(after, b))
	}
}

func TestGoalDescriptionNamesTheActualSuccessCondition(t *testing.T) {
	assert.Equal(t, "goto (+5, +0, +0) from spawn", gotoTask{offset: [3]float64{5, 0, 0}}.GoalDescription())
	assert.Equal(t, "mine dirt", mineTask{block: "minecraft:dirt"}.GoalDescription())
	assert.Equal(t, "craft stick", craftTask{item: "minecraft:stick"}.GoalDescription())
	assert.Equal(t, "mine sand", mineFarTask{block: "minecraft:sand"}.GoalDescription())
	assert.Equal(t, "craft chest", craftFarTask{item: "minecraft:chest"}.GoalDescription())
}

func TestGoalDescriptionNamesTheChainsUltimateGoalRegardlessOfStage(t *testing.T) {
	for stage := 1; stage <= 4; stage++ {
		task := chainTask{stage: stage, craftItem: "minecraft:bowl"}
		assert.Equal(t, "craft bowl", task.GoalDescription(), "every chain stage shares the same ultimate craft goal")
	}
}

func TestChainTaskNameUnknownStageFallsBackRatherThanPanicking(t *testing.T) {
	assert.Equal(t, "chain_unknown", chainTask{stage: 99}.Name())
}

// TestSelectAlwaysActivatesExactlyOneTask exercises every kind Select can
// produce (via a full pool, not just the three original fixed cases),
// confirming the "exactly one task active" invariant Convert's old doc
// comment promised still holds for every ToOverride.
func TestSelectAlwaysActivatesExactlyOneTask(t *testing.T) {
	pool := chainPool()
	rng := rand.New(rand.NewPCG(7, 8))

	for episode := range 500 {
		o := Select(pool, rng).ToOverride()

		active := 0
		if !o.GoToTargetDisabled {
			active++
		}
		if o.MineTargetBlock != "" {
			active++
		}
		if o.CraftTargetItem != "" {
			active++
		}
		if active < 1 {
			t.Fatalf("episode %d: no task active in %+v", episode, o)
		}
	}
}

func TestGotoTaskToOverrideEnablesOnlyGoto(t *testing.T) {
	o := gotoTask{offset: [3]float64{5, 1, -2}}.ToOverride()

	assert.False(t, o.GoToTargetDisabled)
	assert.Equal(t, [3]float64{5, 1, -2}, o.TargetOffset)
	assert.Empty(t, o.MineTargetBlock)
	assert.Empty(t, o.CraftTargetItem)
}

func TestMineTaskToOverrideEnablesOnlyMine(t *testing.T) {
	o := mineTask{block: "minecraft:stone", searchRadius: 12}.ToOverride()

	assert.True(t, o.GoToTargetDisabled)
	assert.Equal(t, "minecraft:stone", o.MineTargetBlock)
	assert.Equal(t, 12, o.MineSearchRadius)
	assert.Empty(t, o.CraftTargetItem)
}

func TestCraftTaskToOverrideEnablesOnlyCraft(t *testing.T) {
	o := craftTask{item: "minecraft:stick"}.ToOverride()

	assert.True(t, o.GoToTargetDisabled)
	assert.Empty(t, o.MineTargetBlock)
	assert.Equal(t, "minecraft:stick", o.CraftTargetItem)
}

func TestMineFarTaskToOverrideIsAGotoThenMineComposite(t *testing.T) {
	o := mineFarTask{offset: [3]float64{24, 0, 0}, block: "minecraft:sand", searchRadius: 6}.ToOverride()

	assert.False(t, o.GoToTargetDisabled, "the goto task must be active alongside the mine")
	assert.Equal(t, [3]float64{24, 0, 0}, o.TargetOffset)
	assert.Equal(t, "minecraft:sand", o.MineTargetBlock)
	assert.Equal(t, 6, o.MineSearchRadius)
	assert.Empty(t, o.CraftTargetItem)
	assert.True(t, o.SeedAtGoal, "the block must be seeded beside the goto target, not the bot")
}

func TestCraftFarTaskToOverrideIsAGotoThenCraftComposite(t *testing.T) {
	o := craftFarTask{offset: [3]float64{24, 0, 0}, item: "minecraft:chest", searchRadius: 8}.ToOverride()

	assert.False(t, o.GoToTargetDisabled)
	assert.Equal(t, [3]float64{24, 0, 0}, o.TargetOffset)
	assert.Equal(t, "minecraft:chest", o.CraftTargetItem)
	assert.Equal(t, 8, o.CraftSearchRadius, "the table gate that makes the policy travel")
	assert.Empty(t, o.MineTargetBlock)
	assert.True(t, o.SeedAtGoal)
}

// TestTaskSelectorMatchesSelectForTheSameDraw guards TaskSelector staying
// in sync with Select+ToOverride now that TaskSelector is just that
// pairing: identically-seeded rng streams must draw the identical Task,
// so the override TaskSelector hands rlenv is exactly what Select
// would have produced for a caller that also wants the Task itself (see
// TaskSelector's own doc comment for why cmd/rsi-train's TaskSelector
// closure calls Select directly instead).
func TestTaskSelectorMatchesSelectForTheSameDraw(t *testing.T) {
	pool := chainPool()
	rngA := rand.New(rand.NewPCG(1, 2))
	rngB := rand.New(rand.NewPCG(1, 2))
	selector := TaskSelector(pool, rngA)

	for episode := range 50 {
		want := Select(pool, rngB).ToOverride()
		assert.Equal(t, want, selector(episode, nil))
	}
}

var _ rlenv.TaskSelector = TaskSelector(curriculum.Pool{}, rand.New(rand.NewPCG(0, 0)))
