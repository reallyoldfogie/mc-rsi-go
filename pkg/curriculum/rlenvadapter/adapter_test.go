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

func buriedPool() curriculum.Pool {
	p := fullPool()
	p.MineBuriedTargetBlocks = []string{"minecraft:coal_ore"}
	p.MineBuriedSearchRadius = 10
	p.MineBuriedRequiredTool = "minecraft:wooden_pickaxe"
	return p
}

func chainPool() curriculum.Pool {
	p := fullPool()
	p.ChainTargetOffsets = [][3]float64{{18, 0, 0}, {0, 0, -18}}
	p.ChainGoalItems = []string{"minecraft:bowl", "minecraft:chest"}
	p.ChainSearchRadius = 8
	p.ChainMineTargetBlocks = []string{"minecraft:coal_ore"}
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

// TestSelectProducesMineBuriedWithItsOwnParameters mirrors
// TestSelectProducesTheFarKindsWithTheirOwnParameters for mine_buried
// (docs/plans/14): it has no goto/far-seed component, but does carry its
// own search radius and required tool, separate from the plain mine task's.
func TestSelectProducesMineBuriedWithItsOwnParameters(t *testing.T) {
	pool := buriedPool()
	rng := rand.New(rand.NewPCG(3, 4))
	seen := map[string]bool{}
	for range 2000 {
		task := Select(pool, rng)
		seen[task.Name()] = true
		if task, ok := task.(mineBuriedTask); ok {
			assert.Contains(t, pool.MineBuriedTargetBlocks, task.block)
			assert.Equal(t, 10, task.searchRadius, "the buried search radius, not the plain mine task's")
			assert.Equal(t, "minecraft:wooden_pickaxe", task.requiredTool)
		}
	}
	for _, name := range []string{"goto", "mine", "craft", "mine_buried"} {
		assert.True(t, seen[name], "%s must be reachable", name)
	}
}

// A pool without MineBuriedTargetBlocks must never select mine_buried —
// Available() gating it out, not an empty-string task sneaking through.
func TestMineBuriedNeedsTargetBlocks(t *testing.T) {
	p := fullPool() // MineBuriedTargetBlocks deliberately left empty
	assert.False(t, mineBuriedFactory{}.Available(p))

	rng := rand.New(rand.NewPCG(7, 8))
	for range 200 {
		assert.NotEqual(t, "mine_buried", Select(p, rng).Name())
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
		case 5:
			assert.Contains(t, pool.ChainGoalItems, task.craftItem, "chain_mine still crafts a goal item first")
			assert.Contains(t, pool.ChainMineTargetBlocks, task.mineTarget)
		}
	}
	for _, name := range []string{"chain_gather", "chain_table", "chain_place", "chain_use", "chain_mine"} {
		assert.True(t, seen[name], "%s must be reachable", name)
	}
}

// TestChainMineNeedsAGoalItemAndABuriedBlock mirrors
// TestChainKindsNeedOffsetsAndARadiusAndStageFourAlsoNeedsGoals for
// chain_mine (docs/plans/14): it needs both a goal item to craft (the
// same requirement stage 4 has) *and* at least one buried block to mine
// afterward — either missing leaves it unavailable.
func TestChainMineNeedsAGoalItemAndABuriedBlock(t *testing.T) {
	p := fullPool()
	p.ChainTargetOffsets = [][3]float64{{18, 0, 0}}
	p.ChainSearchRadius = 8
	assert.False(t, chainMineFactory{}.Available(p), "needs a goal item and a buried block")

	p.ChainGoalItems = []string{"minecraft:wooden_pickaxe"}
	assert.False(t, chainMineFactory{}.Available(p), "a goal item alone isn't enough")

	p.ChainMineTargetBlocks = []string{"minecraft:coal_ore"}
	assert.True(t, chainMineFactory{}.Available(p))
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

// TestChainMineTaskToOverrideCarriesTheMineTargetBlock mirrors
// TestChainTaskToOverrideCarriesTheTreeBlockAndStage for chain_mine's one
// extra field.
func TestChainMineTaskToOverrideCarriesTheMineTargetBlock(t *testing.T) {
	task := chainTask{
		stage:        rlenv.ChainMine,
		offset:       [3]float64{18, 0, 0},
		searchRadius: 8,
		craftItem:    "minecraft:wooden_pickaxe",
		mineTarget:   "minecraft:coal_ore",
	}
	o := task.ToOverride()

	assert.Equal(t, "minecraft:wooden_pickaxe", o.CraftTargetItem)
	assert.Equal(t, rlenv.ChainMine, o.ChainStage)
	assert.Equal(t, "minecraft:coal_ore", o.ChainMineTargetBlock)
	// Everything else is identical to every other chain stage.
	assert.Equal(t, curriculum.ChainTreeBlock, o.MineTargetBlock, "the gather stage's log target is unchanged")
	assert.True(t, o.SeedAtGoal)
	assert.True(t, o.CollectDrops)
}

func TestGoalDescriptionForChainMineNamesTheMiningGoalNotTheCraft(t *testing.T) {
	task := chainTask{stage: rlenv.ChainMine, craftItem: "minecraft:wooden_pickaxe", mineTarget: "minecraft:coal_ore"}
	assert.Equal(t, "mine buried coal_ore", task.GoalDescription(), "chain_mine's actual success condition is mining, not crafting")
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

func TestMineBuriedTaskToOverrideEnablesOnlyMine(t *testing.T) {
	o := mineBuriedTask{block: "minecraft:coal_ore", searchRadius: 10, requiredTool: "minecraft:wooden_pickaxe"}.ToOverride()

	assert.True(t, o.GoToTargetDisabled)
	assert.Equal(t, "minecraft:coal_ore", o.MineTargetBlock)
	assert.Equal(t, 10, o.MineSearchRadius)
	assert.Equal(t, "minecraft:wooden_pickaxe", o.MineRequiredTool)
	assert.True(t, o.MineTargetBuried, "the target is already in the world; DefaultEpisodeSeeder must not place a fresh one")
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

func expansionPool() curriculum.Pool {
	return curriculum.Pool{
		TargetOffsets:             [][3]float64{{5, 0, 0}},
		EquipTargetItems:          []string{"minecraft:elytra", "minecraft:iron_sword"},
		PickupTargetItems:         []string{"minecraft:stick"},
		PickupSearchRadius:        8,
		PickupAnyItem:             true,
		PlaceTargetItems:          []string{"minecraft:dirt"},
		FlyTargetOffsets:          [][3]float64{{24, 8, 0}},
		FlyArrivalRadius:          5,
		RideVehicleTypes:          []string{"minecraft:horse"},
		RideSearchRadius:          16,
		CombatTargetMobTypes:      []string{"minecraft:zombie"},
		CombatRadius:              16,
		CreeperGunpowderActive:    true,
		CreeperGunpowderRadius:    16,
		SpearChargeTargetMobTypes: []string{"minecraft:zombie"},
		SpearChargeRadius:         16,
		MaceSmashTargetMobTypes:   []string{"minecraft:skeleton"},
		MaceSmashRadius:           16,
		FollowTargetPlayerNames:   []string{"companion"},
		SleepSearchRadius:         10,
		PortalSearchRadius:        16,
		LightPortalSearchRadius:   12,
		ArriveSneaking:            true,
	}
}

func TestSelectProducesExpansionKindsOverManySamples(t *testing.T) {
	pool := expansionPool()
	rng := rand.New(rand.NewPCG(9, 10))
	seen := map[string]bool{}
	for range 5000 {
		seen[Select(pool, rng).Name()] = true
	}
	for _, name := range []string{
		"goto", "equip", "pickup", "pickup_any", "place", "fly", "ride",
		"combat", "creeper_gunpowder", "spear_charge", "mace_smash", "follow",
	} {
		assert.True(t, seen[name], "%s must be reachable", name)
	}
}

func TestExpansionTaskOverridesPoseTheirActions(t *testing.T) {
	assert.Equal(t, "minecraft:elytra", equipTask{item: "minecraft:elytra"}.ToOverride().EquipTargetItem)
	assert.True(t, pickupAnyTask{searchRadius: 8}.ToOverride().PickupAnyItem)
	assert.True(t, flyTask{offset: [3]float64{1, 2, 3}, arrivalRadius: 5}.ToOverride().FlyTargetActive)
	assert.Equal(t, "minecraft:horse", rideTask{vehicleType: "minecraft:horse", searchRadius: 16}.ToOverride().RideVehicleType)
	assert.Equal(t, "minecraft:zombie", combatTask{mobType: "minecraft:zombie", radius: 16}.ToOverride().CombatTargetMobType)
	assert.True(t, creeperGunpowderTask{radius: 16}.ToOverride().CreeperGunpowderActive)
	assert.Equal(t, "minecraft:zombie", spearChargeTask{mobType: "minecraft:zombie", radius: 16}.ToOverride().SpearChargeTargetMobType)
	assert.Equal(t, "minecraft:skeleton", maceSmashTask{mobType: "minecraft:skeleton", radius: 16}.ToOverride().MaceSmashTargetMobType)
	assert.Equal(t, "companion", followTask{playerName: "companion"}.ToOverride().FollowTargetPlayerName)
	assert.True(t, gotoTask{offset: [3]float64{5, 0, 0}, arriveSneaking: true}.ToOverride().ArriveSneaking)
}

func TestTaskSelectorInjectsCrossCuttingRadii(t *testing.T) {
	pool := expansionPool()
	rngA := rand.New(rand.NewPCG(1, 2))
	rngB := rand.New(rand.NewPCG(1, 2))
	selector := TaskSelector(pool, rngA)

	for episode := range 30 {
		want := Select(pool, rngB).ToOverride()
		want.SleepSearchRadius = pool.SleepSearchRadius
		want.PortalSearchRadius = pool.PortalSearchRadius
		want.LightPortalSearchRadius = pool.LightPortalSearchRadius
		assert.Equal(t, want, selector(episode, nil))
	}
}

func TestValidatePoolRejectsNegativeTaskWeight(t *testing.T) {
	pool := curriculum.Pool{
		TargetOffsets: [][3]float64{{1, 0, 0}},
		TaskWeights:   map[string]float64{"goto": -1},
	}
	err := ValidatePool(pool)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "negative")
}

func TestValidatePoolRejectsAllZeroTaskWeights(t *testing.T) {
	pool := curriculum.Pool{
		TargetOffsets:    [][3]float64{{1, 0, 0}},
		MineTargetBlocks: []string{"minecraft:stone"},
		TaskWeights:      map[string]float64{"goto": 0, "mine": 0},
	}
	assert.ErrorIs(t, ValidatePool(pool), errNoPositiveWeight)
}

func TestSelectExcludesZeroWeightKinds(t *testing.T) {
	pool := curriculum.Pool{
		TargetOffsets:    [][3]float64{{1, 0, 0}},
		MineTargetBlocks: []string{"minecraft:stone"},
		CraftTargetItems: []string{"minecraft:stick"},
		TaskWeights:      map[string]float64{"goto": 0, "mine": 1, "craft": 0},
	}
	require.NoError(t, ValidatePool(pool))
	rng := rand.New(rand.NewPCG(1, 2))
	for range 200 {
		assert.Equal(t, "mine", Select(pool, rng).Name())
	}
}

func TestSelectWeightsBiasTowardHeavierKinds(t *testing.T) {
	// combat weight 100 vs goto weight 1: combat should dominate without
	// needing an exact-distribution assertion (those are flaky by design).
	pool := curriculum.Pool{
		TargetOffsets:        [][3]float64{{1, 0, 0}},
		CombatTargetMobTypes: []string{"minecraft:zombie"},
		CombatRadius:         16,
		TaskWeights:          map[string]float64{"goto": 1, "combat": 100},
	}
	require.NoError(t, ValidatePool(pool))
	rng := rand.New(rand.NewPCG(3, 4))
	counts := map[string]int{}
	const samples = 2000
	for range samples {
		counts[Select(pool, rng).Name()]++
	}
	assert.Greater(t, counts["combat"], counts["goto"]*10,
		"combat=%d goto=%d; expected combat >> goto under 100:1 weights", counts["combat"], counts["goto"])
	assert.Greater(t, counts["goto"], 0, "goto must still appear occasionally under a positive weight")
}

func TestSelectWithNilTaskWeightsStillDrawsAllAvailableKinds(t *testing.T) {
	// Regression: empty/nil TaskWeights must keep the historical
	// "every available kind is reachable" property (uniform default weight 1).
	pool := fullPool()
	require.Nil(t, pool.TaskWeights)
	require.NoError(t, ValidatePool(pool))
	rng := rand.New(rand.NewPCG(5, 6))
	seen := map[string]bool{}
	for range 500 {
		seen[Select(pool, rng).Name()] = true
	}
	assert.True(t, seen["goto"])
	assert.True(t, seen["mine"])
	assert.True(t, seen["craft"])
}

func TestFactoryKindMatchesDrawnTaskName(t *testing.T) {
	// Weights are keyed by Kind(); a mismatch would silently default a
	// configured weight back to 1. Guard the whole registry.
	pool := expansionPool()
	pool.MineTargetBlocks = []string{"minecraft:stone"}
	pool.MineSearchRadius = 8
	pool.CraftTargetItems = []string{"minecraft:stick"}
	pool.MineFarTargetOffsets = [][3]float64{{24, 0, 0}}
	pool.MineFarBlocks = []string{"minecraft:sand"}
	pool.MineFarSearchRadius = 6
	pool.CraftFarTargetOffsets = [][3]float64{{24, 0, 0}}
	pool.CraftFarItems = []string{"minecraft:chest"}
	pool.CraftFarSearchRadius = 6
	pool.MineBuriedTargetBlocks = []string{"minecraft:coal_ore"}
	pool.MineBuriedSearchRadius = 10
	pool.MineBuriedRequiredTool = "minecraft:wooden_pickaxe"
	pool.ChainTargetOffsets = [][3]float64{{18, 0, 0}}
	pool.ChainSearchRadius = 8
	pool.ChainGoalItems = []string{"minecraft:bowl"}
	pool.ChainMineTargetBlocks = []string{"minecraft:coal_ore"}
	rng := rand.New(rand.NewPCG(7, 8))
	for _, factory := range allFactories {
		if !factory.Available(pool) {
			t.Fatalf("%s reported unavailable against a fully populated pool", factory.Kind())
		}
		task := factory.Draw(pool, rng)
		assert.Equal(t, factory.Kind(), task.Name(), "factory Kind must match Task.Name for weight lookup")
	}
}
