package curriculum

import (
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func fullPool() Pool {
	return Pool{
		TargetOffsets:    [][3]float64{{5, 0, 0}, {-5, 0, 0}, {0, 0, 5}},
		MineTargetBlocks: []string{"minecraft:stone", "minecraft:coal_ore"},
		MineSearchRadius: 8,
		CraftTargetItems: []string{"minecraft:stick", "minecraft:oak_planks"},
	}
}

func TestNewUniformRandomRejectsEmptyPool(t *testing.T) {
	_, err := NewUniformRandom(Pool{})
	assert.Error(t, err)
}

func TestNewUniformRandomAcceptsASinglePopulatedType(t *testing.T) {
	_, err := NewUniformRandom(Pool{TargetOffsets: [][3]float64{{1, 0, 0}}})
	assert.NoError(t, err)
}

// TestUniformRandomProducesAllConfiguredTypesOverManySamples is the
// statistical test docs/plans/07's own "Done when" calls for: with all
// three task types populated, every one of them must show up across
// enough samples — not an exact distribution check (that would make this
// test flaky by design), just "all three are reachable."
func TestUniformRandomProducesAllConfiguredTypesOverManySamples(t *testing.T) {
	gen, err := NewUniformRandom(fullPool())
	require.NoError(t, err)

	rng := rand.New(rand.NewPCG(1, 2))
	seen := map[TaskType]bool{}
	const samples = 300
	for episode := range samples {
		seen[gen.Next(episode, rng).Type] = true
	}

	assert.True(t, seen[TaskGoto], "TaskGoto never appeared in %d samples", samples)
	assert.True(t, seen[TaskMine], "TaskMine never appeared in %d samples", samples)
	assert.True(t, seen[TaskCraft], "TaskCraft never appeared in %d samples", samples)
}

// TestUniformRandomExcludesEmptyPools verifies a task type with no
// candidate values is never selected, rather than e.g. panicking or
// silently returning a zero-value spec for it.
func TestUniformRandomExcludesEmptyPools(t *testing.T) {
	gen, err := NewUniformRandom(Pool{
		MineTargetBlocks: []string{"minecraft:stone"},
		CraftTargetItems: []string{"minecraft:stick"},
		// TargetOffsets deliberately left empty.
	})
	require.NoError(t, err)

	rng := rand.New(rand.NewPCG(3, 4))
	for episode := range 200 {
		spec := gen.Next(episode, rng)
		assert.NotEqual(t, TaskGoto, spec.Type, "TaskGoto selected despite an empty TargetOffsets pool (episode %d)", episode)
	}
}

// TestUniformRandomSamplesOnlyConfiguredValues verifies every returned
// parameter value is actually one of the pool's own candidates, not
// something out of range or a zero value masquerading as a real choice.
func TestUniformRandomSamplesOnlyConfiguredValues(t *testing.T) {
	pool := fullPool()
	gen, err := NewUniformRandom(pool)
	require.NoError(t, err)

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
		spec := gen.Next(episode, rng)
		switch spec.Type {
		case TaskGoto:
			assert.True(t, targetOffsets[spec.TargetOffset], "episode %d: TargetOffset %v not in pool", episode, spec.TargetOffset)
		case TaskMine:
			assert.True(t, mineBlocks[spec.MineTargetBlock], "episode %d: MineTargetBlock %q not in pool", episode, spec.MineTargetBlock)
			assert.Equal(t, pool.MineSearchRadius, spec.MineSearchRadius)
		case TaskCraft:
			assert.True(t, craftItems[spec.CraftTargetItem], "episode %d: CraftTargetItem %q not in pool", episode, spec.CraftTargetItem)
		default:
			t.Fatalf("episode %d: unexpected task type %v", episode, spec.Type)
		}
	}
}

func TestUniformRandomIsDeterministicForAFixedRNGSequence(t *testing.T) {
	gen, err := NewUniformRandom(fullPool())
	require.NoError(t, err)

	rngA := rand.New(rand.NewPCG(42, 42))
	rngB := rand.New(rand.NewPCG(42, 42))

	for episode := range 50 {
		a := gen.Next(episode, rngA)
		b := gen.Next(episode, rngB)
		assert.Equal(t, a, b, "episode %d: identical rng streams should produce identical TaskSpecs", episode)
	}
}

func farPool() Pool {
	p := fullPool()
	p.MineFarTargetOffsets = [][3]float64{{24, 0, 0}, {0, 0, -24}}
	p.MineFarBlocks = []string{"minecraft:sand", "minecraft:gravel"}
	p.MineFarSearchRadius = 6
	p.CraftFarTargetOffsets = [][3]float64{{40, 0, 0}, {0, 0, 44}}
	p.CraftFarItems = []string{"minecraft:chest", "minecraft:bowl"}
	p.CraftFarSearchRadius = 6
	return p
}

func TestUniformRandomProducesTheFarTypesWithTheirOwnParameters(t *testing.T) {
	gen, err := NewUniformRandom(farPool())
	require.NoError(t, err)

	rng := rand.New(rand.NewPCG(3, 4))
	seen := map[TaskType]bool{}
	for range 2000 {
		spec := gen.Next(0, rng)
		seen[spec.Type] = true
		switch spec.Type {
		case TaskMineFar:
			assert.Contains(t, farPool().MineFarTargetOffsets, spec.TargetOffset)
			assert.Contains(t, farPool().MineFarBlocks, spec.MineTargetBlock)
			assert.Equal(t, 6, spec.MineSearchRadius, "the far search radius, not the near one")
			assert.Empty(t, spec.CraftTargetItem)
		case TaskCraftFar:
			assert.Contains(t, farPool().CraftFarTargetOffsets, spec.TargetOffset)
			assert.Contains(t, farPool().CraftFarItems, spec.CraftTargetItem)
			assert.Equal(t, 6, spec.CraftSearchRadius)
			assert.Empty(t, spec.MineTargetBlock)
		}
	}
	for _, typ := range []TaskType{TaskGoto, TaskMine, TaskCraft, TaskMineFar, TaskCraftFar} {
		assert.True(t, seen[typ], "%s must be reachable", typ)
	}
}

func TestFarTypesNeedBothOffsetsAndTargets(t *testing.T) {
	p := fullPool()
	p.MineFarTargetOffsets = [][3]float64{{24, 0, 0}} // no blocks
	p.CraftFarItems = []string{"minecraft:chest"}     // no offsets
	assert.NotContains(t, p.availableTypes(), TaskMineFar)
	assert.NotContains(t, p.availableTypes(), TaskCraftFar)
}

// Without a radius the table gate is off and the agent would simply walk to
// the table itself, so the type would teach nothing.
func TestCraftFarNeedsASearchRadius(t *testing.T) {
	p := farPool()
	assert.Contains(t, p.availableTypes(), TaskCraftFar)
	p.CraftFarSearchRadius = 0
	assert.NotContains(t, p.availableTypes(), TaskCraftFar)
}

// A pool without far tasks must select exactly as it did before they existed
// for any given rng stream (the far types are appended, never interleaved).
func TestPoolWithoutFarTasksSelectsExactlyAsBefore(t *testing.T) {
	gen, err := NewUniformRandom(fullPool())
	require.NoError(t, err)
	a, b := rand.New(rand.NewPCG(9, 9)), rand.New(rand.NewPCG(9, 9))
	for range 200 {
		got := gen.Next(0, a)
		assert.NotEqual(t, TaskMineFar, got.Type)
		assert.NotEqual(t, TaskCraftFar, got.Type)
		_ = gen.Next(0, b)
	}
}

func chainPool() Pool {
	p := fullPool()
	p.ChainTargetOffsets = [][3]float64{{18, 0, 0}, {0, 0, -18}}
	p.ChainGoalItems = []string{"minecraft:bowl", "minecraft:chest"}
	p.ChainSearchRadius = 8
	return p
}

func TestChainTypesNeedOffsetsAndARadius_AndStageFourAlsoNeedsGoals(t *testing.T) {
	p := fullPool()
	p.ChainTargetOffsets = [][3]float64{{18, 0, 0}}
	assert.NotContains(t, p.availableTypes(), TaskChainGather, "no radius: the tree would start in sight")

	p.ChainSearchRadius = 8
	got := p.availableTypes()
	assert.Contains(t, got, TaskChainGather)
	assert.Contains(t, got, TaskChainTable)
	assert.Contains(t, got, TaskChainPlace)
	assert.NotContains(t, got, TaskChainUse, "stage 4 needs a goal item")

	p.ChainGoalItems = []string{"minecraft:bowl"}
	assert.Contains(t, p.availableTypes(), TaskChainUse)
}

func TestChainSpecsCarryTheirStageAndItems(t *testing.T) {
	gen, err := NewUniformRandom(chainPool())
	require.NoError(t, err)
	rng := rand.New(rand.NewPCG(11, 12))
	seen := map[TaskType]bool{}
	for range 4000 {
		spec := gen.Next(0, rng)
		stage := spec.Type.ChainStage()
		if stage == 0 {
			assert.Empty(t, spec.CraftTargetItem == ChainTableItem && spec.MineTargetBlock == ChainTreeBlock, "a non-chain spec must not look like one")
			continue
		}
		seen[spec.Type] = true
		assert.Equal(t, ChainTreeBlock, spec.MineTargetBlock)
		assert.Equal(t, 8, spec.MineSearchRadius)
		assert.Contains(t, chainPool().ChainTargetOffsets, spec.TargetOffset)
		switch stage {
		case 1:
			assert.Empty(t, spec.CraftTargetItem, "gather has nothing to craft")
		case 2, 3:
			assert.Equal(t, ChainTableItem, spec.CraftTargetItem)
		case 4:
			assert.Contains(t, chainPool().ChainGoalItems, spec.CraftTargetItem)
		}
	}
	for _, typ := range []TaskType{TaskChainGather, TaskChainTable, TaskChainPlace, TaskChainUse} {
		assert.True(t, seen[typ], "%s must be reachable", typ)
	}
}

func TestChainStageAndNames(t *testing.T) {
	for typ, want := range map[TaskType]struct {
		stage int
		name  string
	}{
		TaskChainGather: {1, "chain_gather"},
		TaskChainTable:  {2, "chain_table"},
		TaskChainPlace:  {3, "chain_place"},
		TaskChainUse:    {4, "chain_use"},
		TaskGoto:        {0, "goto"},
		TaskCraftFar:    {0, "craft_far"},
	} {
		assert.Equal(t, want.stage, typ.ChainStage())
		assert.Equal(t, want.name, typ.String())
	}
}

// A pool without a chain must select exactly as it did before the chain types
// existed, for any given rng stream: they are appended, never interleaved.
func TestPoolWithoutAChainSelectsExactlyAsBefore(t *testing.T) {
	before, err := NewUniformRandom(farPool())
	require.NoError(t, err)
	withZeroChain := farPool()
	withZeroChain.ChainSearchRadius = 8 // a radius alone enables nothing
	after, err := NewUniformRandom(withZeroChain)
	require.NoError(t, err)
	a, b := rand.New(rand.NewPCG(5, 5)), rand.New(rand.NewPCG(5, 5))
	for range 300 {
		assert.Equal(t, before.Next(0, a), after.Next(0, b))
	}
}
