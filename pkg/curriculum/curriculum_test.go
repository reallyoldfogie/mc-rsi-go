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
