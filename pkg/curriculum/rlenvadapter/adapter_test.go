package rlenvadapter

import (
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/mc-agent/rlenv"
	"github.com/stretchr/testify/assert"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum"
)

// fixedGenerator always returns the same TaskSpec, regardless of episode
// or rng — the simplest possible curriculum.Generator, used here to pin
// down exactly what TaskSelector's conversion does for each task type
// without depending on UniformRandom's own random selection.
type fixedGenerator struct {
	spec curriculum.TaskSpec
}

func (f fixedGenerator) Next(int, *rand.Rand) curriculum.TaskSpec { return f.spec }

func TestTaskSelectorGotoSpecEnablesOnlyGoto(t *testing.T) {
	gen := fixedGenerator{spec: curriculum.TaskSpec{Type: curriculum.TaskGoto, TargetOffset: [3]float64{5, 1, -2}}}
	selector := TaskSelector(gen, rand.New(rand.NewPCG(1, 2)))

	override := selector(0, nil)

	assert.False(t, override.GoToTargetDisabled)
	assert.Equal(t, [3]float64{5, 1, -2}, override.TargetOffset)
	assert.Empty(t, override.MineTargetBlock)
	assert.Empty(t, override.CraftTargetItem)
}

func TestTaskSelectorMineSpecEnablesOnlyMine(t *testing.T) {
	gen := fixedGenerator{spec: curriculum.TaskSpec{Type: curriculum.TaskMine, MineTargetBlock: "minecraft:stone", MineSearchRadius: 12}}
	selector := TaskSelector(gen, rand.New(rand.NewPCG(1, 2)))

	override := selector(0, nil)

	assert.True(t, override.GoToTargetDisabled)
	assert.Equal(t, "minecraft:stone", override.MineTargetBlock)
	assert.Equal(t, 12, override.MineSearchRadius)
	assert.Empty(t, override.CraftTargetItem)
}

func TestTaskSelectorCraftSpecEnablesOnlyCraft(t *testing.T) {
	gen := fixedGenerator{spec: curriculum.TaskSpec{Type: curriculum.TaskCraft, CraftTargetItem: "minecraft:stick"}}
	selector := TaskSelector(gen, rand.New(rand.NewPCG(1, 2)))

	override := selector(0, nil)

	assert.True(t, override.GoToTargetDisabled)
	assert.Empty(t, override.MineTargetBlock)
	assert.Equal(t, "minecraft:stick", override.CraftTargetItem)
}

// TestTaskSelectorAlwaysActivatesExactlyOneTask exercises the adapter
// against UniformRandom (not just fixedGenerator) across many episodes,
// confirming the "exactly one task active" invariant TaskSelector's own
// doc comment promises holds for every TaskSpec UniformRandom can
// actually produce, not just the three fixed cases above.
func TestTaskSelectorAlwaysActivatesExactlyOneTask(t *testing.T) {
	gen, err := curriculum.NewUniformRandom(curriculum.Pool{
		TargetOffsets:    [][3]float64{{5, 0, 0}},
		MineTargetBlocks: []string{"minecraft:stone"},
		CraftTargetItems: []string{"minecraft:stick"},
	})
	if err != nil {
		t.Fatalf("NewUniformRandom: %v", err)
	}
	selector := TaskSelector(gen, rand.New(rand.NewPCG(7, 8)))

	for episode := range 200 {
		override := selector(episode, nil)

		active := 0
		if !override.GoToTargetDisabled {
			active++
		}
		if override.MineTargetBlock != "" {
			active++
		}
		if override.CraftTargetItem != "" {
			active++
		}
		if active != 1 {
			t.Fatalf("episode %d: %d tasks active in %+v, want exactly 1", episode, active, override)
		}
	}
}

// TestTaskSelectorPassesEpisodeNumberThrough verifies the episode number
// rlenv.Environment.Reset passes to the returned TaskSelector reaches
// gen.Next unchanged, so a Generator keying off episode (e.g. a future
// non-random one) would see the real count.
func TestTaskSelectorPassesEpisodeNumberThrough(t *testing.T) {
	var seenEpisodes []int
	gen := recordingGenerator{seen: &seenEpisodes}
	selector := TaskSelector(gen, rand.New(rand.NewPCG(1, 2)))

	for episode := range 3 {
		selector(episode, nil)
	}

	assert.Equal(t, []int{0, 1, 2}, seenEpisodes)
}

type recordingGenerator struct {
	seen *[]int
}

func (g recordingGenerator) Next(episode int, _ *rand.Rand) curriculum.TaskSpec {
	*g.seen = append(*g.seen, episode)
	return curriculum.TaskSpec{Type: curriculum.TaskGoto, TargetOffset: [3]float64{1, 0, 0}}
}

var _ rlenv.TaskSelector = TaskSelector(fixedGenerator{}, rand.New(rand.NewPCG(0, 0)))
