package competition

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum"
)

func TestEvaluationPlanIsDeterministicAndOrdered(t *testing.T) {
	pool := curriculum.Pool{
		TargetOffsets:    [][3]float64{{5, 0, 0}, {-5, 0, 0}},
		MineTargetBlocks: []string{"minecraft:stone"},
		MineSearchRadius: 8,
		CraftTargetItems: []string{"minecraft:stick"},
	}
	a, err := NewEvaluationPlan(pool, 42, 20)
	require.NoError(t, err)
	b, err := NewEvaluationPlan(pool, 42, 20)
	require.NoError(t, err)

	assert.Equal(t, a, b)
	for i, scenario := range a.Scenarios {
		assert.Equal(t, i, scenario.Index)
		assert.NotEmpty(t, scenario.Task)
		assert.NotEmpty(t, scenario.Goal)
	}
	ha, err := a.Hash()
	require.NoError(t, err)
	hb, err := b.Hash()
	require.NoError(t, err)
	assert.Equal(t, ha, hb)
}

func TestEvaluationPlanDiffersForDifferentSeeds(t *testing.T) {
	pool := curriculum.Pool{TargetOffsets: [][3]float64{{5, 0, 0}, {-5, 0, 0}}}
	a, err := NewEvaluationPlan(pool, 1, 8)
	require.NoError(t, err)
	b, err := NewEvaluationPlan(pool, 2, 8)
	require.NoError(t, err)

	ha, err := a.Hash()
	require.NoError(t, err)
	hb, err := b.Hash()
	require.NoError(t, err)
	assert.NotEqual(t, ha, hb)
}

func TestEvaluationPlanRejectsInvalidCountAndEmptyPool(t *testing.T) {
	pool := curriculum.Pool{TargetOffsets: [][3]float64{{1, 0, 0}}}
	_, err := NewEvaluationPlan(pool, 1, 0)
	require.Error(t, err)
	assert.ErrorContains(t, err, "count must be positive")

	_, err = NewEvaluationPlan(curriculum.Pool{}, 1, 1)
	require.Error(t, err)
	assert.ErrorContains(t, err, "no candidate values")
}

func TestEvaluationPlanScenarioBounds(t *testing.T) {
	plan := EvaluationPlan{Scenarios: []EvaluationScenario{{Index: 0, Task: "goto", Goal: "reach"}}}
	scenario, ok := plan.Scenario(0)
	assert.True(t, ok)
	assert.Equal(t, "goto", scenario.Task)
	_, ok = plan.Scenario(1)
	assert.False(t, ok)
	_, ok = plan.Scenario(-1)
	assert.False(t, ok)
}
