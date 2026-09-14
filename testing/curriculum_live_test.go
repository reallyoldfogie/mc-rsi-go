package testing

import (
	"context"
	"math/rand/v2"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/reallyoldfogie/mc-agent/actions"
	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	_ "github.com/reallyoldfogie/mc-agent/handler_versions" // registers version-specific packet handlers
	"github.com/reallyoldfogie/mc-agent/rlenv"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/curriculum/rlenvadapter"
)

// TestCurriculumVariesTaskPerEpisodeLive is docs/plans/07-curriculum-generator.md's
// own "Done when" live integration test: promotes
// testing/rl_train_test.go's newAlternatingMineOrCraftSeeder /
// newAlternatingMineOrCraftTaskSelector proof-of-concept (mc-agent,
// docs/plans/06) one level further — this repo's own real
// pkg/curriculum.Generator, through pkg/curriculum/rlenvadapter, driving
// a live rlenv.Environment across all three task types (not just
// mine-vs-craft), confirming Environment.ActionMask always legalizes
// exactly one of GoToTarget/Mine/Craft each episode and that the goal-
// conditioning observation block (docs/plans/06) agrees with it — an
// off-task action must never be legal, matching the 182/182 live-verified
// result docs/plans/01-curriculum-generator.md item 4 reports for the
// proof-of-concept this generalizes.
//
// Shares this package's MC_RSI_TRAINER_LIVE_CONFIG opt-in gate, and (like
// testing/curriculum_live_test.go's sibling live tests) auto-launches a
// server via EnsureServer if none is already running at that config's
// address.
func TestCurriculumVariesTaskPerEpisodeLive(t *testing.T) {
	configPath := os.Getenv("MC_RSI_TRAINER_LIVE_CONFIG")
	if configPath == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_CONFIG not set; skipping live curriculum integration test")
	}

	settings, err := mcconfig.Load(configPath)
	require.NoError(t, err)
	require.NoError(t, mcconfig.ApplyEnv(&settings, "MCAGENT"))
	require.NotEmpty(t, settings.Connection.Address, "config must set connection.address")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	server, err := EnsureServer(ctx, &settings)
	require.NoError(t, err)
	defer func() {
		_ = server.Close(context.Background())
	}()

	a, err := connectAgent(ctx, settings)
	require.NoError(t, err)
	defer func() {
		_ = a.Close(context.Background())
	}()

	liveAgent, ok := a.(rlenv.LiveAgent)
	require.True(t, ok, "agent does not satisfy rlenv.LiveAgent")

	gen, err := curriculum.NewUniformRandom(curriculum.Pool{
		TargetOffsets:    [][3]float64{{5, 0, 0}, {0, 0, 5}, {-5, 0, 0}},
		MineTargetBlocks: []string{"minecraft:stone"},
		MineSearchRadius: 8,
		CraftTargetItems: []string{"minecraft:stick"},
	})
	require.NoError(t, err)

	selectorRNG := rand.New(rand.NewPCG(1, 2))
	env, err := rlenv.New(liveAgent, actions.NewRegistry(), rlenv.Config{
		ArrivalThreshold: 1.5,
		StepTimeout:      15 * time.Second,
		TaskSelector:     rlenvadapter.TaskSelector(gen, selectorRNG),
		Seeder:           rlenv.DefaultEpisodeSeeder,
	})
	require.NoError(t, err, "construct rlenv.Environment")

	const episodes = 20
	seenGoto, seenMine, seenCraft := false, false, false

	for episode := range episodes {
		obs, err := env.Reset(ctx)
		require.NoError(t, err, "episode %d: Reset", episode)

		mask := env.ActionMask()
		legalCount := 0
		if mask[rlenv.ActionGoToTarget] {
			legalCount++
		}
		if mask[rlenv.ActionMine] {
			legalCount++
		}
		if mask[rlenv.ActionCraft] {
			legalCount++
		}
		require.Equal(t, 1, legalCount, "episode %d: mask = %v, want exactly one of GoToTarget/Mine/Craft legal", episode, mask)

		// Cross-check the mask against the independently-computed
		// goal-conditioning observation block (indices 14-16,
		// docs/plans/06) — both signals derive from the same
		// Config.GoToTargetDisabled/MineTargetBlock/CraftTargetItem
		// state, but computed by separate code paths (actionLegal vs.
		// buildObservation), so agreement here is a real check, not a
		// tautology.
		assert.Equal(t, boolToFloat(mask[rlenv.ActionGoToTarget]), obs.Values[14], "episode %d: goalGoToActive disagrees with ActionMask", episode)
		assert.Equal(t, boolToFloat(mask[rlenv.ActionMine]), obs.Values[15], "episode %d: goalMineActive disagrees with ActionMask", episode)
		assert.Equal(t, boolToFloat(mask[rlenv.ActionCraft]), obs.Values[16], "episode %d: goalCraftActive disagrees with ActionMask", episode)

		switch {
		case mask[rlenv.ActionGoToTarget]:
			seenGoto = true
		case mask[rlenv.ActionMine]:
			seenMine = true
		case mask[rlenv.ActionCraft]:
			seenCraft = true
		}
	}

	assert.True(t, seenGoto, "TaskGoto never came up as the active task in %d episodes", episodes)
	assert.True(t, seenMine, "TaskMine never came up as the active task in %d episodes", episodes)
	assert.True(t, seenCraft, "TaskCraft never came up as the active task in %d episodes", episodes)

	t.Logf("✓ %d episodes, every one had exactly one legal task action, matching the goal-conditioning block; all three task types appeared", episodes)
}

func boolToFloat(b bool) float32 {
	if b {
		return 1
	}
	return 0
}
