package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/reallyoldfogie/mc-agent/models"
	"github.com/reallyoldfogie/mc-agent/rlenv"
	mctesting "github.com/reallyoldfogie/mc-agent/testing"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/leapfrog"
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/lineage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseFlagsRejectsMissingOrInvalidInput(t *testing.T) {
	validArgs := func(overrides ...string) []string {
		args := []string{"-checkpoint-dir", "/tmp/ckpt", "-mc-agent-config", "/tmp/config.json"}
		return append(args, overrides...)
	}

	tests := []struct {
		name string
		args []string
	}{
		{"missing checkpoint-dir", []string{"-mc-agent-config", "/tmp/config.json"}},
		{"missing mc-agent-config", []string{"-checkpoint-dir", "/tmp/ckpt"}},
		{"zero epochs-per-generation", validArgs("-epochs-per-generation", "0")},
		{"negative epochs-per-generation", validArgs("-epochs-per-generation", "-1")},
		{"zero eval-episodes", validArgs("-eval-episodes", "0")},
		{"zero eval-episode-len", validArgs("-eval-episode-len", "0")},
		{"negative checkpoint-interval", validArgs("-checkpoint-interval", "-1")},
		{"negative max-rounds", validArgs("-max-rounds", "-1")},
		{"negative max-step-timeouts", validArgs("-max-step-timeouts", "-1")},
		{"negative time-penalty", validArgs("-time-penalty", "-0.1")},
		{"negative win-z-score", validArgs("-win-z-score", "-1")},
		{"negative min-win-margin", validArgs("-min-win-margin", "-1")},
		{"negative clear-area-radius", validArgs("-clear-area-radius", "-1")},
		{"clear-area-radius over the fill limit", validArgs("-clear-area-radius", "40")},
		{"zero parallel-envs", validArgs("-parallel-envs", "0")},
		{"negative parallel-envs", validArgs("-parallel-envs", "-1")},
		{"zero shared-server-separation-chunks with -shared-server", validArgs("-shared-server", "-shared-server-separation-chunks", "0")},
		{"negative shared-server-separation-chunks with -shared-server", validArgs("-shared-server", "-shared-server-separation-chunks", "-1")},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseFlags(tc.args)
			assert.Error(t, err)
		})
	}
}

func TestParseFlagsIgnoresSeparationChunksValidationWithoutSharedServer(t *testing.T) {
	f, err := parseFlags([]string{
		"-checkpoint-dir", "/tmp/ckpt",
		"-mc-agent-config", "/tmp/config.json",
		"-shared-server-separation-chunks", "0",
	})
	require.NoError(t, err, "-shared-server-separation-chunks is only validated when -shared-server is set")
	assert.False(t, f.sharedServer)
	assert.Equal(t, 0, f.sharedServerSeparationChunks)
}

func TestParseFlagsAcceptsValidInputAndDefaults(t *testing.T) {
	f, err := parseFlags([]string{"-checkpoint-dir", "/tmp/ckpt", "-mc-agent-config", "/tmp/config.json"})
	require.NoError(t, err)
	assert.Equal(t, "/tmp/ckpt", f.checkpointDir)
	assert.Equal(t, "/tmp/config.json", f.mcAgentConfigPath)
	assert.Equal(t, "", f.trainerConfigPath)
	assert.Equal(t, 50, f.epochsPerGeneration)
	assert.Equal(t, 5, f.evalEpisodes)
	assert.Equal(t, 200, f.evalEpisodeLen)
	assert.Equal(t, 10, f.checkpointInterval)
	assert.Equal(t, 0, f.maxRounds)
	assert.False(t, f.autoResetOrigin, "-auto-reset-origin must default to false: opt-in, existing configs unaffected")
	assert.Equal(t, 1, f.parallelEnvs, "-parallel-envs must default to 1: existing single-environment behavior unaffected")
	assert.False(t, f.sharedServer, "-shared-server must default to false: existing per-server behavior unaffected")
	assert.Equal(t, 16, f.sharedServerSeparationChunks)
	assert.Equal(t, 5, f.maxStepTimeouts, "-max-step-timeouts is a harmless safety net, so it defaults on")
	assert.Equal(t, 0, f.clearAreaRadius, "-clear-area-radius deletes blocks (flat worlds only), so it must default off")
	assert.InDelta(t, 1.645, f.winZScore, 1e-9, "promotion must require a statistically real advantage by default")
	assert.Zero(t, f.minWinMargin)
	assert.Zero(t, f.timePenalty, "-time-penalty changes the reward scale, so it must default to rlenv's own 0.01")

	f, err = parseFlags([]string{
		"-checkpoint-dir", "/tmp/ckpt",
		"-mc-agent-config", "/tmp/config.json",
		"-trainer-config", "/tmp/trainer.json",
		"-epochs-per-generation", "10",
		"-eval-episodes", "3",
		"-eval-episode-len", "50",
		"-checkpoint-interval", "0",
		"-max-rounds", "2",
		"-auto-reset-origin",
		"-parallel-envs", "4",
		"-shared-server",
		"-shared-server-separation-chunks", "32",
	})
	require.NoError(t, err)
	assert.Equal(t, "/tmp/trainer.json", f.trainerConfigPath)
	assert.Equal(t, 10, f.epochsPerGeneration)
	assert.Equal(t, 3, f.evalEpisodes)
	assert.Equal(t, 50, f.evalEpisodeLen)
	assert.Equal(t, 0, f.checkpointInterval)
	assert.Equal(t, 2, f.maxRounds)
	assert.True(t, f.autoResetOrigin)
	assert.Equal(t, 4, f.parallelEnvs)
	assert.True(t, f.sharedServer)
	assert.Equal(t, 32, f.sharedServerSeparationChunks)
}

// fakePositionProvider is the minimal positionProvider fake
// applyAutoResetOrigin's own tests need — see that function's doc comment
// for why it depends on this narrow interface rather than the full
// models.Agent.
type fakePositionProvider struct {
	pos         models.V3
	initialized bool
}

func (f fakePositionProvider) GetPositionSimple() (models.V3, bool) { return f.pos, f.initialized }

func TestApplyAutoResetOriginSetsOriginAndDefaultJitterWhenEligible(t *testing.T) {
	cfg := rlenv.Config{}
	agent := fakePositionProvider{pos: models.V3{X: 1, Y: 2, Z: 3}, initialized: true}

	applyAutoResetOrigin(&cfg, agent, "127.0.0.1:25575")

	require.NotNil(t, cfg.ResetOrigin)
	assert.Equal(t, [3]float64{1, 2, 3}, *cfg.ResetOrigin)
	assert.Equal(t, defaultAutoJitter, cfg.Jitter)
	assert.NotZero(t, cfg.JitterSeed)
}

func TestApplyAutoResetOriginNoOpsWithoutRCON(t *testing.T) {
	cfg := rlenv.Config{}
	agent := fakePositionProvider{pos: models.V3{X: 1, Y: 2, Z: 3}, initialized: true}

	applyAutoResetOrigin(&cfg, agent, "")

	assert.Nil(t, cfg.ResetOrigin, "no RCON address means TeleportTo could never work; must not set ResetOrigin")
	assert.Zero(t, cfg.Jitter)
}

func TestApplyAutoResetOriginNoOpsWhenPositionNotYetKnown(t *testing.T) {
	cfg := rlenv.Config{}
	agent := fakePositionProvider{initialized: false}

	applyAutoResetOrigin(&cfg, agent, "127.0.0.1:25575")

	assert.Nil(t, cfg.ResetOrigin)
}

// fakeSpawnQualityAgent extends fakePositionProvider with the world/shape
// access applyAutoResetOrigin's optional spawnQualityChecker check needs
// — its own tests below need a real (mctesting-backed) World/ShapeManager
// to exercise isGoodSpawnPosition through the real function, not a
// hand-rolled stub.
type fakeSpawnQualityAgent struct {
	fakePositionProvider
	world    models.World
	shapeMgr models.BlockShapeManager
}

func (f fakeSpawnQualityAgent) GetWorld() models.World                      { return f.world }
func (f fakeSpawnQualityAgent) BlockShapeManager() models.BlockShapeManager { return f.shapeMgr }

const spawnGroundStateID = 9 // mctesting.NewSimpleBlockRegistry's pre-registered "minecraft:grass_block".

func TestApplyAutoResetOriginAcceptsAGoodSpawn(t *testing.T) {
	registry := mctesting.NewSimpleBlockRegistry()
	world := mctesting.NewWorldBuilder(registry).FlatGroundDirect(-5, -5, 5, 5, -1, spawnGroundStateID).Build()
	agent := fakeSpawnQualityAgent{
		fakePositionProvider: fakePositionProvider{pos: models.V3{X: 1, Y: 0, Z: 2}, initialized: true},
		world:                world,
		shapeMgr:             mctesting.NewMockShapeManager(),
	}

	cfg := rlenv.Config{}
	applyAutoResetOrigin(&cfg, agent, "127.0.0.1:25575")

	require.NotNil(t, cfg.ResetOrigin, "a standable, dry spawn must be accepted")
	assert.Equal(t, [3]float64{1, 0, 2}, *cfg.ResetOrigin)
}

func TestApplyAutoResetOriginRejectsASubmergedSpawn(t *testing.T) {
	registry := mctesting.NewSimpleBlockRegistry()
	waterStateID := registry.GetStateID("minecraft:water", nil)
	world := mctesting.NewWorldBuilder(registry).
		FlatGroundDirect(-5, -5, 5, 5, -1, spawnGroundStateID). // solid lakebed
		SetBlockDirect(1, 0, 2, waterStateID).                  // feet submerged
		SetBlockDirect(1, 1, 2, waterStateID).                  // head submerged
		Build()
	agent := fakeSpawnQualityAgent{
		fakePositionProvider: fakePositionProvider{pos: models.V3{X: 1, Y: 0, Z: 2}, initialized: true},
		world:                world,
		shapeMgr:             mctesting.NewMockShapeManager(),
	}

	cfg := rlenv.Config{}
	applyAutoResetOrigin(&cfg, agent, "127.0.0.1:25575")

	assert.Nil(t, cfg.ResetOrigin, "a spawn submerged in water — solid ground below, but water at feet/head — must be rejected")
}

// TestApplyAutoResetOriginRejectsABoxedInSpawn covers hasWalkableWayOut:
// a spawn standable in the exact sense IsWalkablePosition/water checks
// require, but with ground only under that one footprint and nothing but
// void (no ground loaded anywhere) beyond it in every direction — a
// "one-block pillar in the sky" — must still be rejected, since a bot
// pinned to reusing it every episode has nowhere to walk. This is the
// live-found gap isGoodSpawnPosition's earlier two checks alone missed
// (see hasWalkableWayOut's own doc comment).
func TestApplyAutoResetOriginRejectsABoxedInSpawn(t *testing.T) {
	registry := mctesting.NewSimpleBlockRegistry()
	// Ground only directly under the spawn footprint itself (1,-1,2) -
	// nothing else placed anywhere, so every cardinal direction is void
	// within hasWalkableWayOut's own check radius.
	world := mctesting.NewWorldBuilder(registry).SetBlockDirect(1, -1, 2, spawnGroundStateID).Build()
	agent := fakeSpawnQualityAgent{
		fakePositionProvider: fakePositionProvider{pos: models.V3{X: 1, Y: 0, Z: 2}, initialized: true},
		world:                world,
		shapeMgr:             mctesting.NewMockShapeManager(),
	}

	cfg := rlenv.Config{}
	applyAutoResetOrigin(&cfg, agent, "127.0.0.1:25575")

	assert.Nil(t, cfg.ResetOrigin, "a spawn standable only on its own single footprint, boxed in by void on every side, must be rejected")
}

func TestApplyAutoResetOriginRejectsAnUnwalkableSpawn(t *testing.T) {
	registry := mctesting.NewSimpleBlockRegistry()
	// No ground placed anywhere: air below, so IsWalkablePosition itself
	// already fails, independent of the water check.
	world := mctesting.NewWorldBuilder(registry).Build()
	agent := fakeSpawnQualityAgent{
		fakePositionProvider: fakePositionProvider{pos: models.V3{X: 1, Y: 0, Z: 2}, initialized: true},
		world:                world,
		shapeMgr:             mctesting.NewMockShapeManager(),
	}

	cfg := rlenv.Config{}
	applyAutoResetOrigin(&cfg, agent, "127.0.0.1:25575")

	assert.Nil(t, cfg.ResetOrigin, "a spawn with no ground support must be rejected")
}

// TestApplyAutoResetOriginLiftsABuriedSpawnOutOfTheGround covers the
// live-found post-resume dip: a bot's saved position was inside the floor
// (y=-63 under a surface at y=-60), the spawn check rejected it, and the
// shared-server fallback then teleported the bot back into the ground on
// every reset. It must be lifted to the first good position above it.
func TestApplyAutoResetOriginLiftsABuriedSpawnOutOfTheGround(t *testing.T) {
	registry := mctesting.NewSimpleBlockRegistry()
	wb := mctesting.NewWorldBuilder(registry)
	for _, y := range []float64{-3, -2, -1} { // three layers of solid floor, surface at y=0
		wb = wb.FlatGroundDirect(-5, -5, 5, 5, y, spawnGroundStateID)
	}
	agent := fakeSpawnQualityAgent{
		fakePositionProvider: fakePositionProvider{pos: models.V3{X: 1, Y: -2, Z: 2}, initialized: true},
		world:                wb.Build(),
		shapeMgr:             mctesting.NewMockShapeManager(),
	}

	cfg := rlenv.Config{}
	applyAutoResetOrigin(&cfg, agent, "127.0.0.1:25575")

	require.NotNil(t, cfg.ResetOrigin, "a buried spawn must be lifted, not abandoned")
	assert.Equal(t, [3]float64{1, 0, 2}, *cfg.ResetOrigin, "lifted to the first standable cell above the floor")
}

func TestLiftOutOfGroundDoesNotRescueASubmergedSpawn(t *testing.T) {
	registry := mctesting.NewSimpleBlockRegistry()
	waterStateID := registry.GetStateID("minecraft:water", nil)
	world := mctesting.NewWorldBuilder(registry).
		FlatGroundDirect(-5, -5, 5, 5, -1, spawnGroundStateID).
		SetBlockDirect(1, 0, 2, waterStateID).
		SetBlockDirect(1, 1, 2, waterStateID).
		Build()

	_, ok := liftOutOfGround(world, mctesting.NewMockShapeManager(), models.V3{X: 1, Y: 0, Z: 2})
	assert.False(t, ok, "a water-filled column is not a place to lift a bot into")
}

func TestApplyAutoResetOriginNeverOverridesAnOperatorConfiguredResetOriginOrJitter(t *testing.T) {
	explicitOrigin := [3]float64{9, 9, 9}
	explicitJitter := [3]float64{5, 5, 5}
	cfg := rlenv.Config{ResetOrigin: &explicitOrigin, Jitter: explicitJitter, JitterSeed: 42}
	agent := fakePositionProvider{pos: models.V3{X: 1, Y: 2, Z: 3}, initialized: true}

	applyAutoResetOrigin(&cfg, agent, "127.0.0.1:25575")

	assert.Same(t, &explicitOrigin, cfg.ResetOrigin, "an operator-configured ResetOrigin must be left untouched")
	assert.Equal(t, explicitJitter, cfg.Jitter)
	assert.Equal(t, int64(42), cfg.JitterSeed)
}

func TestApplySharedServerWorkingAreaIndexZeroHasZeroOffset(t *testing.T) {
	cfg := rlenv.Config{}
	agent := fakePositionProvider{pos: models.V3{X: 1, Y: 2, Z: 3}, initialized: true}

	err := applySharedServerWorkingArea(&cfg, agent, 0, 16)

	require.NoError(t, err)
	require.NotNil(t, cfg.ResetOrigin)
	assert.Equal(t, [3]float64{1, 2, 3}, *cfg.ResetOrigin, "index 0's working area must equal the captured reference position exactly")
}

func TestApplySharedServerWorkingAreaOffsetsSubsequentIndices(t *testing.T) {
	cfg := rlenv.Config{}
	agent := fakePositionProvider{pos: models.V3{X: 1, Y: 2, Z: 3}, initialized: true}

	err := applySharedServerWorkingArea(&cfg, agent, 2, 16)

	require.NoError(t, err)
	require.NotNil(t, cfg.ResetOrigin)
	assert.Equal(t, [3]float64{1 + 2*16*16, 2, 3}, *cfg.ResetOrigin)
}

func TestApplySharedServerWorkingAreaUsesAlreadySetResetOriginAsBase(t *testing.T) {
	explicitOrigin := [3]float64{100, 5, -50}
	cfg := rlenv.Config{ResetOrigin: &explicitOrigin}
	agent := fakePositionProvider{pos: models.V3{X: 1, Y: 2, Z: 3}, initialized: true}

	err := applySharedServerWorkingArea(&cfg, agent, 1, 16)

	require.NoError(t, err)
	require.NotNil(t, cfg.ResetOrigin)
	assert.Equal(t, [3]float64{100 + 16*16, 5, -50}, *cfg.ResetOrigin,
		"an already-set ResetOrigin (operator config, or a preceding applyAutoResetOrigin call) must be used as the offset base, not the agent's own live position")
}

func TestApplySharedServerWorkingAreaErrorsWhenPositionNotYetKnownAndNoResetOriginSet(t *testing.T) {
	cfg := rlenv.Config{}
	agent := fakePositionProvider{initialized: false}

	err := applySharedServerWorkingArea(&cfg, agent, 1, 16)

	assert.Error(t, err, "unlike applyAutoResetOrigin, an unknown position must be a hard error here — a bot with no working area assigned would collide with bot 0")
	assert.Nil(t, cfg.ResetOrigin)
}

// TestLoadOrInitTeacherCreatesAndPersistsGenerationZero verifies the
// "fresh checkpoint dir" path: loadOrInitTeacher must both return a
// usable generation-0 Params and actually persist it, so a second call
// against the same directory resumes the same generation instead of
// fabricating a new random one — see loadOrInitTeacher's own doc comment
// for why that matters.
func TestLoadOrInitTeacherCreatesAndPersistsGenerationZero(t *testing.T) {
	dir := t.TempDir()
	const observationSize, hiddenSize, actionSpace = 17, 8, 4
	const environmentID = "mc-agent-rlenv:actions=4:obs=17"

	params, rec, err := loadOrInitTeacher(dir, environmentID, observationSize, hiddenSize, actionSpace)
	require.NoError(t, err)
	require.NotNil(t, params)
	assert.Equal(t, 0, rec.Generation)
	assert.Equal(t, -1, rec.ParentGeneration)
	assert.Equal(t, lineage.ProvenanceLeapfrog, rec.Provenance)
	assert.Equal(t, environmentID, rec.EnvironmentID)
	assert.Equal(t, observationSize, params.InputSize())
	assert.Equal(t, hiddenSize, params.HiddenSize())
	assert.Equal(t, actionSpace, params.OutputSize())

	// A fresh call to lineage.Latest against the same dir must now find
	// exactly the generation just persisted, not report "no generation
	// record found."
	latest, err := lineage.Latest(dir)
	require.NoError(t, err)
	assert.Equal(t, 0, latest.Generation)
}

// TestLoadOrInitTeacherResumesExistingLatestGeneration verifies the
// other half: when a generation already exists on disk, loadOrInitTeacher
// must load and return it (not fabricate a new generation 0 on top of
// it).
func TestLoadOrInitTeacherResumesExistingLatestGeneration(t *testing.T) {
	dir := t.TempDir()
	const observationSize, hiddenSize, actionSpace = 17, 8, 4
	const environmentID = "mc-agent-rlenv:actions=4:obs=17"

	first, firstRec, err := loadOrInitTeacher(dir, environmentID, observationSize, hiddenSize, actionSpace)
	require.NoError(t, err)

	second, secondRec, err := loadOrInitTeacher(dir, environmentID, observationSize, hiddenSize, actionSpace)
	require.NoError(t, err)

	assert.Equal(t, firstRec.CreatedAt.Unix(), secondRec.CreatedAt.Unix())
	assert.Equal(t, first.W0.Data, second.W0.Data, "second call must resume the exact same persisted weights, not a fresh random init")
}

func TestApplyEpisodeLimitsFillsUnsetFieldsAndLeavesOperatorValuesAlone(t *testing.T) {
	cfg := rlenv.Config{}
	applyEpisodeLimits(&cfg, episodeLimits{maxStepTimeouts: 5, clearAreaRadius: 16, timePenalty: 0.2})
	assert.Equal(t, float32(0.2), cfg.TimePenalty)
	assert.Equal(t, 5, cfg.MaxConsecutiveStepTimeouts)
	assert.Equal(t, 16, cfg.ClearAreaRadius)
	assert.Equal(t, clearAreaHeightBlocks, cfg.ClearAreaHeight)

	cfg = rlenv.Config{MaxConsecutiveStepTimeouts: 9, ClearAreaRadius: 3, ClearAreaHeight: 6, TimePenalty: 0.5}
	applyEpisodeLimits(&cfg, episodeLimits{maxStepTimeouts: 5, clearAreaRadius: 16, timePenalty: 0.2})
	assert.Equal(t, float32(0.5), cfg.TimePenalty, "an operator-configured time penalty must win")
	assert.Equal(t, 9, cfg.MaxConsecutiveStepTimeouts, "an operator-configured value must win")
	assert.Equal(t, 3, cfg.ClearAreaRadius)
	assert.Equal(t, 6, cfg.ClearAreaHeight)

	cfg = rlenv.Config{}
	applyEpisodeLimits(&cfg, episodeLimits{})
	assert.Zero(t, cfg.MaxConsecutiveStepTimeouts)
	assert.Zero(t, cfg.ClearAreaRadius)
	assert.Zero(t, cfg.ClearAreaHeight, "clearing stays fully off when the flag is 0")
}

func TestPairedEvalTaskReseederPinsEachEnvironmentToTheSameSeedsForBothSidesAndVariesAcrossEnvironments(t *testing.T) {
	type call struct{ seed1, seed2 uint64 }
	var envA, envB []call
	hook := pairedEvalTaskReseeder([]func(uint64, uint64){
		func(a, b uint64) { envA = append(envA, call{a, b}) },
		func(a, b uint64) { envB = append(envB, call{a, b}) },
	})
	require.NotNil(t, hook)

	hook(leapfrog.EvalTeacher)
	hook(leapfrog.EvalStudent)

	require.Len(t, envA, 2)
	require.Len(t, envB, 2)
	assert.Equal(t, envA[0], envA[1], "Teacher and Student must draw the identical tasks on each environment")
	assert.Equal(t, envB[0], envB[1])
	assert.NotEqual(t, envA[0], envB[0], "different environments should cover different tasks")

	hook(leapfrog.EvalTeacher)
	assert.NotEqual(t, envA[0], envA[2], "each round draws fresh seeds")

	assert.Nil(t, pairedEvalTaskReseeder(nil), "no curriculum means no reseeding hook")
}

func TestTaskNameForOverrideDistinguishesTheFarComposites(t *testing.T) {
	assert.Equal(t, "goto", taskNameForOverride(rlenv.TaskOverride{}))
	assert.Equal(t, "mine", taskNameForOverride(rlenv.TaskOverride{GoToTargetDisabled: true, MineTargetBlock: "minecraft:dirt"}))
	assert.Equal(t, "craft", taskNameForOverride(rlenv.TaskOverride{GoToTargetDisabled: true, CraftTargetItem: "minecraft:stick"}))
	assert.Equal(t, "mine_far", taskNameForOverride(rlenv.TaskOverride{MineTargetBlock: "minecraft:sand", SeedAtGoal: true}))
	assert.Equal(t, "craft_far", taskNameForOverride(rlenv.TaskOverride{CraftTargetItem: "minecraft:chest", SeedAtGoal: true}))
	assert.Equal(t, "unknown", taskNameForOverride(rlenv.TaskOverride{GoToTargetDisabled: true}))
}

func TestLoadCurriculumPoolReadsTheFarFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "curriculum.json")
	require.NoError(t, os.WriteFile(path, []byte(`{
		"target_offsets": [[5,0,0]],
		"mine_far_target_offsets": [[24,0,0],[0,0,-24]],
		"mine_far_blocks": ["minecraft:sand"],
		"mine_far_search_radius": 6,
		"craft_far_target_offsets": [[24,0,0]],
		"craft_far_items": ["minecraft:chest","minecraft:bowl"],
		"craft_far_search_radius": 8
	}`), 0o600))

	pool, err := loadCurriculumPool(path)
	require.NoError(t, err)
	assert.Equal(t, [][3]float64{{24, 0, 0}, {0, 0, -24}}, pool.MineFarTargetOffsets)
	assert.Equal(t, []string{"minecraft:sand"}, pool.MineFarBlocks)
	assert.Equal(t, 6, pool.MineFarSearchRadius)
	assert.Equal(t, [][3]float64{{24, 0, 0}}, pool.CraftFarTargetOffsets)
	assert.Equal(t, []string{"minecraft:chest", "minecraft:bowl"}, pool.CraftFarItems)
	assert.Equal(t, 8, pool.CraftFarSearchRadius)
}

func TestEffectiveTimePenaltyFallsBackToRlenvsDefault(t *testing.T) {
	assert.InDelta(t, 0.01, effectiveTimePenalty(0), 1e-12)
	assert.InDelta(t, 0.2, effectiveTimePenalty(0.2), 1e-12)
}

func TestApplyEpisodeLimitsTurnsOnObservationNormalization(t *testing.T) {
	var cfg rlenv.Config
	applyEpisodeLimits(&cfg, episodeLimits{})
	assert.False(t, cfg.NormalizeObservation, "off unless asked for")

	applyEpisodeLimits(&cfg, episodeLimits{normalizeObservation: true})
	assert.True(t, cfg.NormalizeObservation)
}

func TestTaskNameForOverrideNamesTheChainStages(t *testing.T) {
	for stage, want := range map[int]string{
		rlenv.ChainGather: "chain_gather", rlenv.ChainTable: "chain_table",
		rlenv.ChainPlace: "chain_place", rlenv.ChainUse: "chain_use",
	} {
		ov := rlenv.TaskOverride{ChainStage: stage, MineTargetBlock: "minecraft:oak_log", CraftTargetItem: "minecraft:bowl", SeedAtGoal: true}
		assert.Equal(t, want, taskNameForOverride(ov), "a chain episode must not be counted as mine_far/craft_far")
	}
}

func TestLoadCurriculumPoolReadsTheChainKeys(t *testing.T) {
	path := filepath.Join(t.TempDir(), "curriculum.json")
	require.NoError(t, os.WriteFile(path, []byte(`{
		"target_offsets": [[5,0,0]],
		"chain_target_offsets": [[18,0,0],[0,0,-18]],
		"chain_goal_items": ["minecraft:bowl","minecraft:chest"],
		"chain_search_radius": 8
	}`), 0o600))
	pool, err := loadCurriculumPool(path)
	require.NoError(t, err)
	assert.Equal(t, [][3]float64{{18, 0, 0}, {0, 0, -18}}, pool.ChainTargetOffsets)
	assert.Equal(t, []string{"minecraft:bowl", "minecraft:chest"}, pool.ChainGoalItems)
	assert.Equal(t, 8, pool.ChainSearchRadius)
}

func TestActionNamesIncludePlace(t *testing.T) {
	assert.Equal(t, "place", actionName(rlenv.ActionPlace))
}
