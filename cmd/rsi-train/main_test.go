package main

import (
	"context"
	"errors"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	mcconfig "github.com/reallyoldfogie/mc-agent/config"
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
		{"malformed clone-area-from", validArgs("-clone-area-from", "not,a,point")},
		{"clone-area-from with too few coordinates", validArgs("-clone-area-from", "1,2")},
		{"clone-area-radius over the clone limit", validArgs("-clone-area-from", "0,-61,5000", "-clone-area-radius", "40")},
		{"zero parallel-envs", validArgs("-parallel-envs", "0")},
		{"negative parallel-envs", validArgs("-parallel-envs", "-1")},
		{"zero shared-server-separation-chunks with -shared-server", validArgs("-shared-server", "-shared-server-separation-chunks", "0")},
		{"negative shared-server-separation-chunks with -shared-server", validArgs("-shared-server", "-shared-server-separation-chunks", "-1")},
		{"zero hidden-layers", validArgs("-hidden-layers", "0")},
		{"negative hidden-layers", validArgs("-hidden-layers", "-1")},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseFlags(tc.args)
			assert.Error(t, err)
		})
	}
}

func TestPacketLogsAreOptIn(t *testing.T) {
	f, err := parseFlags([]string{"-checkpoint-dir", "/tmp/ckpt", "-mc-agent-config", "/tmp/config.json"})
	require.NoError(t, err)
	assert.False(t, f.packetLogs, "packet logs reached tens of GB per run, so they default off")
	assert.Equal(t, "", f.packetLogDir(), "no directory means connectAgent writes no packet log")

	f, err = parseFlags([]string{"-checkpoint-dir", "/tmp/ckpt", "-mc-agent-config", "/tmp/config.json", "-packet-logs"})
	require.NoError(t, err)
	assert.Equal(t, "/tmp/ckpt", f.packetLogDir())
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

	params, rec, err := loadOrInitTeacher(dir, environmentID, observationSize, hiddenSize, actionSpace, 2)
	require.NoError(t, err)
	require.NotNil(t, params)
	assert.Equal(t, 2, params.NumHiddenLayers())
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

	first, firstRec, err := loadOrInitTeacher(dir, environmentID, observationSize, hiddenSize, actionSpace, 2)
	require.NoError(t, err)

	second, secondRec, err := loadOrInitTeacher(dir, environmentID, observationSize, hiddenSize, actionSpace, 2)
	require.NoError(t, err)

	assert.Equal(t, firstRec.CreatedAt.Unix(), secondRec.CreatedAt.Unix())
	assert.Equal(t, first.Hidden[0].W.Data, second.Hidden[0].W.Data, "second call must resume the exact same persisted weights, not a fresh random init")
}

// TestLoadOrInitTeacherHonorsNumHiddenLayersWhenInitializingFresh is the
// -hidden-layers flag's own test: a fresh generation 0 must actually be
// built at the requested depth, not silently stuck at 2.
func TestLoadOrInitTeacherHonorsNumHiddenLayersWhenInitializingFresh(t *testing.T) {
	dir := t.TempDir()
	const observationSize, hiddenSize, actionSpace = 17, 8, 4
	const environmentID = "mc-agent-rlenv:actions=4:obs=17"

	params, _, err := loadOrInitTeacher(dir, environmentID, observationSize, hiddenSize, actionSpace, 4)
	require.NoError(t, err)
	assert.Equal(t, 4, params.NumHiddenLayers())
}

// TestLoadOrInitTeacherIgnoresNumHiddenLayersWhenResuming confirms the
// other half of -hidden-layers' documented contract: once a generation
// exists on disk, its own depth always wins, whatever -hidden-layers a
// later run happens to pass (e.g. because the operator forgot to change
// it back after trying a different depth once).
func TestLoadOrInitTeacherIgnoresNumHiddenLayersWhenResuming(t *testing.T) {
	dir := t.TempDir()
	const observationSize, hiddenSize, actionSpace = 17, 8, 4
	const environmentID = "mc-agent-rlenv:actions=4:obs=17"

	first, _, err := loadOrInitTeacher(dir, environmentID, observationSize, hiddenSize, actionSpace, 3)
	require.NoError(t, err)
	require.Equal(t, 3, first.NumHiddenLayers())

	second, _, err := loadOrInitTeacher(dir, environmentID, observationSize, hiddenSize, actionSpace, 5)
	require.NoError(t, err)
	assert.Equal(t, 3, second.NumHiddenLayers(), "an existing generation's own depth must win over whatever -hidden-layers a later run passes")
}

func TestParseFlagsDefaultsHiddenLayersToTwoAndAcceptsAnOverride(t *testing.T) {
	f, err := parseFlags([]string{"-checkpoint-dir", "/tmp/ckpt", "-mc-agent-config", "/tmp/config.json"})
	require.NoError(t, err)
	assert.Equal(t, 2, f.hiddenLayers, "2 is this repo's long-standing default, so an un-set flag must preserve it exactly")

	f, err = parseFlags([]string{"-checkpoint-dir", "/tmp/ckpt", "-mc-agent-config", "/tmp/config.json", "-hidden-layers", "4"})
	require.NoError(t, err)
	assert.Equal(t, 4, f.hiddenLayers)
}

// TestLoadOrInitTeacherAndHealthHandleADeepenedCheckpoint is an
// end-to-end check that this binary's own checkpoint-loading and
// health-measurement paths actually work against a non-2-hidden-layer
// network, not just that the underlying cRL-go/pkg/actorcritic
// primitives do in isolation. It mirrors the real workflow
// cmd/deepen-checkpoint supports: deepen an existing generation into a
// fresh generation 0 in a separate directory, then point rsi-train's own
// -checkpoint-dir at that directory for a later run.
func TestLoadOrInitTeacherAndHealthHandleADeepenedCheckpoint(t *testing.T) {
	const observationSize, hiddenSize, actionSpace = 17, 8, 4
	const environmentID = "mc-agent-rlenv:actions=4:obs=17"

	teacherDir := t.TempDir()
	teacher, _, err := loadOrInitTeacher(teacherDir, environmentID, observationSize, hiddenSize, actionSpace, 2)
	require.NoError(t, err)
	require.Equal(t, 2, teacher.NumHiddenLayers(), "sanity: loadOrInitTeacher's own fresh networks are 2 hidden layers today")

	deeper, err := actorcritic.Net2DeeperNet(teacher, 1)
	require.NoError(t, err)
	require.Equal(t, 3, deeper.NumHiddenLayers())

	deepenedDir := t.TempDir()
	require.NoError(t, lineage.Save(deepenedDir, deeper, lineage.Record{
		Generation: 0, ParentGeneration: -1, Provenance: lineage.ProvenanceDeepened,
		SeededFromGeneration: -1, EnvironmentID: environmentID, Metadata: checkpoint.Metadata{},
	}, lineage.PlainCodec))

	// This is the actual code path a later `-checkpoint-dir deepenedDir`
	// run of this binary takes: load whatever's there, no depth check.
	loaded, rec, err := loadOrInitTeacher(deepenedDir, environmentID, observationSize, hiddenSize, actionSpace, 2)
	require.NoError(t, err)
	assert.Equal(t, 0, rec.Generation)
	assert.Equal(t, 3, loaded.NumHiddenLayers())
	assert.Equal(t, observationSize, loaded.InputSize())
	assert.Equal(t, actionSpace, loaded.OutputSize())

	health := measureNetworkHealth(loaded)
	assert.Len(t, health.LayerActive, 3)
	assert.False(t, health.dead(), "a freshly-deepened (function-preserving) network must still look healthy")
}

// TestLoadOrInitTeacherRecurrentCreatesAndPersistsGenerationZero mirrors
// TestLoadOrInitTeacherCreatesAndPersistsGenerationZero: an empty
// checkpoint dir must initialize a fresh *actorcritic.RecurrentParams
// generation 0, marked Recurrent, and persist it immediately.
func TestLoadOrInitTeacherRecurrentCreatesAndPersistsGenerationZero(t *testing.T) {
	dir := t.TempDir()
	const observationSize, hiddenSize, actionSpace = 17, 8, 4
	const environmentID = "mc-agent-rlenv:actions=4:obs=17"

	params, rec, err := loadOrInitTeacherRecurrent(dir, environmentID, observationSize, hiddenSize, actionSpace, 2)
	require.NoError(t, err)
	require.NotNil(t, params)
	assert.Equal(t, 0, rec.Generation)
	assert.Equal(t, -1, rec.ParentGeneration)
	assert.Equal(t, lineage.ProvenanceLeapfrog, rec.Provenance)
	assert.True(t, rec.Recurrent)
	assert.Equal(t, hiddenSize, params.HiddenSize())

	latest, err := lineage.Latest(dir)
	require.NoError(t, err)
	assert.Equal(t, 0, latest.Generation)
	assert.True(t, latest.Recurrent)
}

// TestLoadOrInitTeacherRecurrentResumesExistingRecurrentGeneration
// mirrors TestLoadOrInitTeacherResumesExistingLatestGeneration: once a
// recurrent generation exists on disk, a second call must resume it
// exactly, not warm-start or re-initialize on top of it.
func TestLoadOrInitTeacherRecurrentResumesExistingRecurrentGeneration(t *testing.T) {
	dir := t.TempDir()
	const observationSize, hiddenSize, actionSpace = 17, 8, 4
	const environmentID = "mc-agent-rlenv:actions=4:obs=17"

	first, firstRec, err := loadOrInitTeacherRecurrent(dir, environmentID, observationSize, hiddenSize, actionSpace, 2)
	require.NoError(t, err)

	second, secondRec, err := loadOrInitTeacherRecurrent(dir, environmentID, observationSize, hiddenSize, actionSpace, 2)
	require.NoError(t, err)

	assert.Equal(t, firstRec.Generation, secondRec.Generation)
	assert.Equal(t, first.Trunk.Hidden[0].W.Data, second.Trunk.Hidden[0].W.Data, "second call must resume the exact same persisted weights, not a fresh random init")
	assert.Equal(t, first.LSTM.WhF.Data, second.LSTM.WhF.Data)
}

// TestLoadOrInitTeacherRecurrentWarmStartsFromANonRecurrentGeneration is
// the auto-trigger's own test: pointing a recurrent-configured run at an
// existing *non*-recurrent checkpoint must warm-start it
// (actorcritic.WarmStartRecurrentParams) into a new generation one past
// the source's, saved as recurrent - and leave the source generation's
// own files completely untouched on disk, matching
// cmd/widen-checkpoint/cmd/deepen-checkpoint's "never touch the source"
// convention (see loadOrInitTeacherRecurrent's own doc comment).
func TestLoadOrInitTeacherRecurrentWarmStartsFromANonRecurrentGeneration(t *testing.T) {
	dir := t.TempDir()
	const observationSize, hiddenSize, actionSpace = 17, 8, 4
	const environmentID = "mc-agent-rlenv:actions=4:obs=17"

	plain, plainRec, err := loadOrInitTeacher(dir, environmentID, observationSize, hiddenSize, actionSpace, 2)
	require.NoError(t, err)
	require.False(t, plainRec.Recurrent)

	before, err := os.ReadFile(filepath.Join(dir, "generation-000000000-params.json"))
	require.NoError(t, err)

	warmStarted, rec, err := loadOrInitTeacherRecurrent(dir, environmentID, observationSize, hiddenSize, actionSpace, 2)
	require.NoError(t, err)
	assert.Equal(t, plainRec.Generation+1, rec.Generation)
	assert.Equal(t, plainRec.Generation, rec.ParentGeneration)
	assert.Equal(t, lineage.ProvenanceWarmStarted, rec.Provenance)
	assert.True(t, rec.Recurrent)
	assert.Equal(t, plain.Hidden[0].W.Data, warmStarted.Trunk.Hidden[0].W.Data, "the warm-started trunk must reuse the source's exact trained weights")

	after, err := os.ReadFile(filepath.Join(dir, "generation-000000000-params.json"))
	require.NoError(t, err)
	assert.Equal(t, before, after, "the source (non-recurrent) generation's own params file must be byte-for-byte untouched")

	latest, err := lineage.Latest(dir)
	require.NoError(t, err)
	assert.Equal(t, rec.Generation, latest.Generation, "lineage.Latest must now report the new warm-started generation, not the source")

	// A third call must now resume the warm-started generation (already
	// recurrent) rather than warm-starting all over again from the
	// now-stale non-recurrent one.
	again, againRec, err := loadOrInitTeacherRecurrent(dir, environmentID, observationSize, hiddenSize, actionSpace, 2)
	require.NoError(t, err)
	assert.Equal(t, rec.Generation, againRec.Generation)
	assert.Equal(t, warmStarted.Trunk.Hidden[0].W.Data, again.Trunk.Hidden[0].W.Data)
	assert.Equal(t, warmStarted.LSTM.WhF.Data, again.LSTM.WhF.Data, "resuming must not re-draw a fresh random LSTM")
}

// TestMeasureNetworkHealthToleratesAFreshNetworkOfAnyDepth is a lighter
// companion check across a spread of depths, independent of the
// checkpoint-loading path above.
func TestMeasureNetworkHealthToleratesAFreshNetworkOfAnyDepth(t *testing.T) {
	for _, numHiddenLayers := range []int{1, 2, 3, 5} {
		p := actorcritic.NewParams(rand.New(rand.NewPCG(1, 2)), 17, 32, 4, numHiddenLayers)
		h := measureNetworkHealth(p)
		assert.Len(t, h.LayerActive, numHiddenLayers)
		assert.False(t, h.dead(), "numHiddenLayers=%d", numHiddenLayers)
	}
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
	assert.Nil(t, cfg.CloneAreaFrom, "cloning stays fully off when -clone-area-from is unset")

	cfg = rlenv.Config{}
	from := [3]float64{0, -61, 5000}
	applyEpisodeLimits(&cfg, episodeLimits{cloneAreaFrom: &from, cloneAreaRadius: 16, cloneAreaHeight: 4, cloneAreaDepth: 3})
	require.NotNil(t, cfg.CloneAreaFrom)
	assert.Equal(t, from, *cfg.CloneAreaFrom)
	assert.Equal(t, 16, cfg.CloneAreaRadius)
	assert.Equal(t, 4, cfg.CloneAreaHeight)
	assert.Equal(t, 3, cfg.CloneAreaDepth)

	operatorFrom := [3]float64{1, 2, 3}
	cfg = rlenv.Config{CloneAreaFrom: &operatorFrom, CloneAreaRadius: 5}
	applyEpisodeLimits(&cfg, episodeLimits{cloneAreaFrom: &from, cloneAreaRadius: 16})
	assert.Equal(t, operatorFrom, *cfg.CloneAreaFrom, "an operator-configured CloneAreaFrom must win")
	assert.Equal(t, 5, cfg.CloneAreaRadius)
}

func TestParseXYZ(t *testing.T) {
	v, err := parseXYZ("0,-61,5000")
	require.NoError(t, err)
	assert.Equal(t, [3]float64{0, -61, 5000}, v)

	v, err = parseXYZ(" 1.5 , -2 , 3 ")
	require.NoError(t, err)
	assert.Equal(t, [3]float64{1.5, -2, 3}, v)

	_, err = parseXYZ("1,2")
	assert.Error(t, err)
	_, err = parseXYZ("1,2,3,4")
	assert.Error(t, err)
	_, err = parseXYZ("a,b,c")
	assert.Error(t, err)
}

func TestFlagsEpisodeLimitsOmitsCloneAreaFromWhenUnset(t *testing.T) {
	f, err := parseFlags([]string{"-checkpoint-dir", "/tmp/ckpt", "-mc-agent-config", "/tmp/config.json"})
	require.NoError(t, err)
	assert.Nil(t, f.episodeLimits().cloneAreaFrom)

	f, err = parseFlags([]string{"-checkpoint-dir", "/tmp/ckpt", "-mc-agent-config", "/tmp/config.json", "-clone-area-from", "0,-61,5000"})
	require.NoError(t, err)
	limits := f.episodeLimits()
	require.NotNil(t, limits.cloneAreaFrom)
	assert.Equal(t, [3]float64{0, -61, 5000}, *limits.cloneAreaFrom)
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

func TestGoalDescriptionNamesTheActualSuccessCondition(t *testing.T) {
	assert.Equal(t, "goto (+5, +0, +0) from spawn", goalDescription(rlenv.TaskOverride{TargetOffset: [3]float64{5, 0, 0}}))
	assert.Equal(t, "mine dirt", goalDescription(rlenv.TaskOverride{GoToTargetDisabled: true, MineTargetBlock: "minecraft:dirt"}))
	assert.Equal(t, "craft stick", goalDescription(rlenv.TaskOverride{GoToTargetDisabled: true, CraftTargetItem: "minecraft:stick"}))
	assert.Equal(t, "mine sand", goalDescription(rlenv.TaskOverride{MineTargetBlock: "minecraft:sand", SeedAtGoal: true}))
	assert.Equal(t, "craft chest", goalDescription(rlenv.TaskOverride{CraftTargetItem: "minecraft:chest", SeedAtGoal: true}))
	assert.Equal(t, "", goalDescription(rlenv.TaskOverride{GoToTargetDisabled: true}), "no active task has no goal to show")
}

func TestGoalDescriptionNamesTheChainsUltimateGoalRegardlessOfStage(t *testing.T) {
	for stage := range map[int]bool{rlenv.ChainGather: true, rlenv.ChainTable: true, rlenv.ChainPlace: true, rlenv.ChainUse: true} {
		ov := rlenv.TaskOverride{ChainStage: stage, MineTargetBlock: "minecraft:oak_log", CraftTargetItem: "minecraft:bowl", SeedAtGoal: true}
		assert.Equal(t, "craft bowl", goalDescription(ov), "every chain stage shares the same ultimate craft goal")
	}
}

func TestGoalForSettingsMirrorsGoalDescriptionForAStaticEnv(t *testing.T) {
	assert.Equal(t, "mine dirt", goalForSettings(mcconfig.EnvSettings{MineTargetBlock: "minecraft:dirt"}))
	assert.Equal(t, "craft stick", goalForSettings(mcconfig.EnvSettings{CraftTargetItem: "minecraft:stick"}))
	assert.Equal(t, "goto (+5, +0, +0) from spawn", goalForSettings(mcconfig.EnvSettings{TargetOffset: [3]float64{5, 0, 0}}))
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

// fakeSeedAgent is the minimum rlenv.SeedAgent (plus an optional
// ClearInventory, like the real agent type has via ChainSeedAgent) for
// craftIngredientSeeder's own tests: tracking calls rather than touching a
// live bot.
type fakeSeedAgent struct {
	clearInventoryCalls     int
	clearInventoryErr       error
	seedCraftIngredients    []string // every itemName SeedCraftIngredients was called with
	seedCraftIngredientsErr error
	seedNearbyBlockCalls    int
}

func (f *fakeSeedAgent) ClearInventory(ctx context.Context) error {
	f.clearInventoryCalls++
	return f.clearInventoryErr
}

func (f *fakeSeedAgent) SeedCraftIngredients(ctx context.Context, itemName string) error {
	f.seedCraftIngredients = append(f.seedCraftIngredients, itemName)
	return f.seedCraftIngredientsErr
}

func (f *fakeSeedAgent) SeedNearbyBlock(ctx context.Context, blockName string, radius int) error {
	f.seedNearbyBlockCalls++
	return nil
}

func TestCraftIngredientSeederNoOpsWhenNoCraftTargetItem(t *testing.T) {
	// The seedAtGoal/chain path (craft_far, chain_*) arrives here with
	// CraftTargetItem already blanked by Reset itself - see this
	// function's own doc comment - so this covers both "plain goto/mine"
	// and "far/chain, already handled elsewhere" in one case.
	agent := &fakeSeedAgent{}
	err := craftIngredientSeeder(context.Background(), agent, rlenv.Config{})
	require.NoError(t, err)
	assert.Zero(t, agent.clearInventoryCalls)
	assert.Empty(t, agent.seedCraftIngredients)
}

func TestCraftIngredientSeederClearsThenSeedsForAPlainCraftTask(t *testing.T) {
	agent := &fakeSeedAgent{}
	err := craftIngredientSeeder(context.Background(), agent, rlenv.Config{CraftTargetItem: "minecraft:stick"})
	require.NoError(t, err)
	assert.Equal(t, 1, agent.clearInventoryCalls)
	assert.Equal(t, []string{"minecraft:stick"}, agent.seedCraftIngredients)
}

func TestCraftIngredientSeederPropagatesAClearInventoryFailure(t *testing.T) {
	agent := &fakeSeedAgent{clearInventoryErr: errors.New("rcon down")}
	err := craftIngredientSeeder(context.Background(), agent, rlenv.Config{CraftTargetItem: "minecraft:stick"})
	require.Error(t, err)
	assert.Empty(t, agent.seedCraftIngredients, "must not seed ingredients on top of a clear that failed")
}

func TestCraftIngredientSeederPropagatesASeedCraftIngredientsFailure(t *testing.T) {
	agent := &fakeSeedAgent{seedCraftIngredientsErr: errors.New("no crafting table within range")}
	err := craftIngredientSeeder(context.Background(), agent, rlenv.Config{CraftTargetItem: "minecraft:chest"})
	require.Error(t, err)
}
