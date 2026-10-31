package pvp

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	crlconfig "github.com/reallyoldfogie/cRL-go/pkg/config"
	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	_ "github.com/reallyoldfogie/mc-agent/handler_versions" // registers version-specific packet handlers - production code (cmd/rsi-pvp-train) already does this; this test lives in pkg/pvp itself, which deliberately does not, so it must here.
	"github.com/stretchr/testify/require"

	rsitesting "github.com/reallyoldfogie/mc-rsi-trainer/testing"
)

// writeJSONFile marshals v to path, failing the test on any error -
// mirrors cmd/rsi-train's own small writeJSON test helper, duplicated
// here since that one is unexported in a different package.
func writeJSONFile(t *testing.T, path string, v any) {
	t.Helper()
	data, err := json.MarshalIndent(v, "", "  ")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(path, data, 0o644))
}

// TestRunCompletesOneMatchAgainstLiveServerLive is pkg/pvp's own smoke
// test, mirroring cmd/rsi-train's TestRunCompletesOneRoundAgainstLiveServerLive
// convention: drive the real Run() entrypoint (not a re-implemented copy)
// through a short, bounded 1v1 match against a real, auto-launched
// Minecraft server (rsitesting.EnsureServer), with deliberately tiny
// trainer settings - this proves the whole new mechanism (shared-arena
// connection, team formation, the synchronized match-reset barrier,
// containment enforcement, checkpointing) works end to end against a
// real server, not just that each piece compiles and unit-tests against
// fakes/toy environments. It is not a claim about learning quality or
// win-rate drift over time - that would need a much longer, separately
// gated run, matching this repo's own smoke-vs-"realistic" live test
// convention (see cmd/rsi-train/main_live_test.go).
func TestRunCompletesOneMatchAgainstLiveServerLive(t *testing.T) {
	if os.Getenv("MC_RSI_TRAINER_LIVE_CONFIG") == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_CONFIG not set; skipping live pkg/pvp smoke test")
	}

	settings := mcconfig.Default()
	settings.Connection = mcconfig.ConnectionSettings{
		Address: "127.0.0.1:34601",
		Offline: true,
		Name:    "PvPSmokeBase",
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	// A superflat, decoration-free world (mc-agent's own
	// testing/framework.go convention) - EnsureServer's own default
	// (normal terrain generation) can hand a match's arena origin a spawn
	// point on top of tree canopy, since ConnectManagedParticipant's
	// origin resolution has no ground-safety check of its own; every
	// reset then teleports bots right back onto the tree, and any
	// movement off it is enough fall damage to kill outright - found
	// live via pkg/tournament's own combat demo, where it looked exactly
	// like every game resolving almost instantly regardless of
	// EpisodeLen/BestOf.
	server, err := rsitesting.EnsureServer(ctx, &settings, rsitesting.WithExtraEnv(map[string]string{
		"LEVEL_TYPE":         "FLAT",
		"GENERATOR_SETTINGS": rsitesting.GeneratorSettingsDeepStoneBand,
	}))
	require.NoError(t, err)
	defer func() { _ = server.Close(context.Background()) }()

	dir := t.TempDir()
	mcConfigPath := filepath.Join(dir, "mc-agent-config.json")
	writeJSONFile(t, mcConfigPath, settings)

	// Deliberately tiny, mirroring cmd/rsi-train's own live smoke test's
	// settings exactly - a live bot's Step is a real network round trip
	// to mc-agent, not a toy in-memory environment, so cRL-go's own
	// config.Default() (RolloutSize 64, EpisodeLen 100, HiddenSize 128)
	// would make this test run far longer than a smoke test should.
	trainerConfigPath := filepath.Join(dir, "trainer-config.json")
	writeJSONFile(t, trainerConfigPath, crlconfig.Settings{
		RolloutSize: 2, EpisodeLen: 30, Gamma: 0.99, LearningRate: 0.05,
		GridSize: 1, HiddenSize: 8, Seed: 42, Workers: 1,
		ClipEpsilon: 0.2, EntropyCoef: 0.01, ValueCoef: 0.5, GAELambda: 0.95,
		PPOEpochs: 1, MinibatchSize: 2,
	})

	checkpointRoot := filepath.Join(dir, "checkpoints")

	cfg := Config{
		RunID:          "pvp-smoke",
		CheckpointRoot: checkpointRoot,
		TrainerConfig:  trainerConfigPath,
		Training: TrainingConfig{
			Duration:           Duration(4 * time.Minute),
			CheckpointInterval: Duration(60 * time.Second),
		},
		Server: ServerConfig{World: WorldSource{Kind: WorldSourceExisting, MCAgentConfig: mcConfigPath}},
		Matches: []MatchConfig{
			{
				ID:     "arena-1",
				Weapon: "minecraft:iron_sword",
				Arena:  ArenaConfig{Containment: ContainmentSoftArea, Radius: 15, CheckInterval: Duration(2 * time.Second)},
				Participants: []Participant{
					{ID: "red", Managed: true, HiddenLayers: 2, HiddenSize: 8, Team: "Red"},
					{ID: "blue", Managed: true, HiddenLayers: 2, HiddenSize: 8, Team: "Blue"},
				},
			},
		},
	}

	start := time.Now()
	err = Run(ctx, cfg)
	elapsed := time.Since(start)
	require.NoError(t, err)
	t.Logf("one bounded pvp match (2 managed combatants) completed in %s", elapsed)

	for _, participantID := range []string{"red", "blue"} {
		path := filepath.Join(checkpointRoot, "arena-1", participantID+".json")
		info, err := os.Stat(path)
		require.NoError(t, err, "expected a checkpoint for participant %q at %s", participantID, path)
		require.Greater(t, info.Size(), int64(0), "checkpoint for %q should not be empty", participantID)
	}
}
