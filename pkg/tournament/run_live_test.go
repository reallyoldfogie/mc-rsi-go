package tournament

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	crlconfig "github.com/reallyoldfogie/cRL-go/pkg/config"
	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	_ "github.com/reallyoldfogie/mc-agent/handler_versions" // registers version-specific packet handlers
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/pvp"
	rsitesting "github.com/reallyoldfogie/mc-rsi-trainer/testing"
	"github.com/stretchr/testify/require"
)

func writeJSONFile(t *testing.T, path string, v any) {
	t.Helper()
	data, err := json.MarshalIndent(v, "", "  ")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(path, data, 0o644))
}

// TestRunCompletesATournamentAgainstLiveServerLive is pkg/tournament's
// own smoke test. It exercises the mechanism this session's Phase 2
// work actually added - bracket construction/advancement, multi-member
// team aggregation, frozen-checkpoint inference play, per-game outcome
// computation, and incremental result persistence - against a real
// server, not the full "4-8 teams incl. an unmanaged entry" bar
// COMBAT_COMPETITION_PLAN.md's own verification section describes.
//
// The unmanaged-entry path is deliberately NOT exercised live here: a
// real (or stand-in) unmanaged participant would need to already be
// positioned near the arena center before FormTeams/SeedMatchReset's own
// VerifyPlayerNearPosition gate will touch them at all (Phase 0's hard
// safety rule) - but today's arena origin is derived dynamically from
// whichever managed bot connects first in a given pairing, so there is
// no fixed, predictable location for an unmanaged participant to have
// gone and stood at ahead of time. That's a real gap for a future
// fixed-origin config option, not something to paper over with a
// workaround here; the unmanaged code path itself stays covered by
// config_test.go's unit tests.
//
// Uses BestOf=1 (not the production default of 3) to keep this a smoke
// test's wall-clock cost, not a "few hundred episodes" claim about
// learning quality - the same scope choice pkg/pvp's own live smoke
// test (docs/plans/19-combat-pvp-training.md) made for Phase 1.
func TestRunCompletesATournamentAgainstLiveServerLive(t *testing.T) {
	if os.Getenv("MC_RSI_TRAINER_LIVE_CONFIG") == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_CONFIG not set; skipping live pkg/tournament smoke test")
	}

	settings := mcconfig.Default()
	settings.Connection = mcconfig.ConnectionSettings{
		Address: "127.0.0.1:34611",
		Offline: true,
		Name:    "TourneySmokeBase",
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

	server, err := rsitesting.EnsureServer(ctx, &settings)
	require.NoError(t, err)
	defer func() { _ = server.Close(context.Background()) }()

	dir := t.TempDir()
	mcConfigPath := filepath.Join(dir, "mc-agent-config.json")
	writeJSONFile(t, mcConfigPath, settings)

	checkpointDir := filepath.Join(dir, "seed-checkpoints")

	// Phase A: a short pkg/pvp training run, just long enough to produce
	// one valid, loadable checkpoint per participant this tournament
	// will need - the SAME five bots this test's bracket competes with,
	// training together in one free-for-all match (who happens to fight
	// whom during this seeding phase doesn't matter; only that every
	// checkpoint file ends up written and environment-ID-compatible
	// with what the tournament config below expects).
	participantIDs := []string{"red1", "blue1", "green1", "green2", "yellow1"}
	seedParticipants := make([]pvp.Participant, len(participantIDs))
	for i, id := range participantIDs {
		seedParticipants[i] = pvp.Participant{ID: id, Managed: true, HiddenLayers: 2, HiddenSize: 8}
	}
	// A small trainer config matching every seedParticipant's own
	// HiddenSize (8): cRL-go's own ppo.NewWithPersistentEnvPool
	// validates that an explicit initial actorcritic.Params' hidden
	// size matches settings.HiddenSize, so leaving this unset (falling
	// back to crlconfig.Default()'s HiddenSize of 128) is a hard
	// construction error, not a silent mismatch - found live.
	trainerConfigPath := filepath.Join(dir, "seed-trainer-config.json")
	writeJSONFile(t, trainerConfigPath, crlconfig.Settings{
		RolloutSize: 2, EpisodeLen: 30, Gamma: 0.99, LearningRate: 0.05,
		GridSize: 1, HiddenSize: 8, Seed: 42, Workers: 1,
		ClipEpsilon: 0.2, EntropyCoef: 0.01, ValueCoef: 0.5, GAELambda: 0.95,
		PPOEpochs: 1, MinibatchSize: 2,
	})

	seedCfg := pvp.Config{
		RunID:          "tourney-seed",
		CheckpointRoot: checkpointDir,
		TrainerConfig:  trainerConfigPath,
		Training: pvp.TrainingConfig{
			Duration:           pvp.Duration(90 * time.Second),
			CheckpointInterval: pvp.Duration(20 * time.Second),
		},
		Server: pvp.ServerConfig{World: pvp.WorldSource{Kind: pvp.WorldSourceExisting, MCAgentConfig: mcConfigPath}},
		Matches: []pvp.MatchConfig{
			{
				ID:           "seed-arena",
				Weapon:       "minecraft:iron_sword",
				Arena:        pvp.ArenaConfig{Containment: pvp.ContainmentSoftArea, Radius: 20, CheckInterval: pvp.Duration(2 * time.Second)},
				Participants: seedParticipants,
			},
		},
	}
	require.NoError(t, pvp.Run(ctx, seedCfg), "seeding checkpoints via a short pvp.Run training burst")

	for _, id := range participantIDs {
		path := filepath.Join(checkpointDir, "seed-arena", id+".json")
		info, err := os.Stat(path)
		require.NoError(t, err, "expected a seed checkpoint for %q at %s", id, path)
		require.Greater(t, info.Size(), int64(0))
	}

	// Phase B: the actual tournament, loading those checkpoints for
	// pure inference play.
	checkpointPath := func(id string) string { return filepath.Join(checkpointDir, "seed-arena", id+".json") }
	tournamentCfg := Config{
		RunID:       "tourney-smoke",
		ResultsRoot: filepath.Join(dir, "results"),
		Greedy:      true,
		BestOf:      1,
		EpisodeLen:  60,
		Weapon:      "minecraft:iron_sword",
		Arena:       pvp.ArenaConfig{Containment: pvp.ContainmentSoftArea, Radius: 20, CheckInterval: pvp.Duration(2 * time.Second)},
		Server:      pvp.ServerConfig{World: pvp.WorldSource{Kind: pvp.WorldSourceExisting, MCAgentConfig: mcConfigPath}},
		Competitors: []Competitor{
			{ID: "red", DisplayName: "Red", Participants: []pvp.Participant{
				{ID: "red1", Managed: true, HiddenLayers: 2, HiddenSize: 8, CheckpointPath: checkpointPath("red1")},
			}},
			{ID: "blue", DisplayName: "Blue", Participants: []pvp.Participant{
				{ID: "blue1", Managed: true, HiddenLayers: 2, HiddenSize: 8, CheckpointPath: checkpointPath("blue1")},
			}},
			{ID: "green", DisplayName: "Green (2-member)", Participants: []pvp.Participant{
				{ID: "green1", Managed: true, HiddenLayers: 2, HiddenSize: 8, CheckpointPath: checkpointPath("green1")},
				{ID: "green2", Managed: true, HiddenLayers: 2, HiddenSize: 8, CheckpointPath: checkpointPath("green2")},
			}},
			{ID: "yellow", DisplayName: "Yellow", Participants: []pvp.Participant{
				{ID: "yellow1", Managed: true, HiddenLayers: 2, HiddenSize: 8, CheckpointPath: checkpointPath("yellow1")},
			}},
		},
	}

	start := time.Now()
	bracket, err := Run(ctx, tournamentCfg)
	elapsed := time.Since(start)
	require.NoError(t, err)
	t.Logf("4-competitor tournament (one 2-member team) completed in %s", elapsed)

	champion, ok := bracket.Champion()
	require.True(t, ok, "want a decided champion")
	t.Logf("champion: %s", champion)

	resultsData, err := os.ReadFile(filepath.Join(dir, "results", "results.json"))
	require.NoError(t, err)
	var results Results
	require.NoError(t, json.Unmarshal(resultsData, &results))
	require.Equal(t, champion, results.Champion)
	require.NotNil(t, results.FinishedAt)
	require.Len(t, results.Pairings, 3, "want exactly 3 pairings: two semifinals and a final")
	for _, p := range results.Pairings {
		require.NotEmpty(t, p.WinnerID, "pairing %+v should have a recorded winner", p)
		require.NotEmpty(t, p.Games, "pairing %+v should have at least one recorded game", p)
	}
}
