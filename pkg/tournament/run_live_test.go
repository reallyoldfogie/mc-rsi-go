package tournament

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	crlconfig "github.com/reallyoldfogie/cRL-go/pkg/config"
	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	_ "github.com/reallyoldfogie/mc-agent/handler_versions" // registers version-specific packet handlers
	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/liveview"
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

// freeLocalAddr returns a "127.0.0.1:<port>" address with an OS-assigned
// free port, for a live test's own throwaway dashboard server. Binding a
// listener and immediately closing it to learn the port carries a small,
// accepted TOCTOU race (another process could grab the port before
// StartDashboardServer gets to it) - fine for an isolated test run, not
// a pattern for anything load-bearing.
func freeLocalAddr(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	addr := l.Addr().String()
	require.NoError(t, l.Close())
	return addr
}

// seedCheckpoints runs a short pkg/pvp training burst, just long enough
// to produce one valid, loadable checkpoint per participantID - shared by
// every pkg/tournament live test that needs real (if barely-trained)
// checkpoints to play inference games against, rather than each
// duplicating this setup. Who happens to fight whom during this seeding
// free-for-all doesn't matter; only that every checkpoint file ends up
// written and environment-ID-compatible with what a tournament config
// built from its returned paths expects. Returns a checkpointPath(id)
// lookup function.
func seedCheckpoints(t *testing.T, ctx context.Context, mcConfigPath, dir string, participantIDs []string) func(id string) string {
	t.Helper()
	checkpointDir := filepath.Join(dir, "seed-checkpoints")

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

	return func(id string) string { return filepath.Join(checkpointDir, "seed-arena", id+".json") }
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

	participantIDs := []string{"red1", "blue1", "green1", "green2", "yellow1"}
	checkpointPath := seedCheckpoints(t, ctx, mcConfigPath, dir, participantIDs)
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

// TestRunWithObservabilityServesLiveDashboardLive is pkg/tournament's
// Phase 3 smoke test. It runs an 8-competitor (four-round) bracket so the
// dashboard has multiple pairings to show progress across, with
// RunWithObservability running in the background while this test polls
// the live /status and /metrics HTTP endpoints from the main goroutine -
// the actual concurrency this feature needs to be safe under (a
// dashboard reader hitting the same *LiveStatus a different goroutine is
// simultaneously calling Refresh on), not just a sequential call-then-
// inspect.
//
// COMBAT_COMPETITION_PLAN.md's Phase 3 verification bar describes
// "dashboard/metrics smoke test with multiple concurrent matches."
// pkg/tournament's own Run plays bracket pairings sequentially by design
// (see Run's own doc comment - a shared server's arena/containment state
// would otherwise need multi-match bookkeeping pkg/pvp's MatchOrigin
// exists for but this package doesn't yet use). This test's own reading
// of "multiple matches" is multiple pairings' state visible together on
// one bracket's live dashboard as the tournament progresses through
// them, not literally concurrent game execution - the latter is a
// separate, larger feature this session's Phase 3 scope does not cover.
func TestRunWithObservabilityServesLiveDashboardLive(t *testing.T) {
	if os.Getenv("MC_RSI_TRAINER_LIVE_CONFIG") == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_CONFIG not set; skipping live pkg/tournament dashboard smoke test")
	}

	settings := mcconfig.Default()
	settings.Connection = mcconfig.ConnectionSettings{
		Address: "127.0.0.1:34612",
		Offline: true,
		Name:    "TourneyDashBase",
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()

	server, err := rsitesting.EnsureServer(ctx, &settings)
	require.NoError(t, err)
	defer func() { _ = server.Close(context.Background()) }()

	dir := t.TempDir()
	mcConfigPath := filepath.Join(dir, "mc-agent-config.json")
	writeJSONFile(t, mcConfigPath, settings)

	participantIDs := []string{"c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"}
	checkpointPath := seedCheckpoints(t, ctx, mcConfigPath, dir, participantIDs)

	competitors := make([]Competitor, len(participantIDs))
	for i, id := range participantIDs {
		competitors[i] = Competitor{ID: id, DisplayName: id, Participants: []pvp.Participant{
			{ID: id, Managed: true, HiddenLayers: 2, HiddenSize: 8, CheckpointPath: checkpointPath(id)},
		}}
	}

	tournamentCfg := Config{
		RunID:       "tourney-dash-smoke",
		ResultsRoot: filepath.Join(dir, "results"),
		Greedy:      true,
		BestOf:      1,
		EpisodeLen:  60,
		Weapon:      "minecraft:iron_sword",
		Arena:       pvp.ArenaConfig{Containment: pvp.ContainmentSoftArea, Radius: 20, CheckInterval: pvp.Duration(2 * time.Second)},
		Server:      pvp.ServerConfig{World: pvp.WorldSource{Kind: pvp.WorldSourceExisting, MCAgentConfig: mcConfigPath}},
		Competitors: competitors,
	}

	metrics, err := NewMetrics(tournamentCfg.RunID)
	require.NoError(t, err)
	status := NewLiveStatus(tournamentCfg.RunID)
	view := liveview.NewRegistry()
	go view.RunPoller(ctx)
	addr := freeLocalAddr(t)
	StartDashboardServer(addr, status, metrics, view)
	baseURL := "http://" + addr

	type runResult struct {
		bracket *Bracket
		err     error
	}
	resultCh := make(chan runResult, 1)
	go func() {
		bracket, err := RunWithObservability(ctx, tournamentCfg, metrics, status, view)
		resultCh <- runResult{bracket, err}
	}()

	// Poll /status and /agents while the tournament runs: /status to
	// confirm the dashboard actually advances rather than just reflecting
	// a single static snapshot, and /agents to confirm each pairing's
	// live bots are actually registered with real positions and that at
	// least one of them has its opponent showing up in Nearby - the
	// user's own request ("render the players that each competitor
	// views"), live-verified end to end: no new tracking code was
	// written for this, it's purely a consequence of GetTrackedEntities
	// already including other players.
	var sawMultiplePairingsDecided, sawPopulatedAgents, sawOpponentInNearby bool
	stopPoll := make(chan struct{})
	pollDone := make(chan struct{})
	go func() {
		defer close(pollDone)
		ticker := time.NewTicker(500 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-stopPoll:
				return
			case <-ticker.C:
				resp, err := http.Get(baseURL + "/status")
				if err != nil {
					continue
				}
				var snap StatusSnapshot
				decodeErr := json.NewDecoder(resp.Body).Decode(&snap)
				resp.Body.Close()
				if decodeErr != nil {
					continue
				}
				decided := 0
				for _, p := range snap.Pairings {
					if p.WinnerID != "" {
						decided++
					}
				}
				if decided >= 2 {
					sawMultiplePairingsDecided = true
				}

				agentsResp, err := http.Get(baseURL + "/agents")
				if err == nil {
					var agents []liveview.Snapshot
					if json.NewDecoder(agentsResp.Body).Decode(&agents) == nil {
						for _, a := range agents {
							if a.X != 0 || a.Y != 0 || a.Z != 0 {
								sawPopulatedAgents = true
							}
							if len(a.Nearby) > 0 {
								sawOpponentInNearby = true
							}
						}
					}
					agentsResp.Body.Close()
				}

				if snap.ChampionDecided {
					return
				}
			}
		}
	}()

	res := <-resultCh
	// Stop the poller promptly once Run itself has returned, rather than
	// waiting for its own ticker-driven ChampionDecided check next fires.
	close(stopPoll)
	<-pollDone

	require.NoError(t, res.err)
	champion, ok := res.bracket.Champion()
	require.True(t, ok, "want a decided champion")
	t.Logf("8-competitor tournament champion: %s", champion)
	require.True(t, sawMultiplePairingsDecided, "dashboard /status should have shown at least 2 decided pairings while the tournament progressed through its bracket")
	require.True(t, sawPopulatedAgents, "/agents should have shown at least one bot with a real (non-zero) position while a pairing was live")
	require.True(t, sawOpponentInNearby, "/agents should have shown at least one bot with a non-empty Nearby list - the opponent should show up as a tracked entity with no combat-specific code")

	dashboardResp, err := http.Get(baseURL + "/dashboard")
	require.NoError(t, err)
	defer dashboardResp.Body.Close()
	require.Equal(t, http.StatusOK, dashboardResp.StatusCode)

	finalStatusResp, err := http.Get(baseURL + "/status")
	require.NoError(t, err)
	defer finalStatusResp.Body.Close()
	var finalStatus StatusSnapshot
	require.NoError(t, json.NewDecoder(finalStatusResp.Body).Decode(&finalStatus))
	require.True(t, finalStatus.ChampionDecided)
	require.Equal(t, champion, finalStatus.Champion)
	require.Len(t, finalStatus.Pairings, 7, "an 8-competitor single-elimination bracket has 7 pairings")

	scoreboardResp, err := http.Get(baseURL + "/")
	require.NoError(t, err)
	defer scoreboardResp.Body.Close()
	require.Equal(t, http.StatusOK, scoreboardResp.StatusCode)

	metricsResp, err := http.Get(baseURL + "/metrics")
	require.NoError(t, err)
	defer metricsResp.Body.Close()
	metricsBody, err := io.ReadAll(metricsResp.Body)
	require.NoError(t, err)
	for _, want := range []string{
		"rsi_tournament_games_played_total",
		"rsi_tournament_pairings_played_total",
		"rsi_tournament_champion_decided",
	} {
		require.Contains(t, string(metricsBody), want, "expected /metrics to expose %s", want)
	}
	require.Contains(t, string(metricsBody), fmt.Sprintf(`run_id="%s"`, tournamentCfg.RunID))
}
