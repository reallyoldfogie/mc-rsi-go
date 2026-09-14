package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	crlconfig "github.com/reallyoldfogie/cRL-go/pkg/config"
	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	"github.com/stretchr/testify/require"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/lineage"
	rsitesting "github.com/reallyoldfogie/mc-rsi-trainer/testing"
)

// TestRunCompletesOneRoundAgainstLiveServerLive is cmd/rsi-train's own
// live smoke test: drives the real run() entrypoint (not a re-implemented
// copy of it) through one full leapfrog round against a real, auto-
// launched Minecraft server (testing.EnsureServer, docs/plans/10) — the
// first actual proof that this repo's whole single-environment loop
// (docs/plans/03 pkg/leapfrog.Round, wired together by docs/plans/04's
// cmd/rsi-train) works end to end, not just that each piece compiles and
// unit-tests in isolation. Written for docs/plans/08's own stated
// precondition: it "depends on 03/04 ... being proven out first," and
// this is that proof, plus the real wall-clock timing docs/plans/08's
// rewrite is grounded in.
//
// Importing github.com/reallyoldfogie/mc-rsi-trainer/testing here (a
// _test.go file only, never main.go) does not add EnsureServer's Docker
// dependency to the shipped rsi-train binary — production code
// (main.go) deliberately never imports that package itself, exactly so
// a real training deployment doesn't need Docker just because this
// smoke test does.
func TestRunCompletesOneRoundAgainstLiveServerLive(t *testing.T) {
	if os.Getenv("MC_RSI_TRAINER_LIVE_CONFIG") == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_CONFIG not set; skipping live cmd/rsi-train smoke test")
	}

	settings := mcconfig.Default()
	settings.Connection = mcconfig.ConnectionSettings{
		Address: "127.0.0.1:34599",
		Offline: true,
		Name:    "RSITrainSmoke", // Minecraft usernames are capped at 16 characters — this is 13; see this test's own recent history for why that margin matters.
	}
	settings.Env = mcconfig.EnvSettings{
		TargetOffset:       [3]float64{5, 0, 0},
		ArrivalThreshold:   1.5,
		StepTimeoutSeconds: 10,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	server, err := rsitesting.EnsureServer(ctx, &settings)
	require.NoError(t, err)
	defer func() {
		_ = server.Close(context.Background())
	}()

	dir := t.TempDir()
	mcConfigPath := filepath.Join(dir, "mc-agent-config.json")
	writeJSON(t, mcConfigPath, settings)

	// Deliberately tiny relative to config.Default()'s toy-environment-
	// scale defaults (RolloutSize 64, EpisodeLen 100): every episode here
	// is a real live-bot round trip, not a cheap in-memory toy env step —
	// this is a smoke test proving the loop works, not a meaningful
	// training run.
	trainerConfigPath := filepath.Join(dir, "trainer-config.json")
	writeJSON(t, trainerConfigPath, crlconfig.Settings{
		RolloutSize:   2,
		EpisodeLen:    30,
		Gamma:         0.99,
		LearningRate:  0.05,
		GridSize:      1, // unused against a live rlenv session; see leapfrog.Config.Trainer's own doc comment.
		HiddenSize:    8,
		Seed:          42,
		Workers:       1,
		ClipEpsilon:   0.2,
		EntropyCoef:   0.01,
		ValueCoef:     0.5,
		GAELambda:     0.95,
		PPOEpochs:     1,
		MinibatchSize: 2,
	})

	checkpointDir := t.TempDir()

	start := time.Now()
	err = run([]string{
		"-checkpoint-dir", checkpointDir,
		"-mc-agent-config", mcConfigPath,
		"-trainer-config", trainerConfigPath,
		"-epochs-per-generation", "2",
		"-eval-episodes", "2",
		"-eval-episode-len", "30",
		"-max-rounds", "1",
	})
	elapsed := time.Since(start)
	require.NoError(t, err)
	t.Logf("one full leapfrog round (2 training epochs x 2 rollouts, plus 2x2 eval episodes) completed in %s", elapsed)

	rec, err := lineage.Latest(checkpointDir)
	require.NoError(t, err, "run() must leave at least a generation-0 checkpoint behind")
	t.Logf("latest generation after the run: %d (provenance=%s)", rec.Generation, rec.Provenance)
}

// TestRunCompletesOneRealisticRoundAgainstLiveServerLive is
// docs/plans/08's own stated next step after the tiny smoke test above
// proved the mechanism: get a real wall-clock number for a
// production-sized round, not a deliberately tiny one, before designing
// any parallel-environment scaling around throughput. Gated by its own
// env var (on top of MC_RSI_TRAINER_LIVE_CONFIG) rather than folded into
// the smoke test above, since it's meaningfully slower and the smoke
// test's whole point is being a fast, always-run-when-live check.
//
// "Realistic" here means larger than the tiny smoke test's toy-scale
// settings in every dimension a live bot session actually pays for per
// step (RolloutSize, EpochsPerGeneration, eval episodes/length), but
// deliberately not cRL-go's own config.Default() (RolloutSize 64,
// EpisodeLen 100, PPOEpochs 4, HiddenSize 128) — those defaults are sized
// for cRL-go's toy grid/snake environments, where a Step is an in-memory
// array update, not a real network round trip to a live Minecraft
// server. Scaling every dimension 8x-10x over the smoke test's tiny
// settings gives a genuinely meaningful round (160 real training steps,
// up to 1000 real eval steps) while staying inside a bounded, known test
// timeout rather than guessing at config.Default()'s cost against a live
// session with no prior data point at all.
//
// Passes -auto-reset-origin (main.go): earlier live runs against a
// non-flat, randomly generated world (no fixed seed here — see
// testing.EnsureServer) surfaced a compounding failure mode without it —
// see docs/plans/08-parallel-environments-and-scaling.md's own "Status"
// for the full data (a standable-but-unreachable ground-snapped target,
// and nothing to recover the bot from getting stuck near one once it
// happened, since TargetOffset's own "wherever Reset found the bot"
// workaround just re-poses almost the same target from almost the same
// stuck position every subsequent episode). rlenv's own reachability gate
// (rlenv/walkability.go, mc-agent) closes the first half of that;
// -auto-reset-origin closes the second by teleporting back to the bot's
// real, live-verified spawn position every episode instead.
func TestRunCompletesOneRealisticRoundAgainstLiveServerLive(t *testing.T) {
	if os.Getenv("MC_RSI_TRAINER_LIVE_CONFIG") == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_CONFIG not set; skipping live cmd/rsi-train smoke test")
	}
	if os.Getenv("MC_RSI_TRAINER_LIVE_REALISTIC_ROUND") == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_REALISTIC_ROUND not set; skipping the slower production-scale round timing test")
	}

	settings := mcconfig.Default()
	settings.Connection = mcconfig.ConnectionSettings{
		Address: "127.0.0.1:34599",
		Offline: true,
		Name:    "RSIRealisticRnd", // 15 chars; Minecraft usernames are capped at 16.
	}
	settings.Env = mcconfig.EnvSettings{
		TargetOffset:       [3]float64{5, 0, 0},
		ArrivalThreshold:   1.5,
		StepTimeoutSeconds: 10,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()

	server, err := rsitesting.EnsureServer(ctx, &settings)
	require.NoError(t, err)
	defer func() {
		_ = server.Close(context.Background())
	}()

	dir := t.TempDir()
	mcConfigPath := filepath.Join(dir, "mc-agent-config.json")
	writeJSON(t, mcConfigPath, settings)

	trainerConfigPath := filepath.Join(dir, "trainer-config.json")
	writeJSON(t, trainerConfigPath, crlconfig.Settings{
		RolloutSize:   16,
		EpisodeLen:    60,
		Gamma:         0.99,
		LearningRate:  0.05,
		GridSize:      1, // unused against a live rlenv session; see leapfrog.Config.Trainer's own doc comment.
		HiddenSize:    32,
		Seed:          42,
		Workers:       1,
		ClipEpsilon:   0.2,
		EntropyCoef:   0.01,
		ValueCoef:     0.5,
		GAELambda:     0.95,
		PPOEpochs:     2,
		MinibatchSize: 8,
	})

	checkpointDir := t.TempDir()

	start := time.Now()
	err = run([]string{
		"-checkpoint-dir", checkpointDir,
		"-mc-agent-config", mcConfigPath,
		"-trainer-config", trainerConfigPath,
		"-epochs-per-generation", "10",
		"-eval-episodes", "5",
		"-eval-episode-len", "100",
		"-max-rounds", "1",
		"-auto-reset-origin",
	})
	elapsed := time.Since(start)
	require.NoError(t, err)
	t.Logf("one production-scale leapfrog round (10 training epochs x 16 rollout steps, plus 5x5 eval episodes up to 100 steps each) completed in %s", elapsed)

	rec, err := lineage.Latest(checkpointDir)
	require.NoError(t, err, "run() must leave at least a generation-0 checkpoint behind")
	t.Logf("latest generation after the run: %d (provenance=%s)", rec.Generation, rec.Provenance)
}

// TestRunCompletesOneRealisticRoundAgainstFlatLiveServerLive is the same
// production-scale round as
// TestRunCompletesOneRealisticRoundAgainstLiveServerLive above, but
// against a superflat world (testing.WithExtraEnv(LEVEL_TYPE=FLAT))
// instead of a normally-generated one. Added after that test's own first
// real run: it completed in ~1s with teacher/student reward both exactly
// -1.000 — not a real timing number at all, but every one of its steps
// hitting "[pathfindAndFollow] FindPath error: path not found ... has no
// walkable cell" (this round's fixed TargetOffset happened to land
// somewhere unreachable on that run's randomly generated terrain), so
// every step paid only the -0.01/step time penalty with zero distance
// progress the entire episode. See docs/plans/08's own "Status" for the
// full writeup — that FindPath loop was previously flagged as a minor
// efficiency curiosity, but this run showed it can silently zero out an
// entire round's training signal while still returning err == nil,
// which is worse. A flat world removes the "target lands on
// unreachable terrain" variable so this test's timing reflects real
// per-step network/inference cost, not a pathfinding failure loop; a
// non-flat run is still worth doing again separately once that
// robustness gap itself is addressed (not attempted here).
func TestRunCompletesOneRealisticRoundAgainstFlatLiveServerLive(t *testing.T) {
	if os.Getenv("MC_RSI_TRAINER_LIVE_CONFIG") == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_CONFIG not set; skipping live cmd/rsi-train smoke test")
	}
	if os.Getenv("MC_RSI_TRAINER_LIVE_REALISTIC_ROUND") == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_REALISTIC_ROUND not set; skipping the slower production-scale round timing test")
	}

	settings := mcconfig.Default()
	settings.Connection = mcconfig.ConnectionSettings{
		Address: "127.0.0.1:34600", // distinct port from the non-flat realistic test, so both can run independently without colliding.
		Offline: true,
		Name:    "RSIRealisticFlat", // 16 chars — exactly at Minecraft's username cap.
	}
	settings.Env = mcconfig.EnvSettings{
		TargetOffset:       [3]float64{5, 0, 0},
		ArrivalThreshold:   1.5,
		StepTimeoutSeconds: 10,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()

	server, err := rsitesting.EnsureServer(ctx, &settings, rsitesting.WithExtraEnv(map[string]string{
		"LEVEL_TYPE": "FLAT",
	}))
	require.NoError(t, err)
	defer func() {
		_ = server.Close(context.Background())
	}()

	dir := t.TempDir()
	mcConfigPath := filepath.Join(dir, "mc-agent-config.json")
	writeJSON(t, mcConfigPath, settings)

	trainerConfigPath := filepath.Join(dir, "trainer-config.json")
	writeJSON(t, trainerConfigPath, crlconfig.Settings{
		RolloutSize:   16,
		EpisodeLen:    60,
		Gamma:         0.99,
		LearningRate:  0.05,
		GridSize:      1, // unused against a live rlenv session; see leapfrog.Config.Trainer's own doc comment.
		HiddenSize:    32,
		Seed:          42,
		Workers:       1,
		ClipEpsilon:   0.2,
		EntropyCoef:   0.01,
		ValueCoef:     0.5,
		GAELambda:     0.95,
		PPOEpochs:     2,
		MinibatchSize: 8,
	})

	checkpointDir := t.TempDir()

	start := time.Now()
	err = run([]string{
		"-checkpoint-dir", checkpointDir,
		"-mc-agent-config", mcConfigPath,
		"-trainer-config", trainerConfigPath,
		"-epochs-per-generation", "10",
		"-eval-episodes", "5",
		"-eval-episode-len", "100",
		"-max-rounds", "1",
	})
	elapsed := time.Since(start)
	require.NoError(t, err)
	t.Logf("one production-scale leapfrog round against a flat world (10 training epochs x 16 rollout steps, plus 5x5 eval episodes up to 100 steps each) completed in %s", elapsed)

	rec, err := lineage.Latest(checkpointDir)
	require.NoError(t, err, "run() must leave at least a generation-0 checkpoint behind")
	t.Logf("latest generation after the run: %d (provenance=%s)", rec.Generation, rec.Provenance)
}

func writeJSON(t *testing.T, path string, v any) {
	t.Helper()
	data, err := json.MarshalIndent(v, "", "  ")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(path, data, 0o644))
}
