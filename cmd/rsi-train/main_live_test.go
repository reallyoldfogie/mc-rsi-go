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

// TestRunCompletesOneRoundAgainstFourFlatLiveServersLive is
// docs/plans/08-parallel-environments-and-scaling.md's first real,
// live confirmation that -parallel-envs actually drives N concurrent
// live bot sessions against N separate servers, not just that the code
// compiles — mirroring TestRunCompletesOneRealisticRoundAgainstFlatLiveServerLive's
// shape (flat world, isolating step-cost from terrain variance) but
// deliberately kept tiny (see the trainer config below): this is a fast
// first confirmation the 4-server wiring works at all, not a
// meaningful training run — a longer, production-scale 4-env run is
// worth doing only after this passes cleanly (four concurrent live
// Minecraft server containers is a real jump in Docker/CPU/RAM usage
// versus any single-server run this repo has attempted so far). Gated
// by its own env var (on top of MC_RSI_TRAINER_LIVE_CONFIG), separate
// from MC_RSI_TRAINER_LIVE_REALISTIC_ROUND, since this is a different
// kind of "heavier" (more servers, not more steps).
//
// Writes the pristine, pre-EnsureServers base settings to
// -mc-agent-config, not any of testing.EnsureServers' own per-index
// derived copies: run()'s own connectEnvironments (main.go) re-derives
// all 4 instances from whatever -mc-agent-config points at via the same
// pkg/parallelenv.DeriveSettings(_, i, 4) call EnsureServers already
// used to launch these 4 containers. Starting both derivations from the
// same base produces byte-identical addresses/RCON/usernames per index;
// starting from an already-derived config instead would double-apply
// DeriveSettings (e.g. re-suffixing an already-suffixed RCON password),
// producing values that don't match any real container. This is also
// why base.RCON.Password is set explicitly below, rather than left for
// EnsureServer to generate randomly per container: DeriveSettings only
// suffixes a non-empty base password identically at both call sites: an
// empty one would make each container's real password unreproducible
// from run()'s own re-derivation.
func TestRunCompletesOneRoundAgainstFourFlatLiveServersLive(t *testing.T) {
	if os.Getenv("MC_RSI_TRAINER_LIVE_CONFIG") == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_CONFIG not set; skipping live cmd/rsi-train smoke test")
	}
	if os.Getenv("MC_RSI_TRAINER_LIVE_PARALLEL_ENVS") == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_PARALLEL_ENVS not set; skipping the heavier 4-server parallel-environment test")
	}

	const n = 4
	base := mcconfig.Default()
	base.Connection = mcconfig.ConnectionSettings{
		Address: "127.0.0.1:34620", // distinct base port from every other live test in this file, so all can run independently without colliding.
		Offline: true,
		Version: "1.21.5",
		Name:    "RSIParallel", // 11 chars; DeriveUsername's "-N" suffix (N is 0-3 here) stays well within Minecraft's 16-character cap.
	}
	base.RCON = mcconfig.RCONSettings{
		Password: "rsi-trainer-parallel-test-pw", // explicit and non-empty — see this test's own doc comment for why.
	}
	base.Env = mcconfig.EnvSettings{
		TargetOffset:       [3]float64{5, 0, 0},
		ArrivalThreshold:   1.5,
		StepTimeoutSeconds: 10,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	servers, _, err := rsitesting.EnsureServers(ctx, base, n, rsitesting.WithExtraEnv(map[string]string{
		"LEVEL_TYPE": "FLAT",
	}))
	require.NoError(t, err)
	defer func() {
		for _, s := range servers {
			_ = s.Close(context.Background())
		}
	}()

	dir := t.TempDir()
	mcConfigPath := filepath.Join(dir, "mc-agent-config.json")
	writeJSON(t, mcConfigPath, base)

	// Deliberately tiny — see this test's own doc comment: proving the
	// 4-server wiring works end to end, not producing a meaningfully
	// trained policy.
	trainerConfigPath := filepath.Join(dir, "trainer-config.json")
	writeJSON(t, trainerConfigPath, crlconfig.Settings{
		RolloutSize:   4,
		EpisodeLen:    15,
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
		MinibatchSize: 4,
	})

	checkpointDir := t.TempDir()

	start := time.Now()
	err = run([]string{
		"-checkpoint-dir", checkpointDir,
		"-mc-agent-config", mcConfigPath,
		"-trainer-config", trainerConfigPath,
		"-epochs-per-generation", "1",
		"-eval-episodes", "1",
		"-eval-episode-len", "15",
		"-max-rounds", "1",
		"-parallel-envs", "4",
	})
	elapsed := time.Since(start)
	require.NoError(t, err)
	t.Logf("one small 4-parallel-environment leapfrog round completed in %s", elapsed)

	rec, err := lineage.Latest(checkpointDir)
	require.NoError(t, err, "run() must leave at least a generation-0 checkpoint behind")
	t.Logf("latest generation after the run: %d (provenance=%s)", rec.Generation, rec.Provenance)
}

// TestRunCompletesOneRoundAgainstSharedServerLive is
// docs/plans/08-parallel-environments-and-scaling.md's live confirmation
// of the shared-server parallel-training design: N bots against ONE
// server, each confined to its own working area
// (pkg/parallelenv.WorkingAreaOffset), instead of N separate servers.
// The isolation itself (does a low view distance plus enough separation
// actually keep bots from ever perceiving each other) was already
// rigorously confirmed live by testing/spike_shared_server_test.go
// (grepped both bots' full logs for any cross-reference — zero hits in
// either direction). This test's narrower job is proving the real
// -shared-server/-shared-server-separation-chunks CLI flags and
// connectEnvironments wiring work end to end through the actual run()
// entrypoint, not just via direct rlenv.Environment calls the way the
// spike exercised it.
//
// Sets settings.RCON.Password explicitly before launch: -shared-server
// requires RCON, and unlike the N-servers test
// (TestRunCompletesOneRoundAgainstFourFlatLiveServersLive) there's no
// double-derivation concern here to design around — every bot uses the
// exact same connection details in shared-server mode, so whatever
// EnsureServer mutates into settings can be written to the config file
// as-is.
func TestRunCompletesOneRoundAgainstSharedServerLive(t *testing.T) {
	if os.Getenv("MC_RSI_TRAINER_LIVE_CONFIG") == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_CONFIG not set; skipping live cmd/rsi-train smoke test")
	}
	if os.Getenv("MC_RSI_TRAINER_LIVE_SHARED_SERVER") == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_SHARED_SERVER not set; skipping the shared-server parallel-environment test")
	}

	settings := mcconfig.Default()
	settings.Connection = mcconfig.ConnectionSettings{
		Address: "127.0.0.1:34650", // distinct base port from every other live test in this file, so all can run independently without colliding.
		Offline: true,
		Version: "1.21.5",
		Name:    "RSIShared", // 9 chars; DeriveUsername's "-N" suffix stays well within Minecraft's 16-character cap.
	}
	settings.RCON = mcconfig.RCONSettings{
		Password: "rsi-trainer-shared-server-test-pw", // explicit and non-empty -- -shared-server requires RCON.
	}
	settings.Env = mcconfig.EnvSettings{
		TargetOffset:       [3]float64{5, 0, 0},
		ArrivalThreshold:   1.5,
		StepTimeoutSeconds: 10,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	server, err := rsitesting.EnsureServer(ctx, &settings,
		rsitesting.WithExtraEnv(map[string]string{"LEVEL_TYPE": "FLAT"}),
		rsitesting.WithViewDistance(4),
	)
	require.NoError(t, err)
	defer func() {
		_ = server.Close(context.Background())
	}()

	dir := t.TempDir()
	mcConfigPath := filepath.Join(dir, "mc-agent-config.json")
	writeJSON(t, mcConfigPath, settings)

	// Deliberately tiny -- see this test's own doc comment: proving the
	// shared-server wiring works end to end, not producing a
	// meaningfully trained policy.
	trainerConfigPath := filepath.Join(dir, "trainer-config.json")
	writeJSON(t, trainerConfigPath, crlconfig.Settings{
		RolloutSize:   4,
		EpisodeLen:    15,
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
		MinibatchSize: 4,
	})

	checkpointDir := t.TempDir()

	start := time.Now()
	err = run([]string{
		"-checkpoint-dir", checkpointDir,
		"-mc-agent-config", mcConfigPath,
		"-trainer-config", trainerConfigPath,
		"-epochs-per-generation", "1",
		"-eval-episodes", "1",
		"-eval-episode-len", "15",
		"-max-rounds", "1",
		"-parallel-envs", "2",
		"-shared-server",
		"-shared-server-separation-chunks", "16",
	})
	elapsed := time.Since(start)
	require.NoError(t, err)
	t.Logf("one small 2-bot shared-server leapfrog round completed in %s", elapsed)

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
