// Package testing holds this repo's live-server integration tests —
// tests that need a real, reachable Minecraft server plus mc-agent bot
// credentials, as opposed to the toy-environment unit tests that live
// alongside the code they test (e.g. pkg/leapfrog/leapfrog_test.go).
// Mirrors mc-agent's own testing/ package for the same reason mc-agent
// uses it: keeping every live-server test in one findable place.
//
// Deliberately gated by an environment variable
// (MC_RSI_TRAINER_LIVE_CONFIG below) rather than a Go build tag. An
// earlier version of this file used a build tag, but build-tag-excluded
// files are invisible to IDE tooling (gopls) by default, so syntax and
// type errors in them go undiscovered until someone remembers to build
// with the tag — an env-var check inside an always-compiled test
// achieves the same "skip by default" behavior while staying visible to
// normal editing/type-checking. See mc-agent's own
// testing/rl_train_test.go (MCAGENT_LONG_RL_TRAIN_TEST) for the
// precedent this follows.
package testing

import (
	"context"
	"fmt"
	"math/rand/v2"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	crlconfig "github.com/reallyoldfogie/cRL-go/pkg/config"

	"github.com/reallyoldfogie/mc-agent/actions"
	"github.com/reallyoldfogie/mc-agent/agent"
	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	_ "github.com/reallyoldfogie/mc-agent/handler_versions" // registers version-specific packet handlers
	"github.com/reallyoldfogie/mc-agent/models"
	"github.com/reallyoldfogie/mc-agent/rlenv"
	"github.com/reallyoldfogie/mc-agent/utils"
	rofutils "github.com/reallyoldfogie/mc-bot-go/utils"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/leapfrog"
)

// TestRoundAgainstLiveRlenv runs one real leapfrog.Round against a live
// mc-agent bot session, satisfying
// docs/plans/03-single-task-leapfrog-evaluation-loop.md's "Done when"
// integration-test requirement: "runs one real round against rlenv end
// to end." Skipped unless MC_RSI_TRAINER_LIVE_CONFIG names a config file
// (mc-agent's own config.Settings JSON shape — see mc-agent's
// cmd/rl-train and configs/config.json for a worked example). This test
// loads it with mc-agent's own config.Load/config.ApplyEnv (MCAGENT_
// env-var prefix), exactly like cmd/rl-train does, so any config
// file/environment already set up for `rl-train` works here too. Does
// not require a server to already be running at config's
// connection.address: EnsureServer (mcserver.go) checks first and
// launches one via Docker if nothing answers, torn down again afterward
// unless MC_RSI_TRAINER_KEEP_SERVER is set. Run explicitly via:
//
//	MC_RSI_TRAINER_LIVE_CONFIG=/path/to/config.json go test ./testing/... -run TestRoundAgainstLiveRlenv -v
func TestRoundAgainstLiveRlenv(t *testing.T) {
	configPath := os.Getenv("MC_RSI_TRAINER_LIVE_CONFIG")
	if configPath == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_CONFIG not set; skipping live rlenv integration test (see this package's doc comment)")
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

	registry := actions.NewRegistry()
	env, err := rlenv.New(liveAgent, registry, settings.Env.ToRlenvConfig())
	require.NoError(t, err)

	rng := rand.New(rand.NewPCG(1, 2))
	const hiddenSize = 16
	teacherParams := actorcritic.NewParams(rng, env.ObservationSize(), hiddenSize, env.ActionSpace())

	cfg := leapfrog.Config{
		Trainer: crlconfig.Settings{
			RolloutSize: 2,
			// Deliberately short: this test's job is to prove the
			// wiring works end to end against a real bot session, not
			// to produce a meaningfully trained policy — a real
			// leapfrog run (docs/plans/04's cmd/rsi-train) uses far
			// larger values.
			EpisodeLen:    50,
			Gamma:         0.99,
			LearningRate:  0.01,
			GridSize:      1, // unused against a live rlenv session; see leapfrog.Config.Trainer's own doc comment.
			HiddenSize:    hiddenSize,
			Seed:          1,
			Workers:       1,
			ClipEpsilon:   0.2,
			EntropyCoef:   0.01,
			ValueCoef:     0.5,
			GAELambda:     0.95,
			PPOEpochs:     1,
			MinibatchSize: 4,
		},
		EpochsPerGeneration: 1,
		EvalEpisodes:        1,
		EvalEpisodeLen:      50,
	}

	result, err := leapfrog.Round(ctx, env, teacherParams, cfg, rng)
	require.NoError(t, err)
	t.Logf("live round result: teacher=%.3f student=%.3f studentWon=%v", result.TeacherReward, result.StudentReward, result.StudentWon)
}

// connectAgent establishes one live bot session from settings. This is a
// trimmed copy of mc-agent's own cmd/rl-train/main.go connectAgent
// (auth resolution, version auto-detect, RCON dial, agent.New/Init/Start)
// — trimmed of packet-log-file rotation and skin fetching, neither of
// which this integration test needs, to avoid pulling their extra
// dependencies (lumberjack, an HTTP skin cache) into this repo just for
// an opt-in test. If mc-agent's own connectAgent shape changes
// meaningfully, re-sync this copy against it.
func connectAgent(ctx context.Context, settings mcconfig.Settings) (models.Agent, error) {
	conn := settings.Connection

	auth, err := agent.ResolveAuth(conn.Offline, conn.Name, conn.UUID, conn.Token, settings.Auth)
	if err != nil {
		return nil, err
	}

	version := conn.Version
	if version == "" {
		detectedVersion, _, err := rofutils.CheckServerVersion(conn.Address, 0)
		if err != nil {
			return nil, fmt.Errorf("auto-detect version from %s: %w", conn.Address, err)
		}
		version = detectedVersion
	}

	rcon, err := agent.DialRCON(ctx, settings.RCON.Address, settings.RCON.Password)
	if err != nil {
		return nil, err
	}

	logLevel, err := utils.ParseLevel(settings.Logging.Level)
	if err != nil {
		return nil, err
	}

	cfg := models.AgentConfig{
		Name:             auth.Name,
		Address:          conn.Address,
		Version:          version,
		Auth:             auth,
		MCDataGenPath:    conn.MCDataGenPath,
		MCProtocolGoPath: conn.MCProtocolGoPath,
		LogLevel:         logLevel,
		RCON:             rcon,
	}

	a, err := agent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("creating agent: %w", err)
	}
	if err := a.Init(ctx); err != nil {
		return nil, fmt.Errorf("init: %w", err)
	}
	if err := a.Start(ctx); err != nil {
		return nil, fmt.Errorf("start: %w", err)
	}
	return a, nil
}
