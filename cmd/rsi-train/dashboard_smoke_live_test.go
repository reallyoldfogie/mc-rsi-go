package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
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

// TestDashboardSmokeAgainstSharedServerLive is a throwaway manual smoke
// test for the GET /agents + /dashboard dashboard (not meant to stay in
// the tree): drives the real run() entrypoint against a dedicated,
// auto-launched Minecraft server (rsitesting.EnsureServer — a different
// container/port than any developer's own live training run), with its
// own -metrics-addr so it can never collide with one already using the
// default :9400. While run() is in flight, it polls /agents and
// /dashboard from a second goroutine and logs what it sees.
func TestDashboardSmokeAgainstSharedServerLive(t *testing.T) {
	if os.Getenv("MC_RSI_TRAINER_LIVE_CONFIG") == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_CONFIG not set; skipping live cmd/rsi-train smoke test")
	}

	settings := mcconfig.Default()
	settings.Connection = mcconfig.ConnectionSettings{
		Address: "127.0.0.1:34651", // distinct from every other live test's own fixed port in this package.
		Offline: true,
		Version: "1.21.5",
		Name:    "RSIDashSmoke",
	}
	settings.RCON = mcconfig.RCONSettings{
		// Explicit, distinct host RCON port: EnsureServer's own
		// resolveRCONAddress defaults to Minecraft's conventional 25575
		// when RCON.Address is empty, which collides with any other
		// server already using that default (e.g. a developer's own live
		// training run's container) — exactly the interference this
		// smoke test exists to avoid.
		Address:  "127.0.0.1:25576",
		Password: "rsi-trainer-dashboard-smoke-test-pw",
	}
	settings.Env = mcconfig.EnvSettings{
		TargetOffset:       [3]float64{5, 0, 0},
		ArrivalThreshold:   1.5,
		StepTimeoutSeconds: 10,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	server, err := rsitesting.EnsureServer(ctx, &settings,
		rsitesting.WithExtraEnv(map[string]string{"LEVEL_TYPE": "FLAT", "GENERATOR_SETTINGS": rsitesting.GeneratorSettingsDeepStoneBand}),
		rsitesting.WithViewDistance(4),
	)
	require.NoError(t, err)
	defer func() {
		_ = server.Close(context.Background())
	}()

	dir := t.TempDir()
	mcConfigPath := filepath.Join(dir, "mc-agent-config.json")
	writeJSON(t, mcConfigPath, settings)

	trainerConfigPath := filepath.Join(dir, "trainer-config.json")
	writeJSON(t, trainerConfigPath, crlconfig.Settings{
		RolloutSize:   4,
		EpisodeLen:    15,
		Gamma:         0.99,
		LearningRate:  0.05,
		GridSize:      1,
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
	const metricsAddr = "127.0.0.1:9401" // distinct from a real run's default :9400.

	runDone := make(chan error, 1)
	go func() {
		runDone <- run([]string{
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
			"-metrics-addr", ":9401",
		})
	}()

	sawAgents, sawDashboard := false, false
	deadline := time.Now().Add(90 * time.Second)
	for time.Now().Before(deadline) {
		select {
		case err := <-runDone:
			require.NoError(t, err)
			rec, lErr := lineage.Latest(checkpointDir)
			require.NoError(t, lErr, "run() must leave at least a generation-0 checkpoint behind")
			t.Logf("run finished; latest generation %d (provenance=%s)", rec.Generation, rec.Provenance)
			require.True(t, sawAgents, "never got a successful GET /agents response while run() was live")
			require.True(t, sawDashboard, "never got a successful GET /dashboard response while run() was live")
			return
		case <-time.After(2 * time.Second):
		}

		if body, ok := httpGet(t, "http://"+metricsAddr+"/agents"); ok {
			var snap []map[string]any
			if err := json.Unmarshal(body, &snap); err == nil {
				sawAgents = true
				t.Logf("GET /agents: %s", body)
			}
		}
		if body, ok := httpGet(t, "http://"+metricsAddr+"/dashboard"); ok {
			sawDashboard = true
			_ = body
		}
	}
	t.Fatal("run() did not finish within the test's own 90s polling window")
}

func httpGet(t *testing.T, url string) ([]byte, bool) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		return nil, false
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil || resp.StatusCode != http.StatusOK {
		return nil, false
	}
	return body, true
}
