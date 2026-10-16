package main

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	crlconfig "github.com/reallyoldfogie/cRL-go/pkg/config"
	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	"github.com/stretchr/testify/require"

	"github.com/reallyoldfogie/mc-rsi-trainer/pkg/lineage"
	rsitesting "github.com/reallyoldfogie/mc-rsi-trainer/testing"
)

// TestWorldStreamLive is GET /agents/world's own live smoke test, mirroring
// TestDashboardSmokeAgainstSharedServerLive's shape: drives the real run()
// entrypoint against a dedicated, auto-launched Minecraft server
// (rsitesting.EnsureServer — a distinct port from every other live test in
// this package, never the live production server), then connects to the
// stream for agent 0 and confirms it actually emits both a "position" event
// and a "blocks" event carrying at least one real, non-air block — the
// thing worldstream.go exists to prove, since agent.FindAllVisibleBlocksInSphere
// itself is already mc-agent's own tested code.
func TestWorldStreamLive(t *testing.T) {
	if os.Getenv("MC_RSI_TRAINER_LIVE_CONFIG") == "" {
		t.Skip("MC_RSI_TRAINER_LIVE_CONFIG not set; skipping live worldstream smoke test")
	}

	settings := mcconfig.Default()
	settings.Connection = mcconfig.ConnectionSettings{
		Address: "127.0.0.1:34655", // distinct from every other live test's own fixed port in this package.
		Offline: true,
		Version: "1.21.5",
		Name:    "RSIWorldStream",
	}
	settings.RCON = mcconfig.RCONSettings{
		Address:  "127.0.0.1:25579",
		Password: "rsi-trainer-worldstream-smoke-test-pw",
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
	const metricsAddr = "127.0.0.1:9402" // distinct from a real run's default :9400 and the other live tests' own metrics ports.

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
			"-parallel-envs", "1",
			"-shared-server",
			"-shared-server-separation-chunks", "16",
			"-metrics-addr", ":9402",
		})
	}()

	// Give run() a little time to connect its bot and start serving before
	// the stream's own 60s read window begins - a connection attempt made
	// before the metrics server is up would just fail outright, not retry.
	time.Sleep(10 * time.Second)

	sawPosition, sawNonEmptyBlocks := readWorldStreamEvents(t, "http://"+metricsAddr+"/agents/world?index=0", 60*time.Second)
	require.True(t, sawPosition, "never saw a 'position' event on the world stream")
	require.True(t, sawNonEmptyBlocks, "never saw a 'blocks' event with at least one visible block")

	select {
	case err := <-runDone:
		require.NoError(t, err)
	case <-time.After(60 * time.Second):
		t.Fatal("run() did not finish within the test's own polling window after the stream check")
	}
	rec, lErr := lineage.Latest(checkpointDir)
	require.NoError(t, lErr, "run() must leave at least a generation-0 checkpoint behind")
	t.Logf("run finished; latest generation %d (provenance=%s)", rec.Generation, rec.Provenance)
}

// readWorldStreamEvents connects to a worldStreamHandler SSE endpoint and
// reads lines until it has seen at least one "position" event and one
// "blocks" event with a non-empty blocks array, or timeout elapses -
// whichever comes first. The request's own context cancels the connection
// on timeout, which unblocks the scanner with an error rather than hanging.
func readWorldStreamEvents(t *testing.T, url string, timeout time.Duration) (sawPosition, sawNonEmptyBlocks bool) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	require.NoError(t, err)
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var event struct {
			Type   string `json:"type"`
			Blocks []any  `json:"blocks"`
		}
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			continue
		}
		t.Logf("worldstream event: %s", payload)
		switch event.Type {
		case "position":
			sawPosition = true
		case "blocks":
			if len(event.Blocks) > 0 {
				sawNonEmptyBlocks = true
			}
		case "gone":
			t.Fatalf("worldstream reported the agent gone before any data arrived: %s", payload)
		}
		if sawPosition && sawNonEmptyBlocks {
			return
		}
	}
	if err := scanner.Err(); err != nil {
		t.Logf("worldstream scan ended: %v", err)
	}
	return
}
