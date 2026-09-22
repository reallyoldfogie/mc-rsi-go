// Operational launcher, not a real test -- kept across
// sessions as a reusable tool for launching one shared-server-mode
// Minecraft server for a long, unattended training run (deliberately
// does NOT close the server, so the container outlives this process).
// Writes the launched connection settings to
// MC_RSI_TRAINER_LAUNCH_CONFIG_OUT for a separate orchestration script
// (e.g. cmd/rsi-train) to consume. Every setting is overridable via
// environment variable so the same launcher serves any run without
// editing this file - see each os.Getenv call below for its default.
package testing

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	mcconfig "github.com/reallyoldfogie/mc-agent/config"
	"github.com/stretchr/testify/require"
)

func TestLaunchLongRunSharedServer(t *testing.T) {
	if os.Getenv("MC_RSI_TRAINER_LAUNCH_LONG_RUN") == "" {
		t.Skip("MC_RSI_TRAINER_LAUNCH_LONG_RUN not set; skipping throwaway server launcher")
	}
	outPath := os.Getenv("MC_RSI_TRAINER_LAUNCH_CONFIG_OUT")
	require.NotEmpty(t, outPath, "MC_RSI_TRAINER_LAUNCH_CONFIG_OUT must name where to write the launched config")

	getenvOr := func(key, fallback string) string {
		if v := os.Getenv(key); v != "" {
			return v
		}
		return fallback
	}

	settings := mcconfig.Default()
	settings.Connection = mcconfig.ConnectionSettings{
		Address: getenvOr("MC_RSI_TRAINER_LAUNCH_ADDRESS", "127.0.0.1:34670"),
		Offline: true,
		Version: getenvOr("MC_RSI_TRAINER_LAUNCH_VERSION", "1.21.5"),
		Name:    getenvOr("MC_RSI_TRAINER_LAUNCH_NAME", "RSILong"),
	}
	settings.RCON = mcconfig.RCONSettings{
		Password: getenvOr("MC_RSI_TRAINER_LAUNCH_RCON_PASSWORD", "rsi-trainer-long-run-pw"),
	}
	settings.Env = mcconfig.EnvSettings{
		TargetOffset:       [3]float64{5, 0, 0},
		ArrivalThreshold:   1.5,
		StepTimeoutSeconds: 10,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	server, err := EnsureServer(ctx, &settings,
		WithExtraEnv(map[string]string{"LEVEL_TYPE": "FLAT"}),
		WithViewDistance(4),
	)
	require.NoError(t, err)
	t.Logf("launched server at %s:%d -- deliberately NOT closing; container persists after this process exits", server.Host, server.Port)

	data, err := json.MarshalIndent(settings, "", "  ")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(outPath, data, 0o644))
	t.Logf("wrote launched connection settings to %s", outPath)
}
